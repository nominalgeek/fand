"""fand-calibrate: PWM perturbation sweep + zones.yaml generation.

Three-phase, operator-supervised. The output is a zones.yaml in the new
sensors+fans schema with phase-3-measured cooling weights baked into each
fan's `cools:` dict — review and tune target_c/critical_c, but it lands
ready to use.

  1. PWM ↔ fan-tach mapping (which physical fan responds to each PWM channel)
  2. pwm_min per channel (lowest PWM at which the fan still spins)
  3. Sensor attribution (which sensors warm up when each fan is throttled —
     produces the seed cooling weights that bootstrap the daemon's control law)

Run with:  fand-calibrate     (after `sudo install.sh`)
Requires root (writes to /sys/class/hwmon).

Phases 1 and 2 want the system IDLE so probes aren't masked by load-driven
heat. Phase 3 wants the OPPOSITE — typical sustained workload so heat sources
are actively dissipating and dropping a fan produces a measurable delta.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .actuators import Actuators
from .sensors import Sensors, find_hwmon_by_name

log = logging.getLogger("fand-calibrate")


# Phase 3 (attribution) defaults
ATTRIBUTION_BASELINE_PWM = 200
ATTRIBUTION_BASELINE_SETTLE_S = 60.0
ATTRIBUTION_DROP_DURATION_S = 90.0
ATTRIBUTION_RECOVERY_S = 30.0
ATTRIBUTION_MIN_DELTA_C = 0.5
ATTRIBUTION_SAMPLES = 5
ATTRIBUTION_SAMPLE_INTERVAL_S = 2.0


def _read_tach(hw: Path, fan_key: str) -> int:
    try:
        return int((hw / f"{fan_key}_input").read_text().strip())
    except (OSError, ValueError):
        return 0


def _settle(seconds: float) -> None:
    log.info("  settle %.0fs...", seconds)
    time.sleep(seconds)


def _check_no_daemon_running() -> None:
    """Refuse to run if fand.service is active — would fight every PWM write."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", "fand.service"],
            check=False,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return
    if result.returncode == 0:
        log.error("fand.service is currently active and will fight every PWM write.")
        log.error("Stop the daemon first, run this, then start it again:")
        log.error("    sudo systemctl stop fand")
        log.error("    sudo fand-calibrate")
        log.error("    sudo systemctl start fand")
        sys.exit(3)


def _sensor_name(chip: str, label: str) -> str:
    """Slug for sensor key. nvme[0]/Composite → nvme0_composite."""
    out: list[str] = []
    prev_under = False
    for c in f"{chip}_{label}":
        if c.isalnum():
            out.append(c.lower())
            prev_under = False
        elif not prev_under:
            out.append("_")
            prev_under = True
    return "".join(out).strip("_") or "sensor"


def _collect_sensors_catalog(s: Sensors) -> dict[str, tuple[str, str]]:
    """All currently-readable sensors that have valid (non-zero, non-bogus)
    temperature readings. The Sensors layer already filters readings outside
    [-50, 150] °C; we additionally skip exactly-0.0 readings which are typical
    of unwired PCH chip thermistors that some boards expose.

    Returns: {sensor_name → (chip, label)}.
    """
    obs = s.read_all()
    catalog: dict[str, tuple[str, str]] = {}
    for chip, temps in obs.hwmon.items():
        for label, val in temps.items():
            if val == 0.0:
                continue
            catalog[_sensor_name(chip, label)] = (chip, label)
    if obs.gpu and "temp_c" in obs.gpu:
        catalog[_sensor_name("gpu", "temp_c")] = ("gpu", "temp_c")
    return catalog


def _average_temps(
    sensors_obj: Sensors,
    samples: int = ATTRIBUTION_SAMPLES,
    interval: float = ATTRIBUTION_SAMPLE_INTERVAL_S,
) -> dict[tuple[str, str], float]:
    """Mean temperature per (chip, label) across `samples` polls. Includes GPU
    via nvidia-smi (special chip key 'gpu')."""
    sums: dict[tuple[str, str], list[float]] = {}
    for i in range(samples):
        obs = sensors_obj.read_all()
        for chip, temps in obs.hwmon.items():
            for label, val in temps.items():
                sums.setdefault((chip, label), []).append(val)
        if obs.gpu and "temp_c" in obs.gpu:
            sums.setdefault(("gpu", "temp_c"), []).append(obs.gpu["temp_c"])
        if i < samples - 1:
            time.sleep(interval)
    return {k: sum(v) / len(v) for k, v in sums.items() if v}


def _sensor_defaults(chip: str, label: str) -> tuple[float, float]:
    """Conservative (target_c, critical_c) per hardware class. Operator can
    tune per machine in zones.yaml."""
    # AMD CPU die (Ryzen throttles ~95°C)
    if chip == "k10temp":
        return (70.0, 90.0)
    # nct sensors mirroring CPU diode or board socket
    if chip.startswith("nct") and label in ("TSI0_TEMP", "CPUTIN"):
        return (70.0, 90.0)
    # NVIDIA GPU via nvidia-smi (Blackwell throttles ~87°C)
    if chip == "gpu":
        return (80.0, 90.0)
    # AMD GPU / iGPU
    if chip == "amdgpu":
        return (75.0, 90.0)
    # DDR5 SPD5118 — JEDEC HIGH alarm at 55°C
    if chip.startswith("spd5118"):
        return (45.0, 55.0)
    # NVMe drives — Sensor 2 (NAND) runs hotter than Composite/controller
    if chip == "nvme" or chip.startswith("nvme["):
        if label == "Sensor 2":
            return (65.0, 80.0)
        return (60.0, 75.0)
    # nct ambient probes
    if chip.startswith("nct"):
        if label == "AUXTIN4":  # external room-ambient probe
            return (30.0, 45.0)
        if label == "SMBUSMASTER 0":
            return (65.0, 80.0)
        return (45.0, 65.0)
    return (60.0, 80.0)


def map_pwm_to_tach(actuators: Actuators, hw: Path, fan_keys: list[str]) -> dict[str, str | None]:
    """For each PWM channel, find which fan tach responds the most.

    Strategy: hold every other PWM at 160; bounce target 80 → 220 → 80;
    record tach with biggest swing. Returns {pwm_channel: fan_key_or_None}.
    """
    mapping: dict[str, str | None] = {}
    channels = list(actuators.handles.keys())

    for target in channels:
        log.info("probing %s …", target)
        for ch in channels:
            actuators.set_pwm(ch, 160)
        _settle(3.0)

        actuators.set_pwm(target, 80)
        _settle(4.0)
        low = {k: _read_tach(hw, k) for k in fan_keys}

        actuators.set_pwm(target, 220)
        _settle(4.0)
        high = {k: _read_tach(hw, k) for k in fan_keys}

        deltas = {k: high[k] - low[k] for k in fan_keys}
        log.info("  deltas: %s", deltas)

        best = max(deltas.items(), key=lambda kv: kv[1])
        if best[1] < 100:
            log.info("  -> %s: no fan responded (deltas all small)", target)
            mapping[target] = None
        else:
            mapping[target] = best[0]
            log.info("  -> %s drives %s (Δ %d RPM)", target, best[0], best[1])

        actuators.set_pwm(target, 160)

    return mapping


def find_pwm_min(actuators: Actuators, hw: Path, channel: str, tach: str | None) -> int:
    """Step PWM down until the fan stalls. Returns the lowest spinning PWM
    plus a small safety margin, or 200 if the fan never spun."""
    if not tach:
        return 80

    actuators.set_pwm(channel, 200)
    _settle(3.0)

    last_spinning: int | None = None
    for v in range(100, 19, -10):
        actuators.set_pwm(channel, v)
        _settle(3.0)
        rpm = _read_tach(hw, tach)
        log.info("  %s @ pwm=%d -> %d RPM", channel, v, rpm)
        if rpm < 200:
            break
        last_spinning = v
    if last_spinning is None:
        log.warning(
            "%s: fan never spun in probe range (100..20) — returning safe-high "
            "default 200; review pwm_min in zones.yaml",
            channel,
        )
        return 200
    return min(100, last_spinning + 20)


def attribute_fans(
    actuators: Actuators,
    sensors_obj: Sensors,
    pwm_to_tach: dict[str, str | None],
    pwm_min: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Phase 3: empirical sensor attribution.

    Holds all managed fans at a high baseline, then walks each one: drops to
    pwm_min for the observation window, measures ΔT per sensor against the
    pre-drop steady state, retains sensors with Δ ≥ noise threshold. Returns
    {fan_name → {sensor_name → ΔT_c}}. These ΔT values become the seed cooling
    weights the daemon uses until it learns runtime coefficients.
    """
    managed = [ch for ch, tach in pwm_to_tach.items() if tach]
    attribution: dict[str, dict[str, float]] = {ch: {} for ch in pwm_to_tach}
    if not managed:
        return attribution

    log.info("phase 3: empirical sensor attribution")
    log.info("  ** this phase wants a typical workload running so heat sources")
    log.info("  ** are actively dissipating. an idle system gives weak signal.")
    log.info("  setting all managed fans to baseline=%d", ATTRIBUTION_BASELINE_PWM)
    for ch in managed:
        actuators.set_pwm(ch, ATTRIBUTION_BASELINE_PWM)
    _settle(ATTRIBUTION_BASELINE_SETTLE_S)

    baseline = _average_temps(sensors_obj)
    log.info("  baseline (top 5 hottest):")
    for (chip, label), val in sorted(baseline.items(), key=lambda kv: -kv[1])[:5]:
        log.info("    %s/%s: %.1f°C", chip, label, val)

    for target in managed:
        log.info(
            "attributing %s: dropping pwm=%d→%d for %.0fs ...",
            target, ATTRIBUTION_BASELINE_PWM, pwm_min[target], ATTRIBUTION_DROP_DURATION_S,
        )
        actuators.set_pwm(target, pwm_min[target])
        _settle(ATTRIBUTION_DROP_DURATION_S)

        post = _average_temps(sensors_obj)
        hits: list[tuple[str, str, float]] = []  # (chip, label, delta)
        for key, post_val in post.items():
            base_val = baseline.get(key)
            if base_val is None:
                continue
            d = post_val - base_val
            if d >= ATTRIBUTION_MIN_DELTA_C:
                hits.append((key[0], key[1], d))
        hits.sort(key=lambda x: -x[2])
        if hits:
            summary = ", ".join(f"{c}/{lbl} (Δ{d:+.1f}°C)" for c, lbl, d in hits[:3])
            log.info("  -> %s cools: %s", target, summary)
        else:
            log.info(
                "  -> %s: no sensor showed Δ ≥ %.1f°C (cools: will be empty)",
                target, ATTRIBUTION_MIN_DELTA_C,
            )
        for c, lbl, d in hits:
            attribution[target][_sensor_name(c, lbl)] = round(d, 2)

        actuators.set_pwm(target, ATTRIBUTION_BASELINE_PWM)
        log.info("  recovering %.0fs", ATTRIBUTION_RECOVERY_S)
        _settle(ATTRIBUTION_RECOVERY_S)

    return attribution


def _yaml_quote(s: str) -> str:
    """Quote a YAML scalar if it contains characters that would confuse the
    parser (brackets for chip indices, spaces, etc.). Otherwise pass through."""
    if any(c in s for c in "[]:#&*!|>'\"%@`,? "):
        # Single-quote string; escape single quotes by doubling
        return "'" + s.replace("'", "''") + "'"
    return s


def render_zones_yaml(
    pwm_to_tach: dict[str, str | None],
    pwm_min: dict[str, int],
    attribution: dict[str, dict[str, float]],
    sensors_catalog: dict[str, tuple[str, str]],
    out_path: Path,
) -> None:
    """Write zones.yaml in v2 (sensors + fans) schema. Backs up an existing
    file to .bak.
    """
    pwm_min_default = min(pwm_min.values()) if pwm_min else 40
    lines: list[str] = []

    # ---- header ----
    lines.append("# Generated by fand-calibrate on " + time.strftime("%Y-%m-%dT%H:%M:%S"))
    lines.append("#")
    lines.append("# defaults: applied to fans that don't override.")
    lines.append("#")
    lines.append("# sensors:  every readable temperature on this host worth tracking.")
    lines.append("#           target_c / critical_c are conservative defaults based on")
    lines.append("#           hardware specs — review and tune for your system.")
    lines.append("#")
    lines.append("# fans:     PWM channels that responded to a fan tach. The cools dict")
    lines.append("#           maps sensor name to phase-3 ΔT (°C) — the seed cooling weight")
    lines.append("#           used until the daemon has learned runtime coefficients.")
    lines.append("#           Add sensors to a fan's cools list if you know it should")
    lines.append("#           affect them but phase 3 missed it (e.g. not enough thermal")
    lines.append("#           load on that sensor during calibration).")
    lines.append("")
    lines.append("defaults:")
    lines.append(f"  pwm_min: {pwm_min_default}")

    # ---- sensors ----
    lines.append("")
    lines.append("sensors:")
    sorted_sensors = sorted(sensors_catalog.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    for name, (chip, label) in sorted_sensors:
        target_c, critical_c = _sensor_defaults(chip, label)
        lines.append(f"  {name}:")
        lines.append(f"    chip: {_yaml_quote(chip)}")
        lines.append(f"    label: {_yaml_quote(label)}")
        lines.append(f"    target_c: {target_c}")
        lines.append(f"    critical_c: {critical_c}")

    # ---- fans ----
    lines.append("")
    lines.append("fans:")
    for pwm_ch, tach in pwm_to_tach.items():
        if tach is None:
            lines.append(f"  # {pwm_ch}: no responding fan detected; not managed")
            continue
        lines.append(f"  {pwm_ch}:")
        lines.append(f"    fan_tach: {tach}_input")
        if pwm_min.get(pwm_ch, pwm_min_default) != pwm_min_default:
            lines.append(f"    pwm_min: {pwm_min[pwm_ch]}")
        attr = attribution.get(pwm_ch, {})
        if attr:
            lines.append(f"    cools:")
            for sn, delta in sorted(attr.items(), key=lambda kv: -kv[1]):
                lines.append(f"      {sn}: {delta:.2f}    # ΔT °C during phase 3")
        else:
            lines.append(f"    # TODO: phase 3 saw no sensor move ≥ {ATTRIBUTION_MIN_DELTA_C}°C for this fan.")
            lines.append(f"    # Add sensor names (with seed weights in °C) you know it cools:")
            lines.append(f"    cools: {{}}")

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        out_path.rename(backup)
        log.info("backed up existing %s -> %s", out_path, backup)
    out_path.write_text(text)
    log.info("wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/etc/fand/zones.yaml")
    ap.add_argument("--chip", default="nct6799")
    ap.add_argument(
        "--skip-attribution",
        action="store_true",
        help="skip phase 3 (empirical sensor attribution); write empty cools dicts",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if os.geteuid() != 0:
        log.error("calibrate must be run as root (PWM writes need it)")
        return 1

    _check_no_daemon_running()

    hw = find_hwmon_by_name(args.chip)
    if hw is None:
        log.error("hwmon chip %s not found", args.chip)
        return 2

    s = Sensors(chip_name=args.chip)
    obs = s.read_all()
    if obs.gpu and obs.gpu["util_pct"] > 5 and args.skip_attribution:
        log.warning(
            "GPU util = %.0f%% — phase 1 wants IDLE; mapping deltas may be noisy",
            obs.gpu["util_pct"],
        )
    if obs.gpu and obs.gpu["temp_c"] > 50 and args.skip_attribution:
        log.warning(
            "GPU temp = %.0f°C — phase 1 wants ~40°C ambient; let it cool first",
            obs.gpu["temp_c"],
        )
    if not args.skip_attribution and (not obs.gpu or obs.gpu["util_pct"] < 5):
        log.warning(
            "no GPU load detected — phase 3 needs heat to produce signal; "
            "consider starting a typical workload before continuing"
        )

    sensors_catalog = _collect_sensors_catalog(s)

    channels: list[str] = []
    for p in sorted(hw.glob("pwm*")):
        if "_" in p.name:
            continue
        channels.append(p.name)
    fan_keys = [p.name.removesuffix("_input") for p in sorted(hw.glob("fan*_input"))]
    log.info("chip %s — pwm channels: %s, fan tachs: %s", args.chip, channels, fan_keys)
    log.info("sensors catalog: %d entries", len(sensors_catalog))

    log.info("starting calibration. Press Ctrl-C to abort (restores BIOS mode).")
    with Actuators(channels, chip_name=args.chip) as actuators:
        log.info("phase 1: PWM ↔ fan-tach mapping")
        pwm_to_tach = map_pwm_to_tach(actuators, hw, fan_keys)

        log.info("phase 2: pwm_min discovery per channel")
        pwm_min: dict[str, int] = {}
        for ch in channels:
            log.info("finding pwm_min for %s ...", ch)
            pwm_min[ch] = find_pwm_min(actuators, hw, ch, pwm_to_tach.get(ch))
            actuators.set_pwm(ch, 160)

        if args.skip_attribution:
            log.info("phase 3 skipped (--skip-attribution); fans will have empty cools dicts")
            attribution: dict[str, dict[str, float]] = {ch: {} for ch in pwm_to_tach}
        else:
            attribution = attribute_fans(actuators, s, pwm_to_tach, pwm_min)

    # Auto-add any attributed sensor that wasn't in the pre-cal catalog (e.g.
    # a sensor that only became readable under load).
    obs_post = s.read_all()
    for fan_attr in attribution.values():
        for sensor_name in fan_attr:
            if sensor_name in sensors_catalog:
                continue
            for chip, temps in obs_post.hwmon.items():
                for label in temps:
                    if _sensor_name(chip, label) == sensor_name:
                        sensors_catalog[sensor_name] = (chip, label)
                        break
            if sensor_name not in sensors_catalog and obs_post.gpu:
                if _sensor_name("gpu", "temp_c") == sensor_name:
                    sensors_catalog[sensor_name] = ("gpu", "temp_c")

    render_zones_yaml(pwm_to_tach, pwm_min, attribution, sensors_catalog, Path(args.out))

    log.info("done.")
    log.info("next steps:")
    log.info("  1. review %s — check sensor target_c/critical_c and fan cools lists", args.out)
    log.info("  2. sudo systemctl start fand")
    log.info("  3. fand-ctl status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
