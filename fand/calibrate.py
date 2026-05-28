"""fand-calibrate: PWM perturbation sweep + zones.yaml generation.

Three-phase, operator-supervised. The output is a zones.yaml with
target_sensors filled in from empirical attribution — operator can still
review and tweak, but it lands ready to use.

  1. PWM ↔ fan-tach mapping (which physical fan responds to each PWM channel)
  2. pwm_min per channel (lowest PWM at which the fan still spins)
  3. Sensor attribution (which sensors warm up when each fan is throttled —
     identifies what each fan is actually cooling without operator input)

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
import sys
import time
from pathlib import Path

from .actuators import Actuators
from .sensors import Sensors, find_hwmon_by_name

log = logging.getLogger("fand-calibrate")


DEFAULT_BOOTSTRAP_CURVE = [(40.0, 100), (60.0, 160), (75.0, 210), (85.0, 255)]

# Phase 3 (attribution) defaults
ATTRIBUTION_BASELINE_PWM = 200
ATTRIBUTION_BASELINE_SETTLE_S = 60.0
ATTRIBUTION_DROP_DURATION_S = 90.0
ATTRIBUTION_RECOVERY_S = 30.0
ATTRIBUTION_MIN_DELTA_C = 0.5
ATTRIBUTION_TOP_N = 2
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


def _average_temps(
    sensors_obj: Sensors,
    samples: int = ATTRIBUTION_SAMPLES,
    interval: float = ATTRIBUTION_SAMPLE_INTERVAL_S,
) -> dict[tuple[str, str], float]:
    """Sample temps multiple times via Sensors.read_all(), return mean per
    (chip, label). Includes hwmon temps and (if available) the NVIDIA GPU
    temp from nvidia-smi (under the special chip key 'gpu').
    """
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
    """Return (target_c, critical_c) for a given chip/label. Defaults are
    conservative — operator should review for their specific hardware.
    """
    # AMD CPU die (k10temp/Tctl). Ryzen throttles around 95°C.
    if chip == "k10temp":
        return (75.0, 90.0)
    # Super-I/O sensors that reflect the CPU (TSI0_TEMP mirrors Tctl;
    # CPUTIN is the nct socket thermistor which lags).
    if chip.startswith("nct") and label in ("TSI0_TEMP", "CPUTIN"):
        return (75.0, 90.0)
    # NVIDIA GPU via nvidia-smi (Blackwell throttles ~87-95°C).
    if chip == "gpu":
        return (75.0, 90.0)
    # AMD GPU / iGPU
    if chip == "amdgpu":
        return (75.0, 90.0)
    # DDR5 SPD5118 — module HIGH alarm fires at 55°C; give 5°C margin.
    if chip == "spd5118":
        return (50.0, 55.0)
    # NVMe — most drives start throttling around 70-80°C.
    if chip == "nvme" or chip.startswith("nvme["):
        return (65.0, 80.0)
    # nct ambient probes (SYSTIN, AUXTIN*). Chassis interior under load.
    if chip.startswith("nct"):
        return (50.0, 70.0)
    # Unknown — be conservative.
    return (60.0, 80.0)


def map_pwm_to_tach(actuators: Actuators, hw: Path, fan_keys: list[str]) -> dict[str, str | None]:
    """For each PWM channel, find which fan tach responds the most.

    Strategy: set every other PWM to 160 (steady), then for the channel under
    test bounce 80 → 220 → 80 and record the tach with the biggest swing.
    Returns {pwm_channel: fan_key_or_None}.
    """
    mapping: dict[str, str | None] = {}
    channels = list(actuators.handles.keys())

    for target in channels:
        log.info("probing %s …", target)
        # Hold every other channel at 160 (60%) so they don't drift during probe
        for ch in channels:
            actuators.set_pwm(ch, 160)
        _settle(3.0)

        # Low step
        actuators.set_pwm(target, 80)
        _settle(4.0)
        low = {k: _read_tach(hw, k) for k in fan_keys}

        # High step
        actuators.set_pwm(target, 220)
        _settle(4.0)
        high = {k: _read_tach(hw, k) for k in fan_keys}

        deltas = {k: high[k] - low[k] for k in fan_keys}
        log.info("  deltas: %s", deltas)

        best = max(deltas.items(), key=lambda kv: kv[1])
        if best[1] < 100:  # tiny swing -> probably not connected
            log.info("  -> %s: no fan responded (deltas all small)", target)
            mapping[target] = None
        else:
            mapping[target] = best[0]
            log.info("  -> %s drives %s (Δ %d RPM)", target, best[0], best[1])

        # Return to neutral
        actuators.set_pwm(target, 160)

    return mapping


def find_pwm_min(actuators: Actuators, hw: Path, channel: str, tach: str | None) -> int:
    """Step PWM down until the fan stalls (or we hit a safety floor).

    Returns the lowest PWM at which the fan still reports ≥200 RPM, plus a
    small safety margin. If no tach is owned by this channel, returns 80 as a
    conservative default.
    """
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
        # Fan never spun in the probe range — could be a high-static-pressure
        # server fan, large 140/200mm, or AIO pump whose real min is >100,
        # or the fan is dead. Return safe-high so runtime doesn't try to
        # under-drive it; operator must review pwm_min in zones.yaml.
        log.warning(
            "%s: fan never spun in probe range (100..20) — returning safe-high "
            "default 200; review pwm_min in zones.yaml",
            channel,
        )
        return 200
    # Margin: +20 PWM over the last spinning value, capped at 100
    return min(100, last_spinning + 20)


def attribute_fans(
    actuators: Actuators,
    sensors_obj: Sensors,
    pwm_to_tach: dict[str, str | None],
    pwm_min: dict[str, int],
) -> dict[str, list[tuple[str, str, float]]]:
    """Phase 3: empirically attribute each managed fan to the sensors it cools.

    Holds all managed fans at a high baseline, then walks each one in turn:
    drops it to its pwm_min for the observation window, records ΔT per sensor
    vs the steady-state baseline, ranks sensors by ΔT, and keeps the top-N
    above the noise threshold. Restores the channel and lets the system
    recover before moving to the next.

    Returns: {pwm_channel: [(chip, label, delta_c), ...]}, sorted by delta_c
    desc. Channels with no responding fan, or with no sensor crossing
    ATTRIBUTION_MIN_DELTA_C, get an empty list (caller falls back to a TODO
    placeholder).
    """
    managed = [ch for ch, tach in pwm_to_tach.items() if tach]
    attribution: dict[str, list[tuple[str, str, float]]] = {ch: [] for ch in pwm_to_tach}
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
        deltas: list[tuple[str, str, float]] = []
        for key, post_val in post.items():
            base_val = baseline.get(key)
            if base_val is None:
                continue
            d = post_val - base_val
            if d >= ATTRIBUTION_MIN_DELTA_C:
                deltas.append((key[0], key[1], d))
        deltas.sort(key=lambda x: -x[2])
        top = deltas[:ATTRIBUTION_TOP_N]
        if top:
            summary = ", ".join(f"{c}/{l} (Δ{d:+.1f}°C)" for c, l, d in top)
            log.info("  -> %s attributed to: %s", target, summary)
        else:
            log.info(
                "  -> %s: no sensor showed Δ ≥ %.1f°C (will mark as TODO)",
                target, ATTRIBUTION_MIN_DELTA_C,
            )
        attribution[target] = top

        # Restore + recover before next channel
        actuators.set_pwm(target, ATTRIBUTION_BASELINE_PWM)
        log.info("  recovering %.0fs", ATTRIBUTION_RECOVERY_S)
        _settle(ATTRIBUTION_RECOVERY_S)

    return attribution


def render_zones_yaml(
    pwm_to_tach: dict[str, str | None],
    pwm_min: dict[str, int],
    attribution: dict[str, list[tuple[str, str, float]]],
    out_path: Path,
) -> None:
    """Write zones.yaml using phase 3 attribution results. Channels with no
    attributed sensors fall back to a SYSTIN placeholder with a TODO comment.
    """
    lines: list[str] = []
    lines.append("# Generated by fand-calibrate on " + time.strftime("%Y-%m-%dT%H:%M:%S"))
    lines.append("# target_sensors auto-filled from phase 3 attribution (top-N by ΔT")
    lines.append("# during PWM drop). Review the temps and adjust target_c/critical_c")
    lines.append("# for your hardware tolerances.")
    lines.append("# - chip: hwmon `name` file (e.g. nct6799, k10temp, nvme, spd5118).")
    lines.append("# - label: sensor label (e.g. Tctl, SYSTIN, Composite, temp1).")
    lines.append("# - target_c: temp we want this fan to hold below.")
    lines.append("# - critical_c: hard floor; if exceeded, all fans -> 255 + alert.")
    lines.append("zones:")
    for pwm_ch, tach in pwm_to_tach.items():
        if tach is None:
            lines.append(f"  # {pwm_ch}: no responding fan detected; not managed")
            continue
        lines.append(f"  - name: {pwm_ch}")
        lines.append(f"    pwm_channel: {pwm_ch}")
        lines.append(f"    fan_tach: {tach}_input")
        lines.append(f"    pwm_min: {pwm_min.get(pwm_ch, 80)}")
        attr = attribution.get(pwm_ch, [])
        if attr:
            lines.append(f"    target_sensors:")
            for i, (chip, label, delta) in enumerate(attr):
                role = "primary" if i == 0 else "secondary critical"
                target_c, critical_c = _sensor_defaults(chip, label)
                lines.append(f"      # {role} (Δ{delta:+.1f}°C during phase 3)")
                lines.append(f"      - chip: {chip}")
                lines.append(f"        label: {label}")
                lines.append(f"        target_c: {target_c}")
                lines.append(f"        critical_c: {critical_c}")
        else:
            lines.append(f"    # TODO: phase 3 found no sensor with Δ ≥ {ATTRIBUTION_MIN_DELTA_C}°C")
            lines.append(f"    # — set target_sensors by hand (check `sensors` for options)")
            lines.append(f"    target_sensors:")
            lines.append(f"      - chip: nct6799")
            lines.append(f"        label: SYSTIN")
            lines.append(f"        target_c: 65.0")
            lines.append(f"        critical_c: 80.0")
        lines.append(f"    bootstrap_curve:  # temp_c -> pwm fallback before learning")
        for t, p in DEFAULT_BOOTSTRAP_CURVE:
            lines.append(f"      - [{t}, {p}]")
        lines.append("")

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
        help="skip phase 3 (empirical sensor attribution); write TODO placeholders",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if os.geteuid() != 0:
        log.error("calibrate must be run as root (PWM writes need it)")
        return 1

    hw = find_hwmon_by_name(args.chip)
    if hw is None:
        log.error("hwmon chip %s not found", args.chip)
        return 2

    # Pre-flight: warn about GPU state. Phases 1+2 want idle for clean delta-RPM
    # measurements; phase 3 wants load running for delta-T signal — flag both.
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
            "no GPU load detected — phase 3 attribution needs heat to produce "
            "signal; consider starting a typical workload before continuing"
        )

    # Find all pwm channels on this chip
    channels: list[str] = []
    for p in sorted(hw.glob("pwm*")):
        if "_" in p.name:
            continue  # skip pwm1_enable etc.
        channels.append(p.name)
    fan_keys = [p.name.removesuffix("_input") for p in sorted(hw.glob("fan*_input"))]
    log.info("chip %s — pwm channels: %s, fan tachs: %s", args.chip, channels, fan_keys)

    log.info("starting calibration. Press Ctrl-C to abort (restores BIOS mode).")
    with Actuators(channels, chip_name=args.chip) as actuators:
        log.info("phase 1: PWM ↔ fan-tach mapping")
        pwm_to_tach = map_pwm_to_tach(actuators, hw, fan_keys)

        log.info("phase 2: pwm_min discovery per channel")
        pwm_min: dict[str, int] = {}
        for ch in channels:
            log.info("finding pwm_min for %s ...", ch)
            pwm_min[ch] = find_pwm_min(actuators, hw, ch, pwm_to_tach.get(ch))
            # Restore to neutral before next channel
            actuators.set_pwm(ch, 160)

        if args.skip_attribution:
            log.info("phase 3 skipped (--skip-attribution); zones.yaml will have TODO placeholders")
            attribution: dict[str, list[tuple[str, str, float]]] = {ch: [] for ch in pwm_to_tach}
        else:
            attribution = attribute_fans(actuators, s, pwm_to_tach, pwm_min)

    # We're back in BIOS mode after the context exit. Write the template.
    render_zones_yaml(pwm_to_tach, pwm_min, attribution, Path(args.out))

    log.info("done.")
    log.info("next steps:")
    log.info("  1. review %s -- check target_c/critical_c per zone", args.out)
    log.info("  2. systemctl start fand")
    log.info("  3. fand-ctl status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
