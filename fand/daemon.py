"""fand main daemon — autonomous multi-sensor multi-fan control.

Run modes:
  fand-daemon                       # main poll loop (systemd Type=notify)
  fand-daemon --restore-bios        # ExecStopPost: restore pwm enable modes and exit

Each tick (default 2 s):

  1. Read every declared sensor (chip+label lookup against the hwmon observation).
  2. Apply per-sensor AR(1) feed-forward correction to predict near-term ΔT
     (disabled while that sensor's α coefs are untrained).
  3. Critical floor: any sensor at/above its `critical_c` → set_all(255) + alarm.
  4. Otherwise, per fan, compute demand via the weighted-stress rule in
     `model.aggregate_demand` and set_pwm.
  5. Per-fan equilibrium detection: when a fan's PWM has been stable, the
     sensors it cools have low dT/dt, and load features are stable, record a
     row to equilibria.jsonl.
  6. Periodic refit: solve per-sensor ridge regression on the rolling sample
     pool; write model.json.

See DESIGN.md for why we got here from a per-zone architecture.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .actuators import Actuators, SAVED_STATE, restore_from_disk
from .model import (
    FEATURE_NAMES,
    EquilibriumSample,
    SensorModel,
    aggregate_demand,
    demand_to_pwm,
    is_fan_equilibrium,
    load_models_into,
    save_models,
)
from .sensors import Observation, Sensors

log = logging.getLogger("fand")

CONFIG_PATH = Path("/etc/fand/config.yaml")
ZONES_PATH = Path("/etc/fand/zones.yaml")
STATE_DIR = Path("/var/lib/fand")
MODEL_PATH = STATE_DIR / "model.json"
EQUILIBRIA_PATH = STATE_DIR / "equilibria.jsonl"
HISTORY_PATH = STATE_DIR / "history.jsonl"
RUN_DIR = Path("/run/fand")
STATUS_PATH = RUN_DIR / "status.json"
HISTORY_RETENTION_DAYS = 30
FAN_DEAD_STREAK = 3  # ~6s at default 2s poll, skips mechanical spin-up on cold start
SENSOR_RECOVER_STREAK = 3  # consecutive good reads to clear assumed-critical (anti-flap)
TICK_FAIL_LIMIT = 5  # consecutive tick exceptions before exiting for systemd restart
EQUILIBRIA_MAXLEN = 20000


# ---- sd_notify ------------------------------------------------------------


def sd_notify(message: str) -> None:
    """Low-level systemd notify protocol. No-op if NOTIFY_SOCKET not set."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.connect(addr)
        sock.sendall(message.encode())
        sock.close()
    except OSError as exc:
        log.debug("sd_notify: %s", exc)


# ---- helpers --------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(path)


def _temp_at(obs: Observation, chip: str, label: str) -> float | None:
    """Look up a temperature reading. Requires explicit chip name (including
    any `[i]` suffix for multi-instance chips like `nvme[0]`, `spd5118[1]`).

    The special chip name `gpu` routes to the NVIDIA reading from nvidia-smi.
    """
    if chip == "gpu":
        return obs.gpu.get(label) if obs.gpu else None
    chip_temps = obs.hwmon.get(chip)
    if chip_temps is None:
        return None
    return chip_temps.get(label)


def _build_features(obs: Observation) -> dict[str, float]:
    return {
        "util_gpu": obs.gpu["util_pct"] if obs.gpu else 0.0,
        "power_gpu_w": obs.gpu["power_w"] if obs.gpu else 0.0,
        "util_cpu": obs.cpu_util if obs.cpu_util is not None else 0.0,
        "ups_w": obs.ups["realpower_w"] if obs.ups else 0.0,
    }


def _ntfy(cmd: str | None, message: str) -> None:
    if not cmd:
        return
    try:
        subprocess.run([cmd, message], timeout=5, check=False)
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        log.warning("ntfy failed: %s", exc)


# ---- history file rotation ------------------------------------------------


def _maybe_rotate_history(path: Path, current_day: list[str]) -> None:
    """Rotate path to path.YYYYMMDD.gz when the date changes."""
    today = time.strftime("%Y%m%d", time.localtime())
    if not current_day:
        if path.exists():
            try:
                mtime_day = time.strftime("%Y%m%d", time.localtime(path.stat().st_mtime))
            except OSError:
                mtime_day = today
            current_day.append(mtime_day)
        else:
            current_day.append(today)
            return
    if today == current_day[0]:
        return
    if path.exists():
        archived = path.with_name(f"{path.name}.{current_day[0]}.gz")
        try:
            with path.open("rb") as src, gzip.open(archived, "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
            log.info("history rotated to %s", archived)
        except OSError as exc:
            log.warning("history rotate failed: %s", exc)
    current_day[0] = today
    cutoff = time.time() - HISTORY_RETENTION_DAYS * 86400
    for old in path.parent.glob(f"{path.name}.*.gz"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
                log.info("history pruned %s", old)
        except OSError:
            pass


# ---- config dataclasses ---------------------------------------------------


@dataclass
class SensorConfig:
    name: str
    chip: str
    label: str
    target_c: float
    critical_c: float
    # What to do when the sensor stays unreadable past sensor_fail_grace_s:
    # 'critical' = assume it's at critical_c (all fans max), 'alarm' = alarm
    # only. None = use the global sensor_fail_assume_critical default. Set
    # 'alarm' on sensors with their own protection story (e.g. a GPU whose
    # board fans aren't host-controllable anyway).
    on_unreadable: str | None = None


@dataclass
class FanConfig:
    name: str           # also the dict key in zones.yaml
    pwm_channel: str    # PWM channel on the Super-I/O (defaults to name)
    fan_tach: str | None
    pwm_min: int
    cools: dict[str, float] = field(default_factory=dict)  # sensor_name → seed weight


# ---- schema parse + auto-translator --------------------------------------


def _is_old_schema(cfg: dict[str, Any]) -> bool:
    """Old schema has a top-level `zones:` list; new has `sensors:` + `fans:`."""
    return "zones" in cfg and "sensors" not in cfg


def _slug(s: str) -> str:
    """Lowercase, alphanumeric + underscore; collapse runs of non-alnum."""
    out_chars: list[str] = []
    prev_under = False
    for c in s:
        if c.isalnum():
            out_chars.append(c.lower())
            prev_under = False
        elif not prev_under:
            out_chars.append("_")
            prev_under = True
    return "".join(out_chars).strip("_") or "sensor"


def _translate_old_schema(cfg: dict[str, Any], default_pwm_min: int = 40) -> dict[str, Any]:
    """Translate old zones.yaml schema to new sensors+fans schema.

    Each unique (chip, label) becomes a sensors entry; each old zone becomes
    a fans entry with `cools:` populated as uniform-weight seed dict. Real
    weights come from re-running `fand-calibrate` after the translation lands.
    """
    sensors_out: dict[str, dict[str, Any]] = {}
    fans_out: dict[str, dict[str, Any]] = {}
    sensor_name_for: dict[tuple[str, str], str] = {}

    # Pass 1: collect sensors from all zones' target_sensors
    for zone in cfg.get("zones") or []:
        for ts in zone.get("target_sensors") or []:
            chip = str(ts.get("chip", ""))
            label = str(ts.get("label", ""))
            if not chip or not label:
                continue
            key = (chip, label)
            if key in sensor_name_for:
                continue
            base_name = _slug(f"{chip}_{label}")
            name = base_name
            i = 2
            while name in sensors_out:
                name = f"{base_name}_{i}"
                i += 1
            sensor_name_for[key] = name
            sensors_out[name] = {
                "chip": chip,
                "label": label,
                "target_c": float(ts.get("target_c", 65.0)),
                "critical_c": float(ts.get("critical_c", 90.0)),
            }

    # Pass 2: build fans entries
    for zone in cfg.get("zones") or []:
        fan_name = str(zone.get("pwm_channel") or zone.get("name") or "")
        if not fan_name:
            continue
        cools: dict[str, float] = {}
        for ts in zone.get("target_sensors") or []:
            chip = str(ts.get("chip", ""))
            label = str(ts.get("label", ""))
            key = (chip, label)
            if key in sensor_name_for:
                cools[sensor_name_for[key]] = 1.0
        fan_entry: dict[str, Any] = {}
        if zone.get("fan_tach"):
            fan_entry["fan_tach"] = zone["fan_tach"]
        if zone.get("pwm_min") is not None:
            fan_entry["pwm_min"] = int(zone["pwm_min"])
        fan_entry["cools"] = cools
        fans_out[fan_name] = fan_entry

    return {
        "defaults": {"pwm_min": default_pwm_min},
        "sensors": sensors_out,
        "fans": fans_out,
    }


def _maybe_auto_translate(zones_path: Path, cfg: dict[str, Any]) -> None:
    """If cfg is old-schema, write a translated file alongside and exit with a
    helpful message. Daemon won't start on old schema."""
    if not _is_old_schema(cfg):
        return
    translated_path = zones_path.with_suffix(zones_path.suffix + ".translated")
    new_cfg = _translate_old_schema(cfg)
    try:
        translated_path.write_text(yaml.safe_dump(new_cfg, sort_keys=False, default_flow_style=False))
    except OSError as exc:
        log.error("auto-translate write failed (%s); refusing to start", exc)
        sys.exit(2)

    log.error("zones.yaml is in the old per-zone schema — refusing to start.")
    log.error("Wrote translated v2 schema to: %s", translated_path)
    log.error("Review it, then:")
    log.error("    sudo mv %s %s", translated_path, zones_path)
    log.error("    sudo systemctl restart fand")
    log.error("After that runs, re-run `sudo fand-calibrate` to populate phase-3 cooling weights")
    log.error("(translator used uniform seed weights of 1.0; calibrate replaces with measured ΔT).")
    sys.exit(2)


# ---- main daemon class ----------------------------------------------------


class Daemon:
    def __init__(self, config_path: Path = CONFIG_PATH, zones_path: Path = ZONES_PATH):
        self.config = yaml.safe_load(config_path.read_text()) or {}
        zones_cfg = yaml.safe_load(zones_path.read_text()) or {}

        # Auto-translator: exits if old schema detected.
        _maybe_auto_translate(zones_path, zones_cfg)

        self.poll_interval = float(self.config.get("poll_interval_s", 2.0))
        self.fit_interval = float(self.config.get("fit_interval_s", 3600.0))
        self.equilibrium_window_s = float(self.config.get("equilibrium_window_s", 30.0))
        self.equilibrium_pwm_stable_s = float(
            self.config.get("equilibrium_pwm_stable_s", 30.0)
        )
        # A window of n samples spans (n-1)*poll seconds; +1 so the window
        # actually covers equilibrium_window_s (the gate requires the window to
        # span at least pwm_stable_s, so an int() truncation here left it one
        # poll short and the equilibrium detector never fired).
        self.equilibrium_window_n = max(
            8, int(self.equilibrium_window_s / self.poll_interval) + 1
        )
        # Equilibrium-detector tolerances (all operator-tunable; defaults are
        # what is_fan_equilibrium uses if absent).
        self.equilibrium_feature_cov = float(
            self.config.get("feature_cov_threshold", 0.10)
        )
        self.equilibrium_feature_abs_tol = float(
            self.config.get("feature_abs_tolerance", 2.0)
        )
        self.equilibrium_temp_slope = float(
            self.config.get("temp_slope_threshold", 0.05)
        )
        self.equilibrium_pwm_jitter = int(
            self.config.get("pwm_jitter_tolerance", 4)
        )
        if "min_r2_to_learn" in self.config:
            log.warning(
                "config key min_r2_to_learn is no longer used — the trained "
                "gate is holdout RMSE now (max_rmse_to_learn_c, default 1.5 °C)"
            )
        self.ntfy_cmd = self.config.get("ntfy_command")
        self.history_enabled = bool(self.config.get("history_enabled", True))
        self.chip_name = self.config.get("chip_name", "nct6799")
        # Missing-sensor fail-safe: hold the last reading for this long, then
        # escalate per sensor policy (assume critical_c → all fans max, or
        # alarm only). An unreadable sensor must never silently lose its
        # critical floor.
        self.sensor_fail_grace_s = float(self.config.get("sensor_fail_grace_s", 30.0))
        self.sensor_fail_assume_critical = bool(
            self.config.get("sensor_fail_assume_critical", True)
        )

        self.sensors = Sensors(
            ups_name=self.config.get("ups_name", "cyberpower"),
            chip_name=self.chip_name,
        )

        defaults = zones_cfg.get("defaults") or {}
        default_pwm_min = int(defaults.get("pwm_min", 40))

        sensors_raw = zones_cfg.get("sensors") or {}
        if not sensors_raw:
            raise RuntimeError(f"no sensors defined in {zones_path}")
        fans_raw = zones_cfg.get("fans") or {}
        if not fans_raw:
            raise RuntimeError(f"no fans defined in {zones_path}")

        self.sensor_configs: dict[str, SensorConfig] = {}
        for name, sc in sensors_raw.items():
            on_unreadable = sc.get("on_unreadable")
            if on_unreadable is not None and on_unreadable not in ("critical", "alarm"):
                raise RuntimeError(
                    f"sensor {name}: on_unreadable must be 'critical' or 'alarm', "
                    f"got {on_unreadable!r}"
                )
            self.sensor_configs[name] = SensorConfig(
                name=str(name),
                chip=str(sc["chip"]),
                label=str(sc["label"]),
                target_c=float(sc["target_c"]),
                critical_c=float(sc["critical_c"]),
                on_unreadable=on_unreadable,
            )

        self.fan_configs: dict[str, FanConfig] = {}
        for name, fc in fans_raw.items():
            cools_raw = fc.get("cools") or {}
            self.fan_configs[name] = FanConfig(
                name=str(name),
                pwm_channel=str(fc.get("pwm_channel") or name),
                fan_tach=str(fc["fan_tach"]) if fc.get("fan_tach") else None,
                pwm_min=int(fc.get("pwm_min", default_pwm_min)),
                cools={str(k): float(v) for k, v in cools_raw.items()},
            )

        # Each sensor's cools_seeds dict: {fan_name → seed_weight from that fan's cools entry}
        cools_seeds_by_sensor: dict[str, dict[str, float]] = {
            sn: {} for sn in self.sensor_configs
        }
        for fn, fc in self.fan_configs.items():
            for sn, weight in fc.cools.items():
                if sn not in self.sensor_configs:
                    log.warning(
                        "fan %s lists unknown sensor %s — ignoring (typo in zones.yaml?)",
                        fn, sn,
                    )
                    continue
                cools_seeds_by_sensor[sn][fn] = weight

        # Per-sensor models
        self.sensor_models: dict[str, SensorModel] = {}
        for name, sc in self.sensor_configs.items():
            self.sensor_models[name] = SensorModel(
                name=name,
                target_c=sc.target_c,
                critical_c=sc.critical_c,
                cools_seeds=cools_seeds_by_sensor.get(name, {}),
                ridge_lambda=float(self.config.get("ridge_lambda", 1.0)),
                min_samples=int(self.config.get("min_samples_to_learn", 200)),
                max_rmse_c=float(self.config.get("max_rmse_to_learn_c", 1.5)),
                ff_alpha=float(self.config.get("ff_alpha", 0.05)),
                fully_trained_n=int(self.config.get("fully_trained_n", 500)),
                min_pwm_spread=float(self.config.get("min_pwm_spread", 10.0)),
                min_feature_cov=float(self.config.get("min_feature_cov", 0.05)),
            )
        load_models_into(MODEL_PATH, self.sensor_models)

        # Missing-sensor fail-safe state. _last_good_* deliberately start
        # empty: "never read this process-life" is distinguishable from
        # "read once, then lost", and only the latter can hold/escalate.
        self._last_good_temp: dict[str, float] = {}
        self._last_good_t: dict[str, float] = {}
        self._sensor_ever_read: dict[str, bool] = {sn: False for sn in self.sensor_configs}
        self._sensor_escalated: dict[str, bool] = {sn: False for sn in self.sensor_configs}
        self._sensor_recover_streak: dict[str, int] = {sn: 0 for sn in self.sensor_configs}

        # Refuse to start on permanent sensor config errors (raises). Runs
        # before Actuators so a bad zones.yaml never gets as far as PWM.
        self._verify_sensors_at_startup()

        # Actuators: PWM channels come from fans
        channels = [fc.pwm_channel for fc in self.fan_configs.values()]
        self.actuators = Actuators(channels, chip_name=self.chip_name)

        # Per-fan equilibrium tracking; equilibria pool is a global rolling deque
        self.windows: dict[str, deque[tuple[float, dict[str, float], dict[str, float], dict[str, int]]]] = {
            fn: deque(maxlen=self.equilibrium_window_n) for fn in self.fan_configs
        }
        self.equilibria: deque[EquilibriumSample] = deque(maxlen=EQUILIBRIA_MAXLEN)
        self._load_equilibria()

        self.last_fit_t = 0.0
        # Staggered refit: when the fit interval elapses, all sensors are
        # queued and one is fit per tick against a shared pool snapshot.
        self._refit_queue: list[str] = []
        self._refit_samples: list[EquilibriumSample] = []
        self.last_history_day: list[str] = []
        self._stop = False
        self._last_alarm_t: dict[str, float] = {}
        self._fan_fault_streak: dict[str, int] = {fn: 0 for fn in self.fan_configs}

        log.info(
            "daemon initialized: %d sensors, %d fans, equilibrium_window=%.0fs",
            len(self.sensor_configs), len(self.fan_configs), self.equilibrium_window_s,
        )

    # ---- startup sensor verification ----------------------------------------

    def _verify_sensors_at_startup(self) -> None:
        """One read of every declared sensor. A hwmon sensor whose chip or
        label doesn't resolve is a permanent zones.yaml error (pulled drive,
        typo) — refuse to start listing every offender, rather than running
        for months with that sensor silently unprotected. A `gpu` sensor with
        nvidia-smi absent only warns: the source is optional and may come up
        later; the runtime fail-safe covers it once it has read at least once.

        Successful reads seed the last-good state so the runtime grace clock
        starts from a real reading.
        """
        obs = self.sensors.read_all()
        fatal: list[str] = []
        for name, sc in self.sensor_configs.items():
            T = _temp_at(obs, sc.chip, sc.label)
            if T is not None:
                self._last_good_temp[name] = T
                self._last_good_t[name] = obs.t
                self._sensor_ever_read[name] = True
                continue
            if sc.chip == "gpu":
                log.warning(
                    "sensor %s: nvidia-smi unavailable at startup — optional "
                    "source; sensor is unprotected until it first reads",
                    name,
                )
                continue
            chip_temps = obs.hwmon.get(sc.chip)
            if chip_temps is None:
                fatal.append(f"{name}: chip {sc.chip!r} not found in hwmon")
            else:
                fatal.append(
                    f"{name}: label {sc.label!r} not found on chip {sc.chip!r} "
                    f"(present: {sorted(chip_temps)})"
                )
        if fatal:
            for msg in fatal:
                log.error("startup sensor check: %s", msg)
            raise RuntimeError(
                f"{len(fatal)} declared sensor(s) unreadable at startup — "
                f"fix zones.yaml or remove them"
            )

    # ---- equilibrium persistence -------------------------------------------

    def _load_equilibria(self) -> None:
        if not EQUILIBRIA_PATH.exists():
            return
        try:
            lines = EQUILIBRIA_PATH.read_text().splitlines()
        except OSError as exc:
            log.warning("equilibria load failed: %s", exc)
            return
        loaded = skipped = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("equilibria line skipped (json): %s", exc)
                skipped += 1
                continue
            # New v2 format: has 'fan', 'features' dict, 'pwm' dict, 'temps' dict.
            # Old v1 format: has 'zone', 'features' list, 'pwm_applied', 'T_z', 'T_amb'.
            if "fan" not in row or "temps" not in row:
                skipped += 1
                continue
            try:
                self.equilibria.append(EquilibriumSample(
                    t=float(row["t"]),
                    fan=str(row["fan"]),
                    features={str(k): float(v) for k, v in (row.get("features") or {}).items()},
                    pwm={str(k): int(v) for k, v in (row.get("pwm") or {}).items()},
                    temps={str(k): float(v) for k, v in (row.get("temps") or {}).items()},
                ))
                loaded += 1
            except (TypeError, KeyError, ValueError) as exc:
                log.warning("equilibria line skipped (shape): %s", exc)
                skipped += 1
        log.info(
            "loaded %d equilibria samples (skipped %d — including any v1-format rows)",
            loaded, skipped,
        )

    def _append_equilibrium(self, sample: EquilibriumSample) -> None:
        self.equilibria.append(sample)
        try:
            EQUILIBRIA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EQUILIBRIA_PATH.open("a") as f:
                f.write(json.dumps(asdict(sample)) + "\n")
        except OSError as exc:
            log.warning("equilibrium persist failed: %s", exc)

    def _persist_equilibria_atomic(self) -> None:
        """Rewrite equilibria.jsonl from the in-memory deque after each refit."""
        EQUILIBRIA_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = EQUILIBRIA_PATH.with_suffix(EQUILIBRIA_PATH.suffix + ".tmp")
        try:
            with tmp.open("w") as f:
                for s in self.equilibria:
                    f.write(json.dumps(asdict(s)) + "\n")
            tmp.replace(EQUILIBRIA_PATH)
        except OSError as exc:
            log.warning("equilibria atomic persist failed: %s", exc)

    # ---- alarms ------------------------------------------------------------

    def _alarm(self, key: str, message: str, cooldown_s: float = 300.0) -> None:
        now = time.monotonic()
        if now - self._last_alarm_t.get(key, 0) < cooldown_s:
            return
        self._last_alarm_t[key] = now
        log.warning("ALARM %s: %s", key, message)
        _ntfy(self.ntfy_cmd, f"[fand] {message}")

    # ---- main tick ---------------------------------------------------------

    def _tick(self) -> None:
        obs = self.sensors.read_all()
        features = _build_features(obs)

        # Read all sensor temperatures; apply per-sensor AR(1) feed-forward
        # correction (only effective once that sensor's α is trained).
        #
        # Missing-sensor fail-safe: an unreadable sensor must never silently
        # lose its critical floor. Within sensor_fail_grace_s the last good
        # reading stands in (the sensor stays inside the stress and critical
        # checks); past the grace window it escalates per policy — assume
        # critical_c (→ all fans max via the existing any_critical path) or
        # alarm-only. fresh_temps holds only values actually read this tick:
        # it feeds equilibrium learning, so held/synthetic temps can never
        # poison the ridge fit.
        sensor_temps: dict[str, float] = {}            # control + status (incl. held)
        fresh_temps: dict[str, float] = {}             # learning (actually read)
        sensor_effective_temps: dict[str, float] = {}
        sensor_states: dict[str, str] = {}             # ok | hold | assumed-critical | missing
        for name, sc in self.sensor_configs.items():
            T = _temp_at(obs, sc.chip, sc.label)
            sm = self.sensor_models[name]
            if T is not None:
                fresh_temps[name] = T
                self._last_good_temp[name] = T
                self._last_good_t[name] = obs.t
                self._sensor_ever_read[name] = True
                T_eff = sm.ff_corrected_temp(T, features)
                if self._sensor_escalated[name]:
                    # Anti-flap: an intermittently-readable sensor stays
                    # escalated until it produces N consecutive good reads.
                    self._sensor_recover_streak[name] += 1
                    if self._sensor_recover_streak[name] >= SENSOR_RECOVER_STREAK:
                        self._sensor_escalated[name] = False
                        log.info(
                            "sensor %s readable again (%d consecutive reads) — "
                            "clearing assumed-critical",
                            name, SENSOR_RECOVER_STREAK,
                        )
                    else:
                        T_eff = max(T_eff, sc.critical_c)
                sensor_temps[name] = T
                sensor_effective_temps[name] = T_eff
                sensor_states[name] = (
                    "assumed-critical" if self._sensor_escalated[name] else "ok"
                )
                continue

            # Unreadable this tick.
            self._sensor_recover_streak[name] = 0
            if not self._sensor_ever_read[name]:
                # Never read this process-life — an optional source that
                # hasn't come up (hwmon sensors can't get here: startup
                # verification either seeded them or refused to start).
                # No reading to hold and no basis to assume critical.
                sensor_states[name] = "missing"
                self._alarm(
                    f"missing:{name}",
                    f"sensor {name} ({sc.chip}/{sc.label}) unreadable "
                    f"(never read since start — unprotected)",
                )
                continue
            age = obs.t - self._last_good_t[name]
            policy = sc.on_unreadable or (
                "critical" if self.sensor_fail_assume_critical else "alarm"
            )
            if age <= self.sensor_fail_grace_s and not self._sensor_escalated[name]:
                # Grace hold: last reading stands in, no FF on a held value
                # (layering a surge prediction onto a stale temp would
                # double-count; the EMA only updates from real reads).
                T_hold = self._last_good_temp[name]
                sensor_temps[name] = T_hold
                sensor_effective_temps[name] = T_hold
                sensor_states[name] = "hold"
                self._alarm(
                    f"missing:{name}",
                    f"sensor {name} ({sc.chip}/{sc.label}) unreadable — "
                    f"holding last reading {T_hold:.1f}°C",
                    cooldown_s=60.0,
                )
            elif policy == "critical":
                self._sensor_escalated[name] = True
                # Synthetic critical_c goes into the *effective* map only:
                # the critical check and stress both read it from there, and
                # keeping it out of sensor_temps keeps it out of status T and
                # equilibrium windows.
                sensor_effective_temps[name] = sc.critical_c
                sensor_states[name] = "assumed-critical"
                self._alarm(
                    f"missing_critical:{name}",
                    f"sensor {name} ({sc.chip}/{sc.label}) unreadable for "
                    f"{age:.0f}s — assuming critical, all fans to max",
                )
            else:
                sensor_states[name] = "missing"
                self._alarm(
                    f"missing:{name}",
                    f"sensor {name} ({sc.chip}/{sc.label}) unreadable for "
                    f"{age:.0f}s — degraded (on_unreadable: alarm)",
                    cooldown_s=60.0,
                )

        # Global critical check (per sensor)
        any_critical = False
        for name, T_eff in sensor_effective_temps.items():
            sc = self.sensor_configs[name]
            if T_eff >= sc.critical_c:
                any_critical = True
                self._alarm(
                    f"critical:{name}",
                    f"sensor {name} ({sc.chip}/{sc.label}) = {T_eff:.1f}°C ≥ critical {sc.critical_c:.1f}°C",
                )

        # Per-sensor stress
        stresses: dict[str, float] = {}
        for name, T_eff in sensor_effective_temps.items():
            stresses[name] = self.sensor_models[name].stress(T_eff)

        # Per-fan PWM decision
        applied: dict[str, int] = {}
        sources: dict[str, str] = {}
        breakdowns: dict[str, list[tuple[str, float, float, float]]] = {}

        if any_critical:
            self.actuators.set_all(255)
            for fn in self.fan_configs:
                applied[fn] = 255
                sources[fn] = "critical"
                breakdowns[fn] = []
        else:
            for fn, fc in self.fan_configs.items():
                demand, parts = aggregate_demand(fn, fc.cools, stresses, self.sensor_models)
                pwm = demand_to_pwm(demand, fc.pwm_min)
                self.actuators.set_pwm(fc.pwm_channel, pwm)
                applied[fn] = pwm
                any_trained_cools = any(
                    sn in self.sensor_models and self.sensor_models[sn].is_trained()
                    for sn in fc.cools
                )
                sources[fn] = "stress-learned" if any_trained_cools else "stress-seed"
                breakdowns[fn] = parts

        # Fan-fault detection (per fan)
        for fn, fc in self.fan_configs.items():
            if not fc.fan_tach:
                continue
            tach_key = fc.fan_tach.removesuffix("_input")
            tach = obs.fans.get(tach_key)
            if tach is None:
                continue
            if applied.get(fn, 0) >= fc.pwm_min and tach < 100:
                streak = self._fan_fault_streak.get(fn, 0) + 1
                self._fan_fault_streak[fn] = streak
                if streak >= FAN_DEAD_STREAK:
                    self._alarm(
                        f"fan_dead:{fn}",
                        f"{fn} commanded pwm={applied[fn]} but tach={tach} RPM",
                    )
            else:
                self._fan_fault_streak[fn] = 0

        # Per-fan equilibrium collection. Windows and samples record only
        # fresh_temps — a held value is flat by construction (dT/dt = 0) and
        # would both fake equilibrium and feed a stale reading to the fit.
        if not any_critical:
            feature_names = list(features.keys())
            for fn, fc in self.fan_configs.items():
                w = self.windows[fn]
                w.append((obs.t, dict(fresh_temps), dict(features), dict(applied)))
                if len(w) < self.equilibrium_window_n:
                    continue
                cooled = list(fc.cools.keys())
                if is_fan_equilibrium(
                    list(w),
                    fan_name=fn,
                    cooled_sensors=cooled,
                    feature_names=feature_names,
                    pwm_stable_s=self.equilibrium_pwm_stable_s,
                    pwm_jitter_tolerance=self.equilibrium_pwm_jitter,
                    temp_slope_threshold=self.equilibrium_temp_slope,
                    feature_cov_threshold=self.equilibrium_feature_cov,
                    feature_abs_tolerance=self.equilibrium_feature_abs_tol,
                ):
                    sample = EquilibriumSample(
                        t=obs.t,
                        fan=fn,
                        features=dict(features),
                        pwm=dict(applied),
                        temps=dict(fresh_temps),
                    )
                    self._append_equilibrium(sample)
                    w.clear()

        # Periodic refit, staggered one sensor per tick. Fitting every sensor
        # against a 20k-sample pool in a single tick can stall the loop past
        # the watchdog — and a watchdog kill *before* the equilibria rewrite
        # below was the one genuinely unbounded growth path for
        # equilibria.jsonl (kill → restart → same heavy first-tick refit →
        # kill, with appends accumulating and the trim never reached). All
        # sensors in a cycle fit against the same pool snapshot for
        # consistency; last_fit_t is set at cycle start so the interval
        # measures refit-to-refit.
        if not self._refit_queue and obs.t - self.last_fit_t > self.fit_interval:
            self._refit_queue = list(self.sensor_models.keys())
            self._refit_samples = list(self.equilibria)
            self.last_fit_t = obs.t
        if self._refit_queue:
            name = self._refit_queue.pop(0)
            self.sensor_models[name].fit(
                self._refit_samples, FEATURE_NAMES, list(self.fan_configs.keys())
            )
            if not self._refit_queue:
                save_models(MODEL_PATH, self.sensor_models)
                self._persist_equilibria_atomic()
                self._refit_samples = []

        # Status JSON v2
        status = self._build_status(
            obs, sensor_temps, sensor_effective_temps, stresses,
            sensor_states, applied, sources, breakdowns, any_critical,
        )
        _atomic_write_json(STATUS_PATH, status)

        if self.history_enabled:
            _maybe_rotate_history(HISTORY_PATH, self.last_history_day)
            try:
                with HISTORY_PATH.open("a") as f:
                    f.write(json.dumps(status, default=str) + "\n")
            except OSError as exc:
                log.warning("history append failed: %s", exc)

    def _build_status(
        self,
        obs: Observation,
        sensor_temps: dict[str, float],
        sensor_effective_temps: dict[str, float],
        stresses: dict[str, float],
        sensor_states: dict[str, str],
        applied: dict[str, int],
        sources: dict[str, str],
        breakdowns: dict[str, list[tuple[str, float, float, float]]],
        any_critical: bool,
    ) -> dict[str, Any]:
        sensors_state = []
        for name, sc in self.sensor_configs.items():
            sm = self.sensor_models[name]
            sensors_state.append({
                "name": name,
                "chip": sc.chip,
                "label": sc.label,
                "T": sensor_temps.get(name),
                "T_eff": sensor_effective_temps.get(name),
                "read_state": sensor_states.get(name, "missing"),
                "target_c": sc.target_c,
                "critical_c": sc.critical_c,
                "stress": stresses.get(name),
                "trained": sm.is_trained(),
                "n_samples": sm.state.n_samples,
                "r2": sm.state.r2,
                "rmse": sm.state.rmse,
            })

        fans_state = []
        for fn, fc in self.fan_configs.items():
            rpm = None
            if fc.fan_tach:
                tach_key = fc.fan_tach.removesuffix("_input")
                rpm = obs.fans.get(tach_key)
            parts = breakdowns.get(fn, [])
            driving = [
                {"sensor": s, "stress": round(st, 3), "relevance": round(r, 3), "contribution": round(c, 3)}
                for s, st, r, c in parts[:3]
            ]
            fans_state.append({
                "name": fn,
                "pwm_channel": fc.pwm_channel,
                "pwm_applied": applied.get(fn),
                "rpm": rpm,
                "source": sources.get(fn),
                "driving_sensors": driving,
            })

        return {
            "version": 2,
            "wall_t": obs.wall_t,
            "poll_interval_s": self.poll_interval,
            "hwmon": obs.hwmon,
            "fans": obs.fans,
            "pwm_observed": obs.pwm,
            "pwm_enable": obs.pwm_enable,
            "gpu": obs.gpu,
            "cpu_util": obs.cpu_util,
            "ups": obs.ups,
            "sensors": sensors_state,
            "fans_state": fans_state,
            "errors": obs.errors,
            "any_critical": any_critical,
        }

    # ---- main loop ---------------------------------------------------------

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        log.info("signal %d received, exiting cleanly", signum)
        self._stop = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        with self.actuators:
            sd_notify("READY=1")
            sd_notify(
                f"STATUS=running, {len(self.sensor_configs)} sensors, "
                f"{len(self.fan_configs)} fans"
            )
            log.info(
                "daemon entering main loop, poll_interval=%.1fs",
                self.poll_interval,
            )
            next_tick = time.monotonic()
            consecutive_failures = 0
            while not self._stop:
                try:
                    self._tick()
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    log.exception("tick failed (%d consecutive)", consecutive_failures)
                    if consecutive_failures >= TICK_FAIL_LIMIT:
                        # A loop that can't tick shouldn't keep running with
                        # stale PWM while the unit reports active. Exit
                        # non-zero: Restart=on-failure restarts us, and if
                        # the failure persists systemd's rate limit leaves
                        # the unit failed with BIOS modes restored (context
                        # manager + ExecStopPost) — if fand can't run, fans
                        # go back to BIOS.
                        sd_notify(
                            f"STATUS=exiting: {consecutive_failures} "
                            f"consecutive tick failures"
                        )
                        log.error(
                            "%d consecutive tick failures — exiting for "
                            "systemd to restart", consecutive_failures,
                        )
                        return 1
                # Watchdog ping covers failed ticks too: a tick that raises
                # is still a live loop (the failure-streak exit above
                # backstops persistent breakage); only a *hung* tick should
                # starve the watchdog. The old ping at the end of _tick()
                # skipped every exception path, so one slow source plus one
                # raise could watchdog-kill a healthy daemon.
                sd_notify("WATCHDOG=1")
                next_tick += self.poll_interval
                sleep_for = max(0.0, next_tick - time.monotonic())
                if sleep_for == 0.0:
                    next_tick = time.monotonic()
                time.sleep(sleep_for)
            sd_notify("STOPPING=1")
        return 0


# ---- entrypoint -----------------------------------------------------------


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore-bios", action="store_true",
                    help="restore pwm_enable modes from saved snapshot and exit")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--zones", default=str(ZONES_PATH))
    args = ap.parse_args(argv)

    _setup_logging()

    if args.restore_bios:
        chip_name = "nct6799"
        try:
            cfg = yaml.safe_load(Path(args.config).read_text()) or {}
            chip_name = cfg.get("chip_name", chip_name)
        except (OSError, yaml.YAMLError) as exc:
            log.warning("restore: config unreadable (%s); using chip=%s", exc, chip_name)
        n = restore_from_disk(chip_name=chip_name)
        log.info("restored %d channels from %s", n, SAVED_STATE)
        return 0

    try:
        daemon = Daemon(config_path=Path(args.config), zones_path=Path(args.zones))
    except RuntimeError as exc:
        # Config / startup-validation errors: clean message, no traceback.
        log.error("%s", exc)
        return 2
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
