"""fand main daemon.

Run modes:
  python -m fand.daemon                       # main poll loop (systemd Type=notify)
  python -m fand.daemon --restore-bios        # ExecStopPost: restore pwm enable modes and exit
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import logging.handlers
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .actuators import Actuators, SAVED_STATE, restore_from_disk
from .model import (
    EquilibriumSample,
    ZoneModel,
    is_equilibrium,
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
    """Look up a temperature reading from the observation's hwmon dict."""
    if obs.hwmon.get(chip) and label in obs.hwmon[chip]:
        return obs.hwmon[chip][label]
    # Some chips appear with [i] suffix when duplicated (e.g. nvme[0], nvme[1]).
    for key, temps in obs.hwmon.items():
        if key.startswith(chip + "[") and label in temps:
            return temps[label]
    # GPU is queried separately
    if chip == "gpu" and obs.gpu and label in obs.gpu:
        return obs.gpu[label]
    return None


def _build_features(obs: Observation) -> list[float]:
    util_gpu = obs.gpu["util_pct"] if obs.gpu else 0.0
    power_gpu = obs.gpu["power_w"] if obs.gpu else 0.0
    util_cpu = obs.cpu_util if obs.cpu_util is not None else 0.0
    ups_w = obs.ups["realpower_w"] if obs.ups else 0.0
    return [util_gpu, power_gpu, util_cpu, ups_w]


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
        current_day.append(today)
        return
    if today == current_day[0]:
        return
    # Date changed: gzip the old file
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
    # Prune old archives
    cutoff = time.time() - HISTORY_RETENTION_DAYS * 86400
    for old in path.parent.glob(f"{path.name}.*.gz"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
                log.info("history pruned %s", old)
        except OSError:
            pass


# ---- main daemon class ----------------------------------------------------


class Daemon:
    def __init__(self, config_path: Path = CONFIG_PATH, zones_path: Path = ZONES_PATH):
        self.config = yaml.safe_load(config_path.read_text()) or {}
        self.zones_cfg = yaml.safe_load(zones_path.read_text()) or {}
        self.poll_interval = float(self.config.get("poll_interval_s", 2.0))
        self.fit_interval = float(self.config.get("fit_interval_s", 3600.0))
        self.equilibrium_window_s = float(self.config.get("equilibrium_window_s", 30.0))
        self.equilibrium_window_n = max(8, int(self.equilibrium_window_s / self.poll_interval))
        self.ntfy_cmd = self.config.get("ntfy_command")
        self.history_enabled = bool(self.config.get("history_enabled", True))

        self.sensors = Sensors(ups_name=self.config.get("ups_name", "cyberpower"))

        zones = self.zones_cfg.get("zones") or []
        if not zones:
            raise RuntimeError(f"no zones defined in {zones_path}")

        channels = [z["pwm_channel"] for z in zones]
        self.actuators = Actuators(channels)

        self.zones: list[dict[str, Any]] = zones
        self.models: dict[str, ZoneModel] = {}
        for z in zones:
            primary = z["target_sensors"][0]
            self.models[z["name"]] = ZoneModel(
                name=z["name"],
                pwm_min=int(z["pwm_min"]),
                target_c=float(primary["target_c"]),
                critical_c=float(primary.get("critical_c", 90.0)),
                bootstrap_curve=[(float(t), int(p)) for t, p in z.get("bootstrap_curve", [
                    (40.0, 100), (60.0, 160), (75.0, 210), (85.0, 255)
                ])],
                ridge_lambda=float(self.config.get("ridge_lambda", 1.0)),
                min_samples=int(self.config.get("min_samples_to_learn", 200)),
                min_r2=float(self.config.get("min_r2_to_learn", 0.7)),
                ff_gain_per_feature=z.get("ff_gain_per_feature"),
                ff_alpha=float(self.config.get("ff_alpha", 0.05)),
                margin_c=float(self.config.get("margin_c", 5.0)),
            )
        load_models_into(MODEL_PATH, self.models)

        # Equilibrium tracking per zone
        self.windows: dict[str, deque[tuple[float, float, list[float]]]] = {
            z["name"]: deque(maxlen=self.equilibrium_window_n) for z in zones
        }
        self.equilibria: dict[str, deque[EquilibriumSample]] = {
            z["name"]: deque(maxlen=5000) for z in zones
        }
        self._load_equilibria()

        self.last_fit_t = 0.0
        self.last_history_day: list[str] = []  # mutable closure for rotation
        self._stop = False
        self._last_alarm_t: dict[str, float] = {}  # de-dup ntfy alerts

    def _load_equilibria(self) -> None:
        if not EQUILIBRIA_PATH.exists():
            return
        try:
            for line in EQUILIBRIA_PATH.read_text().splitlines():
                row = json.loads(line)
                z = row.pop("zone")
                if z in self.equilibria:
                    self.equilibria[z].append(EquilibriumSample(**row))
            log.info(
                "loaded equilibria: %s",
                {k: len(v) for k, v in self.equilibria.items()},
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            log.warning("equilibria load failed: %s", exc)

    def _append_equilibrium(self, zone_name: str, sample: EquilibriumSample) -> None:
        self.equilibria[zone_name].append(sample)
        try:
            EQUILIBRIA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EQUILIBRIA_PATH.open("a") as f:
                row = asdict(sample) | {"zone": zone_name}
                f.write(json.dumps(row) + "\n")
        except OSError as exc:
            log.warning("equilibrium persist failed: %s", exc)

    def _alarm(self, key: str, message: str, cooldown_s: float = 300.0) -> None:
        """Emit an ntfy alert at most once per cooldown for the given key."""
        now = time.monotonic()
        if now - self._last_alarm_t.get(key, 0) < cooldown_s:
            return
        self._last_alarm_t[key] = now
        log.warning("ALARM %s: %s", key, message)
        _ntfy(self.ntfy_cmd, f"[fand] {message}")

    def _tick(self) -> None:
        obs = self.sensors.read_all()

        # ---- determine per-zone temps and check critical floor --------------
        zone_temps: dict[str, float | None] = {}
        zone_critical: dict[str, bool] = {}
        any_critical = False
        any_sensor_missing = False
        for z in self.zones:
            primary = z["target_sensors"][0]
            T_z = _temp_at(obs, primary["chip"], primary["label"])
            zone_temps[z["name"]] = T_z
            critical_c = float(primary.get("critical_c", 90.0))
            crit = T_z is not None and T_z >= critical_c
            zone_critical[z["name"]] = crit
            if crit:
                any_critical = True
                self._alarm(
                    f"critical:{z['name']}",
                    f"{z['name']} primary sensor {primary['chip']}/{primary['label']} "
                    f"= {T_z:.1f}°C ≥ critical {critical_c:.1f}°C",
                )
            if T_z is None:
                any_sensor_missing = True
                self._alarm(
                    f"missing:{z['name']}",
                    f"{z['name']} primary sensor {primary['chip']}/{primary['label']} unreadable",
                )

        # Also check NON-primary sensors per zone for criticals (e.g. NVMe temps
        # might map to the same zone as the case fan; we still want to react)
        for z in self.zones:
            for s in z["target_sensors"][1:]:
                T = _temp_at(obs, s["chip"], s["label"])
                if T is None:
                    continue
                cc = float(s.get("critical_c", 90.0))
                if T >= cc:
                    any_critical = True
                    self._alarm(
                        f"critical:{z['name']}:{s['label']}",
                        f"{z['name']} secondary {s['chip']}/{s['label']} "
                        f"= {T:.1f}°C ≥ critical {cc:.1f}°C",
                    )

        # ---- decide PWMs ----------------------------------------------------
        applied: dict[str, int] = {}
        sources: dict[str, str] = {}
        if any_critical:
            self.actuators.set_all(255)
            applied = {z["name"]: 255 for z in self.zones}
            sources = {z["name"]: "critical" for z in self.zones}
        else:
            T_amb = _temp_at(obs, "nct6799", "SYSTIN") or 25.0
            features = _build_features(obs)
            for z in self.zones:
                T_z = zone_temps[z["name"]]
                if T_z is None:
                    # Sensor missing for this zone — boost only this zone (not all)
                    self.actuators.set_pwm(z["pwm_channel"], 255)
                    applied[z["name"]] = 255
                    sources[z["name"]] = "sensor_fault"
                    continue
                model = self.models[z["name"]]
                pwm, source = model.predict_pwm(features, T_z, T_amb)
                self.actuators.set_pwm(z["pwm_channel"], pwm)
                applied[z["name"]] = pwm
                sources[z["name"]] = source

        # ---- fan-fault detection (tach=0 while PWM≥min for sustained period)
        for z in self.zones:
            tach = obs.fans.get(z.get("fan_tach", "").replace("_input", ""))
            if tach is None:
                continue
            if applied.get(z["name"], 0) >= z["pwm_min"] and tach < 100:
                self._alarm(
                    f"fan_dead:{z['name']}",
                    f"{z['name']} commanded pwm={applied[z['name']]} but tach={tach} RPM",
                )

        # ---- update equilibrium windows + collect training samples ----------
        if not any_critical and not any_sensor_missing:
            features = _build_features(obs)
            T_amb = _temp_at(obs, "nct6799", "SYSTIN") or 25.0
            for z in self.zones:
                T_z = zone_temps[z["name"]]
                if T_z is None:
                    continue
                w = self.windows[z["name"]]
                w.append((obs.t, T_z, features))
                if len(w) >= self.equilibrium_window_n and is_equilibrium(list(w)):
                    sample = EquilibriumSample(
                        t=obs.t,
                        features=features,
                        pwm_applied=applied[z["name"]],
                        T_z=T_z,
                        T_amb=T_amb,
                    )
                    self._append_equilibrium(z["name"], sample)
                    # Clear the window so we don't record N samples for one plateau
                    w.clear()

        # ---- periodic refit -------------------------------------------------
        if obs.t - self.last_fit_t > self.fit_interval:
            for name, model in self.models.items():
                model.fit(list(self.equilibria[name]))
            save_models(MODEL_PATH, self.models)
            self.last_fit_t = obs.t

        # ---- write status + history -----------------------------------------
        status = {
            "version": 1,
            "wall_t": obs.wall_t,
            "poll_interval_s": self.poll_interval,
            "hwmon": obs.hwmon,
            "fans": obs.fans,
            "pwm_observed": obs.pwm,
            "pwm_enable": obs.pwm_enable,
            "gpu": obs.gpu,
            "cpu_util": obs.cpu_util,
            "ups": obs.ups,
            "zones": [
                {
                    "name": z["name"],
                    "T_z": zone_temps[z["name"]],
                    "pwm_applied": applied.get(z["name"]),
                    "source": sources.get(z["name"]),
                    "trained": self.models[z["name"]].is_trained(),
                    "n_samples": self.models[z["name"]].state.n_samples,
                    "r2": self.models[z["name"]].state.r2,
                }
                for z in self.zones
            ],
            "errors": obs.errors,
            "any_critical": any_critical,
        }
        _atomic_write_json(STATUS_PATH, status)

        if self.history_enabled:
            _maybe_rotate_history(HISTORY_PATH, self.last_history_day)
            try:
                with HISTORY_PATH.open("a") as f:
                    f.write(json.dumps(status, default=str) + "\n")
            except OSError as exc:
                log.warning("history append failed: %s", exc)

        sd_notify("WATCHDOG=1")

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        log.info("signal %d received, exiting cleanly", signum)
        self._stop = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        with self.actuators:
            sd_notify("READY=1")
            sd_notify(f"STATUS=running, {len(self.zones)} zones")
            log.info("daemon entering main loop, poll_interval=%.1fs", self.poll_interval)
            next_tick = time.monotonic()
            while not self._stop:
                try:
                    self._tick()
                except Exception:
                    log.exception("tick failed")
                    # Don't bail on transient errors — safety floor is the daemon's job.
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
        n = restore_from_disk()
        log.info("restored %d channels from %s", n, SAVED_STATE)
        return 0

    daemon = Daemon(config_path=Path(args.config), zones_path=Path(args.zones))
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
