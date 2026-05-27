"""Sensor readers: hwmon, nvidia-smi, /proc/stat CPU util, upsc.

All readers are tolerant of missing sources — they return None or empty dicts
and log a warning. The daemon is responsible for safety floors when readers fail.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

HWMON_ROOT = Path("/sys/class/hwmon")


@dataclass
class Observation:
    """Snapshot returned by Sensors.read_all() — one row per poll."""

    t: float  # monotonic timestamp
    wall_t: float  # unix epoch
    hwmon: dict[str, dict[str, float]] = field(default_factory=dict)
    fans: dict[str, int] = field(default_factory=dict)  # nct6799 fan_N: rpm
    pwm: dict[str, int] = field(default_factory=dict)  # nct6799 pwm_N: 0-255
    pwm_enable: dict[str, int] = field(default_factory=dict)
    gpu: dict[str, float] | None = None  # nvidia-smi
    cpu_util: float | None = None  # 0-100
    ups: dict[str, float] | None = None  # upsc-derived
    errors: list[str] = field(default_factory=list)


def _read_float_file(p: Path, scale: float = 1.0) -> float | None:
    try:
        return float(p.read_text().strip()) * scale
    except (OSError, ValueError) as exc:
        log.debug("read %s: %s", p, exc)
        return None


def _read_int_file(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError) as exc:
        log.debug("read %s: %s", p, exc)
        return None


def find_hwmon_by_name(name: str) -> Path | None:
    """Resolve /sys/class/hwmon/hwmonN by chip name (e.g. 'nct6799').

    hwmon indices can shift on kernel update; the chip's name file is stable.
    Returns None if the chip isn't present.
    """
    for hw in sorted(HWMON_ROOT.glob("hwmon*")):
        try:
            if hw.joinpath("name").read_text().strip() == name:
                return hw
        except OSError:
            continue
    return None


def find_all_hwmon() -> dict[str, list[Path]]:
    """Return {chip_name: [hwmon_path, ...]} — some chips appear multiple times (e.g. nvme×3)."""
    out: dict[str, list[Path]] = {}
    for hw in sorted(HWMON_ROOT.glob("hwmon*")):
        try:
            name = hw.joinpath("name").read_text().strip()
        except OSError:
            continue
        out.setdefault(name, []).append(hw)
    return out


def read_hwmon_temps(hw: Path) -> dict[str, float]:
    """Read all temp{N}_input from one hwmon dir, keyed by label (or 'tempN' fallback).

    Values returned in degrees Celsius. Files missing/unreadable are silently skipped.
    """
    temps: dict[str, float] = {}
    for inp in sorted(hw.glob("temp*_input")):
        prefix = inp.name.removesuffix("_input")  # 'temp1'
        val = _read_float_file(inp, scale=1e-3)
        if val is None:
            continue
        label_path = hw / f"{prefix}_label"
        label = label_path.read_text().strip() if label_path.exists() else prefix
        # Filter obvious bogus thermistor reads (nct6799 ships unused thermistors at -60°C).
        if val < -50 or val > 150:
            continue
        temps[label] = val
    return temps


def read_hwmon_fans_pwm(hw: Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Read fan{N}_input + pwm{N} + pwm{N}_enable from one hwmon dir.

    Returns (fans, pwms, enables) keyed by their N (e.g. 'fan1', 'pwm1').
    """
    fans: dict[str, int] = {}
    pwms: dict[str, int] = {}
    enables: dict[str, int] = {}
    for inp in sorted(hw.glob("fan*_input")):
        key = inp.name.removesuffix("_input")
        v = _read_int_file(inp)
        if v is not None:
            fans[key] = v
    for pwm in sorted(hw.glob("pwm*")):
        if pwm.name.endswith("_enable") or "_" in pwm.name:
            continue
        v = _read_int_file(pwm)
        if v is not None:
            pwms[pwm.name] = v
        en = hw / f"{pwm.name}_enable"
        if en.exists():
            ev = _read_int_file(en)
            if ev is not None:
                enables[pwm.name] = ev
    return fans, pwms, enables


def read_nvidia_gpu() -> dict[str, float] | None:
    """Call nvidia-smi for temp / fan% / power / util. Returns None on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        log.warning("nvidia-smi failed: %s", exc)
        return None
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    try:
        temp, fan, power, util = (s.strip() for s in line.split(","))
        return {
            "temp_c": float(temp),
            "fan_pct": float(fan),
            "power_w": float(power),
            "util_pct": float(util),
        }
    except ValueError:
        log.warning("nvidia-smi parse failed: %r", line)
        return None


def _read_proc_stat_cpu() -> tuple[int, int] | None:
    """Return (total_jiffies, idle_jiffies) from the first 'cpu' line of /proc/stat."""
    try:
        first = Path("/proc/stat").read_text().splitlines()[0]
    except OSError as exc:
        log.warning("read /proc/stat: %s", exc)
        return None
    parts = first.split()
    if not parts or parts[0] != "cpu":
        return None
    nums = [int(x) for x in parts[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
    return sum(nums), idle


def read_ups(name: str = "cyberpower") -> dict[str, float] | None:
    """Parse `upsc <name>` output into floats we care about. None on failure."""
    try:
        out = subprocess.run(
            ["upsc", name],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        log.warning("upsc failed: %s", exc)
        return None
    kv: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()
    try:
        load_pct = float(kv.get("ups.load", "0"))
        nominal_w = float(kv.get("ups.realpower.nominal", "0"))
        realpower_w = nominal_w * load_pct / 100.0
        return {
            "load_pct": load_pct,
            "nominal_w": nominal_w,
            "realpower_w": realpower_w,
            "input_v": float(kv.get("input.voltage", "0")),
            "battery_pct": float(kv.get("battery.charge", "0")),
            "on_line": 1.0 if kv.get("ups.status", "").startswith("OL") else 0.0,
        }
    except ValueError as exc:
        log.warning("upsc parse failed: %s", exc)
        return None


class Sensors:
    """Stateful sensor poller.

    Stateful only for CPU util (needs two /proc/stat samples to compute %).
    """

    def __init__(self, ups_name: str = "cyberpower"):
        self.ups_name = ups_name
        self._cpu_last: tuple[int, int] | None = None
        self._chips = find_all_hwmon()
        log.info("hwmon chips found: %s", {k: len(v) for k, v in self._chips.items()})

    def read_all(self) -> Observation:
        obs = Observation(t=time.monotonic(), wall_t=time.time())

        for name, paths in self._chips.items():
            for i, hw in enumerate(paths):
                key = name if len(paths) == 1 else f"{name}[{i}]"
                try:
                    temps = read_hwmon_temps(hw)
                    if temps:
                        obs.hwmon[key] = temps
                except OSError as exc:
                    obs.errors.append(f"hwmon {key}: {exc}")

        nct = find_hwmon_by_name("nct6799")
        if nct is not None:
            try:
                obs.fans, obs.pwm, obs.pwm_enable = read_hwmon_fans_pwm(nct)
            except OSError as exc:
                obs.errors.append(f"nct6799 fans/pwm: {exc}")
        else:
            obs.errors.append("nct6799 not found")

        obs.gpu = read_nvidia_gpu()
        if obs.gpu is None:
            obs.errors.append("nvidia-smi unavailable")

        cur = _read_proc_stat_cpu()
        if cur is not None and self._cpu_last is not None:
            total_d = cur[0] - self._cpu_last[0]
            idle_d = cur[1] - self._cpu_last[1]
            if total_d > 0:
                obs.cpu_util = 100.0 * (1.0 - idle_d / total_d)
        if cur is not None:
            self._cpu_last = cur

        obs.ups = read_ups(self.ups_name)
        if obs.ups is None:
            obs.errors.append("upsc unavailable")

        return obs
