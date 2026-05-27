"""PWM actuators with snapshot/restore.

Writes go to /sys/class/hwmon/<nct>/pwm{N}. Mode is managed via pwm{N}_enable:
  0 = full speed (255)
  1 = manual (we set the value)
  2 = motherboard auto (thermal cruise)
  5 = SmartFan IV / BIOS-defined curve  ← Linux default on this board

We snapshot the original mode + value on startup to /var/lib/fand/saved_pwm.json,
flip to mode=1, and restore on shutdown. If the snapshot file is missing on restore
(e.g. dirty crash recovery via systemctl restart), we fall back to mode=5 which is
the safe BIOS-controlled default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .sensors import find_hwmon_by_name

log = logging.getLogger(__name__)

SAVED_STATE = Path("/var/lib/fand/saved_pwm.json")
DEFAULT_RESTORE_MODE = 5  # BIOS SmartFan


@dataclass
class PWMHandle:
    name: str  # e.g. 'pwm1'
    pwm_path: Path
    enable_path: Path


class Actuators:
    """Owns the nct6799 PWM channels named in the zone config.

    Use as a context manager:

        with Actuators(['pwm1','pwm2','pwm7']) as act:
            act.set_pwm('pwm1', 128)
            ...
    """

    def __init__(self, channels: list[str], chip_name: str = "nct6799"):
        hw = find_hwmon_by_name(chip_name)
        if hw is None:
            raise RuntimeError(f"hwmon chip {chip_name!r} not found")
        self.hwmon = hw
        self.handles: dict[str, PWMHandle] = {}
        for ch in channels:
            pwm_path = hw / ch
            enable_path = hw / f"{ch}_enable"
            if not pwm_path.exists() or not enable_path.exists():
                raise RuntimeError(f"pwm node missing: {pwm_path}")
            self.handles[ch] = PWMHandle(ch, pwm_path, enable_path)
        self._snapshot: dict[str, dict[str, int]] | None = None

    # ----- snapshot / restore ---------------------------------------------

    def _read_state(self) -> dict[str, dict[str, int]]:
        state: dict[str, dict[str, int]] = {}
        for ch, h in self.handles.items():
            try:
                state[ch] = {
                    "pwm": int(h.pwm_path.read_text().strip()),
                    "enable": int(h.enable_path.read_text().strip()),
                }
            except (OSError, ValueError) as exc:
                log.warning("snapshot read %s: %s", ch, exc)
        return state

    def snapshot(self) -> None:
        """Capture current state to memory + disk so ExecStopPost can restore."""
        self._snapshot = self._read_state()
        SAVED_STATE.parent.mkdir(parents=True, exist_ok=True)
        SAVED_STATE.write_text(json.dumps(self._snapshot, indent=2))
        log.info("snapshot saved: %s", self._snapshot)

    def take_over(self) -> None:
        """Flip every managed channel to manual mode (1). Caller is responsible
        for writing a sane initial PWM via set_pwm() before takeover.

        Raises RuntimeError if any channel write failed — otherwise the daemon
        would silently run with the chip still under BIOS control on that
        channel, and set_all(255) on a critical would no-op.
        """
        failed: list[str] = []
        for ch, h in self.handles.items():
            try:
                h.enable_path.write_text("1")
            except OSError as exc:
                log.error("take_over %s: %s", ch, exc)
                failed.append(ch)
        if failed:
            raise RuntimeError(f"take_over failed for channels: {', '.join(failed)}")

    def restore(self) -> None:
        """Restore the modes (and values) captured by snapshot(). If snapshot
        missing, restore mode=5 (BIOS auto) for every managed channel.
        """
        if self._snapshot is None:
            try:
                self._snapshot = json.loads(SAVED_STATE.read_text())
            except (OSError, json.JSONDecodeError):
                self._snapshot = None

        for ch, h in self.handles.items():
            saved = self._snapshot.get(ch, {}) if self._snapshot else {}
            target_mode = saved.get("enable", DEFAULT_RESTORE_MODE)
            try:
                # If returning to manual (mode=1), write the captured pwm first
                # so the channel doesn't sit at fand's last-commanded value
                # (potentially 255) after we flip enable.
                if target_mode == 1 and "pwm" in saved:
                    h.pwm_path.write_text(str(saved["pwm"]))
                h.enable_path.write_text(str(target_mode))
                log.info("restored %s -> enable=%d", ch, target_mode)
            except OSError as exc:
                log.error("restore %s: %s", ch, exc)

    # ----- runtime --------------------------------------------------------

    def set_pwm(self, channel: str, value: int) -> bool:
        """Write a PWM value. Clamps to [0, 255]. Returns True if write succeeded
        AND the read-back matched (within ±2 to allow chip quantization)."""
        h = self.handles.get(channel)
        if h is None:
            raise KeyError(channel)
        v = max(0, min(255, int(value)))
        try:
            h.pwm_path.write_text(str(v))
        except OSError as exc:
            log.error("set_pwm %s=%d: %s", channel, v, exc)
            return False
        try:
            back = int(h.pwm_path.read_text().strip())
        except (OSError, ValueError):
            return False
        if abs(back - v) > 2:
            log.warning("set_pwm %s: wrote %d, read back %d", channel, v, back)
            return False
        return True

    def set_all(self, value: int) -> None:
        """Emergency: drive every managed channel to one value."""
        for ch in self.handles:
            self.set_pwm(ch, value)

    # ----- context-manager glue ------------------------------------------

    def __enter__(self) -> "Actuators":
        self.snapshot()
        # Drive every managed channel to a safe-ish 60% before flipping to manual
        # so we never hit "manual mode with stale 0% pwm" during the transition.
        for ch in self.handles:
            self.set_pwm(ch, 160)
        self.take_over()
        return self

    def __exit__(self, *exc) -> None:
        self.restore()


def restore_from_disk(chip_name: str = "nct6799") -> int:
    """Standalone restore using only the saved-state file.

    Used by `fand --restore-bios` invoked from ExecStopPost (the daemon's
    Actuators object is gone by then). Returns the number of channels restored.
    """
    hw = find_hwmon_by_name(chip_name)
    if hw is None:
        log.error("restore: hwmon %s not found", chip_name)
        return 0
    try:
        saved: dict[str, dict[str, int]] = json.loads(SAVED_STATE.read_text())
        have_snapshot = True
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("restore: saved state missing/invalid (%s); using mode=%d", exc, DEFAULT_RESTORE_MODE)
        saved = {}
        have_snapshot = False
    n = 0
    if have_snapshot:
        # Restore only the channels fand was actually managing — touching
        # unmanaged channels would clobber whatever BIOS curve owns them.
        for ch, vals in saved.items():
            pwm_path = hw / ch
            enable_path = hw / f"{ch}_enable"
            if not enable_path.exists():
                log.warning("restore: %s not present on chip %s; skipping", ch, chip_name)
                continue
            mode = vals.get("enable", DEFAULT_RESTORE_MODE)
            try:
                if mode == 1 and "pwm" in vals and pwm_path.exists():
                    pwm_path.write_text(str(vals["pwm"]))
                enable_path.write_text(str(mode))
                n += 1
            except OSError as exc:
                log.error("restore %s: %s", ch, exc)
    else:
        for pwm_enable in sorted(hw.glob("pwm*_enable")):
            ch = pwm_enable.name.removesuffix("_enable")
            try:
                pwm_enable.write_text(str(DEFAULT_RESTORE_MODE))
                n += 1
            except OSError as exc:
                log.error("restore %s: %s", ch, exc)
    return n
