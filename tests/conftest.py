"""Shared fakes for fand unit tests.

The daemon under test is constructed against real config/zones files in a
tmp dir, with `Sensors` and `Actuators` swapped for fakes and the module's
state paths pointed into the tmp dir. Observations are scripted per test:
startup verification consumes the first one, each `_tick()` consumes one
more. No hardware, no root, no subprocesses.
"""

from __future__ import annotations

import pytest

import fand.daemon as daemon_mod
from fand.sensors import Observation


ZONES_YAML = """\
defaults:
  pwm_min: 40
sensors:
  cpu:
    chip: k10temp
    label: Tctl
    target_c: 50.0
    critical_c: 90.0
fans:
  pwm1:
    fan_tach: fan1_input
    cools: {cpu: 2.0}
"""

CONFIG_YAML = """\
poll_interval_s: 2.0
sensor_fail_grace_s: 30.0
"""


class FakeSensors:
    """Scripted observation source. Tests append to `script`; read_all pops."""

    def __init__(self) -> None:
        self.script: list[Observation] = []

    def read_all(self) -> Observation:
        assert self.script, "FakeSensors script exhausted — append more observations"
        return self.script.pop(0)


class FakeActuators:
    """Records writes; per-channel failure injectable via `fail_channels`."""

    def __init__(self, channels: list[str], chip_name: str = "nct6799") -> None:
        self.handles = dict.fromkeys(channels)
        self.set_calls: list[tuple[str, int]] = []
        self.take_over_calls = 0
        self.fail_channels: set[str] = set()

    def set_pwm(self, channel: str, value: int) -> bool:
        self.set_calls.append((channel, int(value)))
        return channel not in self.fail_channels

    def set_all(self, value: int) -> list[str]:
        return [ch for ch in self.handles if not self.set_pwm(ch, value)]

    def take_over(self) -> None:
        self.take_over_calls += 1


def make_obs(t: float, cpu: float | None = None) -> Observation:
    """Observation with the test sensor present (cpu=°C) or absent (None)."""
    o = Observation(t=t, wall_t=1.7e9 + t)
    if cpu is not None:
        o.hwmon = {"k10temp": {"Tctl": float(cpu)}}
    return o


@pytest.fixture
def make_daemon(tmp_path, monkeypatch):
    """Factory: build a Daemon against fakes + tmp state.

    `script` seeds the fake sensor observations (the first is consumed by
    startup verification). Returns the daemon; its .sensors.script can be
    extended and ._tick() driven directly.
    """

    def _make(
        script: list[Observation],
        zones_yaml: str = ZONES_YAML,
        config_yaml: str = CONFIG_YAML,
    ) -> daemon_mod.Daemon:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(config_yaml)
        zns = tmp_path / "zones.yaml"
        zns.write_text(zones_yaml)

        fake_sensors = FakeSensors()
        fake_sensors.script.extend(script)
        monkeypatch.setattr(daemon_mod, "Sensors", lambda *a, **k: fake_sensors)
        monkeypatch.setattr(daemon_mod, "Actuators", FakeActuators)
        monkeypatch.setattr(daemon_mod, "MODEL_PATH", tmp_path / "model.json")
        monkeypatch.setattr(daemon_mod, "EQUILIBRIA_PATH", tmp_path / "equilibria.jsonl")
        monkeypatch.setattr(daemon_mod, "HISTORY_PATH", tmp_path / "history.jsonl")
        monkeypatch.setattr(daemon_mod, "STATUS_PATH", tmp_path / "status.json")
        return daemon_mod.Daemon(config_path=cfg, zones_path=zns)

    return _make


@pytest.fixture
def status_path(tmp_path):
    return tmp_path / "status.json"
