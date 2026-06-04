"""Actuators against a tmp fake hwmon tree: restore scoping (review issue 1)
and set_all failure reporting (issue 4)."""

from __future__ import annotations

import json

import fand.actuators as act


def _fake_hwmon(tmp_path, channels: dict[str, str]):
    """Build a fake chip dir: {channel: enable_value}. pwm files start at 100."""
    hw = tmp_path / "hwmon0"
    hw.mkdir()
    for ch, enable in channels.items():
        (hw / ch).write_text("100")
        (hw / f"{ch}_enable").write_text(enable)
    return hw


def test_restore_without_snapshot_touches_nothing(tmp_path, monkeypatch):
    hw = _fake_hwmon(tmp_path, {"pwm1": "5", "pwm2": "2"})
    monkeypatch.setattr(act, "find_hwmon_by_name", lambda name: hw)
    monkeypatch.setattr(act, "SAVED_STATE", tmp_path / "saved_pwm.json")  # absent

    assert act.restore_from_disk() == 0
    # The old behavior forced mode 5 onto every pwm*_enable — pwm2 must keep
    # its BIOS-owned mode 2.
    assert (hw / "pwm1_enable").read_text() == "5"
    assert (hw / "pwm2_enable").read_text() == "2"


def test_restore_with_snapshot_restores_only_saved_channels(tmp_path, monkeypatch):
    hw = _fake_hwmon(tmp_path, {"pwm1": "1", "pwm2": "2"})
    saved = tmp_path / "saved_pwm.json"
    saved.write_text(json.dumps({"pwm1": {"pwm": 120, "enable": 5}}))
    monkeypatch.setattr(act, "find_hwmon_by_name", lambda name: hw)
    monkeypatch.setattr(act, "SAVED_STATE", saved)

    assert act.restore_from_disk() == 1
    assert (hw / "pwm1_enable").read_text() == "5"
    assert (hw / "pwm2_enable").read_text() == "2"  # unmanaged, untouched


def test_set_all_reports_failed_channels(tmp_path, monkeypatch):
    hw = _fake_hwmon(tmp_path, {"pwm1": "1"})
    # pwm2's value node is a directory: write_text raises OSError → set_pwm False.
    (hw / "pwm2").mkdir()
    (hw / "pwm2_enable").write_text("1")
    monkeypatch.setattr(act, "find_hwmon_by_name", lambda name: hw)

    a = act.Actuators(["pwm1", "pwm2"])
    assert a.set_all(200) == ["pwm2"]
    assert (hw / "pwm1").read_text() == "200"  # good channel still written


def test_set_pwm_clamps_and_verifies(tmp_path, monkeypatch):
    hw = _fake_hwmon(tmp_path, {"pwm1": "1"})
    monkeypatch.setattr(act, "find_hwmon_by_name", lambda name: hw)
    a = act.Actuators(["pwm1"])

    assert a.set_pwm("pwm1", 999) is True
    assert (hw / "pwm1").read_text() == "255"  # clamped
    assert a.set_pwm("pwm1", -5) is True
    assert (hw / "pwm1").read_text() == "0"
