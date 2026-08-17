"""Missing-sensor fail-safe: hold → assume-critical → recovery (review issue 3)."""

from __future__ import annotations

import json

import pytest

from conftest import CONFIG_YAML, ZONES_YAML, make_obs


def _status(status_path):
    return json.loads(status_path.read_text())


def _sensor(status, name):
    return next(s for s in status["sensors"] if s["name"] == name)


def test_hold_then_escalate_then_recover(make_daemon, status_path):
    d = make_daemon([make_obs(100.0, cpu=60.0)])  # startup verify
    s = d.sensors

    # Fresh read: normal control.
    s.script.append(make_obs(102.0, cpu=60.0))
    d._tick()
    st = _status(status_path)
    assert _sensor(st, "cpu")["read_state"] == "ok"
    assert not st["any_critical"]

    # Unreadable within grace: last reading (60.0) stands in, no roar.
    s.script.append(make_obs(104.0))
    d._tick()
    st = _status(status_path)
    cpu = _sensor(st, "cpu")
    assert cpu["read_state"] == "hold"
    assert cpu["T"] == 60.0  # held value still drives stress + critical checks
    assert not st["any_critical"]
    # Held value must not enter the equilibrium window (learning pool).
    assert "cpu" not in d.windows["pwm1"][-1][1]

    # Past grace (age 36s > 30s): assume critical, all fans max.
    s.script.append(make_obs(138.0))
    d._tick()
    st = _status(status_path)
    assert _sensor(st, "cpu")["read_state"] == "assumed-critical"
    assert st["any_critical"]
    assert ("pwm1", 255) in d.actuators.set_calls

    # Recovery is anti-flap: needs 3 consecutive good reads to clear.
    for t, still_critical in ((140.0, True), (142.0, True), (144.0, False)):
        s.script.append(make_obs(t, cpu=60.0))
        d._tick()
        st = _status(status_path)
        assert st["any_critical"] is still_critical, f"at t={t}"
    assert _sensor(st, "cpu")["read_state"] == "ok"


def test_recovery_streak_resets_on_intermittent_reads(make_daemon, status_path):
    d = make_daemon([make_obs(100.0, cpu=60.0)])
    s = d.sensors
    s.script.append(make_obs(102.0, cpu=60.0))
    d._tick()
    s.script.append(make_obs(140.0))  # escalate (age 38s)
    d._tick()
    assert _status(status_path)["any_critical"]

    # good, good, MISS, good, good — never 3 consecutive: stays escalated.
    for t, cpu in ((142.0, 60.0), (144.0, 60.0), (146.0, None), (148.0, 60.0), (150.0, 60.0)):
        s.script.append(make_obs(t, cpu=cpu))
        d._tick()
        assert _status(status_path)["any_critical"], f"flapped open at t={t}"


def test_on_unreadable_alarm_opts_out_of_roar(make_daemon, status_path):
    zones = ZONES_YAML.replace("critical_c: 90.0", "critical_c: 90.0\n    on_unreadable: alarm")
    d = make_daemon([make_obs(100.0, cpu=60.0)], zones_yaml=zones)
    s = d.sensors
    s.script.append(make_obs(102.0, cpu=60.0))
    d._tick()

    s.script.append(make_obs(140.0))  # past grace
    d._tick()
    st = _status(status_path)
    assert _sensor(st, "cpu")["read_state"] == "missing"
    assert not st["any_critical"]
    # Excluded sensor → no stress → fan slews down toward pwm_min (was 67
    # at stress 0.25; −10/tick), never to 255.
    assert d.actuators.set_calls[-1] == ("pwm1", 57)


def test_never_read_optional_source_excluded_not_escalated(make_daemon, status_path):
    zones = ZONES_YAML.replace("fans:", """\
  gpu_core:
    chip: gpu
    label: temp_c
    target_c: 80.0
    critical_c: 90.0
fans:""")
    # nvidia-smi absent at startup and at runtime: warn + exclude, never roar.
    d = make_daemon([make_obs(100.0, cpu=60.0)], zones_yaml=zones)
    s = d.sensors
    s.script.append(make_obs(102.0, cpu=60.0))
    d._tick()
    st = _status(status_path)
    assert _sensor(st, "gpu_core")["read_state"] == "missing"
    assert not st["any_critical"]


def test_startup_fails_fast_on_bad_label(make_daemon):
    zones = ZONES_YAML.replace("label: Tctl", "label: Tdie")
    with pytest.raises(RuntimeError, match="unreadable at startup"):
        make_daemon([make_obs(100.0, cpu=60.0)], zones_yaml=zones)


def test_startup_fails_fast_on_missing_chip(make_daemon):
    zones = ZONES_YAML.replace("chip: k10temp", "chip: nct9999")
    with pytest.raises(RuntimeError, match="unreadable at startup"):
        make_daemon([make_obs(100.0, cpu=60.0)], zones_yaml=zones)


def test_grace_period_configurable(make_daemon, status_path):
    cfg = CONFIG_YAML.replace("sensor_fail_grace_s: 30.0", "sensor_fail_grace_s: 500.0")
    d = make_daemon([make_obs(100.0, cpu=60.0)], config_yaml=cfg)
    s = d.sensors
    s.script.append(make_obs(102.0, cpu=60.0))
    d._tick()
    s.script.append(make_obs(400.0))  # 298s gap — still inside 500s grace
    d._tick()
    st = _status(status_path)
    assert _sensor(st, "cpu")["read_state"] == "hold"
    assert not st["any_critical"]
