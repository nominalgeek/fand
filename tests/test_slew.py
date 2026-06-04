"""PWM slew limiting: fans follow the demand envelope, not per-tick noise."""

from __future__ import annotations

from conftest import make_obs

# config poll_interval_s=2.0 → per-tick steps: up 60, down 10
CONFIG = """\
poll_interval_s: 2.0
sensor_fail_grace_s: 30.0
pwm_slew_up_per_s: 30.0
pwm_slew_down_per_s: 5.0
"""


def _pwm1(d):
    return [v for ch, v in d.actuators.set_calls if ch == "pwm1"]


def test_slew_clamps_demand_swings(make_daemon):
    d = make_daemon([make_obs(100.0, cpu=45.0)], config_yaml=CONFIG)
    s = d.sensors

    # First tick: cool (stress 0) → unconstrained first write at pwm_min.
    s.script.append(make_obs(102.0, cpu=45.0))
    d._tick()
    assert _pwm1(d)[-1] == 40

    # Hot spike: target jumps to ~247, but slew allows +60/tick from 40.
    s.script.append(make_obs(104.0, cpu=89.0))
    d._tick()
    assert _pwm1(d)[-1] == 100

    # Spike continues: keeps climbing at +60.
    s.script.append(make_obs(106.0, cpu=89.0))
    d._tick()
    assert _pwm1(d)[-1] == 160

    # Sensor noise dips back to cool: decay is slow (−10/tick), no bounce.
    s.script.append(make_obs(108.0, cpu=45.0))
    d._tick()
    assert _pwm1(d)[-1] == 150
    s.script.append(make_obs(110.0, cpu=45.0))
    d._tick()
    assert _pwm1(d)[-1] == 140


def test_critical_bypasses_slew(make_daemon):
    d = make_daemon([make_obs(100.0, cpu=45.0)], config_yaml=CONFIG)
    s = d.sensors
    s.script.append(make_obs(102.0, cpu=45.0))
    d._tick()
    assert _pwm1(d)[-1] == 40

    # At/above critical_c (90): straight to 255, no ramp.
    s.script.append(make_obs(104.0, cpu=95.0))
    d._tick()
    assert _pwm1(d)[-1] == 255

    # Recovery decays from 255 at the slew-down rate, not a cliff.
    s.script.append(make_obs(106.0, cpu=45.0))
    d._tick()
    assert _pwm1(d)[-1] == 245


def test_steady_demand_passes_through(make_daemon):
    d = make_daemon([make_obs(100.0, cpu=70.0)], config_yaml=CONFIG)
    s = d.sensors
    # stress (70−50)/40 = 0.5 → demand 0.5^1.5 ≈ 0.354 → target 116
    s.script.append(make_obs(102.0, cpu=70.0))
    d._tick()
    first = _pwm1(d)[-1]
    assert first == 116
    s.script.append(make_obs(104.0, cpu=70.0))
    d._tick()
    assert _pwm1(d)[-1] == first  # steady demand → steady pwm
