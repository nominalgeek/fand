"""zones.yaml validation at daemon init (review issue 7) + compaction (issue 6)."""

from __future__ import annotations

import json

import pytest

import fand.daemon as daemon_mod
from fand.model import EquilibriumSample

from conftest import ZONES_YAML, make_obs


def test_inverted_critical_target_rejected(make_daemon):
    zones = ZONES_YAML.replace("critical_c: 90.0", "critical_c: 45.0")  # < target 50
    with pytest.raises(RuntimeError, match="validation error"):
        make_daemon([], zones_yaml=zones)


def test_negative_cools_weight_rejected(make_daemon):
    zones = ZONES_YAML.replace("cools: {cpu: 2.0}", "cools: {cpu: -1.0}")
    with pytest.raises(RuntimeError, match="validation error"):
        make_daemon([], zones_yaml=zones)


def test_pwm_min_out_of_range_rejected(make_daemon):
    zones = ZONES_YAML.replace("pwm_min: 40", "pwm_min: 300")
    with pytest.raises(RuntimeError, match="validation error"):
        make_daemon([], zones_yaml=zones)


def test_bad_on_unreadable_rejected(make_daemon):
    zones = ZONES_YAML.replace(
        "critical_c: 90.0", "critical_c: 90.0\n    on_unreadable: shrug"
    )
    with pytest.raises(RuntimeError, match="on_unreadable"):
        make_daemon([], zones_yaml=zones)


def test_equilibria_compaction_trims_file(make_daemon, monkeypatch, tmp_path):
    monkeypatch.setattr(daemon_mod, "EQUILIBRIA_COMPACT_EVERY", 5)
    d = make_daemon([make_obs(100.0, cpu=60.0)])
    eq_path = tmp_path / "equilibria.jsonl"

    # Simulate stale appends from a prior run that the deque no longer holds.
    eq_path.write_text("".join(
        json.dumps({"t": float(i), "fan": "stale"}) + "\n" for i in range(50)
    ))

    def sample(i: int) -> EquilibriumSample:
        return EquilibriumSample(
            t=float(i), fan="pwm1", features={}, pwm={"pwm1": 100}, temps={"cpu": 60.0}
        )

    for i in range(4):
        d._append_equilibrium(sample(i))
    assert len(eq_path.read_text().splitlines()) == 54  # stale + 4, no trim yet

    d._append_equilibrium(sample(4))  # 5th append → forced atomic rewrite
    lines = eq_path.read_text().splitlines()
    assert len(lines) == len(d.equilibria) == 5
    assert all(json.loads(ln)["fan"] == "pwm1" for ln in lines)
