"""SensorModel.fit: the per-fan seed floor on blended γ.

Closed-loop runtime data drags γ toward zero (the controller ramps fans
*because* temps rise), but a calibration seed is direct causal evidence the
fan cools the sensor — fitting must never erase it.
"""

from __future__ import annotations

from fand.model import EquilibriumSample, SensorModel


def _closed_loop_samples(n: int = 400) -> list[EquilibriumSample]:
    """PWM rises WITH temperature (controller chasing heat): the naive γ for
    pwm1 solves negative and clips to 0. pwm2 never moves (unidentified).
    Feature varies so it isn't folded into the baseline."""
    out = []
    for i in range(n):
        load = i / (n - 1)                      # 0 → 1
        temp = 40.0 + 30.0 * load               # heat follows load
        pwm1 = int(40 + 200 * load)             # controller follows heat
        out.append(EquilibriumSample(
            t=float(i),
            fan="pwm1",
            features={"util_gpu": 100.0 * load},
            pwm={"pwm1": pwm1, "pwm2": 100},
            temps={"s": temp},
        ))
    return out


def test_blended_gamma_floored_at_seed():
    m = SensorModel(
        name="s", target_c=50.0, critical_c=90.0,
        cools_seeds={"pwm1": 2.0, "pwm2": 1.0},
        min_samples=200, fully_trained_n=300,
    )
    m.fit(_closed_loop_samples(), ["util_gpu"], ["pwm1", "pwm2"])

    coefs = m.state.cooling_coefs
    assert coefs, "fit did not run"
    # pwm1 was identified and its data-γ clipped to ~0 — the seed (2.0/100)
    # must survive as the floor, not be erased at full trust.
    assert coefs["pwm1"] >= 2.0 / 100.0 - 1e-9
    # pwm2 never moved: pure seed.
    assert coefs["pwm2"] >= 1.0 / 100.0 - 1e-9
    # Relevance therefore keeps the measured ordering, no zeroed cooler.
    assert m.relevance("pwm1") > 0
    assert m.relevance("pwm2") > 0
