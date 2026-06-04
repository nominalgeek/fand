"""Per-sensor learning model and stress-based control helpers.

The daemon's control law is system-level rather than per-zone: every fan
contributes to cooling whichever sensors are stressed, weighted by how
strongly it actually cools each one. The strength values live in this
module's `SensorModel` and are learned from equilibrium samples.

Per sensor `s`, fit:

    T_s ≈ baseline + Σ_k [α_k(s) · feature_k]  −  Σ_p [γ_p(s) · pwm_p]

- baseline      a per-sensor intercept
- α_k(s)        load-feature heating coefficient
- γ_p(s) ≥ 0   fan `p`'s cooling power on sensor `s` (clipped post-solve)

At runtime:

    stress(s)    = max(0, (T_s − target_c) / (critical_c − target_c))
    relevance(s, p) = γ_p(s) / max_q γ_q(s)             (seed-based if untrained)
    demand(p)    = min(1, Σ_s stress(s)^1.5 × relevance(s, p))
    PWM(p)       = pwm_min(p) + demand(p) × (255 − pwm_min(p))

Until enough equilibrium samples accumulate per sensor (`min_samples_to_learn`)
and the holdout RMSE drops under `max_rmse_to_learn_c`, relevance uses the seed
cooling weights from each fan's `cools:` block in zones.yaml — phase 3's
measured ΔT per sensor. Once trained, the learned γ takes over. (Holdout R² is
kept as a diagnostic only: equilibrium holdouts are near-flat by construction,
so 1 − ss_res/ss_tot explodes on small offsets and even a perfect model can't
clear a positive R² bar — it cannot gate anything.)

AR(1) feed-forward: per sensor, an EMA of the load-feature vector. When the
sensor is trained, the difference between current features and the EMA is
turned into a predicted ΔT via the learned α coefs and added to T_s before
the stress calc — fans ramp before temps actually climb on a load surge.
Disabled per-sensor while α is untrained.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES = ["util_gpu", "power_gpu_w", "util_cpu", "ups_w"]


@dataclass
class EquilibriumSample:
    """One row per detected per-fan equilibrium.

    `fan` is the channel whose equilibrium fired (different fans equilibrate
    at different times). `temps` holds every sensor that was readable when
    the row was captured — fits draw from samples regardless of which fan
    triggered them, as long as the target sensor was observed.
    """

    t: float
    fan: str
    features: dict[str, float]
    pwm: dict[str, int]
    temps: dict[str, float]


@dataclass
class SensorState:
    baseline: float | None = None
    load_coefs: dict[str, float] = field(default_factory=dict)   # α
    cooling_coefs: dict[str, float] = field(default_factory=dict)  # γ (≥ 0)
    r2: float | None = None
    rmse: float | None = None
    n_samples: int = 0
    last_fit_t: float = 0.0


@dataclass
class FeedForwardState:
    """Per-sensor EMA of the feature vector. Used by trained sensors to predict
    a near-term ΔT_s from features that are moving away from their recent
    average.
    """

    alpha: float = 0.05  # ~ 1/(20 samples) ≈ 40s window at 2s poll
    ema: dict[str, float] = field(default_factory=dict)
    initialized: bool = False

    def update(self, features: dict[str, float]) -> None:
        if not self.initialized:
            self.ema = dict(features)
            self.initialized = True
            return
        for k, v in features.items():
            prev = self.ema.get(k, v)
            self.ema[k] = self.alpha * v + (1 - self.alpha) * prev

    def delta(self, features: dict[str, float]) -> dict[str, float]:
        """Current features minus EMA — positive = surge. Empty dict until init."""
        if not self.initialized:
            return {k: 0.0 for k in features}
        return {k: v - self.ema.get(k, v) for k, v in features.items()}


class SensorModel:
    """One model per declared sensor. Learns how load features and fan PWMs
    drive the sensor's temperature, and exposes the relevance weights the
    daemon's control law needs.
    """

    def __init__(
        self,
        name: str,
        target_c: float,
        critical_c: float,
        cools_seeds: dict[str, float],
        ridge_lambda: float = 1.0,
        min_samples: int = 200,
        max_rmse_c: float = 1.5,
        ff_alpha: float = 0.05,
        fully_trained_n: int = 500,
        min_pwm_spread: float = 10.0,
        min_feature_cov: float = 0.05,
    ):
        """
        cools_seeds: {fan_name: ΔT °C from phase 3}. Used for relevance
        weights while untrained, and to regularize γ toward physical priors
        during low-sample fits.

        fully_trained_n: sample count at which γ is fully driven by data
        (vs blended with seed). Linear blend below this.

        min_pwm_spread: minimum PWM range (counts) a fan must cover across
        the sample pool for its γ to be fit from data; below it the fan's γ
        stays seed-derived (the coefficient is unidentifiable).

        min_feature_cov: minimum relative variation (std / max(|mean|, 1))
        a load feature must show across the pool for its α to be fit; below
        it the feature's constant contribution folds into the baseline.

        max_rmse_c: holdout RMSE (°C) the fit must beat for the sensor to
        count as trained. RMSE rather than R²: equilibrium holdouts are
        near-flat, so R² explodes on small offsets regardless of model
        quality, while "predicts held-out temps within X °C" stays
        meaningful.
        """
        self.name = name
        self.target_c = target_c
        self.critical_c = critical_c
        self.cools_seeds = dict(cools_seeds)
        self.ridge_lambda = ridge_lambda
        self.min_samples = min_samples
        self.max_rmse_c = max_rmse_c
        self.fully_trained_n = fully_trained_n
        self.min_pwm_spread = min_pwm_spread
        self.min_feature_cov = min_feature_cov
        self.ff = FeedForwardState(alpha=ff_alpha)
        self.state = SensorState()

    # ---- inference -----------------------------------------------------

    def is_trained(self) -> bool:
        return (
            self.state.rmse is not None
            and self.state.n_samples >= self.min_samples
            and self.state.rmse <= self.max_rmse_c
            and bool(self.state.cooling_coefs)
        )

    def stress(self, T: float) -> float:
        """Normalized headroom: 0 at target, 1 at critical, clipped above."""
        span = max(0.1, self.critical_c - self.target_c)
        return max(0.0, min(1.0, (T - self.target_c) / span))

    def ff_corrected_temp(self, T: float, features: dict[str, float]) -> float:
        """Apply AR(1) feed-forward correction: add predicted ΔT from feature
        surge to T_s before stress is computed. Disabled while α is untrained
        (FF on uncalibrated coefs is worse than no FF). Always updates the
        EMA so it stays current.
        """
        d_features = self.ff.delta(features)
        self.ff.update(features)
        if not self.is_trained() or not self.state.load_coefs:
            return T
        dT_pred = sum(
            self.state.load_coefs.get(k, 0.0) * d_features.get(k, 0.0)
            for k in d_features
        )
        return T + max(0.0, dT_pred)

    def relevance(self, fan_name: str) -> float:
        """Fan `fan_name`'s contribution to cooling this sensor, normalized
        to [0, 1] against the strongest cooler. Untrained → uses seed weights;
        trained → uses learned γ.

        A trained model whose γ solved to all-zero (closed-loop data drags γ
        negative — fans ramp *because* temps rise — and the clip floors it at
        0) falls back to seeds rather than returning 0 for every fan: learning
        may refine the cooling map, never erase it. A sensor under stress must
        always retain at least its calibration-measured fan response.
        """
        if self.is_trained() and self.state.cooling_coefs:
            coefs = self.state.cooling_coefs
            max_g = max(coefs.values()) if coefs else 0.0
            if max_g > 0:
                return max(0.0, coefs.get(fan_name, 0.0) / max_g)
            # learned γ carries no signal — fall through to seeds
        seeds = self.cools_seeds
        max_seed = max(seeds.values()) if seeds else 0.0
        if max_seed <= 0:
            return 0.0
        return max(0.0, seeds.get(fan_name, 0.0) / max_seed)

    # ---- training ------------------------------------------------------

    def fit(
        self,
        samples: list[EquilibriumSample],
        feature_names: list[str],
        fan_names: list[str],
    ) -> None:
        """Ridge regression on equilibrium samples. Drops samples missing this
        sensor's temp reading. Solves on standardized columns with an
        unpenalized intercept, over only the PWM columns that actually varied;
        clips γ ≥ 0 post-solve and blends per-fan with seed weights while a
        fan's data is thin.
        """
        usable = [s for s in samples if self.name in s.temps]

        # Identifiability: a column with (near-)constant value is collinear
        # with the intercept — the solver would use it as an intercept
        # surrogate and hallucinate a coefficient for an input that never
        # moved (and the standardized back-transform divides by the tiny σ,
        # exploding it). Only fans whose PWM actually varied and features
        # with real relative variation across the pool enter the regression;
        # excluded fans keep seed-derived γ below, excluded features fold
        # their constant contribution into the baseline. As the rolling pool
        # spans more load regimes, columns re-enter on later refits.
        id_fans: list[str] = []
        id_feats: list[str] = []
        if usable:
            for p in fan_names:
                col = [float(s.pwm.get(p, 0)) for s in usable]
                if max(col) - min(col) >= self.min_pwm_spread:
                    id_fans.append(p)
            for fn in feature_names:
                fcol = np.asarray([s.features.get(fn, 0.0) for s in usable])
                cov = float(fcol.std()) / max(abs(float(fcol.mean())), 1.0)
                if cov >= self.min_feature_cov:
                    id_feats.append(fn)

        n_params = 1 + len(id_feats) + len(id_fans)  # baseline + α + (−γ)
        # Need enough rows to reliably separate the coefficients
        min_for_fit = max(10, 2 * n_params)
        if len(usable) < min_for_fit:
            log.info(
                "[sensor %s] fit: too few samples (%d < %d)",
                self.name, len(usable), min_for_fit,
            )
            return

        # Design matrix without intercept column: [features..., −pwm...].
        # Negative PWM columns so the solved coefficient comes out as γ ≥ 0
        # (matching the model T_s = ... − γ·pwm).
        n = len(usable)
        F = np.zeros((n, len(id_feats) + len(id_fans)))
        y = np.zeros(n)
        for i, s in enumerate(usable):
            for j, fn in enumerate(id_feats):
                F[i, j] = s.features.get(fn, 0.0)
            for j, p in enumerate(id_fans):
                F[i, len(id_feats) + j] = -float(s.pwm.get(p, 0))
            y[i] = s.temps[self.name]

        # Hold out last 20% for honest R²
        n_train = max(int(n * 0.8), n - 50)
        Ftr, ytr = F[:n_train], y[:n_train]
        Fts, yts = F[n_train:], y[n_train:]

        # Standardize for the solve. Ridge's penalty is scale-dependent:
        # explaining a constant 50 °C costs 50² through the intercept but
        # ~(50/680)² through a ups_w-sized column, so an unstandardized solve
        # shoves the DC temperature level into the large columns and α/γ stop
        # meaning marginal effects. Solve in z-scored space (uniform penalty,
        # intercept = column means, unpenalized), back-transform after.
        mu = Ftr.mean(axis=0)
        sigma = Ftr.std(axis=0)
        informative = sigma > 1e-9
        sigma_safe = np.where(informative, sigma, 1.0)
        Z = (Ftr - mu) / sigma_safe
        yc = ytr - ytr.mean()
        k = Z.shape[1]
        try:
            A = Z.T @ Z + self.ridge_lambda * np.eye(k)
            b = Z.T @ yc
            theta_std = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            log.warning("[sensor %s] fit: solve failed: %s", self.name, exc)
            return
        # Back-transform; constant columns carry no information → coef 0.
        theta_raw = np.where(informative, theta_std / sigma_safe, 0.0)
        baseline = float(ytr.mean() - theta_raw @ mu)

        alpha_dict = {
            id_feats[j]: float(theta_raw[j]) for j in range(len(id_feats))
        }
        gamma_raw = {
            id_fans[j]: float(theta_raw[len(id_feats) + j])
            for j in range(len(id_fans))
        }
        gamma_dict = {fn: max(0.0, g) for fn, g in gamma_raw.items()}

        # Diagnostic: a fan with a positive seed that solved to negative γ is a
        # phase-3-vs-runtime disagreement worth flagging.
        for fn, g_raw in gamma_raw.items():
            seed = self.cools_seeds.get(fn, 0.0)
            if g_raw < -1e-6 and seed > 0:
                log.info(
                    "[sensor %s] fit: clipped γ for fan %s (raw=%.4f, seed=%.2f) "
                    "— phase 3 said it cools this sensor but runtime data disagrees",
                    self.name, fn, g_raw, seed,
                )

        # Blend learned γ with seeds per fan. Seeds are in °C ΔT; γ is in °C
        # per PWM unit. Convert via a rough PWM-range divisor so the blend
        # lives on a comparable scale. Identified fans earn trust with sample
        # count; fans whose PWM never moved hold pure seed weight regardless
        # of n — sample count is not information about a coefficient the data
        # can't see.
        trust = min(1.0, n / max(1, self.fully_trained_n))
        seed_pwm_scale = 100.0  # ΔT_phase3 / 100 ≈ γ if fan drops PWM by 100 units
        blended: dict[str, float] = {}
        for fn in fan_names:
            seed_g = self.cools_seeds.get(fn, 0.0) / seed_pwm_scale
            if fn in gamma_dict:
                blended[fn] = (1 - trust) * seed_g + trust * gamma_dict[fn]
            else:
                blended[fn] = seed_g

        # Holdout metrics. RMSE is the trained gate: "predicts held-out temps
        # within max_rmse_c °C" stays meaningful on the near-flat holdouts
        # equilibrium sampling produces. R² is diagnostic only — with
        # ss_tot ≈ 0 it explodes on small offsets no matter how good the fit.
        if len(yts) >= 5:
            yhat = baseline + Fts @ theta_raw
            ss_res = float(np.sum((yts - yhat) ** 2))
            ss_tot = float(np.sum((yts - yts.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse: float | None = float(np.sqrt(ss_res / len(yts)))
        else:
            r2 = float("nan")
            rmse = None

        self.state.baseline = baseline
        self.state.load_coefs = alpha_dict
        self.state.cooling_coefs = blended
        self.state.r2 = r2
        self.state.rmse = rmse
        self.state.n_samples = n
        self.state.last_fit_t = usable[-1].t

        log.info(
            "[sensor %s] fit n=%d rmse=%s r²=%.3f baseline=%.2f id_fans=%s id_feats=%s α=%s γ=%s",
            self.name, n,
            f"{rmse:.2f}" if rmse is not None else "-",
            r2, baseline, id_fans, id_feats,
            {k: round(v, 4) for k, v in alpha_dict.items()},
            {k: round(v, 5) for k, v in blended.items() if v > 1e-5},
        )

    # ---- persistence ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_c": self.target_c,
            "critical_c": self.critical_c,
            "state": {
                "baseline": self.state.baseline,
                "load_coefs": self.state.load_coefs,
                "cooling_coefs": self.state.cooling_coefs,
                "r2": self.state.r2,
                "rmse": self.state.rmse,
                "n_samples": self.state.n_samples,
                "last_fit_t": self.state.last_fit_t,
            },
        }

    def load_state(self, d: dict[str, Any]) -> None:
        s = d.get("state", {})
        self.state.baseline = s.get("baseline")
        self.state.load_coefs = s.get("load_coefs", {}) or {}
        self.state.cooling_coefs = s.get("cooling_coefs", {}) or {}
        self.state.r2 = s.get("r2")
        self.state.rmse = s.get("rmse")
        self.state.n_samples = s.get("n_samples", 0)
        self.state.last_fit_t = s.get("last_fit_t", 0.0)


# ---- module-level control helpers ----------------------------------------


def aggregate_demand(
    fan_name: str,
    cools: dict[str, float],
    stresses: dict[str, float],
    sensor_models: dict[str, SensorModel],
) -> tuple[float, list[tuple[str, float, float, float]]]:
    """Compute fan `fan_name`'s control demand and the per-sensor breakdown
    that drove it. Weighted sum with saturation:

        demand = min(1, Σ_s stress(s)^1.5 × relevance(s, p))

    Returns (demand, contributions) where contributions is a list of
    (sensor_name, stress, relevance, contribution) sorted by contribution
    descending. Empty cools or empty stresses → demand 0.
    """
    if not cools:
        return 0.0, []
    parts: list[tuple[str, float, float, float]] = []
    total = 0.0
    for sensor_name in cools:
        if sensor_name not in sensor_models or sensor_name not in stresses:
            continue
        sm = sensor_models[sensor_name]
        rel = sm.relevance(fan_name)
        if rel <= 0:
            continue
        stress = stresses[sensor_name]
        contrib = (stress ** 1.5) * rel
        if contrib > 0:
            parts.append((sensor_name, stress, rel, contrib))
            total += contrib
    parts.sort(key=lambda x: -x[3])
    return min(1.0, total), parts


def demand_to_pwm(demand: float, pwm_min: int) -> int:
    """Linear map: stress=0 → pwm_min, stress=1 → 255. Demand pre-clipped to [0,1]."""
    pwm = pwm_min + demand * (255 - pwm_min)
    return max(0, min(255, int(round(pwm))))


# ---- persistence (top-level) ---------------------------------------------


def save_models(path: Path, models: dict[str, SensorModel]) -> None:
    payload = {
        "version": 2,
        "sensors": {n: m.to_dict() for n, m in models.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_models_into(path: Path, models: dict[str, SensorModel]) -> None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    version = payload.get("version")
    if version != 2:
        log.info(
            "model.json version %s is not v2 — ignoring (a refit will produce v2)",
            version,
        )
        return
    for name, d in payload.get("sensors", {}).items():
        if name in models:
            models[name].load_state(d)


# ---- equilibrium detection -----------------------------------------------


def is_fan_equilibrium(
    window: list[tuple[float, dict[str, float], dict[str, float], dict[str, int]]],
    fan_name: str,
    cooled_sensors: list[str],
    feature_names: list[str],
    pwm_stable_s: float = 30.0,
    pwm_jitter_tolerance: int = 4,
    temp_slope_threshold: float = 0.05,
    feature_cov_threshold: float = 0.10,
    feature_abs_tolerance: float = 2.0,
) -> bool:
    """Decide if the window represents an equilibrium for `fan_name`'s fit.

    Three conditions, all required:
    1. `fan_name`'s PWM has been within `pwm_jitter_tolerance` across the
       window AND the window covers at least `pwm_stable_s` seconds.
    2. Every sensor in `cooled_sensors` that's present in the window's temps
       has |dT/dt| ≤ `temp_slope_threshold` °C/s.
    3. Every feature is stable: its within-window coefficient of variation
       (std / max(|mean|, 1)) ≤ `feature_cov_threshold`, OR its raw std ≤
       `feature_abs_tolerance`. The absolute fallback keeps inherently
       low-magnitude, jittery signals (idle CPU util) from gating every
       window, while still rejecting genuine load swings.

    Window entries are (t, temps_dict, features_dict, pwm_dict).
    """
    if len(window) < 8:
        return False

    ts = np.asarray([w[0] for w in window])
    dt_total = ts[-1] - ts[0]
    if dt_total < pwm_stable_s:
        return False

    # PWM stability for the fan in question.
    pwms = [w[3].get(fan_name) for w in window]
    pwm_values = [p for p in pwms if p is not None]
    if not pwm_values:
        return False
    if max(pwm_values) - min(pwm_values) > pwm_jitter_tolerance:
        return False

    # Temperature stability per sensor in fan's cooled list.
    t_rel = ts - ts[0]
    for sensor_name in cooled_sensors:
        temps = [w[1].get(sensor_name) for w in window]
        temps_valid = [(t_rel[i], v) for i, v in enumerate(temps) if v is not None]
        if len(temps_valid) < 8:
            continue  # don't gate on under-observed sensors
        ts_v = np.asarray([x[0] for x in temps_valid])
        Tv = np.asarray([x[1] for x in temps_valid])
        try:
            slope = float(np.polyfit(ts_v, Tv, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return False
        if abs(slope) > temp_slope_threshold:
            return False

    # Feature stability: every feature must be either relatively stable (low
    # coefficient of variation) or absolutely near-constant (small raw std).
    if feature_names:
        feats = np.asarray(
            [[w[2].get(fn, 0.0) for fn in feature_names] for w in window]
        )
        std = feats.std(axis=0)
        base = np.maximum(np.abs(feats.mean(axis=0)), 1.0)
        stable = (std / base <= feature_cov_threshold) | (std <= feature_abs_tolerance)
        if not bool(stable.all()):
            return False

    return True
