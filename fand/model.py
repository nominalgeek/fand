"""Per-zone learning model.

Two parts:

(a) Steady-state PWM↔temp identification via ridge regression on equilibrium
    samples. Physical form:

        T_z - T_amb = (k₁·util_GPU + k₂·power_GPU + k₃·util_CPU + k₄·ups_W + b)
                      / max(PWM - PWM₀, 1)

    so given a measured equilibrium row (T_z, T_amb, features, PWM_applied):
        y_i  = (T_z - T_amb) · max(PWM_applied - PWM₀, 1)
        X_i  = [features..., 1]
    fit  y = X · θ   (ridge).  At inference,

        PWM_required = PWM₀ + θ·features / max(T_target - T_amb - margin, ε)

(b) Predictive feed-forward bump: anticipate load surges before temp rises.
    Tracked as an exponential moving average of the same feature vector; when
    current load exceeds recent average by a delta, add an extra PWM bump
    proportional to that delta. The bump's gain is configurable per zone.

Bootstrap: until ≥ `min_samples` equilibrium rows are logged and ridge R² ≥
`min_r2` on a holdout fold, predictions come from a piecewise-linear curve
shipped in the zone config (temp_c → pwm pairs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FEATURE_NAMES = ["util_gpu", "power_gpu_w", "util_cpu", "ups_w"]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class EquilibriumSample:
    t: float
    features: list[float]  # length N_FEATURES
    pwm_applied: int
    T_z: float
    T_amb: float


@dataclass
class ZoneState:
    name: str
    coefs: list[float] | None = None  # length N_FEATURES + 1 (bias)
    r2: float | None = None
    n_samples: int = 0
    last_fit_t: float = 0.0


@dataclass
class FeedForwardState:
    """One per zone. EMA of features for surge detection."""

    ema: list[float] = field(default_factory=lambda: [0.0] * N_FEATURES)
    alpha: float = 0.05  # ~ 1/(20 samples) ≈ 40s window at 2s poll
    initialized: bool = False

    def update(self, features: list[float]) -> None:
        if not self.initialized:
            self.ema = list(features)
            self.initialized = True
            return
        self.ema = [
            self.alpha * f + (1 - self.alpha) * e for f, e in zip(features, self.ema)
        ]

    def surge(self, features: list[float]) -> list[float]:
        """Per-feature positive deviation from EMA (clamped at 0).
        Returns zeros until the EMA is initialized so the first call doesn't
        treat the entire feature vector as a surge.
        """
        if not self.initialized:
            return [0.0] * len(features)
        return [max(0.0, f - e) for f, e in zip(features, self.ema)]


class ZoneModel:
    """One model per fan zone."""

    def __init__(
        self,
        name: str,
        pwm_min: int,
        target_c: float,
        bootstrap_curve: list[tuple[float, int]],
        critical_c: float,
        ridge_lambda: float = 1.0,
        min_samples: int = 200,
        min_r2: float = 0.7,
        ff_gain_per_feature: list[float] | None = None,
        ff_alpha: float = 0.05,
        margin_c: float = 5.0,
    ):
        self.name = name
        self.pwm_min = pwm_min
        self.target_c = target_c
        self.critical_c = critical_c
        self.bootstrap_curve = sorted(bootstrap_curve)  # list of (temp_c, pwm)
        self.ridge_lambda = ridge_lambda
        self.min_samples = min_samples
        self.min_r2 = min_r2
        self.margin_c = margin_c
        # Feed-forward bump = sum(gain_i · surge_i). Defaults: GPU power dominates.
        self.ff_gain = ff_gain_per_feature or [0.5, 0.05, 0.2, 0.02]
        self.ff = FeedForwardState(alpha=ff_alpha)
        self.state = ZoneState(name=name)

    # ---- inference ----------------------------------------------------------

    def is_trained(self) -> bool:
        return (
            self.state.coefs is not None
            and self.state.r2 is not None
            and self.state.n_samples >= self.min_samples
            and self.state.r2 >= self.min_r2
        )

    def predict_pwm(
        self,
        features: list[float],
        T_z: float,
        T_amb: float,
    ) -> tuple[int, str]:
        """Return (PWM_value, source) where source ∈ {'bootstrap','model','critical'}."""

        if T_z >= self.critical_c:
            return 255, "critical"

        # Update feed-forward EMA before computing surge so we don't see ourselves.
        surge = self.ff.surge(features)
        ff_bump = int(sum(g * s for g, s in zip(self.ff_gain, surge)))

        self.ff.update(features)

        if not self.is_trained():
            base = self._bootstrap_pwm(T_z)
            return min(255, max(self.pwm_min, base + ff_bump)), "bootstrap"

        theta = np.asarray(self.state.coefs)
        x = np.asarray(features + [1.0])
        heat = float(theta @ x)
        dT_eff = max(self.target_c - T_amb - self.margin_c, 5.0)
        pwm_required = self.pwm_min + heat / dT_eff
        return min(255, max(self.pwm_min, int(pwm_required) + ff_bump)), "model"

    def _bootstrap_pwm(self, T_z: float) -> int:
        """Piecewise-linear interpolation over the bootstrap curve."""
        pts = self.bootstrap_curve
        if not pts:
            return 160  # safe-ish 63%
        if T_z <= pts[0][0]:
            return pts[0][1]
        if T_z >= pts[-1][0]:
            return pts[-1][1]
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            if t0 <= T_z <= t1:
                frac = (T_z - t0) / (t1 - t0)
                return int(p0 + frac * (p1 - p0))
        return pts[-1][1]

    # ---- training -----------------------------------------------------------

    def fit(self, samples: list[EquilibriumSample]) -> None:
        """Ridge regression on equilibrium samples. Updates self.state."""
        if len(samples) < 10:
            log.info("[%s] fit: too few samples (%d)", self.name, len(samples))
            return

        X = np.asarray([s.features + [1.0] for s in samples])
        # y_i = (T_z - T_amb) * max(PWM - PWM₀, 1)
        y = np.asarray(
            [(s.T_z - s.T_amb) * max(s.pwm_applied - self.pwm_min, 1) for s in samples]
        )

        # Hold out last 20% for R² estimation
        n = len(samples)
        n_train = max(int(n * 0.8), n - 50)
        Xtr, ytr = X[:n_train], y[:n_train]
        Xts, yts = X[n_train:], y[n_train:]

        # Closed-form ridge: θ = (X·Xᵀ + λI)⁻¹ Xᵀ y
        d = Xtr.shape[1]
        A = Xtr.T @ Xtr + self.ridge_lambda * np.eye(d)
        b = Xtr.T @ ytr
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            log.warning("[%s] fit: solve failed: %s", self.name, exc)
            return

        # R² on held-out fold
        if len(yts) >= 5:
            yhat = Xts @ theta
            ss_res = float(np.sum((yts - yhat) ** 2))
            ss_tot = float(np.sum((yts - yts.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            r2 = float("nan")

        self.state.coefs = theta.tolist()
        self.state.r2 = r2
        self.state.n_samples = n
        self.state.last_fit_t = samples[-1].t
        log.info(
            "[%s] fit n=%d r²=%.3f coefs=%s",
            self.name,
            n,
            r2,
            [round(c, 4) for c in theta],
        )

    # ---- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pwm_min": self.pwm_min,
            "target_c": self.target_c,
            "critical_c": self.critical_c,
            "state": {
                "coefs": self.state.coefs,
                "r2": self.state.r2,
                "n_samples": self.state.n_samples,
                "last_fit_t": self.state.last_fit_t,
            },
        }

    def load_state(self, d: dict) -> None:
        s = d.get("state", {})
        self.state.coefs = s.get("coefs")
        self.state.r2 = s.get("r2")
        self.state.n_samples = s.get("n_samples", 0)
        self.state.last_fit_t = s.get("last_fit_t", 0.0)


def save_models(path: Path, models: dict[str, ZoneModel]) -> None:
    payload = {"version": 1, "zones": {n: m.to_dict() for n, m in models.items()}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_models_into(path: Path, models: dict[str, ZoneModel]) -> None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    for name, d in payload.get("zones", {}).items():
        if name in models:
            models[name].load_state(d)


# ---- equilibrium detection (used by daemon to assemble training set) -------


def is_equilibrium(
    window: list[tuple[float, float, list[float]]],
    temp_threshold: float = 0.05,
    feature_threshold_pct: float = 0.05,
) -> bool:
    """Given a sliding window of (t, T_z, features), decide if the system is at
    steady state — temp rate of change small AND feature variance small.
    """
    if len(window) < 8:
        return False
    ts = np.asarray([w[0] for w in window])
    Tz = np.asarray([w[1] for w in window])
    feats = np.asarray([w[2] for w in window])  # (N, F)

    # Linear fit slope of T_z vs t
    dt = ts[-1] - ts[0]
    if dt <= 0:
        return False
    slope = float(np.polyfit(ts - ts[0], Tz, 1)[0])  # °C / s
    if abs(slope) > temp_threshold:
        return False

    # Feature variance: max(|range| / max(mean, 1)) across features
    rng = feats.max(axis=0) - feats.min(axis=0)
    base = np.maximum(np.abs(feats.mean(axis=0)), 1.0)
    if float((rng / base).max()) > feature_threshold_pct:
        return False

    return True
