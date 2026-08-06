"""Mode A — training-free analytic gating (primary strict few-shot mode).

The gate weight for a candidate box is

    w = sigma( -alpha * sigma_tilde^2_visual + beta * sigma_tilde^2_text
               + gamma * a_tilde_visual )

where all inputs are normalized to [0, 1]:

  * sigma_tilde^2_visual : normalized visual uncertainty (mean pairwise cosine
    distance over k support features; zero for k=1).
  * sigma_tilde^2_text   : normalized text uncertainty (mean pairwise cosine
    distance over the M=20 prompt-template ensemble).
  * a_tilde_visual       : normalized per-box visual affinity
                           a = (1 + cos(f_box, p_visual)) / 2.

The fused score is S_final = (1 - w) * S_text + w * S_visual.

Pre-registration guarantees (docs/pre_registration.md):
  * Coefficients are FIXED at alpha = beta = gamma = 1 (not learned from the
    target domain).
  * Temperature T = 1 — no learned scaling, no calibration data.
  * The rule is purely analytic; there are no trainable parameters and no
    MC Dropout.

Beta-regression fallback (pre-registration D5 contingency, §10): when the
normalized variances cluster at the extremes (D5 flags >30% of values below
0.25 or above 0.75), the first-order (Taylor-style) logit approximation is
least trustworthy. ``beta_regression_gate`` / :class:`BetaGate` replace it
with a Beta-linked gate whose weight is the mean of a Beta distribution with
input-linked precision — see the section below. Still training-free with
fixed coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 1.0
DEFAULT_TEMPERATURE = 1.0
EPS = 1e-9


def analytic_gate_logit(
    norm_text_variance: float,
    norm_visual_variance: float,
    norm_affinity: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Pre-activation logit of the analytic gate:
    -alpha * v_visual + beta * v_text + gamma * a_visual.

    Sign convention (per proposal): high visual uncertainty *lowers* the
    weight on the visual branch (negative term), high text uncertainty
    *raises* it (positive term), and high visual affinity *raises* it
    (positive term, bias-variance correction).
    """
    return (
        -alpha * float(norm_visual_variance)
        + beta * float(norm_text_variance)
        + gamma * float(norm_affinity)
    )


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def gate_weight(
    norm_text_variance: float,
    norm_visual_variance: float,
    norm_affinity: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Gate weight w in (0, 1) via sigmoid of the analytic logit."""
    logit = analytic_gate_logit(
        norm_text_variance,
        norm_visual_variance,
        norm_affinity,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return float(_sigmoid(logit))


def fuse_scores(s_text: float, s_visual: float, w: float) -> float:
    """Fuse modality scores: S_final = (1 - w) * S_text + w * S_visual."""
    return (1.0 - w) * float(s_text) + w * float(s_visual)


@dataclass
class ModeAGate:
    """Stateless training-free gate with fixed coefficients (alpha=beta=gamma=1)
    and temperature T=1. Instantiate once per experiment; holds no learnable
    state so it is inherently training-free."""

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    gamma: float = DEFAULT_GAMMA
    temperature: float = DEFAULT_TEMPERATURE  # fixed T=1 in Mode A

    def __post_init__(self) -> None:
        if self.temperature != DEFAULT_TEMPERATURE:
            raise ValueError(
                "Mode A uses T=1 (no calibration data); learned temperatures "
                "belong to Mode B."
            )

    def weight(
        self,
        norm_text_variance: float,
        norm_visual_variance: float,
        norm_affinity: float,
    ) -> float:
        return gate_weight(
            norm_text_variance,
            norm_visual_variance,
            norm_affinity,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

    def fused_score(
        self,
        s_text: float,
        s_visual: float,
        norm_text_variance: float,
        norm_visual_variance: float,
        norm_affinity: float,
    ) -> float:
        w = self.weight(norm_text_variance, norm_visual_variance, norm_affinity)
        return fuse_scores(s_text, s_visual, w)

    def predict_batch(
        self,
        inputs: List[Mapping[str, float]],
    ) -> np.ndarray:
        """Vectorized gate weights for a list of input dicts with keys
        ``s_text``, ``s_visual``, ``norm_text_variance``,
        ``norm_visual_variance``, ``norm_affinity``."""
        X = np.asarray(
            [
                [
                    r["norm_text_variance"],
                    r["norm_visual_variance"],
                    r["norm_affinity"],
                ]
                for r in inputs
            ],
            dtype=np.float64,
        )
        logits = (
            -self.alpha * X[:, 1] + self.beta * X[:, 0] + self.gamma * X[:, 2]
        )
        return np.asarray(_sigmoid(logits), dtype=np.float64)


# ---------------------------------------------------------------------------
# Beta-regression fallback gate (pre-registered D5 contingency, §10)
# ---------------------------------------------------------------------------
# D5 (Taylor-validity sentinel) flags when >30% of normalized variances
# cluster below 0.25 or above 0.75; the pre-registration then falls back to a
# Beta-regression variant of the gate. The analytic gate's logit is a
# first-order (Taylor-style) combination of the variance proxies — least
# trustworthy exactly in the boundary regime D5 detects. The Beta gate treats
# the weight as the mean of a Beta distribution:
#
#     eta       = -alpha*v_visual + beta*v_text + gamma*a_visual  (same logit)
#     mu        = sigmoid(eta)                                    (Beta mean)
#     precision = precision_max / (1 + slope*(v_text + v_visual)) (Beta phi)
#     w         = (precision*mu + prior_precision*prior_weight)
#                 / (precision + prior_precision)
#
# The Beta precision (concentration) is linked to the input variances: when
# they are extreme the Beta is diffuse and the weight is pulled toward the
# neutral prior w0 = 0.5 (naive averaging), hedging the gate's commitment
# exactly where the Taylor approximation is invalid. In the low-variance
# limit (precision -> precision_max large) it recovers the analytic gate.
# All coefficients stay fixed (alpha = beta = gamma = 1) — still
# training-free, no calibration data.

DEFAULT_PRECISION_MAX = 20.0
DEFAULT_PRECISION_SLOPE = 5.0
DEFAULT_PRIOR_PRECISION = 1.0
NEUTRAL_PRIOR_WEIGHT = 0.5


def beta_regression_gate(
    norm_text_variance: float,
    norm_visual_variance: float,
    norm_affinity: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    precision_max: float = DEFAULT_PRECISION_MAX,
    precision_slope: float = DEFAULT_PRECISION_SLOPE,
    prior_precision: float = DEFAULT_PRIOR_PRECISION,
    prior_weight: float = NEUTRAL_PRIOR_WEIGHT,
) -> float:
    """Beta-regression gate weight in [0, 1] (pre-registered D5 fallback).

    The weight is the posterior-mean of a Beta distribution: the Beta mean
    ``mu = sigmoid(eta)`` with the SAME analytic logit ``eta``, blended with
    the neutral prior ``prior_weight`` in proportion to the Beta precision
    ``precision = precision_max / (1 + slope*(v_text + v_visual))``.

    Args:
        norm_text_variance: normalized text variance in [0, 1].
        norm_visual_variance: normalized visual variance in [0, 1].
        norm_affinity: normalized visual affinity in [0, 1].
        alpha/beta/gamma: fixed Mode A coefficients (default 1).
        precision_max: Beta precision when both variances are zero
            (recovers the analytic gate as precision_max grows).
        precision_slope: how strongly extreme variances reduce precision.
        prior_precision: precision of the neutral-prior component.
        prior_weight: neutral weight the gate hedges toward (default 0.5 =
            naive averaging).

    Returns:
        Gate weight in [0, 1] (a convex combination of two [0, 1] values, so
        the bound holds for ANY inputs, including exact 0.0/1.0 variances).
    """
    eta = analytic_gate_logit(
        norm_text_variance,
        norm_visual_variance,
        norm_affinity,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    mu = float(_sigmoid(eta))
    precision = precision_max / (
        1.0 + precision_slope * (float(norm_text_variance) + float(norm_visual_variance))
    )
    w = (precision * mu + prior_precision * prior_weight) / (
        precision + prior_precision
    )
    return float(min(max(w, 0.0), 1.0))


@dataclass
class BetaGate:
    """Training-free Beta-regression gate (D5 fallback; fixed coefficients).

    Same inputs and coefficient semantics as :class:`ModeAGate`, but the
    weight is the mean of a Beta distribution whose precision shrinks with
    the input variances, pulling the weight toward the neutral 0.5 prior in
    the boundary regime D5 flags (Taylor approximation invalid). Holds no
    learnable state — inherently training-free.
    """

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    gamma: float = DEFAULT_GAMMA
    precision_max: float = DEFAULT_PRECISION_MAX
    precision_slope: float = DEFAULT_PRECISION_SLOPE
    prior_precision: float = DEFAULT_PRIOR_PRECISION
    prior_weight: float = NEUTRAL_PRIOR_WEIGHT

    def weight(
        self,
        norm_text_variance: float,
        norm_visual_variance: float,
        norm_affinity: float,
    ) -> float:
        return beta_regression_gate(
            norm_text_variance,
            norm_visual_variance,
            norm_affinity,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            precision_max=self.precision_max,
            precision_slope=self.precision_slope,
            prior_precision=self.prior_precision,
            prior_weight=self.prior_weight,
        )

    def fused_score(
        self,
        s_text: float,
        s_visual: float,
        norm_text_variance: float,
        norm_visual_variance: float,
        norm_affinity: float,
    ) -> float:
        w = self.weight(norm_text_variance, norm_visual_variance, norm_affinity)
        return fuse_scores(s_text, s_visual, w)

    def predict_batch(
        self,
        inputs: List[Mapping[str, float]],
    ) -> np.ndarray:
        """Vectorized Beta gate weights (same input schema as ModeAGate)."""
        X = np.asarray(
            [
                [
                    r["norm_text_variance"],
                    r["norm_visual_variance"],
                    r["norm_affinity"],
                ]
                for r in inputs
            ],
            dtype=np.float64,
        )
        logits = -self.alpha * X[:, 1] + self.beta * X[:, 0] + self.gamma * X[:, 2]
        mu = np.asarray(_sigmoid(logits), dtype=np.float64)
        precision = self.precision_max / (
            1.0 + self.precision_slope * (X[:, 0] + X[:, 1])
        )
        return (precision * mu + self.prior_precision * self.prior_weight) / (
            precision + self.prior_precision
        )
