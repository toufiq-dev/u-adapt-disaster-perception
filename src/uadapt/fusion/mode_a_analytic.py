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
