"""Uncertainty estimators (proposal Section 5.4.1).

Mode A (training-free proxy uncertainty)
  * sigma^2_text    : mean pairwise cosine distance across the M=20 prompt
                      template embeddings (per class).
  * sigma^2_visual  : mean pairwise cosine distance across the k support
                      features; for k=1 the configured k1 prior is returned
                      (default 0.0 = maximum-likelihood treatment of a
                      degenerate sample; pre-registered ablation 0.5 =
                      max-entropy prior).
  * a_visual        : per-box visual affinity (1 + cos)/2 in [0, 1].

Mode B (learned MC Dropout)
  * mc_dropout_estimate: T stochastic forward passes with dropout active;
                        returns mean and variance of the score.

All raw variances are normalized to [0, 1] via min-max scaling using
support-set (or calibration-set) statistics before entering the gate
(normalization is an ablation: none | min-max | percentile rank).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6  # clamp epsilon for normalized variances


# ----------------------------------------------------------------------
# Mode A signals
# ----------------------------------------------------------------------
def mean_pairwise_cosine_distance(features: np.ndarray) -> float:
    """2/(M(M-1)) * sum_{i<j} (1 - cos(e_i, e_j)); 0 for < 2 samples."""
    feats = _l2_normalize_rows(np.asarray(features, dtype=np.float64))
    n = len(feats)
    if n < 2:
        return 0.0
    sim = feats @ feats.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sim[iu]))


def normalized_text_variance(prompt_embeddings: np.ndarray) -> float:
    """Raw text uncertainty from a prompt-template ensemble (unitless, [0, 2]
    for cosine similarity in [-1, 1]); normalized to [0, 1] downstream."""
    return mean_pairwise_cosine_distance(prompt_embeddings)


def normalized_visual_variance(
    support_features: np.ndarray, k1_prior: float = 0.0
) -> float:
    """Raw visual uncertainty across k support features.

    k=1 (single exemplar): returns ``k1_prior`` instead of estimating
    dispersion. Default 0.0 is the pre-registered maximum-likelihood treatment
    of a degenerate sample; the max-entropy-prior ablation (config
    ``k1_max_entropy_prior: 0.5``, pre-registration §2) passes 0.5.
    """
    features = np.asarray(support_features, dtype=np.float64)
    if len(features) < 2:
        return float(k1_prior)
    return mean_pairwise_cosine_distance(features)


def visual_affinity(box_feature: np.ndarray, prototype: np.ndarray) -> float:
    """a_visual = (1 + cos(f_box, p_visual)) / 2 in [0, 1]."""
    cos = _cosine(np.asarray(box_feature, dtype=np.float64), prototype)
    return float((1.0 + cos) / 2.0)


def min_max_normalize(
    values: np.ndarray,
    support_stats: Optional[Tuple[float, float]] = None,
    eps: float = EPS,
) -> np.ndarray:
    """Min-max normalize to [0, 1] using support-set statistics
    x_tilde = (x - min) / (max - min + eps)."""
    values = np.asarray(values, dtype=np.float64)
    if support_stats is not None:
        vmin, vmax = support_stats
    else:
        vmin, vmax = float(values.min()), float(values.max())
    denom = (vmax - vmin) + eps
    return np.clip((values - vmin) / denom, 0.0, 1.0)


# ----------------------------------------------------------------------
# Mode B signals (MC Dropout)
# ----------------------------------------------------------------------
def mc_dropout_estimate(
    predict_fn: Callable[[], np.ndarray],
    t_passes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ``predict_fn`` T times (dropout active during each pass) and return
    (mean, variance) across passes.

    Args:
        predict_fn: zero-arg callable returning a score array (N,) or (N, D).
        t_passes: T (10 primary; 50 stability check on one subset only).
    """
    samples = np.stack([np.asarray(predict_fn(), dtype=np.float64) for _ in range(t_passes)])
    mean = samples.mean(axis=0)
    var = samples.var(axis=0)
    return mean, var


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
