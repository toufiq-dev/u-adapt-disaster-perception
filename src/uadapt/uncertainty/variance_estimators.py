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

REAL-cache per-proposal estimators (change_log.md 2026-08-05; deviation on
pre-registration §2 / §7.6): the class-level quantities above yield exactly
C distinct values per dataset, so D1/D2 lose statistical power at C=3 even
when pooled across LADD + D-Fire (and the 0.5 text placeholder made them
constant). On the real-cache path we therefore use per-proposal estimators:
  * proposal_text_variance   : normalized entropy of the per-box class-
                               similarity vector (continuous in [0, 1]).
  * proposal_visual_variance : mean (1 - cos) between the box feature and
                               the class support set (continuous in [0, 2]).
Both feed the SAME gate terms (beta for text, -alpha for visual) and the
D1/D2 diagnostics, but with proposal-level resolution.

Mode B (learned MC Dropout)
  * mc_dropout_estimate: T stochastic forward passes with dropout active;
                        returns mean and variance of the score.

All raw variances are normalized to [0, 1] before entering the gate
(normalization is an ablation: none | min-max | percentile rank | absolute;
see pre-registration section 2). Min-max uses support-set (or
calibration-set) statistics; absolute scaling (`x / 2.0`) is
class-count-independent and is the pre-registered deviation fixing the
2-class degeneracy (2026-08-03, change_log.md).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6  # clamp epsilon for normalized variances

# Raw mean pairwise cosine distance (1 - cos, cos in [-1, 1]) ranges in
# [0, 2]; absolute scaling divides by this constant (class-count-independent).
COSINE_DISTANCE_MAX = 2.0


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


def proposal_text_variance(text_similarities: Optional[np.ndarray]) -> float:
    """Per-proposal text uncertainty from the class-similarity vector.

    The real-cache backbone (Grounding DINO) stores, per proposal, the
    per-class text similarities ``s`` (C,) (sigmoid query-to-token logits).
    When the model cannot discriminate between classes ``s`` is flat; when it
    is confident one class dominates. We measure this with the NORMALIZED
    entropy of the relative class weights ``p = s / sum(s)``:

        H_norm = -sum_i p_i ln p_i / ln C     in [0, 1]

    0.0 = one class dominates (confident text assignment -> text reliable),
    1.0 = perfectly flat (classes indistinguishable -> text unreliable, the
    gate should route toward the visual branch).

    This is the REAL-cache estimator that replaces the 0.5 placeholder: the
    class-level template-ensemble variance yields only C distinct values and
    underpowered D1 at C=3, while the per-proposal entropy is a continuous
    signal with full statistical power (change_log.md 2026-08-05).

    Args:
        text_similarities: (C,) similarity vector; None or empty returns 0.5
            (the neutral "no signal" value, preserving the old placeholder
            semantics for degenerate records).

    Returns:
        Normalized entropy in [0, 1].
    """
    if text_similarities is None:
        return 0.5
    s = np.asarray(text_similarities, dtype=np.float64).ravel()
    if s.size == 0:
        return 0.5
    if s.size == 1:
        return 0.0  # a single class: nothing to be uncertain about
    s = np.clip(s, 0.0, None)  # sigmoid sims are >= 0; guard negatives
    if s.sum() <= 0.0:
        return 1.0  # all similarities zero: fully uninformative
    p = s / s.sum()
    # 0 * log(0) -> 0 (zero-mass classes contribute no entropy)
    log_p = np.zeros_like(p)
    nz = p > 0
    log_p[nz] = np.log(p[nz])
    h = -float(np.sum(p * log_p))
    return float(np.clip(h / np.log(s.size), 0.0, 1.0))


def proposal_visual_variance(
    box_feature: Optional[np.ndarray],
    support_features: Optional[np.ndarray],
    k1_prior: float = 0.0,
) -> float:
    """Per-proposal visual uncertainty: mean (1 - cos) between the box
    feature and the class's k support features (raw range [0, 2]).

    Same dispersion measure as the pre-registered ``sigma^2_visual`` (mean
    pairwise cosine distance across supports), evaluated per-proposal against
    the support set instead of across it. Unlike the class-level
    ``sigma_visual`` (exactly C distinct values -> D2 underpowered at C=3),
    this is a continuous per-proposal signal: a box far from the class
    support set has high visual uncertainty (and, consistently, low
    affinity), so D2 has statistical power (change_log.md 2026-08-05).

    k=1 is well defined here (a single observed distance — no dispersion to
    estimate), so the ``k1_prior`` only applies when the support set is empty
    (degenerate record); the pre-registered k1-prior semantics are unchanged
    for the class-level ``normalized_visual_variance``.

    Args:
        box_feature: (D,) box feature from the cache.
        support_features: (k, D) prototype support features.
        k1_prior: fallback when there are no support features (default 0.0).

    Returns:
        Mean pairwise (1 - cos) distance in [0, 2].
    """
    box = np.asarray(box_feature, dtype=np.float64).ravel()
    sup = np.asarray(support_features, dtype=np.float64)
    if (
        box_feature is None
        or sup.ndim != 2
        or sup.shape[0] == 0
        or box.size == 0
    ):
        return float(k1_prior)
    box = _l2_normalize_rows(box[None, :])[0]
    sup = _l2_normalize_rows(sup)
    sims = sup @ box  # (k,) cosine similarities, both normalized
    return float(np.mean(1.0 - sims))


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


def absolute_normalize(
    values: np.ndarray,
    scale: float = COSINE_DISTANCE_MAX,
) -> np.ndarray:
    """Class-count-independent absolute scaling: x_tilde = x / scale.

    For the cosine-distance terms (``sigma^2_text`` / ``sigma^2_visual``) the
    raw mean pairwise cosine distance has a FIXED range [0, 2]
    (1 - cos, cos in [-1, 1]), so ``scale=2.0`` maps [0, 2] exactly onto
    [0, 1] without any support-set statistics.

    Unlike min-max, the normalized value of a class does NOT depend on how
    many classes are in the set — it is invariant to the class count. This
    fixes the 2-class degeneracy (D-Fire): min-max across C classes yields
    only C distinct normalized values, so with C=2 the variance terms
    collapse to {0, 1} and the D1/D2 diagnostics lose statistical power
    (pre-registration deviation, change_log.md 2026-08-03).
    """
    values = np.asarray(values, dtype=np.float64)
    return np.clip(values / float(scale), 0.0, 1.0)


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
