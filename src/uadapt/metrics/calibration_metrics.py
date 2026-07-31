"""Calibration and reliability metrics (RQ4).

  * ECE with 15 bins (pre-registered; proposal Phase 5).
  * Brier score.
  * Reliability table for reliability diagrams.
  * Uncertainty AUROC: rank-based discrimination of incorrect predictions by
    their uncertainty (uncertainty is high when the model is wrong).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

DEFAULT_ECE_BINS = 15


def ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Expected Calibration Error with ``n_bins`` equal-width bins.

    confidences: predicted confidence in [0, 1].
    correct: 0/1 correctness flags.
    """
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf >= lo) & (conf < hi)
        if i == n_bins - 1:  # include upper edge in the last bin
            in_bin = (conf >= lo) & (conf <= hi)
        if not in_bin.any():
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(conf[in_bin].mean())
        total += (len(conf[in_bin]) / n) * abs(bin_acc - bin_conf)
    return float(total)


def brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between confidence and outcome."""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    return float(np.mean((conf - correct) ** 2))


def reliability_table(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin_conf, bin_acc, bin_count) for reliability diagrams."""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf: List[float] = []
    bin_acc: List[float] = []
    bin_count: List[float] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if in_bin.any():
            bin_conf.append(float(conf[in_bin].mean()))
            bin_acc.append(float(correct[in_bin].mean()))
            bin_count.append(float(in_bin.sum()))
        else:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            bin_count.append(0.0)
    return (
        np.asarray(bin_conf),
        np.asarray(bin_acc),
        np.asarray(bin_count),
    )


def uncertainty_auroc(uncertainties: np.ndarray, correct: np.ndarray) -> float:
    """AUC that ranks uncertain predictions above correct ones (higher is
    better: uncertainty discriminates errors). Computed as the Mann-Whitney U
    statistic normalized to [0, 1]."""
    unc = np.asarray(uncertainties, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(unc) == 0 or not correct.any() or correct.all():
        return 0.5  # undefined without both classes
    err = unc[~correct]
    ok = unc[correct]
    n_err, n_ok = len(err), len(ok)
    # rank-based U: P(err_unc > ok_unc) + 0.5 * P(tie)
    ranks = _rankdata(np.concatenate([err, ok]))
    u_err = float(ranks[:n_err].sum()) - n_err * (n_err + 1) / 2.0
    return float(u_err / (n_err * n_ok))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float((i + 1 + j + 1) / 2.0)
        i = j + 1
    return ranks
