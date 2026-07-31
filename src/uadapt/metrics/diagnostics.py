"""Pre-registered diagnostics D1-D5 (proposal Section 7.5).

These are sanity checks on the core assumptions of the uncertainty-gated
fusion mechanism — NOT contributions. They are computed AFTER main results are
collected and never influence experimental choices or filtering criteria. If a
diagnostic reveals a failure of a core assumption, the finding is reported
honestly (docs/pre_registration.md).

  D1 (Text uncertainty-accuracy correlation): per-bin error rate of proposal
     correctness (IoU >= 0.5 with correct class label) vs bin of
     sigma_tilde^2_text (10 bins) + Spearman rho.
  D2 (Visual uncertainty-accuracy correlation): analogous for
     sigma_tilde^2_visual.
  D3 (Gate favorability): fraction of disagreeing cases where the gate assigns
     higher weight to the more accurate modality; binomial test vs 0.5.
  D4 (Affinity diagnostic): mean signed Delta w = w_full - w_{gamma=0} binned
     by a_visual (validates the bias-variance model).
  D5 (Distribution of normalized variances): histogram + quartiles; flags the
     Taylor expansion if >30% of values fall below 0.25 or above 0.75.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Pre-registered constants
D1_N_BINS = 10
D2_N_BINS = 10
D5_FLAG_FRACTION = 0.30
D5_LOWER = 0.25
D5_UPPER = 0.75
D3_ALPHA = 0.05


@dataclass
class DiagnosticResult:
    """Bagged result of one diagnostic (json-serializable)."""

    name: str
    summary: Dict[str, float]
    flag: Optional[str] = None
    raw: Optional[Dict] = None


def _bin_means(values: np.ndarray, targets: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bin ``values`` into n_bins and return (bin_edges_mid, mean target per bin)."""
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    means = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (values >= lo) & (values <= hi) if i == n_bins - 1 else (values >= lo) & (values < hi)
        means[i] = float(targets[in_bin].mean()) if in_bin.any() else 0.0
    return mids, means


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation in pure numpy (ties averaged)."""
    if len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
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


def _binomial_pvalue(n: int, k: int, p0: float = 0.5) -> float:
    """Two-sided binomial test: 2 * min(P(X<=k), P(X>=k)), clamped to 1.

    Exact via math.comb for n <= 5000; normal approximation above that.
    """
    import math

    if n == 0:
        return 1.0
    if n <= 5000:
        pmf = [
            math.comb(n, i) * (p0**i) * ((1 - p0) ** (n - i)) for i in range(n + 1)
        ]
        p_lo = sum(pmf[: k + 1])   # P(X <= k)
        p_hi = sum(pmf[k:])        # P(X >= k)
        return float(min(1.0, 2.0 * min(p_lo, p_hi)))
    se = math.sqrt(n * p0 * (1 - p0))
    z = (k - n * p0) / se
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))


def d1_text_uncertainty_accuracy(
    norm_text_variances: np.ndarray,
    correct: np.ndarray,
) -> DiagnosticResult:
    """D1: text uncertainty vs proposal-level error rate (10 bins) + Spearman."""
    correct = np.asarray(correct, dtype=float)
    mids, err_rates = _bin_means(
        np.asarray(norm_text_variances, dtype=float), 1.0 - correct, D1_N_BINS
    )
    rho = _spearman_rho(
        np.asarray(norm_text_variances, dtype=float), 1.0 - correct
    )
    return DiagnosticResult(
        name="D1_text_uncertainty_accuracy",
        summary={"spearman_rho": rho, "n": float(len(correct))},
        flag="positive rho expected" if rho > 0 else "rho <= 0: proxy may be uninformative",
        raw={"bin_mid": mids.tolist(), "error_rate": err_rates.tolist()},
    )


def d2_visual_uncertainty_accuracy(
    norm_visual_variances: np.ndarray,
    correct: np.ndarray,
) -> DiagnosticResult:
    """D2: visual uncertainty vs visual-only error rate + Spearman."""
    correct = np.asarray(correct, dtype=float)
    mids, err_rates = _bin_means(
        np.asarray(norm_visual_variances, dtype=float), 1.0 - correct, D2_N_BINS
    )
    rho = _spearman_rho(
        np.asarray(norm_visual_variances, dtype=float), 1.0 - correct
    )
    return DiagnosticResult(
        name="D2_visual_uncertainty_accuracy",
        summary={"spearman_rho": rho, "n": float(len(correct))},
        flag=(
            "weak/absent correlation: pairwise support variance is a poor proxy"
            if rho <= 0.05
            else "ok"
        ),
        raw={"bin_mid": mids.tolist(), "error_rate": err_rates.tolist()},
    )


def d3_gate_favorability(
    w_text_better: np.ndarray,
    w_visual_better: np.ndarray,
) -> DiagnosticResult:
    """D3: gate favorability.

    Args:
        w_text_better: gate weights w on cases where the TEXT modality is the
            more accurate one (expected: w < 0.5).
        w_visual_better: gate weights on cases where the VISUAL modality is
            the more accurate one (expected: w > 0.5).

    Favorability fraction = fraction of cases where the gate points at the
    better modality; binomial test vs 0.5 (alpha=0.05).
    """
    n_fav = int(np.sum(np.asarray(w_text_better) < 0.5) + np.sum(np.asarray(w_visual_better) > 0.5))
    n_total = int(len(w_text_better) + len(w_visual_better))
    frac = n_fav / n_total if n_total else 0.0
    pvalue = _binomial_pvalue(n_total, n_fav)
    return DiagnosticResult(
        name="D3_gate_favorability",
        summary={
            "favorability_fraction": frac,
            "n": float(n_total),
            "binomial_pvalue": pvalue,
        },
        flag=None if pvalue < D3_ALPHA and frac > 0.5 else "gate not significantly favoring better modality",
    )


def d4_affinity_diagnostic(
    w_full: np.ndarray,
    w_gamma0: np.ndarray,
    affinities: np.ndarray,
    n_bins: int = D2_N_BINS,
) -> DiagnosticResult:
    """D4: mean signed Delta w = w_full - w_{gamma=0} binned by a_visual.

    Validates the bias-variance model: higher affinity should shift the gate
    toward visual (Delta w > 0 at high affinity, < 0 at low affinity).
    """
    delta = np.asarray(w_full, dtype=float) - np.asarray(w_gamma0, dtype=float)
    mids, mean_delta = _bin_means(np.asarray(affinities, dtype=float), delta, n_bins)
    corr = _spearman_rho(np.asarray(affinities, dtype=float), delta)
    return DiagnosticResult(
        name="D4_affinity_diagnostic",
        summary={"affinity_delta_spearman": corr, "n": float(len(delta))},
        flag=(
            "affinity shifts gate in the correct direction"
            if corr > 0
            else "affinity shift inconsistent with bias-variance model"
        ),
        raw={"bin_mid": mids.tolist(), "mean_delta_w": mean_delta.tolist()},
    )


def d5_variance_distribution(
    norm_text_variances: np.ndarray,
    norm_visual_variances: np.ndarray,
) -> DiagnosticResult:
    """D5: empirical distribution of normalized variances (Taylor validity).

    Flags if >30% of values fall below 0.25 or above 0.75 -> pre-registered
    Beta-regression fallback analysis.
    """
    all_v = np.concatenate(
        [
            np.asarray(norm_text_variances, dtype=float).ravel(),
            np.asarray(norm_visual_variances, dtype=float).ravel(),
        ]
    )
    if len(all_v) == 0:
        return DiagnosticResult(name="D5_variance_distribution", summary={})
    frac_boundary = float(np.mean((all_v < D5_LOWER) | (all_v > D5_UPPER)))
    q = np.percentile(all_v, [25, 50, 75])
    return DiagnosticResult(
        name="D5_variance_distribution",
        summary={
            "frac_below_0.25_or_above_0.75": frac_boundary,
            "q25": float(q[0]),
            "median": float(q[1]),
            "q75": float(q[2]),
            "n": float(len(all_v)),
        },
        flag=(
            "Taylor expansion validity FLAGGED: >30% of normalized variances "
            "cluster near 0/1; run Beta-regression fallback"
            if frac_boundary > D5_FLAG_FRACTION
            else "expansion validity ok"
        ),
        raw={"hist": np.histogram(all_v, bins=10, range=(0, 1))[0].tolist()},
    )
