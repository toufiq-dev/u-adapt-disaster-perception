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

Pooling (pre-registration deviation 2026-08-03, docs/change_log.md):
D1/D2/D3 computed on D-Fire alone are STRUCTURALLY UNDERPOWERED — its 2
classes (fire, smoke) yield only 2 distinct normalized variance values, so a
meaningful Spearman rank correlation / gate-favorability trend cannot be
computed from 2 data points. The pre-registered protocol therefore computes
D1/D2/D3 POOLED across LADD + D-Fire (3 distinct classes: pedestrian, fire,
smoke). Each of d1_text_uncertainty_accuracy, d2_visual_uncertainty_accuracy
and d3_gate_favorability accepts an optional ``pool_with`` argument (the
second dataset's arrays) and returns a structured dict with per-dataset
(``primary`` / ``secondary``) and ``pooled`` results; pooled values are the
PRIMARY diagnostic claim, per-dataset values are still reported. Existing
calls without ``pool_with`` are unchanged (backward compatible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    """Spearman rank correlation in pure numpy (ties averaged).

    Returns 0.0 when either input is constant (undefined correlation — e.g.
    a single-class dataset contributes exactly one distinct variance value),
    so results stay finite and JSON-serializable.
    """
    if len(x) < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.ptp(rx) == 0.0 or np.ptp(ry) == 0.0:
        return 0.0
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


def _validate_pool_arrays(
    primary: Sequence[np.ndarray],
    secondary: Sequence[np.ndarray],
    diag_name: str,
    paired: bool = True,
) -> None:
    """Validate arrays passed to a ``pool_with`` call (deviation 2026-08-03).

    Raises a clear ValueError when a caller tries to pool INCOMPATIBLE
    datasets: every array must be 1-D, and for the paired D1/D2 case each
    dataset's variance array must have the same length as its correctness
    array (one variance value per proposal). D3 (``paired=False``) only
    enforces the 1-D requirement — its two weight subsets legitimately have
    different lengths.
    """
    for label, arrays in (
        ("primary dataset", primary),
        ("pool_with dataset", secondary),
    ):
        for arr in arrays:
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError(
                    f"{diag_name}: cannot pool incompatible datasets — {label} "
                    f"contributed a {arr.ndim}-D array (shape {arr.shape}); pooled "
                    "diagnostics are defined at the proposal level and need 1-D arrays."
                )
    if paired:
        (v1, e1), (v2, e2) = primary, secondary
        for label, (v, e) in (("primary", (v1, e1)), ("pool_with", (v2, e2))):
            if len(v) != len(e):
                raise ValueError(
                    f"{diag_name}: cannot pool incompatible datasets — {label} "
                    f"variance array has length {len(v)} but its correctness array has "
                    f"length {len(e)}. Each dataset must pair exactly one variance value "
                    "with one correctness label per proposal."
                )


def d1_text_uncertainty_accuracy(
    norm_text_variances: np.ndarray,
    correct: np.ndarray,
    pool_with: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[DiagnosticResult, Dict[str, DiagnosticResult]]:
    """D1: text uncertainty vs proposal-level error rate (10 bins) + Spearman.

    Args:
        norm_text_variances: normalized per-proposal text variance in [0, 1].
        correct: per-proposal correctness (IoU >= 0.5 with same-class GT).
        pool_with: optional ``(norm_text_variances, correct)`` of a SECOND
            dataset (e.g. LADD when the primary is D-Fire). When provided,
            returns a structured dict::

                {"primary":   DiagnosticResult,  # this dataset alone
                 "secondary": DiagnosticResult,  # pool_with dataset alone
                 "pooled":    DiagnosticResult}  # concatenated -> PRIMARY claim

            Pooling is mandated by the pre-registration deviation of
            2026-08-03 (docs/change_log.md §10): on 2-class D-Fire there are
            only 2 distinct normalized variance values, so the Spearman
            correlation is structurally underpowered. Pooled across
            LADD + D-Fire (3 distinct classes) it has statistical power.

    Returns:
        A :class:`DiagnosticResult` when ``pool_with`` is None (unchanged,
        backward compatible), else the structured per-dataset + pooled dict.

    Raises:
        ValueError: if the arrays to pool are not 1-D or a dataset's
            variance/error arrays differ in length (incompatible datasets).
    """
    if pool_with is not None:
        primary_var = np.asarray(norm_text_variances, dtype=float)
        primary_correct = np.asarray(correct, dtype=float)
        other_var = np.asarray(pool_with[0], dtype=float)
        other_correct = np.asarray(pool_with[1], dtype=float)
        _validate_pool_arrays(
            (primary_var, primary_correct), (other_var, other_correct),
            "D1_text_uncertainty_accuracy",
        )
        return {
            "primary": d1_text_uncertainty_accuracy(norm_text_variances, correct),
            "secondary": d1_text_uncertainty_accuracy(pool_with[0], pool_with[1]),
            "pooled": d1_text_uncertainty_accuracy(
                np.concatenate([primary_var, other_var]),
                np.concatenate([primary_correct, other_correct]),
            ),
        }
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
    pool_with: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[DiagnosticResult, Dict[str, DiagnosticResult]]:
    """D2: visual uncertainty vs visual-only error rate + Spearman.

    Same contract as :func:`d1_text_uncertainty_accuracy` (including the
    optional ``pool_with`` for the pooled LADD + D-Fire protocol, deviation
    2026-08-03); returns a structured per-dataset + pooled dict when pooling.

    Args:
        norm_visual_variances: normalized per-proposal visual variance in
            [0, 1].
        correct: per-proposal correctness.
        pool_with: optional ``(norm_visual_variances, correct)`` of a second
            dataset; pooled Spearman ρ is the PRIMARY claim.

    Returns:
        A :class:`DiagnosticResult` when ``pool_with`` is None, else the
        structured per-dataset + pooled dict.

    Raises:
        ValueError: if the arrays to pool are incompatible (see
            :func:`_validate_pool_arrays`).
    """
    if pool_with is not None:
        primary_var = np.asarray(norm_visual_variances, dtype=float)
        primary_correct = np.asarray(correct, dtype=float)
        other_var = np.asarray(pool_with[0], dtype=float)
        other_correct = np.asarray(pool_with[1], dtype=float)
        _validate_pool_arrays(
            (primary_var, primary_correct), (other_var, other_correct),
            "D2_visual_uncertainty_accuracy",
        )
        return {
            "primary": d2_visual_uncertainty_accuracy(norm_visual_variances, correct),
            "secondary": d2_visual_uncertainty_accuracy(pool_with[0], pool_with[1]),
            "pooled": d2_visual_uncertainty_accuracy(
                np.concatenate([primary_var, other_var]),
                np.concatenate([primary_correct, other_correct]),
            ),
        }
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
    pool_with: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[DiagnosticResult, Dict[str, DiagnosticResult]]:
    """D3: gate favorability.

    Args:
        w_text_better: gate weights w on cases where the TEXT modality is the
            more accurate one (expected: w < 0.5).
        w_visual_better: gate weights on cases where the VISUAL modality is
            the more accurate one (expected: w > 0.5).
        pool_with: optional ``(w_text_better, w_visual_better)`` of a SECOND
            dataset (e.g. LADD when the primary is D-Fire). Same structured
            return contract as :func:`d1_text_uncertainty_accuracy`; the
            binomial test runs on the concatenated favorability counts, so
            the pooled result is the PRIMARY claim (deviation 2026-08-03).

    Favorability fraction = fraction of cases where the gate points at the
    better modality; binomial test vs 0.5 (alpha=0.05).

    Returns:
        A :class:`DiagnosticResult` when ``pool_with`` is None, else the
        structured per-dataset + pooled dict.

    Raises:
        ValueError: if the arrays to pool are not 1-D (the two weight
            subsets of a dataset legitimately differ in length).
    """
    if pool_with is not None:
        primary_text = np.asarray(w_text_better, dtype=float)
        primary_visual = np.asarray(w_visual_better, dtype=float)
        other_text = np.asarray(pool_with[0], dtype=float)
        other_visual = np.asarray(pool_with[1], dtype=float)
        _validate_pool_arrays(
            (primary_text, primary_visual), (other_text, other_visual),
            "D3_gate_favorability", paired=False,
        )
        return {
            "primary": d3_gate_favorability(w_text_better, w_visual_better),
            "secondary": d3_gate_favorability(pool_with[0], pool_with[1]),
            "pooled": d3_gate_favorability(
                np.concatenate([primary_text, other_text]),
                np.concatenate([primary_visual, other_visual]),
            ),
        }
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
