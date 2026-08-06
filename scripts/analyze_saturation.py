#!/usr/bin/env python
"""analyze_saturation.py — quantify Mode A gate saturation on the real pilot.

Loads the ``proposal_level.json`` files written by
``scripts/demo_mode_a_end_to_end.py`` (real-cache path) and quantifies the
gate-saturation hypothesis from the n=100 pilot review:

  * the gate logit is ``eta = -alpha*vv + beta*tv + gamma*affinity``;
  * on real RoI-pooled features the affinity term clusters near 1.0, which
    pins the sigmoid weight ``w = sigma(eta)`` high and leaves the variance
    terms with too little influence to move ``w`` away from the visual
    branch.

For every dataset the script prints summary statistics (mean/std/min/max) for
``w``, ``affinity``, ``norm_text_var`` and ``norm_visual_var``, the fraction
of proposals with ``w > 0.55``, a counterfactual analysis (gate weight if the
variance terms were inert, i.e. ``w = sigma(affinity)``, and how far the
variance terms actually move ``w``), and a one-line verdict. It also saves
one 2x2 histogram figure per dataset.

Usage:
    python scripts/analyze_saturation.py
    python scripts/analyze_saturation.py \\
        --proposals outputs/real_data/ladd/proposal_level.json \\
                     outputs/real_data/dfire/proposal_level.json \\
        --out-dir outputs/real_data/saturation_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless (no display / no Qt)
import matplotlib.pyplot as plt

# Default pilot artifacts (gitignored; written by demo_mode_a_end_to_end.py).
DEFAULT_PROPOSALS = [
    Path("outputs/real_data/ladd/proposal_level.json"),
    Path("outputs/real_data/dfire/proposal_level.json"),
]

# Fields to summarize + histogram. Keys match the proposal rows written by
# src/uadapt/demo/pipeline.py.
FIELDS = [
    ("w", "gate weight w"),
    ("affinity", "visual affinity a_visual"),
    ("norm_text_var", "normalized text variance"),
    ("norm_visual_var", "normalized visual variance"),
]

W_SATURATION_THRESHOLD = 0.55
AFFINITY_CLUSTER_THRESHOLD = 0.8

# Mode A default coefficients (alpha=beta=gamma=1, pre-registration §2).
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def load_proposals(path: Path) -> List[Dict]:
    """Load the ``proposals`` list from a demo proposal_level.json."""
    payload = json.loads(Path(path).read_text())
    proposals = payload.get("proposals")
    if not proposals:
        raise ValueError(
            f"{path} contains no 'proposals' list (expected output of "
            "scripts/demo_mode_a_end_to_end.py)"
        )
    return proposals


def summary_stats(values: np.ndarray) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0.0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": float(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def analyze(proposals: List[Dict], label: str) -> Dict[str, str]:
    """Print stats + counterfactual decomposition + verdict for one dataset."""
    w = np.asarray([p["w"] for p in proposals], dtype=float)
    affinity = np.asarray([p["affinity"] for p in proposals], dtype=float)
    tv = np.asarray([p["norm_text_var"] for p in proposals], dtype=float)
    vv = np.asarray([p["norm_visual_var"] for p in proposals], dtype=float)

    lines: List[str] = []
    lines.append(f"=== {label} ({len(proposals)} proposals) ===")
    lines.append(f"{'field':<22}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    lines.append("-" * 58)
    arrays = {"w": w, "affinity": affinity,
              "norm_text_var": tv, "norm_visual_var": vv}
    stats = {name: summary_stats(arrays[name]) for name, _ in FIELDS}
    for name, label_col in FIELDS:
        s = stats[name]
        lines.append(
            f"{label_col:<22}{s['mean']:>9.4f}{s['std']:>9.4f}"
            f"{s['min']:>9.4f}{s['max']:>9.4f}"
        )

    frac_w_sat = float(np.mean(w > W_SATURATION_THRESHOLD))
    frac_aff_hi = float(np.mean(affinity > AFFINITY_CLUSTER_THRESHOLD))
    lines.append("")
    lines.append(
        f"w > {W_SATURATION_THRESHOLD:.2f}: {frac_w_sat * 100:.1f}%  |  "
        f"affinity > {AFFINITY_CLUSTER_THRESHOLD:.2f}: {frac_aff_hi * 100:.1f}%"
    )

    # --- counterfactual decomposition --------------------------------------
    # If the variance terms were inert (tv = vv = 0), the weight would be
    # w_aff = sigma(affinity): the affinity term alone, per proposal. Compare
    # the realized weight against this counterfactual to isolate how much the
    # uncertainty terms actually move the gate.
    w_aff_only = _sigmoid(GAMMA * affinity)
    mean_w_aff_only = float(w_aff_only.mean())
    var_shift = float(np.mean(np.abs(w - w_aff_only)))
    min_aff = float(affinity.min())
    floor_w = float(_sigmoid(GAMMA * min_aff))
    lines.append("")
    lines.append("counterfactual: gate weight if the variance terms were inert")
    lines.append(
        f"  mean w = sigma(affinity) alone: {mean_w_aff_only:.4f}  "
        f"(min affinity {min_aff:.4f} -> w >= {floor_w:.4f} for every proposal)"
    )
    lines.append(
        f"  realized mean w: {float(w.mean()):.4f}  ->  variance terms move w "
        f"by on average +/-{var_shift:.4f}"
    )

    # --- verdict ------------------------------------------------------------
    aff_mean = float(affinity.mean())
    if frac_w_sat > 0.9 and (frac_aff_hi > 0.9 or mean_w_aff_only > 0.6):
        if var_shift < 0.05:
            verdict = (
                f"CONFIRMED: affinity (mean {aff_mean:.2f}) alone pins mean w at "
                f"{mean_w_aff_only:.2f}; the variance terms move w by only "
                f"+/-{var_shift:.3f} on average, so {frac_w_sat * 100:.0f}% of "
                f"proposals sit above {W_SATURATION_THRESHOLD:.2f} — the gate is "
                "effectively a fixed high weight dominated by affinity; the "
                "uncertainty terms are inert."
            )
        else:
            direction = "up (high text uncertainty pushes toward visual)" \
                if float(tv.mean()) > float(vv.mean()) else "down"
            verdict = (
                f"CONFIRMED (with nuance): affinity (mean {aff_mean:.2f}) sets a "
                f"floor of w >= {floor_w:.2f} for every proposal; the variance "
                f"terms move w by +/-{var_shift:.3f} on average — on this dataset "
                f"mostly {direction} — but the net range of w is narrow "
                f"(std {float(w.std()):.3f}) and {frac_w_sat * 100:.0f}% of "
                f"proposals still sit above {W_SATURATION_THRESHOLD:.2f}. The "
                "gate leans visual for almost every proposal regardless of the "
                "uncertainty signals."
            )
    else:
        verdict = (
            f"NOT CONFIRMED: w > {W_SATURATION_THRESHOLD:.2f} for only "
            f"{frac_w_sat * 100:.1f}% of proposals (mean affinity "
            f"{aff_mean:.2f}) — the gate is not pinned to the visual branch."
        )
    lines.append(f"  verdict: {verdict}")
    lines.append("")

    for line in lines:
        print(line)

    return {
        "label": label,
        "n": str(len(proposals)),
        "w_mean": f"{float(w.mean()):.4f}",
        "w_std": f"{float(w.std()):.4f}",
        "affinity_mean": f"{aff_mean:.4f}",
        "frac_w_gt_0.55": f"{frac_w_sat * 100:.1f}%",
        "frac_affinity_gt_0.8": f"{frac_aff_hi * 100:.1f}%",
        "w_if_variance_inert": f"{mean_w_aff_only:.4f}",
        "variance_shift_mean": f"{var_shift:.4f}",
        "verdict": verdict,
    }


def plot_histograms(proposals: List[Dict], label: str, out_dir: Path) -> Path:
    """One 2x2 histogram figure per dataset (4 variables), saved to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Mode A gate inputs/outputs — {label} ({len(proposals)} proposals)",
        fontsize=13, fontweight="bold",
    )

    for ax, (key, title) in zip(axes.ravel(), FIELDS):
        values = np.asarray([p[key] for p in proposals], dtype=float)
        ax.hist(values, bins=30, range=(0.0, 1.0), color="#1f77b4",
                edgecolor="white", alpha=0.85)
        ax.axvline(values.mean(), color="#d62728", ls="--", lw=1.2,
                   label=f"mean {values.mean():.3f}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(title)
        ax.set_ylabel("proposals")
        ax.set_title(f"{key}  (std {values.std():.3f})", fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{label.lower().replace(' ', '_')}_saturation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify Mode A gate saturation on real pilot proposal data."
    )
    parser.add_argument(
        "--proposals", nargs="*", type=Path, default=DEFAULT_PROPOSALS,
        help="proposal_level.json files (default: the real n=100 pilot outputs)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/real_data/saturation_analysis"),
        help="directory for histogram figures + summary.txt",
    )
    args = parser.parse_args()

    missing = [p for p in args.proposals if not p.exists()]
    if missing:
        print("ERROR: missing proposal files: " + ", ".join(map(str, missing)),
              file=sys.stderr)
        sys.exit(1)

    print("=" * 74)
    print("Mode A gate saturation analysis (real n=100 pilot proposals)")
    print("=" * 74)
    summaries: List[Dict[str, str]] = []
    saved_figures: List[Path] = []
    for path in args.proposals:
        proposals = load_proposals(path)
        label = path.parent.name.upper()  # e.g. LADD / DFIRE
        summaries.append(analyze(proposals, label))
        saved_figures.append(plot_histograms(proposals, label, args.out_dir))

    # --- final cross-dataset verdict ---------------------------------------
    print("=" * 74)
    print("SATURATION SUMMARY")
    print("=" * 74)
    for s in summaries:
        print(f"- {s['label']}: {s['verdict']}")

    lines: List[str] = []
    for s in summaries:
        lines += [
            f"label: {s['label']}",
            f"n: {s['n']}",
            f"w_mean: {s['w_mean']}",
            f"w_std: {s['w_std']}",
            f"affinity_mean: {s['affinity_mean']}",
            f"frac_w_gt_0.55: {s['frac_w_gt_0.55']}",
            f"frac_affinity_gt_0.8: {s['frac_affinity_gt_0.8']}",
            f"w_if_variance_inert: {s['w_if_variance_inert']}",
            f"variance_shift_mean: {s['variance_shift_mean']}",
            f"verdict: {s['verdict']}",
            "",
        ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.txt").write_text("\n".join(lines))
    print(f"\nsummary -> {args.out_dir / 'summary.txt'}")
    for fig in saved_figures:
        print(f"histogram -> {fig}")


if __name__ == "__main__":
    main()
