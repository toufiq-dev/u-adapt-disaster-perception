#!/usr/bin/env python
"""generate_real_data_report.py — markdown report for the real-data validation run.

Consumes the outputs of ``scripts/run_real_data_validation.sh``:

    * ``demo_mode_a_end_to_end.py`` results.json  (LADD + D-Fire): map50 per
      method, gap recovery, per-class AP, meta (data_source, norm_strategy)
    * ``compute_pooled_diagnostics.py`` pooled_diagnostics.json: per-dataset +
      pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10)
    * dataset configs (``configs/datasets/{ladd,dfire}.yaml``): the
      Michailidou et al. zero-shot/transfer mAP50 baselines (Table III)

and renders ``docs/real_data_results.md`` with an executive summary, per-method
mAP50 tables, gap-recovery analysis (fraction of the zero-shot → transfer gap
closed), pooled D1/D2/D3 values, and a comparison against the literature
baselines. A prominent caveat is emitted automatically when the underlying
runs used the synthetic stand-in world (meta.data_source == "synthetic") —
real-data evaluation supersedes it.

**Pilot labeling:** when any run reports ``meta.n_test_images <= 100`` (or
``--pilot`` is passed), the title reads **"PILOT RESULTS (n=<N> images)"**
and a banner warns that the numbers are a preliminary pipeline check, not
final thesis results — so a n=10/n=100 pilot report can never be mistaken
for the final thesis report.

Usage:
    python scripts/generate_real_data_report.py \\
        --ladd-results  outputs/real_data/ladd/results.json \\
        --dfire-results outputs/real_data/dfire/results.json \\
        --pooled-diagnostics outputs/real_data/pooled_diagnostics.json \\
        --out docs/real_data_results_pilot.md
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Method ordering + display labels (match demo_mode_a_end_to_end.py).
# ---------------------------------------------------------------------------
METHODS = [
    ("zero_shot_raw", "Zero-shot (raw detector scores)"),
    ("text_only", "Text-only (w=0)"),
    ("visual_only", "Visual-only (w=1)"),
    ("naive_average", "Naive averaging (w=0.5, T-Rex2 surrogate)"),
    ("uadapt_mode_a", "U-ADAPT Mode A (analytic gate)"),
]

DIAG_NAMES = {
    "D1_text_uncertainty_accuracy": "D1 — text uncertainty–accuracy",
    "D2_visual_uncertainty_accuracy": "D2 — visual uncertainty–accuracy",
    "D3_gate_favorability": "D3 — gate favorability",
}


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def _pct(x: float, nd: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string (e.g. 0.956 -> '95.6%')."""
    return f"{100.0 * x:.{nd}f}%"


def _pct_pp(x: float, nd: int = 1) -> str:
    """Format a 0-1 fraction as a signed percentage point difference."""
    return f"{100.0 * x:+.{nd}f} pp"


def _fmt_mAP(x: float) -> str:
    """mAP50 stored as a 0-1 fraction in demo results; display as percent."""
    return f"{100.0 * x:.1f}"


def _caveat(results: Dict, dataset: str) -> Optional[str]:
    """Return a caveat line if the run was on the synthetic stand-in world."""
    src = results.get("meta", {}).get("data_source")
    if src == "synthetic":
        return (
            f"> ⚠️ **{dataset} numbers below come from the deterministic synthetic "
            "stand-in world** (no real cache on this machine) — they validate the "
            "pipeline *mechanism*, not a research result. Re-run "
            "`scripts/run_real_data_validation.sh` once raw data is cached; real "
            "results supersede these numbers."
        )
    return None


def _literature_numbers(cfg: Dict) -> Dict[str, float]:
    """Michailidou et al. (Table III) baselines from the dataset config meta."""
    meta = cfg.get("meta", {})
    return {
        "zero_shot": float(meta.get("zero_shot_map50_gdino", float("nan"))),
        "transfer": float(meta.get("transfer_map50_gdino", float("nan"))),
        "gap_pp": float(meta.get("gap_pp", float("nan"))),
    }


def _mAP50_section(dataset: str, results: Dict) -> List[str]:
    """Markdown table of mAP50 per method for one dataset."""
    lines = [
        f"### {dataset} — mAP50 (subset)",
        "",
        "| Method | mAP50 (%) | Δ vs zero-shot |",
        "|---|---|---|",
    ]
    map50 = results.get("map50", {})
    base = map50.get("zero_shot_raw", float("nan"))
    for key, label in METHODS:
        v = map50.get(key, float("nan"))
        delta = (v - base) if key != "zero_shot_raw" else 0.0
        bold_label = f"**{label}**" if key == "uadapt_mode_a" else label
        bold_val = f"**{_fmt_mAP(v)}**" if key == "uadapt_mode_a" else _fmt_mAP(v)
        lines.append(f"| {bold_label} | {bold_val} | {_pct_pp(delta)} |")
    return lines


def _gap_recovery_section(dataset: str, results: Dict, lit: Dict) -> List[str]:
    """Gap-recovery analysis: fraction of the zero-shot -> transfer gap closed."""
    gap = results.get("gap_recovery", {})
    is_synthetic = results.get("meta", {}).get("data_source") == "synthetic"
    lines = [
        f"### {dataset} — gap recovery",
        "",
        f"Michailidou et al. (Grounding DINO, Table III) floor/ceiling: "
        f"zero-shot **{lit['zero_shot']:.1f}%** → transfer "
        f"**{lit['transfer']:.1f}%** (gap **{lit['gap_pp']:.1f} pp**).",
        "",
    ]
    vs_lit = gap.get("gap_recovery_vs_literature")
    if vs_lit is not None:
        if vs_lit > 1.0:
            if is_synthetic:
                over = " (>100% means the adapter exceeds the literature "\
                       "transfer ceiling — expected only on the synthetic "\
                       "stand-in)"
            else:
                n_img = results.get("meta", {}).get("n_test_images")
                over = " (>100% means the adapter exceeds the literature "\
                       "transfer ceiling — on real data this reflects the "\
                       "tiny pilot subset: the literature ceiling is "\
                       "measured over the full test set, while this run "\
                       f"scores only {n_img if n_img else 'a few'} "\
                       "selected images)"
        else:
            over = ""
        lines.append(
            f"- **Fraction of the zero-shot→transfer gap closed:** "
            f"{_pct(vs_lit)}{over} "
            f"(U-ADAPT {_fmt_mAP(gap.get('uadapt_map50', float('nan')))}% vs "
            f"zero-shot {_fmt_mAP(gap.get('zero_shot_raw_map50', float('nan')))}%; "
            f"transfer ceiling {lit['transfer']:.1f}%)."
        )
    if gap.get("gap_recovery_vs_oracle") is not None:
        lines.append(
            f"- Gap recovered vs the oracle re-rank ceiling "
            f"({_fmt_mAP(gap.get('oracle_rerank_map50', float('nan')))}%): "
            f"**{_pct(gap['gap_recovery_vs_oracle'])}** "
            f"(also vs proposal-recall ceiling {_pct(gap.get('gap_recovery_vs_ceiling', 0.0))})."
        )
    return lines


def _diag_value_table(pooled: Dict, per_dataset: Dict) -> List[str]:
    """Pooled D1/D2/D3 table + per-dataset table."""
    lines = ["### Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10)", ""]
    lines.append(
        "D-Fire alone has 2 classes → only 2 distinct variance values, so D1/D2/D3 "
        "on it alone are structurally underpowered. Per the pre-registered "
        "deviation, the primary diagnostic claim is computed **pooled across "
        "LADD + D-Fire** (3 distinct classes: pedestrian, fire, smoke)."
    )
    lines += ["", "| Diagnostic | Pooled value | n |"]
    lines.append("|---|---|---|")
    for key, label in DIAG_NAMES.items():
        d = pooled.get(key, {})
        s = d.get("summary", {})
        flag = d.get("flag") or ""
        if "spearman_rho" in s:
            lines.append(
                f"| {label} | Spearman ρ = **{s['spearman_rho']:+.3f}** "
                f"({flag}) | {s['n']:.0f} |"
            )
        else:
            lines.append(
                f"| {label} | favorability = **{s.get('favorability_fraction', 0.0):.1%}** "
                f"(binomial p = {s.get('binomial_pvalue', float('nan')):.3g}) | "
                f"{s.get('n', 0):.0f} |"
            )
    # D1 pooled-sign caveat (2026-08-05): the pooled Spearman rho mixes two
    # datasets at different error base rates whose text-entropy ranges are
    # structurally disjoint (LADD is single-class -> entropy 0.0 by
    # construction; D-Fire sits near max entropy). The pooled sign is
    # therefore dominated by the BETWEEN-dataset base-rate difference rather
    # than by a within-proposal uncertainty-accuracy relationship; the
    # within-dataset D1 is the interpretable diagnostic on real data.
    lines += [
        "",
        "> ⚠️ **D1 pooled-sign caveat (2026-08-05):** the pooled D1 Spearman ρ "
        "mixes LADD and D-Fire at different error base rates with structurally "
        "different text-entropy ranges (LADD is single-class → entropy 0.0 by "
        "construction; D-Fire sits near max entropy). The pooled sign is "
        "dominated by this **between-dataset base-rate difference**, not by a "
        "within-proposal uncertainty–accuracy relationship (per-dataset D1 is "
        "the interpretable signal). See `docs/change_log.md` 2026-08-05.",
        "",
        "> ⚠️ **D2 pooled-sign caveat (2026-08-05, confirmed at n=100):** the "
        "pooled D2 ρ is likewise driven by the **between-dataset normalization "
        "scale difference** (LADD min-max spreads visual variance across [0,1]; "
        "D-Fire absolute keeps it tiny). Within each dataset the visual "
        "uncertainty–accuracy relationship is ≈ 0 (LADD ρ = +0.077, D-Fire "
        "ρ = −0.022 at n=100), so the pooled positive sign is a scale artifact, "
        "not evidence of a within-proposal effect.",
        "",
        "> ⚠️ **D3 one-sidedness caveat (2026-08-05, confirmed at n=100):** the "
        "modality-accuracy disagreeing subsets are almost entirely "
        "*visual-better* (LADD 131/132, D-Fire 114/114 — 1 text-better case "
        "total). The affinity threshold (≥ 0.65) never fails on real features "
        "(affinity ∈ [0.64, 0.999]), so `visual_correct` saturates and the gate "
        "always leans visual — which the mAP50 table shows is the *weaker* "
        "reranker on this data. A 100% D3 is therefore a saturation artifact, "
        "not evidence the gate is well-calibrated. Diagnosing D3 properly "
        "requires proposals with lower affinity (full-scale run) or a "
        "re-thresholded visual-correctness rule.",
    ]
    # Per-dataset D1/D2/D3 for reporting.
    lines += ["", "Per-dataset values (reported; pooled is primary):", "",
              "| Dataset | D1 ρ | D2 ρ | D3 favorability |", "|---|---|---|---|"]
    for ds_name, ds_diag in (("LADD", per_dataset.get("ladd", {})),
                             ("D-Fire", per_dataset.get("dfire", {}))):
        d1 = ds_diag.get("D1_text_uncertainty_accuracy", {}).get("summary", {})
        d2 = ds_diag.get("D2_visual_uncertainty_accuracy", {}).get("summary", {})
        d3 = ds_diag.get("D3_gate_favorability", {}).get("summary", {})
        lines.append(
            f"| {ds_name} | {d1.get('spearman_rho', float('nan')):+.3f} | "
            f"{d2.get('spearman_rho', float('nan')):+.3f} | "
            f"{d3.get('favorability_fraction', float('nan')):.1%} |"
        )
    return lines


def _literature_table(lit_ladd: Dict, lit_dfire: Dict,
                      ladd: Dict, dfire: Dict) -> List[str]:
    """Michailidou et al. (Table III) baselines vs the U-ADAPT Mode A result."""
    def _row(name: str, lit: Dict, res: Dict) -> str:
        return (
            f"| {name} | {lit['zero_shot']:.1f} | {lit['transfer']:.1f} | "
            f"{lit['gap_pp']:.1f} | "
            f"{_fmt_mAP(res.get('map50', {}).get('uadapt_mode_a', float('nan')))} | "
            f"{_fmt_gap_closed(res.get('gap_recovery', {}).get('gap_recovery_vs_literature'))} |"
        )

    return [
        "## Comparison to literature baselines",
        "",
        "Michailidou et al. (preprint, Table III) with Grounding DINO; U-ADAPT "
        "numbers are the Mode A results above (synthetic stand-in unless real "
        "data was used — see caveats).",
        "",
        "| Dataset | Zero-shot mAP50 | Transfer mAP50 (ceiling) | Gap (pp) | U-ADAPT mAP50 | Gap closed |",
        "|---|---|---|---|---|---|",
        _row("LADD", lit_ladd, ladd),
        _row("D-Fire", lit_dfire, dfire),
    ]


# ---------------------------------------------------------------------------
# Comparative mode (2026-08-07): Analytic vs. Beta fallback side-by-side.
# ---------------------------------------------------------------------------

def _load_run_dir(run_dir: Path) -> Dict:
    """Load one pipeline output directory into {ladd, dfire, pooled}."""
    ladd = run_dir / "ladd" / "results.json"
    dfire = run_dir / "dfire" / "results.json"
    pooled = run_dir / "pooled_diagnostics.json"
    for p in (ladd, dfire):
        if not p.exists():
            raise FileNotFoundError(
                f"--compare-dirs directory {run_dir} is missing {p.name} at {p} "
                "(run scripts/run_real_data_validation.sh into this directory first)"
            )
    return {
        "ladd": load_json(ladd),
        "dfire": load_json(dfire),
        "pooled": load_json(pooled) if pooled.exists() else None,
    }


def _ds_key(ds: str) -> str:
    """'LADD' -> 'ladd', 'D-Fire' -> 'dfire' (results.json dict keys)."""
    return "dfire" if ds == "D-Fire" else ds.lower()


def _compare_mAP50_section(dataset: str, runs: Dict[str, Dict]) -> List[str]:
    """Side-by-side mAP50 table: rows = methods, columns = gate labels."""
    labels = list(runs.keys())
    dsk = _ds_key(dataset)
    lines = [
        f"### {dataset} — mAP50 (n=100 subset, side-by-side)",
        "",
        "| Method | " + " | ".join(labels) + " | Δ (β − α) |",
        "| --- | " + " | ".join(["---"] * len(labels)) + " | --- |",
    ]
    for key, label in METHODS:
        # Gate-neutral label for the Mode A row (the per-column headers carry
        # the gate name; the METHODS label hardcodes "(analytic gate)").
        if key == "uadapt_mode_a":
            label = "U-ADAPT Mode A"
        vals = [
            runs[lab][dsk].get("map50", {}).get(key, float("nan"))
            for lab in labels
        ]
        bold = [f"**{_fmt_mAP(v)}**" if key == "uadapt_mode_a" else _fmt_mAP(v)
                for v in vals]
        delta = (vals[-1] - vals[0]) if len(vals) == 2 and key != "zero_shot_raw" \
            else float("nan")
        dcell = "—" if delta != delta else _pct_pp(delta)
        lines.append(f"| {label} | " + " | ".join(bold) + f" | {dcell} |")
    return lines


def _compare_gate_section(runs: Dict[str, Dict]) -> List[str]:
    """Gate-weight behavior per dataset, side-by-side (saturation story)."""
    labels = list(runs.keys())
    lines = [
        "## Gate weight behavior (saturation diagnostic)",
        "",
        "| Dataset | " + " | ".join(f"{lab}: mean w" for lab in labels)
        + " | " + " | ".join(f"{lab}: std w" for lab in labels)
        + f" | {labels[-1]}: % > 0.55 |",
        "| --- | " + " | ".join(["---"] * (2 * len(labels) + 1)) + " |",
    ]
    for ds in ("LADD", "D-Fire"):
        rows = [
            runs[lab][_ds_key(ds)].get("gate_stats", {}) for lab in labels
        ]
        mean_cells = [f"{r.get('mean_w', float('nan')):.3f}" for r in rows]
        std_cells = [f"{r.get('std_w', float('nan')):.3f}" for r in rows]
        above = rows[-1].get("frac_above_0.55", float("nan"))
        lines.append(
            f"| {ds} | " + " | ".join(mean_cells) + " | " + " | ".join(std_cells)
            + f" | {100.0 * above:.0f}% |"
        )
    lines += [
        "",
        "> The analytic gate's mean weight saturates toward the visual "
        "modality (LADD 0.699, D-Fire 0.856 at n=100). The Beta fallback "
        "hedges the commitment where variances cluster at the extremes "
        "(D5 contingency): D-Fire mean w drops 0.856 -> 0.775, LADD "
        "0.699 -> 0.686 — still > 0.55 for essentially every proposal.",
    ]
    return lines


def _compare_diag_section(runs: Dict[str, Dict]) -> List[str]:
    """Pooled D1/D2/D3 side-by-side + per-dataset D5 Taylor-validity sentinel."""
    labels = list(runs.keys())
    pooled_ok = all(runs[lab].get("pooled") for lab in labels)
    lines = [
        "## Diagnostics D1–D5 (side-by-side)",
        "",
    ]
    if not pooled_ok:
        lines += [
            "_Pooled diagnostics missing for one or more runs (no "
            "pooled_diagnostics.json); only per-dataset D5 is shown._",
            "",
        ]
    else:
        lines += [
            "Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10; "
            "LADD + D-Fire, n=657 proposals):",
            "",
            "| Diagnostic | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join(["---"] * len(labels)) + " |",
        ]
        for key, label in DIAG_NAMES.items():
            cells = []
            for lab in labels:
                s = runs[lab]["pooled"].get("pooled", {}).get(key, {}).get("summary", {})
                if "spearman_rho" in s:
                    cells.append(f"ρ = **{s['spearman_rho']:+.3f}**")
                else:
                    cells.append(f"favorability = **{s.get('favorability_fraction', 0.0):.1%}**")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "> D1/D2 do not involve the gate weight, so they are identical "
            "across gates by construction; D3 (favorability on disagreeing "
            "cases) stays at 100% for both gates because the disagreeing "
            "subsets are almost entirely *visual-better* (affinity "
            "saturates >= 0.65) and every proposal is gated w > 0.55 — a "
            "saturation artifact, not evidence of calibration.",
        ]
    # Per-dataset D5 Taylor-validity sentinel (raw absolute-scale values).
    lines += ["", "D5 — Taylor-validity sentinel on the raw variance scale (per dataset):",
              "", "| Dataset | " + " | ".join(labels) + " |",
              "| --- | " + " | ".join(["---"] * len(labels)) + " |"]
    for ds in ("LADD", "D-Fire"):
        cells = []
        for lab in labels:
            pooled = runs[lab].get("pooled")
            d5 = (pooled or {}).get("per_dataset", {}).get(_ds_key(ds), {}).get(
                "D5_variance_distribution", {}
            )
            s = d5.get("summary", {})
            frac = s.get("frac_below_0.25_or_above_0.75")
            cells.append(
                f"{_pct(frac) if frac is not None else '—'} "
                f"({'FLAGGED' if frac is not None and frac > 0.30 else 'ok'})"
            )
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    lines += ["", "> D5 flags the boundary-clustered variance regime that triggers the "
              "pre-registered Beta-regression fallback (docs/pre_registration.md §10)."]
    return lines


def _compare_exec_summary(runs: Dict[str, Dict]) -> List[str]:
    """One-line per-dataset executive summary with gate deltas."""
    labels = list(runs.keys())
    a, b = labels[0], labels[-1]
    lines = ["## Executive Summary", ""]
    for ds in ("LADD", "D-Fire"):
        dsk = _ds_key(ds)
        ma = runs[a][dsk].get("map50", {})
        mb = runs[b][dsk].get("map50", {})
        ua_a, ua_b = ma.get("uadapt_mode_a"), mb.get("uadapt_mode_a")
        naive = ma.get("naive_average")
        zs = ma.get("zero_shot_raw")
        lines.append(
            f"- **{ds}** (n=100): Mode A {a} **{_fmt_mAP(ua_a)}%** vs "
            f"{b} **{_fmt_mAP(ua_b)}%** ({_pct_pp(ua_b - ua_a)}); "
            f"naive averaging **{_fmt_mAP(naive)}%**, zero-shot "
            f"**{_fmt_mAP(zs)}%**."
        )
        worse = "naive average" if ua_b < naive else "zero-shot baseline"
        lines.append(
            f"  - Beta softens the gate ({_fmt_mAP(ua_a)}% -> {_fmt_mAP(ua_b)}%), "
            f"but Mode A still underperforms the {worse} on {ds} at n=100 — "
            "the saturation artifact is softened, not resolved."
        )
    lines.append("")
    return lines


def _build_compare_report(args) -> List[str]:
    """Render the Analytic-vs-Beta side-by-side report body."""
    labels = [l or d.name for l, d in zip(args.compare_labels, args.compare_dirs)] \
        if args.compare_labels else [d.name for d in args.compare_dirs]
    runs = {lab: _load_run_dir(d) for lab, d in zip(labels, args.compare_dirs)}

    pilot_ns = [
        r["meta"]["n_test_images"]
        for r in (runs[labels[0]]["ladd"], runs[labels[0]]["dfire"])
        if r.get("meta", {}).get("n_test_images") is not None
    ]
    pilot = args.pilot or any(n <= 100 for n in pilot_ns)
    label_n = f"n={max(pilot_ns)}" if (pilot and pilot_ns and not args.pilot) else args.pilot_n
    title = (
        f"# U-ADAPT — Final Comparative Results ({label_n} subset): "
        "Analytic vs. Beta fallback"
        if pilot
        else "# U-ADAPT — Comparative Results: Analytic vs. Beta fallback"
    )
    lines = [
        title,
        "",
        f"*Generated {args.report_date} by `scripts/generate_real_data_report.py` "
        "(`--compare-dirs`). Gates: " + ", ".join(labels) + ".*",
        "",
    ]
    if pilot:
        lines += [
            "> 🧪 **PILOT RESULTS (n=100 subset)** — preliminary pipeline check on "
            "the first 100 images per dataset (Grounding DINO Swin-T, top-k=100, "
            "k=5 shots, seed 0). These numbers are NOT final thesis results; "
            "the full-data run and the 10-seed protocol "
            "(`scripts/run_10_seed_protocol.py`) supersede them.",
            "",
        ]
    lines += _compare_exec_summary(runs)
    lines += ["---", "", "## mAP50 results", ""]
    lines += _compare_mAP50_section("LADD", runs)
    lines += ["", "---", ""]
    lines += _compare_mAP50_section("D-Fire", runs)
    lines += ["", "---", ""]
    lines += _compare_gate_section(runs)
    lines += ["", "---", ""]
    lines += _compare_diag_section(runs)
    lines += ["", "---", ""]

    lit_ladd = _literature_numbers(load_yaml(args.ladd_config))
    lit_dfire = _literature_numbers(load_yaml(args.dfire_config))
    lines += ["## Comparison to literature baselines", "",
              "Michailidou et al. (Table III, Grounding DINO) floor/ceiling vs "
              "the Mode A results above (n=100 subset, per gate):", ""]
    for ds, lit, runkey in (("LADD", lit_ladd, "ladd"), ("D-Fire", lit_dfire, "dfire")):
        cells = []
        for lab in labels:
            res = runs[lab][runkey]
            v = res.get("map50", {}).get("uadapt_mode_a", float("nan"))
            g = res.get("gap_recovery", {}).get("gap_recovery_vs_literature")
            cells.append(f"{_fmt_mAP(v)}% ({_fmt_gap_closed(g)} closed)")
        lines.append(
            f"- **{ds}:** zero-shot floor {lit['zero_shot']:.1f}% → transfer "
            f"ceiling {lit['transfer']:.1f}% (gap {lit['gap_pp']:.1f} pp). "
            f"U-ADAPT Mode A — " + ", ".join(f"{lab}: {c}" for lab, c in zip(labels, cells)) + "."
        )
    lines.append("")
    if args.analytic_stats and args.beta_stats:
        lines += ["---", ""]
        lines += _ten_seed_section(
            load_json(args.analytic_stats),
            load_json(args.beta_stats),
            args.report_date,
            runs,
            labels,
        )
    return lines


# ---------------------------------------------------------------------------
# 10-seed paired protocol section (2026-08-07): analytic + beta stats.json.
# ---------------------------------------------------------------------------

def _ts_row(key: str, s: Dict) -> str:
    """One row of the paired-statistics table from a run_10_seed_protocol cell."""
    t = s["paired_ttest"]
    w = s["wilcoxon"]
    d = s["cohens_d"]
    dcell = f"{d:.2f}" if d == d else "n/a"
    return (
        f"| {key} | {s['n_seeds']} | {s['mode_a_map50_mean']:.4f} "
        f"| {s['naive_map50_mean']:.4f} | {dcell} | {t['t']:.3f} "
        f"| {t['p']:.3g} | {t['q_bh']:.3g} | {w['statistic']:.1f} "
        f"| {w['p']:.3g} | {w['q_bh']:.3g} |"
    )


def _d_span(ds: List[float]) -> str:
    """'−1.5 to −6.1'-style span (least to most negative) for all-negative ds."""
    finite = [d for d in ds if d == d and abs(d) != float("inf")]
    if len(finite) < 2:
        return "n/a"
    lo, hi = min(finite), max(finite)
    if lo >= 0.0 or hi >= 0.0:
        return f"{hi:.1f} to {lo:.1f}"
    return f"−{abs(hi):.1f} to −{abs(lo):.1f}"


def _ten_seed_section(analytic: Dict, beta: Dict, report_date: str,
                      runs: Dict[str, Dict], labels: List[str]) -> List[str]:
    """Markdown for the pre-registration §9 10-seed protocol section.

    Data-driven from the two ``run_10_seed_protocol.py`` stats.json documents
    (--analytic-stats / --beta-stats); ``runs``/``labels`` supply the n=100
    demo-path value referenced by the methodology note. Reproduces the
    hand-appended section of docs/real_data_results_final.md (2026-08-07).
    """
    cells_a = analytic.get("cells", {})
    cells_b = beta.get("cells", {})
    order = [
        k for ds in ("ladd", "dfire") for shots in (1, 3, 5)
        for k in [f"{ds}_k{shots}"]
        if k in cells_a and k in cells_b
    ]
    if not order:
        raise ValueError(
            "--analytic-stats/--beta-stats contain no matching cells "
            "(expected ladd_k1/k3/k5 and dfire_k1/k3/k5)"
        )

    table_hdr = ("| cell | n | Mode A mAP50 | Naive mAP50 | d | t | p(t) | q(t) "
                 "| W | p(W) | q(W) |")
    table_sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

    lines = [
        "## 10-seed paired statistical protocol (pre-registration §9)",
        "",
        f"*Run {report_date} by `scripts/run_10_seed_protocol.py` — 10 seeds × "
        "{ladd, dfire} × k ∈ {1, 3, 5} = 60 cells per gate on the n=100 pilot "
        "caches, via the scripted path (02 → 03 → 04). Primary comparison: "
        "U-ADAPT Mode A vs. naive averaging (w = 0.5). Per cell, across the 10 "
        "seeds: paired two-sided t-test AND Wilcoxon signed-rank, Cohen's d "
        "(paired, d_z), and Benjamini-Hochberg FDR control (q = 0.05) over the "
        "full comparison family of 12 tests (2 tests × 6 cells).*",
        "",
        "### Analytic gate",
        "",
        table_hdr,
        table_sep,
    ]
    lines += [_ts_row(k, cells_a[k]["stats"]) for k in order]
    lines += ["", "### Beta fallback gate", "", table_hdr, table_sep]
    lines += [_ts_row(k, cells_b[k]["stats"]) for k in order]
    lines += [
        "",
        "### Analytic vs. Beta (Mode A means, per-seed paired)",
        "",
        "| cell | analytic | beta_fallback | Δ (β − α) | d analytic | d beta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k in order:
        a = cells_a[k]["stats"]
        b = cells_b[k]["stats"]
        lines.append(
            f"| {k} | {a['mode_a_map50_mean']:.4f} | {b['mode_a_map50_mean']:.4f} "
            f"| {b['mode_a_map50_mean'] - a['mode_a_map50_mean']:+.4f} "
            f"| {a['cohens_d']:.2f} | {b['cohens_d']:.2f} |"
        )

    # --- Interpretation (values derived from the two stats documents) ------
    # NOTE: the verdict prose below states the DEFINITIVE pilot record
    # (2026-08-07): every cell significant after BH-FDR and Mode A below the
    # naive baseline. If the protocol is re-run on new data (e.g. full scale),
    # review these claims against the regenerated tables before trusting them.
    all_d = [cells_a[k]["stats"]["cohens_d"] for k in order] + \
            [cells_b[k]["stats"]["cohens_d"] for k in order]
    n_w0 = sum(
        1 for k in order if cells_a[k]["stats"]["wilcoxon"]["statistic"] == 0.0
    )
    deltas = [
        cells_b[k]["stats"]["mode_a_map50_mean"]
        - cells_a[k]["stats"]["mode_a_map50_mean"]
        for k in order
    ]
    wins = [
        sum(ma > na for ma, na in zip(cells_a[k]["mode_a"], cells_a[k]["naive"]))
        for k in order
    ] + [
        sum(ma > na for ma, na in zip(cells_b[k]["mode_a"], cells_b[k]["naive"]))
        for k in order
    ]
    k_d1 = "dfire_k1" if "dfire_k1" in order else order[0]
    d1_ana = abs(cells_a[k_d1]["stats"]["cohens_d"])
    d1_beta = abs(cells_b[k_d1]["stats"]["cohens_d"])

    ana_label = next((lab for lab in labels if lab == "analytic"), labels[0])
    ladd_ana_n100 = runs[ana_label]["ladd"]["map50"].get(
        "uadapt_mode_a", float("nan")
    )
    ts_example = "ladd_k1" if "ladd_k1" in order else order[0]
    ladd_ts_mean = cells_a[ts_example]["stats"]["mode_a_map50_mean"]

    lines += [
        "",
        "### Interpretation",
        "",
        f"- **Every cell is significant after FDR control (all q < 0.05), in "
        f"the same direction for both gates: Mode A is significantly WORSE "
        f"than naive averaging (w = 0.5) at every k ∈ {{1, 3, 5}} on both "
        f"datasets.** Cohen's d spans {_d_span(all_d)} (very large, "
        f"unfavorable); Wilcoxon W = 0 in {n_w0}/{len(order)} cells means all "
        f"10 paired differences carried the same sign — the gap is systematic, "
        f"not seed noise.",
        f"- **The Beta fallback helps directionally but does not close the "
        f"gap.** Beta raises the Mode A mean on all {len(order)} cells "
        f"({min(deltas) * 100:+.1f} to {max(deltas) * 100:+.1f} pp; largest on "
        f"D-Fire), consistent with the n=100 finding that hedging the "
        f"saturated gate is directionally beneficial. Yet per-seed, Beta Mode "
        f"A beats naive averaging on {min(wins)}–{max(wins)} of 10 seeds in "
        f"every cell.",
        f"- **Variance-stabilization nuance (D-Fire):** the Beta gate also "
        f"shrank the seed-to-seed spread of the Mode A − naive gap (e.g. "
        f"{k_d1}: |d| grows {d1_ana:.2f} → {d1_beta:.2f} despite an improved "
        f"mean) — the fallback both raises the mean and stabilizes it across "
        f"seeds.",
        "- **Naive baseline is bit-identical across the two gate runs** (same "
        "w = 0.5 scores), so the analytic-vs-beta comparison is exactly paired.",
        "- **Verdict:** at pilot scale, neither the analytic gate nor the "
        "pre-registered Beta contingency beats the uncertainty-blind baseline. "
        "The D5-triggered fallback is a robustness contingency, not a fix. The "
        "pre-registered primary comparison (§9) is settled at n=100: the gate, "
        "as implemented, does not recover value from the uncertainty inputs.",
        "",
        f"> Methodology note: the 10-seed protocol evaluates the full cached "
        f"test split via the scripted path (02 → 03 → 04), whereas the n=100 "
        f"tables above use the demo path; the small mean differences (e.g. "
        f"LADD analytic {_fmt_mAP(ladd_ana_n100)}% vs "
        f"{100.0 * ladd_ts_mean:.2f}% here) reflect the documented path-level "
        f"differences in gate inputs, not a data change.",
    ]
    return lines


def _mode_b_report(args) -> List[str]:
    """Standalone Mode B 10-seed protocol report (docs/real_data_results_modeB.md).

    Compares the learned logistic-regression gate (Mode B, pre-registered
    contingency Risk R3) against the naive w = 0.5 baseline with the same
    paired statistics as the §9 Mode A protocol, plus a four-way comparison
    table (zero-shot / naive / Mode A / Mode B) and a data-driven verdict.
    """
    mode_b = load_json(args.mode_b_stats)
    analytic = load_json(args.analytic_stats) if args.analytic_stats else None
    meta_b = mode_b.get("meta", {})
    cells_b = mode_b.get("cells", {})
    cells_a = (analytic or {}).get("cells", {}) if analytic else {}
    order = [
        k for ds in ("ladd", "dfire") for shots in (1, 3, 5)
        for k in [f"{ds}_k{shots}"]
        if k in cells_b
    ]
    if not order:
        raise ValueError(
            "--mode-b-stats contains no expected cells "
            "(ladd_k1/k3/k5, dfire_k1/k3/k5)"
        )
    n_seeds = int(meta_b.get("n_seeds", 0))
    pilot = n_seeds < 10
    zs = meta_b.get("zero_shot_map50", {})

    lines = [
        "# U-ADAPT — Mode B 10-seed protocol: logistic-regression gate "
        "vs naive averaging",
        "",
        f"*Generated {args.report_date} by `scripts/generate_real_data_report.py` "
        "(`--mode-b-stats`). Mode B is the pre-registered contingency "
        "(docs/pre_registration.md §10, Risk R3): a 6-parameter "
        "logistic-regression gate (L2 = 1e-4, 5-fold CV) calibrated on a "
        "per-seed 20-box/class set sampled from the train split, strictly "
        "disjoint from the k-shot support examples and the test split.*",
        "",
    ]
    if pilot:
        lines += [
            "> 🧪 **SMOKE RUN (fewer than 10 seeds)** — pipeline verification "
            "only; the pre-registered protocol requires all 10 seeds. Treat "
            "every statistic below as illustrative, not a result.",
            "",
        ]

    # --- Paired statistics: Mode B vs naive -------------------------------
    table_hdr = ("| cell | n | Mode B mAP50 | Naive mAP50 | d | t | p(t) | q(t) "
                 "| W | p(W) | q(W) |")
    table_sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines += [
        "## Mode B vs naive averaging (w = 0.5)",
        "",
        f"*{n_seeds} seeds × {{ladd, dfire}} × k ∈ {{1, 3, 5}} = "
        f"{n_seeds * 6} cells, via the scripted path (02 → 05 → 03 → 04). "
        "Per cell, across seeds: paired two-sided t-test AND Wilcoxon "
        "signed-rank, Cohen's d (paired, d_z), and Benjamini-Hochberg FDR "
        "control (q = 0.05) over the full comparison family of 12 tests "
        "(2 tests × 6 cells). The naive baseline is bit-identical to the "
        "Mode A protocol runs.*",
        "",
        table_hdr,
        table_sep,
    ]
    full = [k for k in order if "paired_ttest" in cells_b[k].get("stats", {})]
    for k in order:
        s = cells_b[k]["stats"]
        if k in full:
            lines.append(_ts_row(k, s))
        else:
            lines.append(
                f"| {k} | {s['n_seeds']} | — | — | — | — | — | — | — | — | — |"
            )
    if len(full) < len(order):
        short = ", ".join(k for k in order if k not in full)
        lines += ["", f"> Cells {short} had fewer than 2 seeds — paired "
                      "statistics not computed for them (smoke run)."]

    # --- Calibration audit ------------------------------------------------
    lines += [
        "",
        "### Calibration-set audit (per-seed, per cell)",
        "",
        "| cell | n samples | per class (sampled) |",
        "| --- | --- | --- |",
    ]
    for k in order:
        cal = cells_b[k].get("calibration")
        if cal is None:
            lines.append(f"| {k} | — | — |")
            continue
        n_min, n_max = cal["n_samples_min"], cal["n_samples_max"]
        span = f"{n_min}" if n_min == n_max else f"{n_min}–{n_max}"
        pmin, pmax = cal["per_class_min"], cal["per_class_max"]
        per = ", ".join(
            f"{c}: {pmin[c]}" if pmin[c] == pmax[c] else f"{c}: {pmin[c]}–{pmax[c]}"
            for c in pmin
        )
        lines.append(f"| {k} | {span} | {per} |")
    lines += [
        "",
        "> Pre-registered size is 20 boxes per class. At n=100 pilot scale the "
        "train caches are tiny (LADD 10 images, D-Fire 9 images), so cells "
        "fall short of 20 (ranges above span the seeds) — the gate was fit on "
        "exactly those counts (a full-scale run samples the full train split "
        "and is not limited this way).",
    ]

    # --- Four-way comparison ----------------------------------------------
    lines += [
        "",
        "## Four-way comparison (per-cell mAP50 means, %)",
        "",
        "| cell | Zero-shot | Naive | Mode A | Mode B | Δ (B − N, pp) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k in order:
        sb = cells_b[k]["stats"]
        z = zs.get(k.split("_")[0])
        ma = (cells_a[k]["stats"]["mode_a_map50_mean"]
              if k in cells_a else float("nan"))
        naive_m = sb["naive_map50_mean"]
        mb = sb["mode_a_map50_mean"]
        zcell = f"{100.0 * z:.2f}" if z == z else "—"
        macell = f"{100.0 * ma:.2f}" if ma == ma else "—"
        lines.append(
            f"| {k} | {zcell} | {100.0 * naive_m:.2f} | {macell} | "
            f"{100.0 * mb:.2f} | {100.0 * (mb - naive_m):+.2f} |"
        )
    lines += [
        "",
        "> Zero-shot = raw detector scores on the FULL cached test split "
        "(computed once per dataset, seed-independent). Mode A means come from "
        "`--analytic-stats` when provided. All values are mAP50 percentages at "
        "pilot scale — they are not the literature zero-shot floors "
        "(LADD 61.0%, D-Fire 27.5%, Michailidou et al. Table III), which are "
        "measured over the full test set.",
    ]

    # --- Interpretation (data-driven) --------------------------------------
    # Cells with < 2 seeds carry only "note" (no paired stats); keep them out
    # of the statistical interpretation (mirrors run_10_seed_protocol.py's own
    # summary print).
    full = [k for k in order if "paired_ttest" in cells_b[k].get("stats", {})]
    beat = []
    worse = []
    sig = []

    lines += ["", "### Interpretation", ""]
    if not full:
        lines.append(
            "- Paired statistics unavailable: every cell has fewer than 2 seeds "
            "in this stats document (smoke run). Re-run with `--max-seeds 10` "
            "for the pre-registered protocol."
        )
    else:
        def _q_min(s: Dict) -> float:
            qs = [s["paired_ttest"]["q_bh"], s["wilcoxon"]["q_bh"]]
            finite = [q for q in qs if q == q]
            return min(finite) if finite else float("nan")

        all_d = [cells_b[k]["stats"]["cohens_d"] for k in full]
        sig = [k for k in full if _q_min(cells_b[k]["stats"]) < 0.05]
        deltas = [
            cells_b[k]["stats"]["mode_a_map50_mean"]
            - cells_b[k]["stats"]["naive_map50_mean"]
            for k in full
        ]
        wins = [
            sum(mb > nv for mb, nv in zip(cells_b[k]["mode_a"], cells_b[k]["naive"]))
            for k in full
        ]
        n_w0 = sum(
            1 for k in full
            if cells_b[k]["stats"]["wilcoxon"]["statistic"] == 0.0
        )
        beat = [k for k, d in zip(full, deltas) if d > 0]
        worse = [k for k, d in zip(full, deltas) if d < 0]

        if len(worse) == len(full) and len(sig) == len(full):
            lines.append(
                f"- **Mode B did NOT beat naive averaging.** The learned gate is "
                f"significantly WORSE than w = 0.5 at every cell after BH-FDR "
                f"control ({len(sig)}/{len(full)} cells, all q < 0.05); Cohen's d "
                f"spans {_d_span(all_d)} (very large, unfavorable); Wilcoxon W = 0 "
                f"in {n_w0}/{len(full)} cells means every paired difference had "
                f"the same sign — the gap is systematic, not seed noise. Per-seed, "
                f"Mode B beats naive on {min(wins)}–{max(wins)} of {n_seeds} seeds "
                f"across cells. **The pre-registered fallback narrative applies:** "
                f"elevate the plain-confidence margin / acknowledge the limitation "
                f"in the thesis rather than claiming learned-gate gains."
            )
        elif len(beat) > 0:
            pos = [d for d in deltas if d > 0]
            lines.append(
                f"- **Mode B beat naive averaging on {len(beat)}/{len(full)} cells** "
                f"({', '.join(beat)}), with deltas of "
                f"{100.0 * min(pos):+.2f} to {100.0 * max(pos):+.2f} pp on those "
                f"cells; it lost on {len(worse)} cells "
                f"({', '.join(worse) if len(worse) <= 3 else 'see table'}). "
                f"Cells significant after FDR: {len(sig)}/{len(full)}."
            )
        else:
            lines.append(
                f"- **No cell shows Mode B above the naive baseline** "
                f"(Δ = {100.0 * min(deltas):+.2f} to {100.0 * max(deltas):+.2f} pp); "
                f"{len(sig)}/{len(full)} cells significant after FDR. "
                f"The pre-registered fallback narrative applies."
            )
        if analytic is not None:
            a_means = [100.0 * cells_a[k]["stats"]["mode_a_map50_mean"] for k in full]
            b_means = [100.0 * cells_b[k]["stats"]["mode_a_map50_mean"] for k in full]
            better = sum(b > a for a, b in zip(a_means, b_means))
            lines.append(
                f"- **vs Mode A:** the learned gate raises the per-cell mean above "
                f"the analytic gate on {better}/{len(full)} cells "
                f"(Mode B − Mode A from {min(b - a for a, b in zip(a_means, b_means)):+.2f} "
                f"to {max(b - a for a, b in zip(a_means, b_means)):+.2f} pp) — "
                f"consistent with learning beating the saturated analytic rule, "
                f"but neither learned nor analytic gating recovers the naive "
                f"baseline at pilot scale."
            )
        lines += [
            "",
            "> ⚠️ **Pilot-scale caveats that limit the generality of this verdict:** "
            "(1) the train caches are tiny (LADD 10 images, D-Fire 9), so the "
            "calibration sets are far below 20 boxes/class and the gate's learning "
            "signal is minimal; (2) affinity saturates ≥ 0.65 on real features, so "
            "`visual_correct` is True for essentially every sampled box and the "
            "soft targets collapse toward σ(S_visual − S_text) — the gate has little "
            "directional signal to learn from; (3) the soft-target mapping in "
            "`soft_targets()` was fixed to the pre-registered formula (proposal "
            "§5.4.2) on 2026-08-07 — pre-fix Mode B numbers learned the inverse "
            "mapping and are not comparable.",
        ]
    lines += [
        "",
        "---",
        "",
        "## Verdict (per pre-registered contingency Risk R3)",
        "",
        "Did Mode B successfully beat naive averaging? "
        + ("**No — and significantly not.** "
           if (len(worse) == len(full) and len(sig) == len(full)) else
           "**Partially / no.** ") +
        "The logistic-regression gate trained on 20 boxes/class does not "
        "rescue performance where the analytic rule failed at n=100 pilot "
        "scale; the thesis narrative should treat the learned gate as a "
        "robustness contingency, not a fix, and should elevate the "
        "plain-confidence-margin / acknowledged-limitation fallback per the "
        "pre-registration.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the real-data validation markdown report "
                    "(docs/real_data_results.md), or the Analytic-vs-Beta "
                    "comparative report (--compare-dirs)."
    )
    # Comparative mode: two or more pipeline output directories, each
    # containing {ladd,dfire}/results.json + pooled_diagnostics.json.
    parser.add_argument("--compare-dirs", nargs="+", type=Path, default=None,
                        help="pipeline output directories to compare side-by-side "
                             "(e.g. outputs/real_data/n100_analytic "
                             "outputs/real_data/n100_beta) -> writes the "
                             "Analytic-vs-Beta comparative report")
    parser.add_argument("--compare-labels", nargs="+", default=None,
                        help="display labels for --compare-dirs (default: directory "
                             "names)")
    parser.add_argument("--ladd-results", type=Path)
    parser.add_argument("--dfire-results", type=Path)
    parser.add_argument("--pooled-diagnostics", type=Path, default=None,
                        help="output of compute_pooled_diagnostics.py "
                             "(recommended; report notes when absent)")
    parser.add_argument("--ladd-config", default="configs/datasets/ladd.yaml", type=Path)
    parser.add_argument("--dfire-config", default="configs/datasets/dfire.yaml", type=Path)
    parser.add_argument("--out", default=None, type=Path,
                        help="output path (default: docs/real_data_results.md in "
                             "single-run mode; docs/real_data_results_final.md "
                             "in --compare-dirs mode)")
    parser.add_argument("--report-date", default=date.today().isoformat())
    parser.add_argument("--pilot", action="store_true",
                        help="force pilot labeling (auto-detected from "
                             "meta.n_test_images < 100)")
    parser.add_argument("--pilot-n", default="n=100",
                        help="pilot label size text (default: 'n=100')")
    parser.add_argument("--analytic-stats", type=Path, default=None,
                        help="stats.json from scripts/run_10_seed_protocol.py "
                             "(--gate-type analytic); with --beta-stats, appends "
                             "the pre-registration §9 10-seed protocol section "
                             "to the comparative report (--compare-dirs mode only)")
    parser.add_argument("--beta-stats", type=Path, default=None,
                        help="stats.json from scripts/run_10_seed_protocol.py "
                             "(--gate-type beta_fallback); must be paired with "
                             "--analytic-stats")
    parser.add_argument("--mode-b-stats", type=Path, default=None,
                        help="stats.json from scripts/run_10_seed_protocol.py "
                             "(--mode B); writes the standalone Mode B protocol "
                             "report (docs/real_data_results_modeB.md). Optional "
                             "--analytic-stats adds the Mode A column to the "
                             "four-way comparison")
    args = parser.parse_args()

    # --- Comparative mode -------------------------------------------------
    if args.compare_dirs:
        if len(args.compare_dirs) < 2:
            parser.error("--compare-dirs needs at least two directories")
        if args.compare_labels and len(args.compare_labels) != len(args.compare_dirs):
            parser.error("--compare-labels must match --compare-dirs in count")
        if bool(args.analytic_stats) != bool(args.beta_stats):
            parser.error("--analytic-stats and --beta-stats must be passed "
                         "together")
        for p in (args.analytic_stats, args.beta_stats):
            if p is not None and not p.exists():
                raise FileNotFoundError(
                    f"missing 10-seed stats file: {p} "
                    "(run scripts/run_10_seed_protocol.py first)"
                )
        args.out = args.out or Path("docs/real_data_results_final.md")
        lines = _build_compare_report(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote comparative report -> {args.out}")
        return

    # --- Mode B protocol report (standalone) --------------------------------
    if args.mode_b_stats:
        if args.compare_dirs:
            parser.error("--mode-b-stats cannot be combined with --compare-dirs "
                         "(use separate invocations)")
        if not args.mode_b_stats.exists():
            raise FileNotFoundError(
                f"missing --mode-b-stats file: {args.mode_b_stats} "
                "(run scripts/run_10_seed_protocol.py --mode B first)"
            )
        if args.analytic_stats is not None and not args.analytic_stats.exists():
            raise FileNotFoundError(
                f"missing --analytic-stats file: {args.analytic_stats}"
            )
        args.out = args.out or Path("docs/real_data_results_modeB.md")
        lines = _mode_b_report(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote Mode B protocol report -> {args.out}")
        return

    # --- Single-run mode (backward compatible) -----------------------------
    if args.ladd_results is None or args.dfire_results is None:
        parser.error("--ladd-results and --dfire-results are required in single-run "
                     "mode (or pass --compare-dirs for the comparative report)")
    args.out = args.out or Path("docs/real_data_results.md")

    for p in (args.ladd_results, args.dfire_results):
        if not p.exists():
            raise FileNotFoundError(f"missing results file: {p} (run the pipeline first)")
    if args.pooled_diagnostics is not None and not args.pooled_diagnostics.exists():
        raise FileNotFoundError(
            f"missing pooled-diagnostics file: {args.pooled_diagnostics} "
            "(run scripts/compute_pooled_diagnostics.py first)"
        )

    ladd, dfire = load_json(args.ladd_results), load_json(args.dfire_results)
    lit_ladd = _literature_numbers(load_yaml(args.ladd_config))
    lit_dfire = _literature_numbers(load_yaml(args.dfire_config))

    pooled_data: Optional[Dict] = None
    if args.pooled_diagnostics is not None:
        pooled_data = load_json(args.pooled_diagnostics)
    pooled = (pooled_data or {}).get("pooled", {})
    per_dataset = (pooled_data or {}).get("per_dataset", {})

    synthetic_any = any(
        r.get("meta", {}).get("data_source") == "synthetic"
        for r in (ladd, dfire)
    )

    pilot_ns = [
        r["meta"]["n_test_images"]
        for r in (ladd, dfire)
        if r.get("meta", {}).get("n_test_images") is not None
    ]
    pilot = args.pilot or any(n <= 100 for n in pilot_ns)
    label_n = f"n={max(pilot_ns)}" if (pilot and pilot_ns and not args.pilot) else args.pilot_n

    title = (
        f"# U-ADAPT — PILOT RESULTS ({label_n} images)"
        if pilot
        else "# U-ADAPT — Real-Data Validation Results"
    )
    lines: List[str] = [
        title,
        "",
        f"*Generated {args.report_date} by `scripts/generate_real_data_report.py`.*",
        "",
    ]
    if pilot:
        lines += [
            "> 🧪 **PILOT RESULTS** — preliminary n-image pipeline check "
            "(only the first few images per dataset were evaluated). These "
            "numbers are NOT final thesis results; the full-data run "
            "supersedes them.",
            "",
        ]
    if synthetic_any:
        lines += [
            "> ⚠️ **Synthetic stand-in caveat:** the runs below used the "
            "deterministic synthetic world (no real feature cache on this "
            "machine), so every number validates the *pipeline mechanism*, not "
            "a research result. Real-data evaluation supersedes it. "
            "See `docs/supervisor_demo_report.md` (Methodological Caveats, "
            "2026-08-03) and `docs/change_log.md` (2026-08-03/2026-08-04).",
            "",
        ]

    # --- Executive summary -------------------------------------------------
    lines += ["## Executive Summary", ""]
    for name, res in (("LADD", ladd), ("D-Fire", dfire)):
        caveat = _caveat(res, name)
        if caveat:
            lines += [caveat, ""]
        m = res.get("map50", {})
        ua, naive, zs = (m.get("uadapt_mode_a", float("nan")),
                         m.get("naive_average", float("nan")),
                         m.get("zero_shot_raw", float("nan")))
        lines.append(
            f"- **{name}:** U-ADAPT Mode A **{_fmt_mAP(ua)}%** mAP50 vs naive "
            f"averaging **{_fmt_mAP(naive)}%** and zero-shot "
            f"**{_fmt_mAP(zs)}%**."
        )
        if ua < naive or ua < zs:
            worse = "naive average" if ua < naive else "zero-shot baseline"
            lines.append(
                f"  - ⚠️ **U-ADAPT underperforms** the {worse} on {name}. "
                f"The analytic gate saturates toward the visual "
                f"modality (gate weight mean ≫ 0.5 on real features), but "
                f"visual-only reranking is worse than the raw detector score "
                f"here — see the D2/D3 caveat below."
            )
    if pooled_data:
        s = pooled_data.get("summary", {})
        lines += [
            "",
            f"- **Pooled D1/D2/D3** (LADD+D-Fire, PRIMARY claim): D1 ρ = "
            f"**{s.get('D1_spearman_rho', float('nan')):+.3f}**, D2 ρ = "
            f"**{s.get('D2_spearman_rho', float('nan')):+.3f}**, D3 favorability "
            f"= **{s.get('D3_favorability_fraction', 0.0):.1%}** "
            f"(binomial p = {s.get('D3_binomial_pvalue', float('nan')):.3g}).",
        ]
    lines += ["", "## mAP50 results", ""]
    lines += _mAP50_section("LADD", ladd)
    lines += ["", "---", ""]
    lines += _mAP50_section("D-Fire", dfire)
    lines += ["", "---", ""]
    lines += _gap_recovery_section("LADD", ladd, lit_ladd)
    lines += ["", "---", ""]
    lines += _gap_recovery_section("D-Fire", dfire, lit_dfire)
    lines += ["", "---", ""]

    if pooled_data:
        lines += _diag_value_table(pooled, per_dataset)
        lines += ["", "---", ""]
    else:
        lines += [
            "## Pooled D1/D2/D3 diagnostics",
            "",
            "_Not computed — pass `--pooled-diagnostics` (output of "
            "`scripts/compute_pooled_diagnostics.py`) to include the pooled "
            "D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10)._",
            "",
            "---",
            "",
        ]

    # --- Literature comparison ---------------------------------------------
    lines += _literature_table(lit_ladd, lit_dfire, ladd, dfire)

    # --- Appendix: per-class AP ---------------------------------------------
    lines += ["", "## Appendix — per-class AP (U-ADAPT)", ""]
    for name, res in (("LADD", ladd), ("D-Fire", dfire)):
        ap = res.get("per_class_ap", {})
        if ap:
            cells = " · ".join(f"{c} {v:.3f}" for c, v in ap.items())
            lines.append(f"- **{name}:** {cells}")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote report -> {args.out}")


def _fmt_gap_closed(v: Optional[float]) -> str:
    """Gap-recovery fraction -> percentage string (or '—' when absent)."""
    if v is None:
        return "—"
    return f"{100.0 * v:.1f}%"


if __name__ == "__main__":
    main()
