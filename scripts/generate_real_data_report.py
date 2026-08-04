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

**Pilot labeling:** when any run reports ``meta.n_test_images < 100`` (or
``--pilot`` is passed), the title reads **"PILOT RESULTS (n=<N> images)"**
and a banner warns that the numbers are a preliminary pipeline check, not
final thesis results — so a n=10 pilot report can never be mistaken for the
final thesis report.

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
        over = " (>100% means the adapter exceeds the literature transfer "\
               "ceiling — expected only on the synthetic stand-in)" \
            if vs_lit > 1.0 else ""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the real-data validation markdown report "
                    "(docs/real_data_results.md)."
    )
    parser.add_argument("--ladd-results", required=True, type=Path)
    parser.add_argument("--dfire-results", required=True, type=Path)
    parser.add_argument("--pooled-diagnostics", type=Path, default=None,
                        help="output of compute_pooled_diagnostics.py "
                             "(recommended; report notes when absent)")
    parser.add_argument("--ladd-config", default="configs/datasets/ladd.yaml", type=Path)
    parser.add_argument("--dfire-config", default="configs/datasets/dfire.yaml", type=Path)
    parser.add_argument("--out", default="docs/real_data_results.md", type=Path)
    parser.add_argument("--report-date", default=date.today().isoformat())
    parser.add_argument("--pilot", action="store_true",
                        help="force pilot labeling (auto-detected from "
                             "meta.n_test_images < 100)")
    parser.add_argument("--pilot-n", default="n=10",
                        help="pilot label size text (default: 'n=10')")
    args = parser.parse_args()

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
    pilot = args.pilot or any(n < 100 for n in pilot_ns)
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
        fh.write("\n".join(lines))
    print(f"wrote report -> {args.out}")


def _fmt_gap_closed(v: Optional[float]) -> str:
    """Gap-recovery fraction -> percentage string (or '—' when absent)."""
    if v is None:
        return "—"
    return f"{100.0 * v:.1f}%"


if __name__ == "__main__":
    main()
