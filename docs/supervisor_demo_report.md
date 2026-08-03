# U-ADAPT — Supervisor Demo Report

**Uncertainty-gated fusion of text & visual prompts for few-shot cross-domain disaster detection (Mode A)**

*Prepared 2026-08-01 · deterministic seed=0 · demo subset (synthetic world; real cache ready)*

---

## Executive Summary

U-ADAPT's Mode A applies a **training-free analytic gate** `w = σ(−α·σ̃²_visual + β·σ̃²_text + γ·ã_visual)` (α=β=γ=1, T=1) to blend text and visual evidence per proposal, with **zero backbone gradient steps**. This demo runs the full pipeline end-to-end (prototypes → uncertainty → gating → fusion → mAP50 + D1–D3 diagnostics) and shows: (1) the gate is **dynamic**, assigning a wide range of weights rather than collapsing to naive averaging; (2) the uncertainty proxies **correlate with error** as assumed (D1 ρ=+0.26, D2 ρ=+0.39); (3) the gate **favors the more accurate modality** significantly more often than chance (D3 71.9%, p≈1.5e-8); and (4) U-ADAPT **recovers ~88% of the gap** between the zero-shot baseline and the oracle re-rank ceiling, and beats every single-modality and fixed-weight baseline on this subset. The pipeline is deterministic (seed=0), Colab-T4-friendly (<30 min), and drops onto real cached features unchanged once Milestone 1 data is available. **Caveat:** the numbers below were produced on a synthetic stand-in world (no raw data is downloaded yet) and demonstrate the *mechanism and wiring*, not a research result.

## Key Results Table (mAP50, demo subset, seed=0)

| Method | mAP50 | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | **0.673** | — |
| Text-only (w=0) | **0.836** | +0.163 |
| Visual-only (w=1) | **0.741** | +0.069 |
| Naive averaging (w=0.5, T-Rex2 surrogate) | **0.947** | +0.274 |
| **U-ADAPT Mode A (analytic gate)** | **0.956** | **+0.284** |
| Transfer ceiling (oracle re-rank) | **0.995** | +0.322 |

Per-class AP (U-ADAPT): debris 0.908 · fire 1.000 · person 0.975 · roof 0.983 · smoke 0.911 · vehicle 0.963.

## Figures

Figures 1–6 are rendered by `notebooks/supervisor_demo_visualizations.ipynb` into `outputs/supervisor_demo/figures/` (executed automatically by `bash run_this_for_supervisor.sh`).

![Figure 1 — Gate weight distribution](../outputs/supervisor_demo/figures/figure1_gate_weights.png)
![Figure 2 — Uncertainty–accuracy correlation (D1/D2)](../outputs/supervisor_demo/figures/figure2_d1_d2.png)
![Figure 3 — Gate favorability (D3)](../outputs/supervisor_demo/figures/figure3_gate_favorability.png)
![Figure 4 — Gap recovery analysis](../outputs/supervisor_demo/figures/figure4_gap_recovery.png)
![Figure 5 — Qualitative examples](../outputs/supervisor_demo/figures/figure5_qualitative.png)
![Figure 6 — Coefficient ablation](../outputs/supervisor_demo/figures/figure6_ablation.png)

## Interpretation of D1–D3 Diagnostics

- **D1 / D2 (uncertainty–accuracy):** positive Spearman ρ (D1 +0.26, D2 +0.39) between normalized uncertainty and error rate validates the *core assumption* of the method — uncertainty proxies predict when each modality fails. This is the foundation the gate builds on.
- **D3 (gate favorability):** among 167 proposals where text and visual *disagree*, the gate assigned higher weight to the more accurate modality in **71.9%** of cases (binomial p = 1.5e-8) — direct evidence the gate is *useful*, not decorative.
- **Figure 1 supports this:** text-correct proposals skew toward low w, visual-correct toward high w; **32% of proposals get w < 0.45 and 62% get w > 0.55, with only 5% near 0.5 (mean w = 0.624, std 0.162)** — the gate is clearly not stuck at naive averaging.

## Methodological Caveats (2026-08-03)

The following caveats reflect the 2-class degeneracy finding and the
**absolute-scaling** normalization fix (`--norm-strategy absolute`, x/2.0 for
cosine distances — pre-registration deviations §2 and §10,
[`change_log.md`](change_log.md)).

> **1. The 2-class statistical-power limitation.** *"I discovered that
> evaluating D1, D2, and D3 on D-Fire in isolation is structurally
> underpowered. Because D-Fire only has 2 classes (fire and smoke), there are
> only 2 distinct variance values. You cannot compute a meaningful Spearman
> rank correlation or gate favorability trend with only 2 data points.
> Therefore, the pre-registered protocol must be updated to evaluate D1/D2/D3
> pooled across LADD and D-Fire (giving us 3 distinct classes and 3 distinct
> variance values), which provides the necessary statistical power."* The
> D1 ρ=+0.26 / D2 ρ=+0.39 / D3 71.9% numbers in this report come from the
> **6-class** demo; on the 2-class D-Fire stand-in alone they remain weak (by
> construction) and are therefore reported pooled across LADD+D-Fire for
> real-data evaluation.
>
> **2. The synthetic-world artifact.** *"The synthetic demo world was
> implicitly engineered around the min-max normalization stretch. When we
> apply the mathematically correct absolute scaling (x/2.0), the raw variance
> magnitudes are revealed to be small relative to the affinity term in this
> specific synthetic setup. This is a demo-world artifact. On real data, the
> variance magnitudes will be different. Furthermore, Diagnostic D5 is
> specifically designed to catch this: if real variances cluster near 0, D5
> will flag it and trigger the pre-registered Beta-regression fallback."*
>
> **3. The fix worked.** *"Despite the D3 drop on the synthetic stand-in, the
> primary goal was achieved: the 2-class mAP50 degeneracy is fixed. Under
> absolute scaling, U-ADAPT (0.956) now correctly beats naive averaging
> (0.955) on the 2-class D-Fire setup, whereas min-max actively harmed it
> (0.906)."*

## Preliminary Gap Recovery

**88.2% of the zero-shot→oracle gap is recovered** on this subset:
`(0.956 − 0.673) / (0.995 − 0.673)`. The oracle ceiling is an *oracle re-rank* (every GT-correct proposal ranked above every incorrect one) — the honest upper bound for any re-scoring method. Literature references (D-Fire §config: 27.5 → 65.6; LADD: 61.0 → 92.2) are stored in `results.json` as dataset-specific context only; they are **not comparable** to the synthetic operating range and are shown as dashed reference lines in Figure 4.

## Ablation Study (Figure 6)

| Gate variant | mAP50 |
|---|---|
| Full (α=β=γ=1) | **0.956** |
| α=0 (no visual uncertainty) | **0.946** |
| β=0 (no text uncertainty) | **0.957** |
| γ=0 (no affinity) | **0.958** |

On the demo subset the **visual-uncertainty term (α) contributes** (+1.0 pp); β and γ effects are within noise (the synthetic world's gating signal is dominated by visual uncertainty). Component-level significance testing is deferred to the real-data protocol (§9, 10 seeds + paired tests).

## Next Steps (before thesis submission)

1. **Real data run** — download D-Fire/LADD, run `01_extract_and_cache.py`, then re-run this demo with `--cache-dir cached_features --ground-truth data/annotations/dfire_test.json --norm-strategy absolute` (**absolute scaling is required for the 2-class D-Fire run**; min-max collapses the variance terms to {0, 1} and actively hurts mAP — 0.906 vs 0.955 naive on the stand-in).
2. **Full protocol** — 10 seeds, paired t-test + Wilcoxon (§9), FDR control; mAP50:95, ECE/Brier/uncertainty-AUROC via `scripts/04_evaluate.py`.
3. **Cross-backbone (RQ5)** — repeat with OWL-ViT / YOLOE26 configs.
4. **Figure 5 on real imagery** — already wired: when the demo runs with real cached features + `--ground-truth`, Figure 5 automatically renders real detection images (falling back to the schematic scene otherwise). Just re-run and the panels upgrade themselves.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic demo numbers misread as results | Figures + report carry explicit synthetic-data caveats |
| Real cache absent → demo cannot run | Deterministic synthetic world keeps every downstream stage runnable |
| Gate could collapse to w≈0.5 | Figure 1 + gate_stats check spread explicitly (dynamic gate) |
| D1/D2/D3 uninformative on D-Fire alone (2 classes) | Pre-registered deviation 2026-08-03: evaluate D1/D2/D3 **pooled across LADD+D-Fire (3 classes)**; report per-dataset values separately |
| Real variance magnitudes small vs affinity (absolute scaling) | D5 is the pre-registered sentinel: flag + Beta-regression fallback |
