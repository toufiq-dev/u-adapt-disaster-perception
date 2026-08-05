# U-ADAPT — PILOT RESULTS (n=100 images)

*Generated 2026-08-06 by `scripts/generate_real_data_report.py`.*

> 🧪 **PILOT RESULTS** — preliminary n-image pipeline check (only the first few images per dataset were evaluated). These numbers are NOT final thesis results; the full-data run supersedes them.

## Executive Summary

- **LADD:** U-ADAPT Mode A **78.3%** mAP50 vs naive averaging **80.4%** and zero-shot **81.3%**.
  - ⚠️ **U-ADAPT underperforms** the naive average on LADD. The analytic gate saturates toward the visual modality (gate weight mean ≫ 0.5 on real features), but visual-only reranking is worse than the raw detector score here — see the D2/D3 caveat below.
- **D-Fire:** U-ADAPT Mode A **66.9%** mAP50 vs naive averaging **71.8%** and zero-shot **73.4%**.
  - ⚠️ **U-ADAPT underperforms** the naive average on D-Fire. The analytic gate saturates toward the visual modality (gate weight mean ≫ 0.5 on real features), but visual-only reranking is worse than the raw detector score here — see the D2/D3 caveat below.

- **Pooled D1/D2/D3** (LADD+D-Fire, PRIMARY claim): D1 ρ = **-0.056**, D2 ρ = **+0.051**, D3 favorability = **100.0%** (binomial p = 1.77e-74).

## mAP50 results

### LADD — mAP50 (subset)

| Method | mAP50 (%) | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | 81.3 | +0.0 pp |
| Text-only (w=0) | 81.3 | +0.0 pp |
| Visual-only (w=1) | 66.9 | -14.4 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 80.4 | -0.9 pp |
| **U-ADAPT Mode A (analytic gate)** | **78.3** | -3.0 pp |

---

### D-Fire — mAP50 (subset)

| Method | mAP50 (%) | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | 73.4 | +0.0 pp |
| Text-only (w=0) | 73.4 | +0.0 pp |
| Visual-only (w=1) | 54.7 | -18.7 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 71.8 | -1.6 pp |
| **U-ADAPT Mode A (analytic gate)** | **66.9** | -6.5 pp |

---

### LADD — gap recovery

Michailidou et al. (Grounding DINO, Table III) floor/ceiling: zero-shot **61.0%** → transfer **92.2%** (gap **31.2 pp**).

- **Fraction of the zero-shot→transfer gap closed:** 55.4% (U-ADAPT 78.3% vs zero-shot 81.3%; transfer ceiling 92.2%).
- Gap recovered vs the oracle re-rank ceiling (99.8%): **-16.3%** (also vs proposal-recall ceiling 11.9%).

---

### D-Fire — gap recovery

Michailidou et al. (Grounding DINO, Table III) floor/ceiling: zero-shot **27.5%** → transfer **65.6%** (gap **38.1 pp**).

- **Fraction of the zero-shot→transfer gap closed:** 103.4% (>100% means the adapter exceeds the literature transfer ceiling — on real data this reflects the tiny pilot subset: the literature ceiling is measured over the full test set, while this run scores only 100 selected images) (U-ADAPT 66.9% vs zero-shot 73.4%; transfer ceiling 65.6%).
- Gap recovered vs the oracle re-rank ceiling (81.0%): **-86.3%** (also vs proposal-recall ceiling 51.7%).

---

### Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10)

D-Fire alone has 2 classes → only 2 distinct variance values, so D1/D2/D3 on it alone are structurally underpowered. Per the pre-registered deviation, the primary diagnostic claim is computed **pooled across LADD + D-Fire** (3 distinct classes: pedestrian, fire, smoke).

| Diagnostic | Pooled value | n |
|---|---|---|
| D1 — text uncertainty–accuracy | Spearman ρ = **-0.056** (rho <= 0: proxy may be uninformative) | 657 |
| D2 — visual uncertainty–accuracy | Spearman ρ = **+0.051** (ok) | 657 |
| D3 — gate favorability | favorability = **100.0%** (binomial p = 1.77e-74) | 246 |

> ⚠️ **D1 pooled-sign caveat (2026-08-05):** the pooled D1 Spearman ρ mixes LADD and D-Fire at different error base rates with structurally different text-entropy ranges (LADD is single-class → entropy 0.0 by construction; D-Fire sits near max entropy). The pooled sign is dominated by this **between-dataset base-rate difference**, not by a within-proposal uncertainty–accuracy relationship (per-dataset D1 is the interpretable signal). See `docs/change_log.md` 2026-08-05.

> ⚠️ **D2 pooled-sign caveat (2026-08-05, confirmed at n=100):** the pooled D2 ρ is likewise driven by the **between-dataset normalization scale difference** (LADD min-max spreads visual variance across [0,1]; D-Fire absolute keeps it tiny). Within each dataset the visual uncertainty–accuracy relationship is ≈ 0 (LADD ρ = +0.077, D-Fire ρ = −0.022 at n=100), so the pooled positive sign is a scale artifact, not evidence of a within-proposal effect.

> ⚠️ **D3 one-sidedness caveat (2026-08-05, confirmed at n=100):** the modality-accuracy disagreeing subsets are almost entirely *visual-better* (LADD 131/132, D-Fire 114/114 — 1 text-better case total). The affinity threshold (≥ 0.65) never fails on real features (affinity ∈ [0.64, 0.999]), so `visual_correct` saturates and the gate always leans visual — which the mAP50 table shows is the *weaker* reranker on this data. A 100% D3 is therefore a saturation artifact, not evidence the gate is well-calibrated. Diagnosing D3 properly requires proposals with lower affinity (full-scale run) or a re-thresholded visual-correctness rule.

Per-dataset values (reported; pooled is primary):

| Dataset | D1 ρ | D2 ρ | D3 favorability |
|---|---|---|---|
| LADD | +0.000 | +0.077 | 100.0% |
| D-Fire | -0.061 | -0.022 | 100.0% |

---

## Comparison to literature baselines

Michailidou et al. (preprint, Table III) with Grounding DINO; U-ADAPT numbers are the Mode A results above (synthetic stand-in unless real data was used — see caveats).

| Dataset | Zero-shot mAP50 | Transfer mAP50 (ceiling) | Gap (pp) | U-ADAPT mAP50 | Gap closed |
|---|---|---|---|---|---|
| LADD | 61.0 | 92.2 | 31.2 | 78.3 | 55.4% |
| D-Fire | 27.5 | 65.6 | 38.1 | 66.9 | 103.4% |

## Appendix — per-class AP (U-ADAPT)

- **LADD:** person 0.783
- **D-Fire:** fire 0.790 · smoke 0.548
