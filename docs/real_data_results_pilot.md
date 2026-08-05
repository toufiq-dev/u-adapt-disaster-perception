# U-ADAPT — PILOT RESULTS (n=10 images)

*Generated 2026-08-05 by `scripts/generate_real_data_report.py`.*

> 🧪 **PILOT RESULTS** — preliminary n-image pipeline check (only the first few images per dataset were evaluated). These numbers are NOT final thesis results; the full-data run supersedes them.

## Executive Summary

- **LADD:** U-ADAPT Mode A **64.5%** mAP50 vs naive averaging **73.6%** and zero-shot **73.6%**.
- **D-Fire:** U-ADAPT Mode A **67.3%** mAP50 vs naive averaging **80.3%** and zero-shot **82.1%**.

- **Pooled D1/D2/D3** (LADD+D-Fire, PRIMARY claim): D1 ρ = **-0.339**, D2 ρ = **+0.175**, D3 favorability = **94.1%** (binomial p = 0.000275).

## mAP50 results

### LADD — mAP50 (subset)

| Method | mAP50 (%) | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | 73.6 | +0.0 pp |
| Text-only (w=0) | 73.6 | +0.0 pp |
| Visual-only (w=1) | 41.5 | -32.1 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 73.6 | +0.0 pp |
| **U-ADAPT Mode A (analytic gate)** | **64.5** | -9.1 pp |

---

### D-Fire — mAP50 (subset)

| Method | mAP50 (%) | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | 82.1 | +0.0 pp |
| Text-only (w=0) | 82.1 | +0.0 pp |
| Visual-only (w=1) | 59.5 | -22.6 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 80.3 | -1.8 pp |
| **U-ADAPT Mode A (analytic gate)** | **67.3** | -14.8 pp |

---

### LADD — gap recovery

Michailidou et al. (Grounding DINO, Table III) floor/ceiling: zero-shot **61.0%** → transfer **92.2%** (gap **31.2 pp**).

- **Fraction of the zero-shot→transfer gap closed:** 11.1% (U-ADAPT 64.5% vs zero-shot 73.6%; transfer ceiling 92.2%).
- Gap recovered vs the oracle re-rank ceiling (100.0%): **-34.5%** (also vs proposal-recall ceiling 28.6%).

---

### D-Fire — gap recovery

Michailidou et al. (Grounding DINO, Table III) floor/ceiling: zero-shot **27.5%** → transfer **65.6%** (gap **38.1 pp**).

- **Fraction of the zero-shot→transfer gap closed:** 104.4% (>100% means the adapter exceeds the literature transfer ceiling — on real data this reflects the tiny pilot subset: the literature ceiling is measured over the full test set, while this run scores only 10 selected images) (U-ADAPT 67.3% vs zero-shot 82.1%; transfer ceiling 65.6%).
- Gap recovered vs the oracle re-rank ceiling (86.5%): **-332.1%** (also vs proposal-recall ceiling 30.9%).

---

### Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10)

D-Fire alone has 2 classes → only 2 distinct variance values, so D1/D2/D3 on it alone are structurally underpowered. Per the pre-registered deviation, the primary diagnostic claim is computed **pooled across LADD + D-Fire** (3 distinct classes: pedestrian, fire, smoke).

| Diagnostic | Pooled value | n |
|---|---|---|
| D1 — text uncertainty–accuracy | Spearman ρ = **-0.339** (rho <= 0: proxy may be uninformative) | 44 |
| D2 — visual uncertainty–accuracy | Spearman ρ = **+0.175** (ok) | 44 |
| D3 — gate favorability | favorability = **94.1%** (binomial p = 0.000275) | 17 |

> ⚠️ **D1 pooled-sign caveat (2026-08-05):** the pooled D1 Spearman ρ mixes LADD and D-Fire at different error base rates with structurally different text-entropy ranges (LADD is single-class → entropy 0.0 by construction; D-Fire sits near max entropy). The pooled sign is dominated by this **between-dataset base-rate difference**, not by a within-proposal uncertainty–accuracy relationship (per-dataset D1 is the interpretable signal). See `docs/change_log.md` 2026-08-05.

Per-dataset values (reported; pooled is primary):

| Dataset | D1 ρ | D2 ρ | D3 favorability |
|---|---|---|---|
| LADD | +0.000 | -0.196 | 90.0% |
| D-Fire | +0.019 | -0.164 | 100.0% |

---

## Comparison to literature baselines

Michailidou et al. (preprint, Table III) with Grounding DINO; U-ADAPT numbers are the Mode A results above (synthetic stand-in unless real data was used — see caveats).

| Dataset | Zero-shot mAP50 | Transfer mAP50 (ceiling) | Gap (pp) | U-ADAPT mAP50 | Gap closed |
|---|---|---|---|---|---|
| LADD | 61.0 | 92.2 | 31.2 | 64.5 | 11.1% |
| D-Fire | 27.5 | 65.6 | 38.1 | 67.3 | 104.4% |

## Appendix — per-class AP (U-ADAPT)

- **LADD:** person 0.645
- **D-Fire:** fire 0.703 · smoke 0.642
