# U-ADAPT — PILOT RESULTS (n=10 images)

*Generated 2026-08-05 by `scripts/generate_real_data_report.py`.*

> 🧪 **PILOT RESULTS** — preliminary n-image pipeline check (only the first few images per dataset were evaluated). These numbers are NOT final thesis results; the full-data run supersedes them.

## Executive Summary

- **LADD:** U-ADAPT Mode A **73.6%** mAP50 vs naive averaging **73.6%** and zero-shot **73.6%**.
- **D-Fire:** U-ADAPT Mode A **74.2%** mAP50 vs naive averaging **80.3%** and zero-shot **82.1%**.

- **Pooled D1/D2/D3** (LADD+D-Fire, PRIMARY claim): D1 ρ = **+0.000**, D2 ρ = **+0.000**, D3 favorability = **0.0%** (binomial p = 1).

## mAP50 results

### LADD — mAP50 (subset)

| Method | mAP50 (%) | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | 73.6 | +0.0 pp |
| Text-only (w=0) | 73.6 | +0.0 pp |
| Visual-only (w=1) | 41.5 | -32.1 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 73.6 | +0.0 pp |
| **U-ADAPT Mode A (analytic gate)** | **73.6** | +0.0 pp |

---

### D-Fire — mAP50 (subset)

| Method | mAP50 (%) | Δ vs zero-shot |
|---|---|---|
| Zero-shot (raw detector scores) | 82.1 | +0.0 pp |
| Text-only (w=0) | 82.1 | +0.0 pp |
| Visual-only (w=1) | 59.5 | -22.6 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 80.3 | -1.8 pp |
| **U-ADAPT Mode A (analytic gate)** | **74.2** | -7.9 pp |

---

### LADD — gap recovery

Michailidou et al. (Grounding DINO, Table III) floor/ceiling: zero-shot **61.0%** → transfer **92.2%** (gap **31.2 pp**).

- **Fraction of the zero-shot→transfer gap closed:** 40.3% (U-ADAPT 73.6% vs zero-shot 73.6%; transfer ceiling 92.2%).
- Gap recovered vs the oracle re-rank ceiling (100.0%): **0.0%** (also vs proposal-recall ceiling -0.0%).

---

### D-Fire — gap recovery

Michailidou et al. (Grounding DINO, Table III) floor/ceiling: zero-shot **27.5%** → transfer **65.6%** (gap **38.1 pp**).

- **Fraction of the zero-shot→transfer gap closed:** 122.5% (>100% means the adapter exceeds the literature transfer ceiling — expected only on the synthetic stand-in) (U-ADAPT 74.2% vs zero-shot 82.1%; transfer ceiling 65.6%).
- Gap recovered vs the oracle re-rank ceiling (86.5%): **-177.1%** (also vs proposal-recall ceiling 16.5%).

---

### Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10)

D-Fire alone has 2 classes → only 2 distinct variance values, so D1/D2/D3 on it alone are structurally underpowered. Per the pre-registered deviation, the primary diagnostic claim is computed **pooled across LADD + D-Fire** (3 distinct classes: pedestrian, fire, smoke).

| Diagnostic | Pooled value | n |
|---|---|---|
| D1 — text uncertainty–accuracy | Spearman ρ = **+0.000** (rho <= 0: proxy may be uninformative) | 44 |
| D2 — visual uncertainty–accuracy | Spearman ρ = **+0.000** (weak/absent correlation: pairwise support variance is a poor proxy) | 44 |
| D3 — gate favorability | favorability = **0.0%** (binomial p = 1) | 0 |

Per-dataset values (reported; pooled is primary):

| Dataset | D1 ρ | D2 ρ | D3 favorability |
|---|---|---|---|
| LADD | +0.000 | +0.000 | 0.0% |
| D-Fire | +0.000 | +0.000 | 0.0% |

---

## Comparison to literature baselines

Michailidou et al. (preprint, Table III) with Grounding DINO; U-ADAPT numbers are the Mode A results above (synthetic stand-in unless real data was used — see caveats).

| Dataset | Zero-shot mAP50 | Transfer mAP50 (ceiling) | Gap (pp) | U-ADAPT mAP50 | Gap closed |
|---|---|---|---|---|---|
| LADD | 61.0 | 92.2 | 31.2 | 73.6 | 40.3% |
| D-Fire | 27.5 | 65.6 | 38.1 | 74.2 | 122.5% |

## Appendix — per-class AP (U-ADAPT)

- **LADD:** person 0.736
- **D-Fire:** fire 0.725 · smoke 0.758
