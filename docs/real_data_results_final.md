# U-ADAPT — Final Comparative Results (n=100 subset): Analytic vs. Beta fallback

*Generated 2026-08-07 by `scripts/generate_real_data_report.py` (`--compare-dirs`). Gates: analytic, beta_fallback.*

> 🧪 **PILOT RESULTS (n=100 subset)** — preliminary pipeline check on the first 100 images per dataset (Grounding DINO Swin-T, top-k=100, k=5 shots, seed 0). These numbers are NOT final thesis results; the full-data run and the 10-seed protocol (`scripts/run_10_seed_protocol.py`) supersede them.

## Executive Summary

- **LADD** (n=100): Mode A analytic **78.3%** vs beta_fallback **78.4%** (+0.1 pp); naive averaging **80.4%**, zero-shot **81.3%**.
  - Beta softens the gate (78.3% -> 78.4%), but Mode A still underperforms the naive average on LADD at n=100 — the saturation artifact is softened, not resolved.
- **D-Fire** (n=100): Mode A analytic **66.9%** vs beta_fallback **68.3%** (+1.4 pp); naive averaging **71.8%**, zero-shot **73.4%**.
  - Beta softens the gate (66.9% -> 68.3%), but Mode A still underperforms the naive average on D-Fire at n=100 — the saturation artifact is softened, not resolved.

---

## mAP50 results

### LADD — mAP50 (n=100 subset, side-by-side)

| Method | analytic | beta_fallback | Δ (β − α) |
| --- | --- | --- | --- |
| Zero-shot (raw detector scores) | 81.3 | 81.3 | — |
| Text-only (w=0) | 81.3 | 81.3 | +0.0 pp |
| Visual-only (w=1) | 66.9 | 66.9 | +0.0 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 80.4 | 80.4 | +0.0 pp |
| U-ADAPT Mode A | **78.3** | **78.4** | +0.1 pp |

---

### D-Fire — mAP50 (n=100 subset, side-by-side)

| Method | analytic | beta_fallback | Δ (β − α) |
| --- | --- | --- | --- |
| Zero-shot (raw detector scores) | 73.4 | 73.4 | — |
| Text-only (w=0) | 73.4 | 73.4 | +0.0 pp |
| Visual-only (w=1) | 54.7 | 54.7 | +0.0 pp |
| Naive averaging (w=0.5, T-Rex2 surrogate) | 71.8 | 71.8 | +0.0 pp |
| U-ADAPT Mode A | **66.9** | **68.3** | +1.4 pp |

---

## Gate weight behavior (saturation diagnostic)

| Dataset | analytic: mean w | beta_fallback: mean w | analytic: std w | beta_fallback: std w | beta_fallback: % > 0.55 |
| --- | --- | --- | --- | --- | --- |
| LADD | 0.699 | 0.686 | 0.028 | 0.027 | 99% |
| D-Fire | 0.856 | 0.775 | 0.013 | 0.006 | 100% |

> The analytic gate's mean weight saturates toward the visual modality (LADD 0.699, D-Fire 0.856 at n=100). The Beta fallback hedges the commitment where variances cluster at the extremes (D5 contingency): D-Fire mean w drops 0.856 -> 0.775, LADD 0.699 -> 0.686 — still > 0.55 for essentially every proposal.

---

## Diagnostics D1–D5 (side-by-side)

Pooled D1/D2/D3 (PRIMARY claim, deviation 2026-08-03 §10; LADD + D-Fire, n=657 proposals):

| Diagnostic | analytic | beta_fallback |
| --- | --- | --- |
| D1 — text uncertainty–accuracy | ρ = **-0.056** | ρ = **-0.056** |
| D2 — visual uncertainty–accuracy | ρ = **+0.051** | ρ = **+0.051** |
| D3 — gate favorability | favorability = **100.0%** | favorability = **100.0%** |

> D1/D2 do not involve the gate weight, so they are identical across gates by construction; D3 (favorability on disagreeing cases) stays at 100% for both gates because the disagreeing subsets are almost entirely *visual-better* (affinity saturates >= 0.65) and every proposal is gated w > 0.55 — a saturation artifact, not evidence of calibration.

D5 — Taylor-validity sentinel on the raw variance scale (per dataset):

| Dataset | analytic | beta_fallback |
| --- | --- | --- |
| LADD | 99.7% (FLAGGED) | 99.7% (FLAGGED) |
| D-Fire | 95.7% (FLAGGED) | 95.7% (FLAGGED) |

> D5 flags the boundary-clustered variance regime that triggers the pre-registered Beta-regression fallback (docs/pre_registration.md §10).

---

## Comparison to literature baselines

Michailidou et al. (Table III, Grounding DINO) floor/ceiling vs the Mode A results above (n=100 subset, per gate):

- **LADD:** zero-shot floor 61.0% → transfer ceiling 92.2% (gap 31.2 pp). U-ADAPT Mode A — analytic: 78.3% (55.4% closed), beta_fallback: 78.4% (55.7% closed).
- **D-Fire:** zero-shot floor 27.5% → transfer ceiling 65.6% (gap 38.1 pp). U-ADAPT Mode A — analytic: 66.9% (103.4% closed), beta_fallback: 68.3% (107.0% closed).

---

## 10-seed paired statistical protocol (pre-registration §9)

*Run 2026-08-07 by `scripts/run_10_seed_protocol.py` — 10 seeds × {ladd, dfire} × k ∈ {1, 3, 5} = 60 cells per gate on the n=100 pilot caches, via the scripted path (02 → 03 → 04). Primary comparison: U-ADAPT Mode A vs. naive averaging (w = 0.5). Per cell, across the 10 seeds: paired two-sided t-test AND Wilcoxon signed-rank, Cohen's d (paired, d_z), and Benjamini-Hochberg FDR control (q = 0.05) over the full comparison family of 12 tests (2 tests × 6 cells).*

### Analytic gate

| cell | n | Mode A mAP50 | Naive mAP50 | d | t | p(t) | q(t) | W | p(W) | q(W) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ladd_k1 | 10 | 0.7808 | 0.8025 | -2.37 | -7.487 | 3.75e-05 | 0.00015 | 0.0 | 0.00195 | 0.00213 |
| ladd_k3 | 10 | 0.7893 | 0.8030 | -3.72 | -11.775 | 9.04e-07 | 5.43e-06 | 0.0 | 0.00195 | 0.00213 |
| ladd_k5 | 10 | 0.7887 | 0.8028 | -5.74 | -18.150 | 2.13e-08 | 2.56e-07 | 0.0 | 0.00195 | 0.00213 |
| dfire_k1 | 10 | 0.6817 | 0.7190 | -1.51 | -4.782 | 0.000999 | 0.002 | 1.0 | 0.00391 | 0.00391 |
| dfire_k3 | 10 | 0.6840 | 0.7200 | -2.11 | -6.658 | 9.29e-05 | 0.000279 | 0.0 | 0.00195 | 0.00213 |
| dfire_k5 | 10 | 0.6932 | 0.7205 | -1.99 | -6.286 | 0.000143 | 0.000344 | 0.0 | 0.00195 | 0.00213 |

### Beta fallback gate

| cell | n | Mode A mAP50 | Naive mAP50 | d | t | p(t) | q(t) | W | p(W) | q(W) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ladd_k1 | 10 | 0.7822 | 0.8025 | -2.30 | -7.285 | 4.64e-05 | 0.000111 | 0.0 | 0.00195 | 0.00213 |
| ladd_k3 | 10 | 0.7912 | 0.8030 | -3.61 | -11.428 | 1.17e-06 | 6.99e-06 | 0.0 | 0.00195 | 0.00213 |
| ladd_k5 | 10 | 0.7905 | 0.8028 | -6.08 | -19.222 | 1.29e-08 | 1.55e-07 | 0.0 | 0.00195 | 0.00213 |
| dfire_k1 | 10 | 0.6885 | 0.7190 | -1.88 | -5.933 | 0.00022 | 0.00044 | 1.0 | 0.00391 | 0.00391 |
| dfire_k3 | 10 | 0.6923 | 0.7200 | -3.38 | -10.686 | 2.05e-06 | 8.22e-06 | 0.0 | 0.00195 | 0.00213 |
| dfire_k5 | 10 | 0.6973 | 0.7205 | -3.15 | -9.961 | 3.7e-06 | 1.11e-05 | 0.0 | 0.00195 | 0.00213 |

### Analytic vs. Beta (Mode A means, per-seed paired)

| cell | analytic | beta_fallback | Δ (β − α) | d analytic | d beta |
| --- | --- | --- | --- | --- | --- |
| ladd_k1 | 0.7808 | 0.7822 | +0.0014 | -2.37 | -2.30 |
| ladd_k3 | 0.7893 | 0.7912 | +0.0019 | -3.72 | -3.61 |
| ladd_k5 | 0.7887 | 0.7905 | +0.0018 | -5.74 | -6.08 |
| dfire_k1 | 0.6817 | 0.6885 | +0.0068 | -1.51 | -1.88 |
| dfire_k3 | 0.6840 | 0.6923 | +0.0083 | -2.11 | -3.38 |
| dfire_k5 | 0.6932 | 0.6973 | +0.0041 | -1.99 | -3.15 |

### Interpretation

- **Every cell is significant after FDR control (all q < 0.05), in the same direction for both gates: Mode A is significantly WORSE than naive averaging (w = 0.5) at every k ∈ {1, 3, 5} on both datasets.** Cohen's d spans −1.5 to −6.1 (very large, unfavorable); Wilcoxon W = 0 in 5/6 cells means all 10 paired differences carried the same sign — the gap is systematic, not seed noise.
- **The Beta fallback helps directionally but does not close the gap.** Beta raises the Mode A mean on all 6 cells (+0.1 to +0.8 pp; largest on D-Fire), consistent with the n=100 finding that hedging the saturated gate is directionally beneficial. Yet per-seed, Beta Mode A beats naive averaging on 0–1 of 10 seeds in every cell.
- **Variance-stabilization nuance (D-Fire):** the Beta gate also shrank the seed-to-seed spread of the Mode A − naive gap (e.g. dfire_k1: |d| grows 1.51 → 1.88 despite an improved mean) — the fallback both raises the mean and stabilizes it across seeds.
- **Naive baseline is bit-identical across the two gate runs** (same w = 0.5 scores), so the analytic-vs-beta comparison is exactly paired.
- **Verdict:** at pilot scale, neither the analytic gate nor the pre-registered Beta contingency beats the uncertainty-blind baseline. The D5-triggered fallback is a robustness contingency, not a fix. The pre-registered primary comparison (§9) is settled at n=100: the gate, as implemented, does not recover value from the uncertainty inputs.

> Methodology note: the 10-seed protocol evaluates the full cached test split via the scripted path (02 → 03 → 04), whereas the n=100 tables above use the demo path; the small mean differences (e.g. LADD analytic 78.3% vs 78.08% here) reflect the documented path-level differences in gate inputs, not a data change.
