# Pre-Registration

This document freezes the experimental protocol **before** any main results
are collected. All commitments below are binding; any deviation requires an
entry in [`change_log.md`](change_log.md) with justification.

> Snapshot date: 2026-08-01 (repository bootstrap). Source of truth: the
> thesis proposal (§5 Method, §7.5 Diagnostics, §11 Timeline).

## 1. Evaluation Modes (frozen)

| Mode | Data beyond k support | Backbone gradients | Training | Status |
|------|----------------------|--------------------|----------|--------|
| **A** | None | None | **Training-free** | **Primary strict few-shot claim** |
| **B** | 20 labeled boxes per class (calibration) | None (frozen backbone) | Lightweight calibration | Secondary; reported separately |
| **C** | Source-domain episodic simulation only (COCO/LVIS) | None (frozen backbone) | Source-domain meta-training | Exploratory only |

1. **Mode A is the primary strict few-shot claim.**
2. **Mode A uses no calibration data and no learned temperature; T = 1.**
3. **Mode B uses 20 boxes per class for lightweight calibration and is
   reported separately** — never averaged or conflated with Mode A.
   The logistic-regression gate (6 parameters) is the primary Mode B claim;
   the MLP (5→128→1, ≈650 params, dropout p=0.3, L2 1e-4, early stopping,
   5-fold CV) is a secondary variant. If the MLP fails to beat the logistic
   gate, that finding is reported honestly.
4. **Mode C is exploratory source-domain transfer only.** If it does not
   transfer, this is neither a failure of the primary method nor a weakness
   of the proposal.
5. **All modes share the same frozen backbone and cached features**; only the
   computation of the gate weight `w` differs.

## 2. Mode A Analytic Gating Rule (frozen)

```
w      = sigma( -alpha * sigma_tilde^2_visual + beta * sigma_tilde^2_text + gamma * a_tilde_visual )
S_final = (1 - w) * S_text + w * S_visual
```

* Default coefficients **alpha = beta = gamma = 1**, fixed — **not learned**
  from the target domain.
* All inputs normalized to [0, 1] (min-max with epsilon 1e-6; normalization
  strategy is an ablation: none | min-max | percentile rank).
* `sigma_tilde^2_text` = min-max normalized mean pairwise cosine distance over
  the M=20 prompt-template ensemble (sensitivity check: M=50 on one subset).
* `sigma_tilde^2_visual` = min-max normalized mean pairwise cosine distance
  over the k support features; **zero for k=1** (maximum-likelihood treatment
  of a degenerate sample). Ablation: replace with maximum-entropy prior 0.5.
* `a_tilde_visual` = per-box visual affinity `(1 + cos(f_box, p_visual)) / 2`.
* Sigmoid input is bounded in [-1, 2] at alpha=beta=gamma=1; neither modality
  is ever fully suppressed.
* Coefficients learned on a **source** domain and frozen at target evaluation
  belong to Mode C — never Mode A.

### Ablations (pre-registered)

| Variant | α | β | γ |
|---------|---|---|---|
| Full (default) | 1 | 1 | 1 |
| No visual uncertainty | 0 | 1 | 1 |
| No text uncertainty | 1 | 0 | 1 |
| No affinity | 1 | 1 | 0 |
| Visual uncertainty only | 1 | 0 | 0 |
| Text uncertainty only | 0 | 1 | 0 |
| Affinity only | 0 | 0 | 1 |

## 3. Feature Caching (frozen)

* **Features are cached after one backbone pass** per image (top-k proposals
  only). No backbone or encoder is run repeatedly per proposal.
* Cache lives **outside the repository** (default `cached_features/`,
  gitignored; `--cache-dir` configurable). Model weights, caches, checkpoints,
  and Colab outputs are never uploaded.
* Encoder choice (CLIP vs DINOv2 vs detector-internal features) is an
  ablation; extraction is one additional frozen-encoder pass at most.

## 4. Top-k Proposal Limiting (frozen)

* **Candidate proposals are limited to top-k = 100 for primary experiments.**
* **Top-k = 300 may be used only as an upper-bound ablation.**
* The choke point is `uadapt.models.backbone_loader.limit_top_k`.

## 5. Mode B MC Dropout (frozen)

* **MC Dropout passes are T = 10 for Mode B where applicable** (stability
  check with T = 50 on one dataset subset only).
* Score variances are computed per modality **before** the gate processes them
  (no circularity).

## 6. Mask-to-Box Filtering (frozen)

**Mask-to-box filtering rules are frozen before evaluation** (implemented in
`data/mask_to_box/filter.py`, constants at module level):

| Rule | Value |
|------|-------|
| Minimum box area | **32 px²** |
| Maximum box area | **< 50% of image area** |
| Aspect ratio | **between 1:10 and 10:1** |
| Pure stuff classes (grass, tree, road, water, sand) | **excluded** |
| Damage-level classes | **region-level targets** where appropriate; reported separately |

* **The final class list is frozen before main experiments** (RescueNet
  retained: building, pool, vehicle, debris, roof; FloodNet+ retained:
  building-flooded, building-non-flooded, road-flooded, road-non-flooded,
  vehicle, pool).
* Outlier rejection on visual prototypes: Mahalanobis (shrinkage covariance,
  2σ) for k ≥ 5; cosine threshold 0.5 for k < 5.

## 7. Calibration & Reliability (frozen)

* Mode A: **T = 1 (no learned scaling).** Mode A+ (temperature learned on the
  k support examples) is a clearly-labeled post-hoc ablation, **not** part of
  the primary claim.
* Mode B: T optimized on the calibration split (NLL). Mode C: T learned on the
  source-domain calibration set.
* Reported metrics: **ECE (15 bins)**, reliability diagrams, **Brier score**,
  **uncertainty AUROC**.

## 8. Baselines (frozen)

Zero-shot Grounding DINO, text-only (`S_text`), visual-only (`S_visual`),
naive averaging (T-Rex2-style, `w = 0.5`), and U-ADAPT modes. Raw proposal
recall is reported as the ceiling analysis (no post-hoc method can exceed it).

## 9. Statistical Testing (frozen)

* 10 random seeds; paired t-test **and** Wilcoxon signed-rank for the primary
  comparison (Mode A vs best baseline) across seeds.
* Benjamini–Hochberg FDR control across comparisons; Cohen's d effect size.
* D3 uses an exact binomial test vs 0.5 (α = 0.05).
* **Negative gap recovery is pre-registered**: adapted < zero-shot is reported
  as-is (a finding, not an error).

## 10. Diagnostics D1–D5 (frozen; computed AFTER main results)

* **D1** Text uncertainty–accuracy correlation: 10 bins of σ̃²_text vs
  proposal error rate (correct = IoU ≥ 0.5 with same-class GT); Spearman ρ > 0
  expected.
* **D2** Visual uncertainty–accuracy correlation (same protocol for
  σ̃²_visual).
* **D3** Gate favorability: on disagreeing cases, fraction where the gate
  weights the more accurate modality higher; binomial test vs 0.5. Tests the
  H-fail hypothesis (both modalities unreliable → gate no better than chance).
* **D4** Affinity diagnostic: mean signed Δw = w_full − w_{γ=0} binned by
  a_visual (validates the bias-variance model).
* **D5** Distribution of normalized variances: if >30% of σ̃² values fall
  below 0.25 or above 0.75, the Taylor expansion is flagged and a
  Beta-regression gate variant is pre-registered as fallback.

### Contingency if D1/D2 fail (pre-registered)

(a) report the failure honestly and demote Mode A uncertainty terms to
ablations; (b) fall back to the plain-confidence baseline (cosine-similarity
margin); (c) rely on the Mode B logistic gate; (d) pivot to the comparative
calibration-and-reliability study.

## 11. Licenses & Data Policy (frozen)

* **Dataset and model licenses must be checked before experiments.**
* **Grounding DINO is Apache 2.0 and permits research use and feature
  caching.**
* **If any dataset license restricts academic use, the dataset is replaced or
  dropped and logged in `docs/change_log.md`.**
* See [`licenses.md`](licenses.md) for the full table (issue #1).

## 12. Compute Constraint (frozen)

All experiments must be **Google Colab T4 feasible** (16 GB). The pilot
(notebook `00_pilot_colab_memory.ipynb`) validates memory/time before main
experiments; fallback backbones (OWL-ViT, YOLO-World-small, YOLO11-small) are
used only if the primary backbone exceeds the budget.
