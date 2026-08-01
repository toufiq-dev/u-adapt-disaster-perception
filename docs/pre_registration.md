# Pre-Registration

This document freezes the experimental protocol **before** any main results
are collected. All commitments below are binding; any deviation requires an
entry in [`change_log.md`](change_log.md) with justification.

> Snapshot date: 2026-08-01 (repository bootstrap, synced to proposal
> Revision 3). Source of truth: the thesis proposal (§5 Method, §7.6
> Diagnostics, §11 Timeline). Mode C no longer exists as a separate mode —
> source-domain meta-training is folded into Mode B as a gate-initialization
> ablation (proposal §5.4.3).

## 1. Evaluation Modes (frozen)

| Mode | Data beyond k support | Backbone gradients | Training | Initialization | Status |
|------|----------------------|--------------------|----------|----------------|--------|
| **A** | None | None | **Training-free** | N/A (analytic rule) | **Primary strict few-shot claim** |
| **B** | 20 labeled boxes per class (calibration) | None (frozen backbone) | Lightweight calibration | **Random** (default) or **COCO/LVIS-pretrained** (ablation; the former Mode C) | Secondary; reported separately |

1. **Mode A is the primary strict few-shot claim.**
2. **Mode A uses no calibration data and no learned temperature; T = 1.**
3. **Mode B uses 20 boxes per class for lightweight calibration and is
   reported separately** — never averaged or conflated with Mode A.
   The logistic-regression gate (6 parameters) is the primary Mode B claim;
   the MLP (5→128→1, ≈900 params, dropout p=0.3, L2 1e-4, early stopping,
   5-fold CV) is a secondary variant. If the MLP fails to beat the logistic
   gate, that finding is reported honestly.
4. **The former Mode C is a Mode B gate-initialization ablation, not a
   separate mode.** The Mode B gate may be initialized randomly (default) or
   from COCO/LVIS-pretrained weights, then calibrated on the same 20-box-per-
   class target set (proposal §5.4.3). If the pretrained initialization does
   not transfer, this is neither a failure of the primary method nor a
   weakness of the proposal.
5. **All modes share the same frozen backbone and cached features**; only the
   computation of the gate weight `w` differs.

### Cross-Domain Transfer Protocol (RQ3, proposal §7.2)

Cross-domain transfer semantics differ by evaluation mode:

* **Mode A (strict few-shot):** transfer means using the same fixed analytic
  coefficients (α = β = γ = 1) on both source and target domains, with only
  the prototypes updating from k target-domain support examples. No training
  or tuning is involved — transfer is assessed purely by the robustness of
  the hand-designed gating rule across domain shift.
* **Mode B (calibrated):** the gate (logistic regression primary, or the
  small MLP variant) is trained on the source domain (e.g., LADD) using the
  full calibration set, then frozen and applied to the target domain (e.g.,
  D-Fire), with only the prototypes updating from k target-domain support
  examples. This tests whether the learned gating pattern generalizes across
  disaster types. Within this same transfer test, the gate may be initialized
  either randomly (default) or from COCO/LVIS-pretrained weights (ablation;
  the former Mode C): if the pretrained initialization does not beat random
  init on the target domain, the optimal gating strategy is largely
  domain-specific.
* This mode-specific distinction ensures cross-domain transfer is evaluated
  consistently, and that failure to transfer in one mode does not reflect on
  the other.

**Dataset-size asymmetry (acknowledged):** LADD (1,365 images) is roughly
16× smaller than D-Fire (21,527 images), so LADD → D-Fire and D-Fire → LADD
are **not** symmetric transfer tests. Transfer is therefore reported in **both
 directions**, and LADD-trained → D-Fire-evaluated results are interpreted
with this imbalance caveat stated explicitly. If asymmetric transfer is
observed, the direction consistent with the larger source domain is treated
as the more reliable estimate of transferability, and the discrepancy is
discussed as a finding rather than averaged away.

**Directional hypothesis (pre-registered):** transfer is expected to be
asymmetric but not along naive class-complexity intuition alone. For Mode A
(fixed analytic coefficients), transfer should be roughly symmetric because
nothing is learned. For Mode B (frozen trained gate), we expect
D-Fire-trained → LADD-evaluated to transfer more favorably than the reverse,
because the D-Fire training set (21,527 images) provides far more visual
diversity for learning the gate than LADD (1,365 images). Class complexity
cuts the other way — fire/smoke are diffuse and segmentation-shaped while
pedestrians are compact — so a gate trained on D-Fire may over-weight the
visual branch when applied to LADD. Tested in both directions.

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
  belong to the Mode B COCO/LVIS-pretrained initialization ablation — never
  Mode A.

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
| Minimum valid boxes per class | **≥ 10** across the whole dataset, else excluded and reported |
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
* Mode B: T optimized on the calibration split (NLL). Mode B (COCO/LVIS
  pretrained init): T calibrated on the target 20-box-per-class set
  (source-domain temperature is not carried over).
* Reported metrics: **ECE (15 bins)**, reliability diagrams, **Brier score**,
  **uncertainty AUROC**.

## 8. Baselines (frozen)

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Zero-shot text-only | OV detector with text prompts only, no visual exemplars | Performance floor |
| Naive text/visual averaging | Score-level averaging with `w = 0.5` (`(S_text + S_visual)/2`); embedding-level averaging surrogate also implemented where feasible | Direct comparison to prior art; separates score-level fusion from embedding-level averaging |
| Visual-only nearest-prototype | Visual prototype matching without text | Isolates visual prompt contribution |
| U-ADAPT w/o uncertainty gating | Fixed `w = 0.5` (equal averaging) | Ablation of core contribution |
| U-ADAPT w/o temperature scaling | Gating without final calibration | Ablation of calibration |
| U-ADAPT w/o MC Dropout | Plain cosine-similarity margin as uncertainty proxy | Tests whether MC Dropout is necessary |
| Mode A: full U-ADAPT (strict few-shot) | Analytic gating, no additional training | Primary claim |
| Mode B: U-ADAPT with calibration (logistic regression) | 6-parameter logistic gate, 20-box calibration | Simpler Mode B alternative |
| Mode B: U-ADAPT with calibration (MLP) | ≈900-parameter MLP gate, 20-box calibration | Full Mode B variant |
| Mode B: U-ADAPT with COCO/LVIS-pretrained init | Gate initialized from COCO/LVIS-pretrained weights, then calibrated on target | Initialization ablation (formerly Mode C) |
| Transfer-learning reference | Transfer-learning upper bound from Michailidou et al. | Performance ceiling context |
| Supervised detectors | YOLOv11l, YOLO26L, RT-DETRv2-L (Michailidou Table III) | Absolute performance ceiling |

Raw proposal recall is reported as the ceiling analysis (no post-hoc method
can exceed it). The "w/o MC Dropout" baseline tests whether a much simpler
uncertainty proxy (the normalized cosine distance margin between `S_text` and
`S_visual`) achieves comparable gating performance — if it does, the MC
Dropout overhead may not be justified, and this is reported honestly.

## 9. Statistical Testing (frozen)

* **Primary comparison: U-ADAPT Mode A vs. naive averaging (w = 0.5)** per
  dataset and per shot (proposal §7.1, §7.6). 10 random seeds; paired t-test
  **and** Wilcoxon signed-rank across seeds. Power analysis is not formalized:
  with 10 paired observations a paired t-test detects a large effect
  (d ≈ 1.0) at α = 0.05 with power ≈ 0.8; effect sizes are reported alongside
  p-values and non-significant differences are treated as evidence of
  comparable performance, not superiority.
* Secondary comparisons: Mode A vs text-only and vs visual-only; Mode B
  (logistic) vs Mode A. Same paired protocol.
* Benjamini–Hochberg FDR control (q = 0.05) across ~18 comparisons
  (2 datasets × 3 shots × ~3 primary comparisons); uncorrected p and adjusted
  q both reported; Cohen's d for every significant comparison.
* D3 uses an exact binomial test vs 0.5 (α = 0.05).
* **Negative gap recovery is pre-registered**: adapted < zero-shot is reported
  as-is (a finding, not an error).
* **RQ5 numeric definition (pre-registered):** a gain is backbone-agnostic if
  U-ADAPT's relative improvement over its own zero-shot baseline is within a
  factor of 2× across all tested backbones (ratio of largest to smallest
  relative improvement ≤ 2), evaluated per dataset.

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

### Compound contingency: pilot failure + Colab memory failure (proposal §10)

The individual contingencies above are pre-registered separately, but the
compound case is addressed explicitly: if the pilot (Week 3) reveals that
Mode A's uncertainty proxies do not correlate with error (D1/D2 fail) **and**
Grounding DINO exceeds Colab memory limits at the same time, the plan is:

1. Switch the primary backbone to the fallback (OWL-ViT or YOLO11-small)
   rather than fighting the memory ceiling.
2. Demote Mode A's uncertainty terms to ablations and elevate the
   plain-confidence baseline (cosine-similarity margin) plus the Mode B
   logistic gate, which learns the weighting from data rather than assuming
   the proxy is informative.
3. Absorb the re-baselining cost by trimming the cross-backbone matrix to a
   single secondary backbone, since the core RQ1/RQ2 claim depends only on
   Mode A on two datasets.

This compound case is decided at the pilot's post-mortem; the decision tree
is fixed before the pilot runs.

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
experiments; fallback backbones (OWL-ViT, YOLOE26, YOLO-World-small,
YOLO11-small) are used only if the primary backbone exceeds the budget.

## 13. Ethics & Scope (frozen)

* **Research-ethics exemption:** No institutional ethics approval is required:
  publicly available, de-identified datasets, no human subjects, no
  personally identifiable information (proposal §Ethics). The exact name and
  process of the university's ethics review is confirmed before submission.
* **License verification is committed** before experiments (see
  [`licenses.md`](licenses.md), issue #1); any restricted dataset/model is
  dropped or replaced and logged in [`change_log.md`](change_log.md).
* **Out of scope (delimitations, proposal §13):** online/streaming adaptation;
  gating-network architecture search; panoptic/instance segmentation;
  backbone fine-tuning; video or non-aerial imagery; TENT/MEMO-style test-time
  gradient adaptation.
