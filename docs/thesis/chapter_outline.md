# U-ADAPT Thesis — Chapter Outline (v1, 2026-08-07)

*Title (working):* **U-ADAPT: Uncertainty-Aware Post-Hoc Adaptation of Open-Vocabulary Detectors for Few-Shot Cross-Domain Disaster Perception**

> **Status of the numbers in this outline.** Every result cited here is the
> **definitive pilot-scale record** (n=100 caches, real Grounding DINO Swin-T
> features): the n=100 single-seed comparison, the **10-seed paired statistical
> protocol** (analytic + Beta fallback), and the D1–D5 diagnostics. All numbers
> are reproduced in `docs/real_data_results_final.md` (byte-identical from
> `scripts/generate_real_data_report.py` with `--analytic-stats` /
> `--beta-stats`). The full-scale run replaces the n=100 figures where marked
> **[FULL-SCALE]**. Pre-registration commitments are cited by section (§) of
> `docs/pre_registration.md`; every deviation already has a `docs/change_log.md`
> entry.

**Narrative spine.** The thesis is deliberately structured as the honest
execution of a pre-registered experiment whose primary hypothesis fails — and
whose **failure is informative because the diagnostics explain it**. The
scientific claim is *not* "U-ADAPT improves few-shot detection" (pilot data
reject this) but "**a pre-registered uncertainty-gating framework can be built,
its assumptions empirically falsified at pilot scale, the failure mechanism
isolated (gate saturation), and a pre-registered contingency (Beta-regression
fallback) executed and characterized**." That arc — hypothesis → falsification
→ mechanism → contingency — is the contribution.

- **H (primary, §9):** uncertainty-gated fusion (Mode A) beats fixed
  averaging (w = 0.5) in mAP50, per dataset, per shot, over 10 seeds.
  **Pilot verdict: rejected** (all 6 cells, q < 0.05 after BH-FDR, d ≤ −1.5).
- **H-fail (D3, §10):** when both modalities are unreliable, the gate is no
  better than chance at weighting the more accurate modality. Pilot evidence:
  D3 favorability = 100% — but this is a **saturation artifact** (disagreeing
  subsets are almost entirely visual-better, so the 100% is a degenerate
  consequence of the gate always leaning visual).
- **D5 contingency (§10):** Taylor-expansion validity flagged (>30% of
  variances outside [0.25, 0.75]) → Beta-regression fallback pre-registered →
  **executed** (analytic vs. beta 10-seed comparison). Verdict: directionally
  beneficial (+0.1–0.8 pp, variance-stabilizing) but does not beat w = 0.5.

---

## Chapter 1 — Introduction

1.1 **Problem statement.** Few-shot, cross-domain detection in aerial disaster
    perception; open-vocabulary detectors (OVDs) transfer poorly zero-shot; the
    zero-shot → transfer gap (Michailidou et al.: LADD 61.0% → 92.2%, D-Fire
    27.5% → 65.6%). Gap decomposition into vocabulary mismatch,
    feature-distribution shift, and proposal-quality degradation (proposal §1).
1.2 **Research questions** (proposal §3), each with its pilot-scale status:
    - RQ1 (primary): does uncertainty-gated fusion help under 1/3/5-shot?
      **Pilot: no** — Mode A < naive (w = 0.5) at all cells.
    - RQ2 (gap recovery): what fraction of the gap is closed? **Pilot: not
      achieved** — Mode A trails its own zero-shot on *both* datasets
      (LADD 78.3 < 81.3; D-Fire 66.9 < 73.4).
    - RQ3 (transfer): not assessable at pilot scale **[FULL-SCALE]**.
    - RQ4 (reliability/calibration): **pending** (ECE/Brier/AUROC not yet the
      focus) **[FULL-SCALE]**.
    - RQ5 (backbone sensitivity): **pending** (single backbone so far).
1.3 **The H-fail hypothesis and the thesis's real claim.** State H (§9) and
    H-fail (§10 D3) plainly. Announce that the thesis reports a
    pre-registered negative result, the isolation of its mechanism (gate
    saturation), and the executed D5 contingency — i.e., a contribution in
    **diagnostics and protocol**, not in claimed SOTA.
1.4 **Contributions** (honest reframing):
    (1) training-free Mode A gating framework with MVUE-motivated fusion;
    (2) a five-part diagnostic suite (D1–D5) that isolates *why* a
    theoretically-motivated gate fails on real data;
    (3) execution of the full pre-registration lifecycle (deviation log,
    10-seed paired protocol, BH-FDR, contingency);
    (4) the Beta-regression fallback as a principled boundary-safe gate.
1.5 **Thesis outline** (one paragraph per chapter).
- **Figures/Tables:** Fig 1.1 — high-level U-ADAPT pipeline schematic;
  Table 1.1 — RQ → hypothesis → diagnostic → verdict mapping (the "scorecard"
  that previews Chapter 5).

## Chapter 2 — Related Work

2.1 Open-vocabulary object detection (Grounding DINO family, OWL-ViT,
    YOLO-World; Michailidou et al. benchmark, Table III baselines).
2.2 Text–visual prompt fusion (T-Rex2 as the principal score-level fusion
    comparison; embedding-level averaging surrogate).
2.3 Confidence-gated multimodal fusion — the lineage U-ADAPT extends
    (weighted fusion with confidence/uncertainty controls).
2.4 Uncertainty estimation in vision–language models (MC Dropout, entropy,
    cosine-distance margins; distinguish aleatoric/epistemic).
2.5 Few-shot and cross-domain detection (prototype methods, prompt tuning).
2.6 Calibration for detection (ECE, Brier, reliability diagrams).
2.7 Test-time adaptation (TENT/MEMO; why U-ADAPT is post-hoc but not
    gradient-based — out of scope per §13).
2.8 Prototype-based few-shot detection (k-shot support sets, outlier
    rejection).
2.9 **Why disaster domains break the standard assumptions** (the chapter's
    thesis-forward synthesis, backed by the pilot findings):
    - single-class entropy collapse (LADD pedestrian-only → text entropy ≡ 0
      by construction → text uncertainty term is inert);
    - diffuse, segmentation-shaped targets (fire/smoke) vs. compact targets
      (pedestrians) — different failure modes per modality;
    - feature-space affinity saturation (real RoI features cluster ≥ 0.87),
      which makes *any* affinity-threshold rule degenerate.
- **Figures/Tables:** Table 2.1 — taxonomy of fusion/uncertainty methods with
  columns (modality signals, trainable?, calibration data, post-hoc?);
  Fig 2.1 — sketch of the affinity/entropy regimes in aerial vs. natural
  imagery (informed by §5.1 data).

## Chapter 3 — Method: U-ADAPT

3.1 **Overview** — five phases (candidate generation → feature extraction →
    prototype construction → uncertainty-gated fusion → calibration) on a
    frozen backbone; only prototypes update from k support examples.
3.2 **Candidate generation** — frozen Grounding DINO Swin-T, top-k = 100
    (§4); design rationale (best zero-shot floor, largest gap, internal
    consistency of floor/ceiling).
3.3 **Feature caching** — one pass per image, RAM-safe batch streaming
    (change_log 2026-08-06), cache outside the repo (§3).
3.4 **Prototype construction** — M = 20 text-prompt ensemble; k-shot visual
    prototypes with outlier rejection (Mahalanobis 2σ for k ≥ 5, cosine
    threshold for k < 5) (§2, §6).
3.5 **Mode A: analytic uncertainty-gated fusion and the MVUE derivation.**
    - Per-proposal score model `S_final = (1 − w)·S_text + w·S_visual`; the
      fused score as a **minimum-variance combination of two estimators**; the
      oracle weight is the inverse-variance (precision) ratio;
    - the training-free proxy gate
      `w = σ(−α·σ̃²_visual + β·σ̃²_text + γ·ã_visual)` with frozen
      α = β = γ = 1 (§2);
    - **Taylor-expansion caveat** — the sigmoid linearization is valid only
      when the normalized variance inputs lie in (0.25, 0.75); D5 is the
      sentinel that tests this.
3.6 **The pre-registered Beta-regression fallback** (§10, D5 contingency).
    - Same logit, same inputs, but the weight is the **mean of a Beta
      distribution** with input-linked precision
      `φ = φ_max / (1 + slope·(v_text + v_visual))` blended with a neutral 0.5
      prior: always a valid probability, boundary-safe where the sigmoid/
      Taylor approximation saturates, and it recovers the analytic gate in the
      low-variance limit (training-free, still strictly Mode A-compatible).
3.7 **Mode B** (secondary, reported separately per §1): 6-parameter logistic
    gate with 20-box calibration; MLP variant; COCO/LVIS-init ablation.
- **Figures/Tables:** Fig 3.1 — full pipeline with tensor shapes;
  Fig 3.2 — gate weight surface `w(v_text, v_visual, a_visual)` for the
  analytic vs. Beta gate (3-D surface + boundary slices);
  Table 3.1 — notation table; Table 3.2 — pre-registered ablations (α/β/γ
  variants, §2).

## Chapter 4 — Experimental Setup

4.1 **Pre-registration discipline** — frozen protocol (§1–§13), the
    deviation log (`docs/change_log.md`) as an auditable record: absolute
    scaling (§2, 2026-08-03), pooled D1/D2/D3 (§10, 2026-08-03), per-proposal
    estimators (§2/§7.6, 2026-08-05), D4 counterfactual (§10, 2026-08-07),
    10-seed protocol execution (§9, 2026-08-07).
4.2 **Datasets** — LADD (1,365 imgs, 1 class: pedestrian) and D-Fire
    (21,527 imgs, 2 classes: fire/smoke); mask-to-box filtering (§6:
    min area 32 px², aspect 1:10–10:1, ≥10 valid boxes/class); train/test
    splits; n=100 pilot subset extraction.
4.3 **Normalization strategies** — min-max (LADD) vs. absolute (D-Fire)
    and the 2-class degeneracy that motivated absolute scaling.
4.4 **Baselines** (§8) — zero-shot text-only, text-only (w=0), visual-only
    (w=1), naive averaging (w=0.5, T-Rex2 surrogate), plus U-ADAPT w/o
    uncertainty gating (the `NaiveGate` w=0.5 scripted baseline used by the
    10-seed protocol).
4.5 **Metrics** (§7.4) — mAP50 (primary), mAP50:95, per-class AP; ECE/Brier/
    AUROC for RQ4.
4.6 **The 10-seed paired statistical protocol** (§9) — 2 datasets × 3 shots
    × 10 seeds = 60 cells per gate; paired two-sided t-test **and** Wilcoxon
    signed-rank; Cohen's d (d_z); **Benjamini–Hochberg FDR (q = 0.05)** over
    the family of 12 tests; power note (10 pairs → large effects detectable);
    the scripted pipeline (02 → 03 → 04) with the `--gate-type` variants
    (analytic / beta_fallback / naive).
4.7 **Diagnostics D1–D5** (§10) — definitions, the pooled-deviation rationale
    (2-class underpowering), D3 as the H-fail test, D4 as the γ=0
    counterfactual (Δw = w − w_{γ=0}), D5 as the Taylor-validity sentinel.
4.8 **Implementation** — Grounding DINO Swin-T, caching, MPS/GPU, RAM-safe
    streaming, reproducibility (seeds, `run_10_seed_protocol.py`).
- **Figures/Tables:** Table 4.1 — dataset summary (images, classes, boxes,
  license, norm strategy); Table 4.2 — protocol parameter card (backbone,
  top-k, k-shots, seeds, tests, FDR family); Fig 4.1 — 10-seed protocol
  flowchart; Table 4.3 — D1–D5 definition card (input, statistic, threshold,
  verdict, contingency).

## Chapter 5 — Results & Diagnostics

### 5.1 The gate saturation phenomenon (n=100, seed 0)
- mAP50 side-by-side (analytic vs. beta): LADD zero-shot **81.3**, text 81.3,
  visual 66.9, naive 80.4, Mode A **78.3 / 78.4**; D-Fire zero-shot **73.4**,
  text 73.4, visual 54.7, naive 71.8, Mode A **66.9 / 68.3**.
  - **Key observation:** visual-only reranking is *worse* than the raw
    detector (LADD 66.9 < 81.3; D-Fire 54.7 < 73.4), yet the gate leans visual.
- **Saturation evidence** (from `scripts/analyze_saturation.py`):
  - LADD: affinity mean 0.943, 99.4% > 0.8; w > 0.55 for 99.4% of proposals;
    variance terms move w by only ±0.021 (counterfactual w with inert
    variances = 0.720 vs realized 0.699).
  - D-Fire: affinity mean 0.946 sets a w ≥ 0.71 floor for *every* proposal;
    100% of w > 0.55; text entropy (mean 0.905) *reinforces* the visual lean.
  - LADD `norm_text_var = 0.0` for all proposals — the single-class entropy
    collapse that makes the text uncertainty term structurally inert.
- **D1–D3, D5 diagnostics:**
  - D1 pooled ρ = −0.056 (within-dataset ≈ 0; LADD constant-entropy by
    construction, D-Fire ρ = −0.061) → no within-proposal text
    uncertainty–accuracy signal;
  - D2 pooled ρ = +0.051 — a **between-dataset normalization-scale artifact**
    (within-dataset ≈ 0), not an effect;
  - D3 favorability = 100% (n=246; 131/132 + 114/114 visual-better, 1
    text-better) — **the H-fail test degenerates**: the affinity threshold
    never fails on real features, so the "gate favors the more accurate
    modality" claim is untestable at this scale;
  - D5 **FLAGGED** both datasets (LADD 99.7%, D-Fire 95.7% of variances at the
    boundaries) → the Taylor expansion is invalid → **Beta fallback triggered
    as pre-registered**.
- **Figures/Tables:** Fig 5.1 — the 2×2 saturation histograms per dataset
  (`outputs/real_data/saturation_analysis/{ladd,dfire}_saturation.png`);
  Fig 5.2 — gate-weight distributions analytic vs. beta; Fig 5.3 — D1/D2
  binned uncertainty–error plots; Table 5.1 — n=100 mAP50 (both datasets,
  both gates); Table 5.2 — saturation summary statistics; Table 5.3 —
  D1–D5 verdict card.

### 5.2 The 10-seed statistical verdict (FDR-corrected)
- Full analytic + beta tables (from `docs/real_data_results_final.md`,
  §10-seed section). Headline: **all 6 cells significant after BH-FDR
  (q < 0.05) and all unfavorable** — Mode A < naive at every k on both
  datasets; Cohen's d spans −1.5 to −6.1 (very large, negative); Wilcoxon
  W = 0 in 5/6 cells (uniform sign across seeds); per-seed wins 0–1 of 10.
- The naive baseline is bit-identical across gate runs → exactly paired
  comparison.
- **Interpretation discipline:** the rejection of H is *statistically
  definitive at pilot scale* (10 seeds, paired tests, FDR), matching the
  deterministic n=100 picture.
- **Figures/Tables:** Fig 5.4 — per-seed paired mAP50 lines (Mode A vs. naive,
  one panel per cell); Fig 5.5 — forest plot of Cohen's d with 95% CIs,
  FDR-significant cells marked; Fig 5.6 — FDR q-value waterfall (12 tests);
  Table 5.4 — the two gate tables (analytic, beta) as printed in
  `real_data_results_final.md`; Table 5.5 — analytic-vs-beta comparison.

### 5.3 The Beta fallback contingency
- **Directional improvement:** Beta raises the Mode A mean on all 6 cells
  (+0.1 to +0.8 pp; largest on D-Fire: 0.6817 → 0.6885 @k1, 0.6840 → 0.6923
  @k3, 0.6932 → 0.6973 @k5), consistent with the n=100 picture (D-Fire
  66.9 → 68.3).
- **Variance stabilization:** the Beta gate shrinks the seed-to-seed spread of
  the Mode A − naive gap (dfire_k1 |d| 1.51 → 1.88 despite the improved mean)
  — a *reliability* benefit separable from the *accuracy* verdict.
- **Ultimate limitation:** still significantly below naive at every cell
  (q < 0.05); wins 0–1 of 10 seeds. The fallback is a robustness contingency,
  not a fix.
- **Figures/Tables:** Fig 5.7 — analytic vs. Beta gate-weight overlays per
  dataset (saturation softening); Fig 5.8 — Δ(β−α) scatter vs. cell; Table
  5.6 — analytic-vs-beta summary (mean w, d, q, wins).

### 5.4 Literature comparison & gap recovery (context)
- LADD: 61.0 → 92.2 (gap 31.2 pp), Mode A closes 55.4% (analytic) / 55.7%
  (beta) — but from a below-zero-shot start; D-Fire: 27.5 → 65.6 (gap 38.1
  pp), closes 103.4% / 107.0% — the >100% is the tiny-subset artifact noted
  in the report. Frame honestly: gap-recovery is not yet a claim **[FULL-SCALE]**.
- **Figures/Tables:** Table 5.7 — floor/ceiling/gap-closed per dataset and
  gate; Fig 5.9 — gap-recovery bars vs. literature ceiling.

## Chapter 6 — Discussion

6.1 **Why the analytic proxies fail in disaster domains** — synthesize §5.1:
    (a) single-class entropy collapse makes text uncertainty structurally
    inert (LADD); (b) RoI feature-space affinity saturation makes the
    affinity term dominate and the gate lean on the *weaker* reranker;
    (c) normalization-scale artifacts corrupt pooled correlations (D2);
    (d) the Taylor expansion's validity window is violated exactly where the
    real data sits (D5) — the failure is a **feature-space regime mismatch**,
    not an implementation bug.
6.2 **The value of the pre-registered fallback** — auditability (the deviation
    log, the executed contingency), the Beta gate's principled boundary
    safety, and the separable *stabilization* benefit (5.3); argue that a
    pre-registered negative result with an executed contingency is stronger
    evidence than an unregistered positive one.
6.3 **What the diagnostics recovered** — the diagnostic suite is the
    contribution: pooled-power fix (D1/D2/D3), the D4 γ=0 counterfactual, the
    D5 sentinel, and the saturation decomposition; each isolated a distinct
    mechanism that a single mAP50 number would have hidden.
6.4 **Limitations** — pilot scale (n=100 subset; ~80-image evaluated split),
    single backbone, frozen coefficients α=β=γ=1, affinity threshold 0.65
    degeneracy, D3 untestable at this scale, no calibration-metric results
    yet (RQ4) **[FULL-SCALE]**.
- **Figures/Tables:** Fig 6.1 — "regime diagram" (validity of each proxy vs.
  feature-space regime: entropy spread, affinity level, variance location);
  Table 6.1 — claim-vs-evidence table (what each RQ can and cannot claim at
  pilot scale).

## Chapter 7 — Conclusion & Future Work

7.1 **Answers to the RQs** — the honest scorecard: RQ1 negative at pilot
    scale; RQ2 not achieved; RQ3/RQ5 deferred; RQ4 deferred; the diagnostic
    RQs (why / how much / is the gate trustworthy) answered.
7.2 **Future work**:
    - re-weight/re-parameterize the gate inputs (variance terms on the
      absolute scale; affinity de-saturation) with the *pre-registered*
      coefficient ablations (§2 table) — a bounded, pre-specified rescue path;
    - elevate the Mode B learned gate (logistic, 20-box calibration) per the
      pre-registered contingency (a)–(d) — learns the weighting where the
      proxy is uninformative;
    - full-scale run (complete test splits) to re-test D3 with a
      non-degenerate disagreeing set and to measure RQ4 calibration metrics;
    - backbone ablations (OWL-ViT, YOLOE26) for RQ5.
7.3 **Broader impact** — methodological: pre-registration + diagnostics +
    contingency as a template for honest few-shot domain-adaptation research;
    practical: the stabilization result suggests uncertainty gating can
    *de-risk* fusion even when it cannot improve accuracy.
- **Figures/Tables:** Fig 7.1 — roadmap (pilot → full-scale → ablations →
    Mode B elevation) with decision gates; Table 7.1 — open items mapped to
    pre-registration contingencies.

---

## Appendix A — Artifact map (tables/figures → sources)

| Thesis figure/table | Source artifact |
| --- | --- |
| Fig 3.2 gate surfaces | `src/uadapt/fusion/mode_a_analytic.py` (analytic + BetaGate), replot |
| Fig 5.1 saturation histograms | `outputs/real_data/saturation_analysis/{ladd,dfire}_saturation.png` |
| Fig 5.2/5.7 gate-weight distributions | `scripts/demo_mode_a_end_to_end.py` / `03_run_fusion.py` outputs + `notebooks/supervisor_demo_visualizations.ipynb` |
| Fig 5.3 D1/D2 bins | `outputs/real_data/n100_{analytic,beta}/pooled_diagnostics.json` + `scripts/compute_pooled_diagnostics.py` |
| Fig 5.4–5.6, Tables 5.4–5.5 | `outputs/real_data/ten_seed_protocol{,_beta}/stats.json`; regenerable via `scripts/generate_real_data_report.py --analytic-stats ... --beta-stats ...` |
| Table 5.2 saturation stats | `scripts/analyze_saturation.py` |
| Tables 5.1/5.7, D1–D5 card | `docs/real_data_results_final.md` (fully reproducible) |
| Diagnostic caveats | `docs/change_log.md` 2026-08-05/06/07; `docs/pre_registration.md` §10 |

## Appendix B — Open methodological items before full scale

1. Re-check the D-Fire mirror downloads and LADD license provenance
   (`docs/licenses.md`, change_log 2026-08-06 status note).
2. Decide whether to keep frozen α=β=γ=1 (pre-registered) or run the
   coefficient-ablation rescue path and log it as a deviation.
3. Extend the 10-seed protocol to the full test splits (same driver,
   `--max-seeds 10`).
4. Add ECE/Brier/AUROC (RQ4) to the evaluation script for the full run.
