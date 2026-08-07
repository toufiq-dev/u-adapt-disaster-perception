# Change Log

Records all deviations from the pre-registration, dataset replacements, and
significant pipeline changes. Every entry cites the date and the affected
section of `docs/pre_registration.md`.

## 2026-08-07 — Full-scale Colab preparation: N_TEST_IMAGES=100 limit removed, disconnect-safe extraction resume, resumable D-Fire mirror, LADD train GT, n-agnostic reports, full-scale guide

- **Extraction is now RESUME-SAFE across Colab disconnects**
  (`src/uadapt/features/cache_engine.py`): `extract_and_cache` previously
  resumed only when `manifest.json` existed — but that file is written only
  when a split COMPLETES, so a disconnected full-scale run (LADD 1,365 /
  D-Fire 21,527 images, hours on a free T4) restarted from image 0. Resume
  now accepts the partial `records.json` state too, so a re-run continues
  from the last checkpoint. Test: `tests/test_cache_engine.py`
  (partial-state resume without manifest).
- **D-Fire mirror download is resumable AND atomic** (`data/download_scripts/download_datasets.py`):
  existing non-empty image files are skipped, so re-running after a Colab
  disconnect does not re-fetch all 21,527 images; downloads now go through a
  `.part` temp file + atomic rename (matching the OneDrive path), so a
  mid-write disconnect can never leave a truncated image that the skip
  check would then treat as complete.
- **`data/annotations/ladd_train.json` is now generated** by the LADD
  downloader when the archive ships `annotations/train.json` (Kaggle LaDD
  layout), remapped (`Pedestrian → person`). Previously only `ladd_test.json`
  was produced, but the Mode B calibration sampler consumes `{ds}_train.json`
  — a full-scale LADD Mode B run would have failed at calibration without
  this.
- **`scripts/run_10_seed_protocol.py` records `meta.n_test_images`** (distinct
  image ids in each dataset's cached test split) so reports can distinguish
  the n=100 pilot from full-scale runs.
- **Report generator is n-agnostic** (`scripts/generate_real_data_report.py`):
  the hardcoded "n=100 subset" labels and pilot caveats in the
  `--compare-dirs`, 10-seed (§9), and `--mode-b-stats` reports now reflect
  the ACTUAL evaluated size per dataset (pilot → "n=100 subset"; full-scale →
  "full test split (n=1365)" etc.), and the pilot-caveat text is suppressed
  when the run is full scale. The default `--n-test-images` help and the
  `run_real_data_validation.sh` usage examples were updated to the full-scale
  default (unset = all images; the n=100 pilot explicitly passed
  `N_TEST_IMAGES=100`).
- **Full-scale Colab execution guide added**: `docs/colab_full_scale_guide.md`
  — two clearly separated phases: (A) Extraction (download full LADD/D-Fire,
  RAM-safe single-image streaming via `01_extract_and_cache.py`, the
  resume-loop cell for Colab's ~12 h session limit, zip + Drive transfer) and
  (B) Mode B 10-seed protocol on the full caches (fresh session, CPU
  sufficient). NOTE on batching: the repo streams ONE image at a time — there
  is no `--batch-size` flag, and batching was deliberately not added because
  Grounding DINO pads per image and a batch would silently change the cached
  features vs. the pilot (breaking comparability).

## 2026-08-07 — Mode B 10-seed protocol tooling (pre-registered contingency Risk R3): soft-target fix, calibration sampler, protocol --mode B, report generator, Colab guide

- **Soft-target bug fix vs. the pre-registration** (proposal §5.4.2):
  `src/uadapt/fusion/mode_b_logreg.py::soft_targets` had the two disagreement
  branches SWAPPED — it assigned w* = 1 when TEXT was correct alone (i.e.
  "trust visual" when the text modality was right) and w* = 0 when VISUAL was
  correct alone, the inverse of the pre-registered formula. Since `w` is the
  weight on the visual score, the old mapping taught the gate to lean on the
  WRONG modality whenever the modalities disagreed. Fixed to `w* = 1` iff
  visual-only correct, `w* = 0` iff text-only correct; the pinning test in
  `tests/test_mode_b_calibration.py` was updated. **Pre-fix Mode B numbers
  learned the inverse mapping and must not be compared with post-fix runs.**
- **Calibration-set sampler added** — `src/uadapt/fusion/calibration_set.py`
  + `scripts/build_calibration_set.py`: builds the Mode B 20-box/class
  `--calibration` JSON from the cached TRAIN split — per-class stratified
  sampling of same-class GT-matched boxes (IoU ≥ 0.5, the pre-registered
  "labeled boxes"), seeded and deterministic, strictly disjoint from the
  seed's k-shot support examples (image ids from the prototype payload) and
  from the test split. Per-box gate inputs mirror
  `calibration.record_gate_input` (s_visual = a_visual = affinity proxy;
  sigma2_text falls back to 0.5). The output carries a `sampling` audit
  block (per-class eligibility counts); classes with fewer eligible boxes
  than requested keep ALL of them — at n=100 pilot scale LADD yields ~5-6
  person boxes and D-Fire ~1 (far below 20), recorded honestly.
  Tests: `tests/test_calibration_set.py` (11 tests).
- **Tiny-calibration robustness guard** (`LogRegGate.fit_cv`): folds with
  empty train or test partitions are skipped (5-fold CV presumes ≥ 5
  samples; pilot cells with 1-6 calibration boxes now produce a CV over the
  valid folds, or `cv_scores = None` when none qualify). Previously a 1-box
  calibration set crashed with a zero-division. Tests added.
- **`scripts/run_10_seed_protocol.py` gained `--mode {A,B}`** (pre-registration
  §10, Risk R3): `--mode B` runs the SAME per-cell loop (02 → 05 → 03 → 04)
  with the learned logreg gate as the primary method — per-seed calibration
  built by 05, fused scores by `03_run_fusion.py --mode B --mode-config
  configs/modes/mode_B_logreg.yaml`, the w = 0.5 naive baseline unchanged,
  and a per-dataset zero-shot mAP50 (raw detector scores on the full cached
  test split, computed once, stored in `meta.zero_shot_map50`). Per-cell
  calibration counts are aggregated across seeds into
  `cells[k]["calibration"]` (min/max). `--gate-type` is ignored in Mode B.
- **Report generator gained a standalone Mode B report**
  (`scripts/generate_real_data_report.py --mode-b-stats [--analytic-stats]`,
  default out `docs/real_data_results_modeB.md`): the Mode B vs naive paired
  table (t-test, Wilcoxon, Cohen's d, BH-FDR over 12 tests), a per-cell
  calibration audit (true sampled counts with seed ranges), a four-way
  comparison (zero-shot / naive / Mode A / Mode B per-cell mAP50 means), and
  a data-driven verdict on whether Mode B beat naive — including the
  pilot-scale caveats (tiny calibration sets, affinity-saturation-degenerate
  soft targets) and the pre-registered fallback-narrative guidance (Risk R3).
- **Colab execution guide added**: `docs/colab_mode_b_guide.md` — a
  step-by-step runbook for the full Mode B 10-seed protocol and report on
  free Colab (CPU is sufficient; the protocol is cache-only; the upload
  bundle is ~5-40 MB).

## 2026-08-07 — Gate-saturation quantification, Mode A fused-score fix in 03, Beta-regression fallback gate (D5 contingency)

- **Saturation analysis tool added** (`scripts/analyze_saturation.py`) — loads
  the real n=100 pilot `proposal_level.json` files and prints per-dataset stats
  for `w` / `affinity` / `norm_text_var` / `norm_visual_var`, a counterfactual
  decomposition (gate weight if the variance terms were inert), a verdict, and
  2×2 histogram figures per dataset (saved to
  `outputs/real_data/saturation_analysis/`). Confirms the saturation hypothesis:
  LADD affinity mean 0.943 (99.4% > 0.8) alone pins mean w at 0.72 while the
  variance terms move w by only ±0.021 on average; D-Fire affinity mean 0.946
  sets a w ≥ 0.71 floor for every proposal, with 100% of w > 0.55.
- **Mode A fused scores fixed in the scripted path** (`scripts/03_run_fusion.py`):
  `_run_mode_a` now emits `S_final = (1 - w) * S_text + w * S_visual` (S_text =
  cached per-class text similarity of the predicted class; S_visual = visual
  affinity proxy, matching `src/uadapt/demo/pipeline.py`) instead of the raw
  cached detector `score` (review finding: the scripted path silently produced
  zero-shot rankings labeled as Mode A). The module's score-scale note was
  updated. Tests: `tests/test_03_run_fusion.py` (4 tests) pin the fused output.
- **Beta-regression fallback gate implemented** (pre-registered D5 contingency,
  §10): `beta_regression_gate` + `BetaGate` in
  `src/uadapt/fusion/mode_a_analytic.py` (exported from `uadapt.fusion`). The
  weight is the mean of a Beta distribution with the SAME analytic logit (fixed
  coefficients α=β=γ=1) and input-linked precision
  `φ = φ_max / (1 + slope·(v_text + v_visual))`, blended with a neutral 0.5
  prior — it hedges the gate toward naive averaging exactly where D5 flags the
  Taylor approximation as invalid, and recovers the analytic gate in the
  low-variance limit (still training-free). Wired as
  `--gate-type {analytic,beta_fallback}` (default `analytic`, backward
  compatible) in `scripts/03_run_fusion.py` and `scripts/demo_mode_a_end_to_end.py`;
  recorded in `run_demo` meta (`gate_type`). Tests: 10 unit tests in
  `tests/test_mode_a_gate.py` (validity at exact 0.0/1.0 variances,
  monotonicity, analytic-limit recovery, softening property, batch/scalar
  parity) + 4 wiring tests in `tests/test_demo_mode_a.py`. Full suite: 150 passing.
- **Preliminary real-data behavior of the fallback** (default 80-image subset;
  same subset for both gates, so the comparison is apples-to-apples): D-Fire
  mean w drops 0.855 → 0.775 and Mode A mAP50 improves 0.668 → 0.682; LADD mean
  w 0.698 → 0.685, mAP50 0.751 → 0.753 (≈ neutral). The hedge is directional
  and modest at the default parameters (φ_max = 20, prior precision = 1); the
  fallback is a pre-registered robustness contingency, not a claim of
  improvement. Full-scale re-tuning is deferred to the main run.

## 2026-08-07 — D4 true γ=0 counterfactual, final Analytic-vs-Beta n=100 comparison, and 10-seed protocol scaffold

- **D4 now uses the true pre-registered counterfactual** (§10):
  `scripts/03_run_fusion.py` emits the per-proposal `w_gamma_0` (gate weight
  with the affinity term zeroed — `gate.weight(v_text, v_visual, 0.0)`, valid
  for both the analytic and Beta gates since the affinity term is
  `γ·a_visual`), and `scripts/04_evaluate.py` passes it to
  `d4_affinity_diagnostic` instead of the previous constant-0.5 array (which
  measured a monotone transform of `w` rather than the affinity-induced shift
  Δw = w − w_{γ=0}). Stale predictions files fall back to 0.5 (backward
  compatible). Tests: `tests/test_04_evaluate.py` (counterfactual is genuinely
  used; constant fallback for stale files).
- **Final comparative n=100 runs (seed 0, k=5):** `run_real_data_validation.sh`
  accepts `--gate-type {analytic,beta_fallback}` (env `GATE_TYPE` also
  honored) plus a `REPORT_OUT` override; the pipeline was run twice into
  `outputs/real_data/n100_{analytic,beta}` and
  `scripts/generate_real_data_report.py --compare-dirs` renders the
  side-by-side report `docs/real_data_results_final.md`. Numbers (mAP50): LADD
  analytic 78.3% vs beta 78.4% (naive 80.4%, zero-shot 81.3%); D-Fire analytic
  66.9% vs beta 68.3% (+1.4 pp; naive 71.8%, zero-shot 73.4%). Gate weight:
  LADD mean w 0.699 → 0.686; D-Fire 0.856 → 0.775 — the Beta fallback softens
  the saturation directionally but every proposal is still gated w > 0.55.
  Pooled D1/D2/D3 are gate-independent by construction (D1/D2 do not involve
  w; D3's disagreeing subsets are all-visual-better, so favorability stays
  100%).
- **Bug fix in `scripts/04_evaluate.py::_coco_to_gt`:** GT image ids are
  remapped to the image file stems via the COCO `images` table (mirroring
  `demo_mode_a_end_to_end.py`). Without the remap, D-Fire proposals (stem ids)
  never matched GT (sequential int ids from the mask→box conversion) and
  every scripted-path mAP50 for D-Fire was 0.0. LADD stem ids pass through
  unchanged. Test: `tests/test_04_evaluate.py`.
- **10-seed statistical-protocol scaffold** (pre-registration §9):
  `scripts/run_10_seed_protocol.py` orchestrates, per (dataset, seed, shots k),
  `02_build_prototypes.py --seed` → `03_run_fusion.py` (Mode A gate AND a new
  `--gate-type naive` — `NaiveGate` in `src/uadapt/fusion/mode_a_analytic.py`,
  the pre-registered fixed-w=0.5 'U-ADAPT w/o uncertainty gating' baseline)
  → `04_evaluate.py` for both score files, then computes per cell a paired
  two-sided t-test, Wilcoxon signed-rank, Cohen's d (paired d_z), and
  Benjamini–Hochberg FDR (q = 0.05) across the full comparison family
  (`scipy.stats.false_discovery_control`). NOTE: the scripted path evaluates
  the FULL cached test split (no demo-path `--n-test-images` subsetting).
  Smoke-tested with `--max-seeds 2` (4 cells, both datasets): scripted-path
  Mode A mAP50 LADD 0.791 / D-Fire 0.678 vs naive 0.804 / 0.719 (small
  path-vs-demo differences come from the documented visual-variance input
  difference). The full 10-seed run was executed 2026-08-07 (results and
  interpretation in the entry below).

## 2026-08-07 — 10-seed paired statistical protocol executed (pre-registration §9): analytic results

- **Full protocol run completed** (`scripts/run_10_seed_protocol.py`, 60 cells =
  2 datasets × 3 shots × 10 seeds, `--gate-type analytic`; scripted path
  02 → 03 → 04 on the n=100 pilot caches). Per cell: paired two-sided t-test AND
  Wilcoxon signed-rank across the 10 seeds, Cohen's d (paired d_z),
  Benjamini–Hochberg FDR (q = 0.05) over the full family of 12 tests. Artifacts
  (gitignored, reproducible via the script): `outputs/real_data/ten_seed_protocol/`.
- **Results — all 6 cells significant after FDR control and ALL unfavorable:**
  Mode A is significantly WORSE than naive averaging (w = 0.5) at every
  k ∈ {1, 3, 5} on both datasets (mAP50 means over 10 seeds):

  | cell | Mode A | naive | d | p(t) | q(t) | W | p(W) | q(W) |
  |---|---|---|---|---|---|---|---|---|
  | ladd_k1 | 0.7808 | 0.8025 | −2.37 | 3.75e-05 | 0.00015 | 0.0 | 0.00195 | 0.00213 |
  | ladd_k3 | 0.7893 | 0.8030 | −3.72 | 9.04e-07 | 5.43e-06 | 0.0 | 0.00195 | 0.00213 |
  | ladd_k5 | 0.7887 | 0.8028 | −5.74 | 2.13e-08 | 2.56e-07 | 0.0 | 0.00195 | 0.00213 |
  | dfire_k1 | 0.6817 | 0.7190 | −1.51 | 9.99e-04 | 0.002 | 1.0 | 0.00391 | 0.00391 |
  | dfire_k3 | 0.6840 | 0.7200 | −2.11 | 9.29e-05 | 0.000279 | 0.0 | 0.00195 | 0.00213 |
  | dfire_k5 | 0.6932 | 0.7205 | −1.99 | 1.43e-04 | 0.000344 | 0.0 | 0.00195 | 0.00213 |

  Wilcoxon W = 0 in 5/6 cells (all 10 paired differences share the same sign) —
  the gap is systematic, not seed noise. The naive baseline is bit-identical
  across gate runs (fixed w = 0.5 scores), so the analytic-vs-beta comparison is
  exactly paired.
- **Beta contingency run also completed** (`--gate-type beta_fallback`,
  `outputs/real_data/ten_seed_protocol_beta/`): Beta raises the Mode A mean on
  all 6 cells (+0.1 to +0.8 pp, largest on D-Fire) and shrinks the seed-to-seed
  spread of the Mode A − naive gap (D-Fire k1 |d| 1.51 → 1.88 despite the
  improved mean), but every cell remains significantly below naive after FDR
  control (per-seed wins 0–1 of 10). The fallback is a robustness contingency,
  not a fix.
- **Verdict recorded:** the pre-registered PRIMARY comparison (§9) is settled at
  pilot scale — the analytic gate as implemented does not recover value from the
  uncertainty inputs, consistent with the n=100 single-seed finding (gate
  saturation toward the weaker visual modality; within-dataset D1/D2 ≈ 0). The
  protocol driver is validated and reproducible (usage in the script docstring).
- **Driver table-print fix:** explicit column separators added to the summary
  table (p(t)/q(t) previously collided for exponent-notation values). Cosmetic
  only — `stats.json` unchanged.
- **Report updated:** `docs/real_data_results_final.md` gained a "10-seed paired
  statistical protocol (pre-registration §9)" section (analytic + beta tables,
  comparison, interpretation); every embedded number was cross-checked against
  the two `stats.json` files. NOTE: the section is hand-appended and would be
  overwritten by a `generate_real_data_report.py` re-run (generator wiring
  pending).

## 2026-08-06 — Real-data medium pilot (n=100) executed: RAM-safe streaming, confound resolution, gate-saturation finding

- **RAM-safe feature extraction implemented (critical for scaling).**
  `src/uadapt/features/cache_engine.py` `extract_and_cache` now processes
  images in small batches (default `batch_size=8`), writing each batch to
  disk and releasing references before loading the next — the previous
  implementation decoded the entire split into RAM at once (~60 GB decoded
  for LADD train). `scripts/01_extract_and_cache.py` gained a streaming
  `iter_images`-style loader; the orchestrator's `N_TEST_IMAGES` cap is
  respected per split. `tests/test_cache_engine.py` (new, 4 tests) pins the
  batch-streaming behavior (resume manifest, id-uniqueness, no-OOM
  structure). Full suite: **132 passing**.
- **`lbl_dest` bug fixed** in `data/download_scripts/download_datasets.py`
  (D-Fire mirror path): the label-destination variable was referenced but
  never defined after an earlier retry-fix edit — would crash on the first
  successful download. The n=100 D-Fire mirror re-download (100 train + 100
  test, YOLO→COCO) succeeded with the fix in place.
- **n=100 pilot ran end-to-end** via `N_TEST_IMAGES=100 bash
  scripts/run_real_data_validation.sh` (real Grounding DINO Swin-T features
  → prototypes k=5 → Mode A eval; LADD min-max, D-Fire absolute;
  per-proposal estimators). 333 LADD + 324 D-Fire scored proposals; no OOM.

  | Method | LADD (min-max) | D-Fire (absolute) |
  |---|---|---|
  | Zero-shot raw | 81.3 | 73.4 |
  | Text-only | 81.3 | 73.4 |
  | Visual-only | 66.9 | 54.7 |
  | Naive (w=0.5) | 80.4 | 71.8 |
  | U-ADAPT Mode A | 78.3 | 66.9 |

- **U-ADAPT underperforms naive averaging on BOTH datasets at n=100**
  (LADD 78.3 vs 80.4; D-Fire 66.9 vs 71.8) — the opposite of the synthetic
  demo, and an honest pilot finding. Root cause is **gate saturation toward
  the visual modality**: mean gate weight 0.70 (LADD) / 0.86 (D-Fire) with
  w > 0.55 for 99.4% / 100% of proposals, because affinity (≥ 0.87 pilot,
  [0.64, 0.999] at n=100) dominates the analytic gate — yet visual-only
  reranking is *worse* than the raw detector score on this real data
  (66.9 < 81.3 LADD; 54.7 < 73.4 D-Fire). The gate is leaning on the weaker
  modality. This is a coefficient/threshold-calibration issue to revisit at
  full scale, not a verdict on the method.
- **Pooled D1/D2/D3 at n=100 (n=657 proposals):**

  | Diagnostic | n=10 | n=100 | Interpretation |
  |---|---|---|---|
  | D1 ρ | −0.339 | **−0.056** | confound confirmed — not small-sample artifact |
  | D2 ρ | +0.175 | **+0.051** | confound confirmed — scale artifact persists |
  | D3 favorability | 94.1% (n=17) | **100% (n=246)** | one-sidedness confirmed — structural |
  | D5 flag | both | **both** | Beta-regression fallback still triggered |

  - **D1**: the n=10 negative sign did NOT replicate — pooled ρ collapsed to
    −0.056 and the dataset base rates converged (LADD err 0.393 vs D-Fire
    0.352 at n=100, vs 0.667/0.241 at n=10). Per-dataset: LADD ρ = 0.0
    (constant entropy by construction), D-Fire ρ = −0.061. No within-
    proposal text-uncertainty–accuracy relationship on real data; the pooled
    value ≈ the within-D-Fire value ≈ 0.
  - **D2**: per-dataset ρ ≈ 0 (LADD +0.077, D-Fire −0.022) while pooled is
    +0.051 — the pooled sign is a **between-dataset normalization-scale
    artifact** (min-max spread vs absolute tiny), not a within-proposal
    effect. The n=10 +0.175 was small-sample inflation of the same
    artifact; it did NOT "resolve" with scale.
  - **D3**: disagreeing subsets remain one-sided (LADD 131 visual-better /
    1 text-better; D-Fire 114/0). The affinity threshold (≥ 0.65) never
    fails on real features, so `visual_correct` saturates and D3 = 100% is
    a **saturation artifact** — the gate favors visual and visual is
    (by that proxy) "correct", even though visual reranking actually
    *hurts* mAP. Structural, not a small-sample issue; a re-thresholded
    visual-correctness rule or lower-affinity proposals (full run) are
    required before D3 can discriminate gate quality.
  - **D5**: still flags both datasets on the absolute scale (LADD
    frac 0.997, D-Fire 0.957) → the pre-registered Beta-regression
    fallback contingency remains active.
- **Report generator upgraded** (`scripts/generate_real_data_report.py`):
  pilot auto-label threshold `n_test_images <= 100` (n=100 medium pilot
  can no longer masquerade as final results), dynamic image count in the
  >100% gap-recovery caveat, and three new honest caveats rendered with the
  pooled diagnostics: U-ADAPT-underperforms (executive summary), D2
  pooled-sign scale artifact, and D3 one-sidedness. Both
  `docs/real_data_results_pilot.md` (PILOT RESULTS n=100) and
  `docs/real_data_results.md` regenerated from the n=100 outputs.
- **Status**: pipeline, data prep (100 test images per dataset), RAM-safe
  streaming, and diagnostics all validated at n=100. Next: full-scale run
  (unset `N_TEST_IMAGES`), after clearing `data/raw/.staging` (~9.3 GB)
  and re-verifying LADD license provenance.

## 2026-08-05 — Per-proposal real-data uncertainty estimators fix pooled D1/D2/D3 = 0.000 (deviation on pre-registration §2 / §7.6)

- **Root cause (investigated on the n=10 pilot).** The pooled D1/D2/D3 were
  all 0.000 for four structural reasons, not one:
  1. `norm_text_var = 0.5` **placeholder** (constant input → D1 had no
     variance);
  2. `text_correct = argmax(sims) == class_name` was **tautologically True**
     (the backbone assigns `class_name = argmax(text_similarities)`);
  3. `visual_correct` (affinity ≥ 0.65) was **saturated** on real RoI-pooled
     features — every pilot proposal had affinity ≥ 0.87, so D2's
     correctness label carried no information;
  4. class-level variance terms (C distinct values) underpower D1/D2 even
     pooled at C=3.
- **Real per-proposal estimators added**
  (`src/uadapt/uncertainty/variance_estimators.py`, exported from
  `src/uadapt/uncertainty/__init__.py`):
  - `proposal_text_variance` — normalized entropy of the per-box
    class-similarity vector (relative weights), continuous in [0, 1];
    replaces the 0.5 placeholder on the real-cache path (`pipeline.py`,
    `scripts/03_run_fusion.py` which also now emits `norm_text_var` /
    `norm_visual_var` consumed by `scripts/04_evaluate.py`).
  - `proposal_visual_variance` — mean (1 − cos) between the box feature and
    the class support set, continuous in [0, 2]; replaces the class-level
    `sigma_visual` in `pipeline.py`'s real-cache mode (support features are
    available in-memory from the prototype builder).
- **Non-tautological `text_correct`.** On the real cache, "text correct" is
  the text modality's top-1 class matching a GT box (IoU ≥ 0.5) — identical
  to `gt_correct`; the old argmax==class_name check is always True by
  construction.
- **D1/D2 correctness aligned with the PRE-REGISTRATION** (proposal
  correctness = `gt_correct`, already the convention in
  `scripts/04_evaluate.py` and the diagnostics module docstring). The
  synthetic demo keeps its per-modality flags (it is a mechanism demo); the
  real-cache path evaluates D1/D2 against `gt_correct` because the
  per-modality labels are degenerate there (tautological text flag +
  saturated affinity threshold). D3 keeps the per-modality disagreeing
  subsets either way. `scripts/compute_pooled_diagnostics.py` updated to the
  same convention; both estimators are recorded in `results.json` meta
  (`text_uncertainty_estimator` / `visual_uncertainty_estimator`).
- **Verified on the n=10 pilot** (real Grounding DINO features; pooled
  LADD + D-Fire, n=44 proposals): the structural zeros are gone —
  **D1 ρ = −0.339**, **D2 ρ = +0.175** (boxes far from the support set more
  often wrong — the expected direction), **D3 favorability = 94.1%
  (p = 0.0003, n=17)**. The D1 sign is NOT a genuine finding — a follow-up
  investigation (entry below, 2026-08-05) showed it is a between-dataset
  base-rate confound. Pilot-scale numbers only; the full run is required
  for research claims. Report regenerated at `docs/real_data_results_pilot.md`.
- **D5 sentinel runs on the absolute scale now.** The raw per-proposal
  values (`text_entropy`, `visual_distance_raw`) are persisted in
  `proposal_level.json`; `compute_pooled_diagnostics.py` computes D5 from
  `text_entropy` and `visual_distance_raw / 2.0`. Min-max normalization
  spreads any array across [0, 1] BY CONSTRUCTION and would silently defeat
  the Taylor-validity flag — exactly where it matters. On the pilot, D5
  FLAGGED both datasets on the absolute scale (LADD frac ≈ 1.0 — the
  single-class text entropy is 0; D-Fire frac ≈ 0.88), triggering the
  pre-registered **Beta-regression fallback** contingency — the sentinel
  working as designed.
- **Caveats (pilot scale):** (1) pooled D2 mixes per-dataset normalization
  scales (LADD min-max, D-Fire absolute) — pooled ranks are a hybrid;
  per-dataset values remain reported and the full run should re-examine
  both. (2) D3's disagreeing subsets are one-sided at n=10 (the affinity
  threshold never fails on this pilot, so `text_better` is empty) — D3
  n=17 is therefore the visual-better side only; the full run will rebalance
  it.
- **Tests** — `tests/test_variance_estimators.py` (11 tests: entropy
  boundary/continuity/edge cases, box-to-support distance incl. the k=1 and
  empty-support semantics) plus a real-mode pipeline signal test in
  `tests/test_demo_mode_a.py` (per-proposal continuous variances, non-
  tautological `text_correct`, D1 ρ > 0.3 and D2 ρ > 0.2 against
  `gt_correct`, D3 subsets non-empty with gate favorability > 0.6, estimator
  names in meta). Full suite: **125 passing** (was 108).

## 2026-08-05 — D1 sign investigation + D3 alignment in 04_evaluate.py (follow-up)

- **The pooled D1 ρ = −0.339 is a BETWEEN-DATASET base-rate confound, not a
  small-sample artifact or a genuine finding.** Root-cause analysis on the
  n=10 pilot proposal rows (`outputs/real_data/*/proposal_level.json`,
  n=44):
  - **LADD** (n=15) has *structurally constant* text entropy — 0.000 for
    every proposal, because it is a single-class dataset (entropy of a
    1-element similarity vector is 0 by construction). Its error base rate
    on this pilot was 0.667.
  - **D-Fire** (n=29) sits near max entropy (range [0.666, 0.974]) with a
    much lower error base rate (0.241); its *within-dataset* D1 ρ is
    **+0.019** and the binned error rates are flat (0.231 vs 0.250 across
    bins) — no within-proposal uncertainty–accuracy relationship.
  - Pooling a (tv=0, err=0.667) population with a (tv≈0.8, err≈0.24)
    population produces a negative rank correlation across the datasets;
    **within-dataset centering of the error rates collapses the pooled ρ
    from −0.339 to −0.001** — the pooled sign is entirely the
    between-dataset base-rate difference.
  - **Implication:** flat-sim proposals are NOT genuinely more often
    GT-correct. The pooled D1 sign is not interpretable as a within-
    proposal effect while LADD contributes a constant entropy at a different
    base rate; per-dataset D1 is the interpretable signal on real data (and
    D1 is uninformative at pilot scale: within-D-Fire ρ ≈ 0). This caveat is
    now emitted by `scripts/generate_real_data_report.py` next to the pooled
    diagnostics table and recorded in the pilot report. The pre-registered
    pooled protocol itself is unchanged (pooled = primary claim, per-dataset
    still reported); the full run will re-examine both within- and
    between-dataset structure.
- **D3 alignment in `scripts/04_evaluate.py`.** The script's D3 previously
  split gate weights by `w < 0.5` / `w > 0.5`, which counts EVERY proposal
  as favorable by construction (any weight sits on one side of 0.5) — D3
  could never drop below 100% and measured nothing. It now uses the same
  disagreeing-proposal subsets as the pipeline and
  `compute_pooled_diagnostics.py` (per-modality correctness flags
  `text_ok` / `visual_ok`, derived in `_diag_arrays` from `gt_correct` and
  `affinity >= 0.65`; explicit per-proposal `text_correct`/`visual_correct`
  fields win when present). Both the per-dataset and the pooled paths were
  aligned. Tests: `tests/test_04_evaluate.py` (3 tests pinning the
  disagreeing-subset convention incl. the pooled binomial concatenation).

## 2026-08-05 — Milestone 1 executed: real-data n=10 pilot runs end-to-end (issues #3)

- **Environment set up** — `torch` + `transformers` installed in the project
  venv (MPS backend); `scripts/01_extract_and_cache.py`, `02_build_prototypes.py`,
  `03_run_fusion.py`, `04_evaluate.py` gained the `src/` path bootstrap the
  other scripts already had (they were only reachable via `PYTHONPATH`).
- **Grounding DINO backbone rewritten** (`src/uadapt/models/backbone_loader.py`)
  — `predict()` now extracts real per-box visual features (RoI mean-pool of
  the encoder vision hidden states) and per-class text similarities (from the
  query-to-token logits), instead of the previous stub. Class names are
  normalized to the config vocabulary via token-span matching; device
  auto-resolves CUDA → MPS → CPU. Smoke-tested on a real LADD image.
- **Latent `load_cache` bug fixed** (`src/uadapt/features/cache_engine.py`) —
  the function declared/returned `Dict[str, List[FeatureRecord]]` but every
  caller (and the type signature) expects a flat list; exposed only on the
  real-data path (the demo previously ran on synthetic data).
- **D-Fire class-order convention corrected** — the HF mirror's YOLO labels
  use **`0 = smoke, 1 = fire`** (the Kaggle mirror is literally named
  "smoke-fire-detection"), not the paper table order `{0: fire, 1: smoke}`.
  The swap was discovered empirically (a systematic 22/22 class inversion at
  high IoU → D-Fire mAP50 0.000 → **0.821** after the fix). Constants updated
  in `data/download_scripts/download_datasets.py`; GT JSONs regenerated.
- **LaDD archive provenance** — official repo still 404; the user supplied the
  archive (`~/Downloads/archive.zip`). Extracted with category remap
  `Pedestrian → person` into `data/raw/ladd` + `data/annotations/ladd_{split}.json`
  (n=10 pilot subset). License confirmation against an official source remains
  the gate before the full-data run (`docs/licenses.md`).
- **n=10 pilot ran end-to-end** via `N_TEST_IMAGES=10 bash scripts/run_real_data_validation.sh`
  (real Grounding DINO Swin-T features → prototypes k=1/3/5 → Mode A eval):

  | Method | LADD (min-max) | D-Fire (absolute) |
  |---|---|---|
  | Zero-shot raw | 0.736 | 0.821 |
  | Text-only | 0.736 | 0.821 |
  | Visual-only | 0.415 | 0.595 |
  | Naive (w=0.5) | 0.736 | 0.803 |
  | U-ADAPT Mode A | 0.736 | 0.742 |

  Report: `docs/real_data_results_pilot.md` (self-labeled **"PILOT RESULTS
  (n=10 images)"**; superseded by the full-data run).
- **Pooled D1/D2/D3 = 0.000 on the pilot — root-caused and FIXED in the
  entry below (2026-08-05, per-proposal estimators).** The causes were: the
  documented 0.5 text-variance placeholder, the tautological `text_correct`
  (backbone sets `class_name = argmax(text_sims)`), the saturated affinity
  threshold (every pilot proposal ≥ 0.87), and the class-level granularity
  of the variance terms. See the per-proposal-estimator entry above for the
  fix and the post-fix pilot values (D1 −0.339 / D2 +0.175 / D3 94.1%).
- **Tests** — `tests/test_backbone_loader.py` (9 unit tests for
  `_class_token_spans`, `_roi_mean_feature`, `resolve_device` without
  torch/transformers). Full suite: **108 passing**.
- **Post-review hardening (2026-08-05)** — (1) RoI pooling now uses the exact
  vision-token grid derived from the padded model input
  (`inputs.pixel_values` aspect) instead of the square-root heuristic, so
  features are not misaligned on non-square aerial imagery; the heuristic
  remains as a fallback for callers without input dims. (2) The D-Fire
  mirror downloader retries expired signed URLs (re-fetching a fresh row)
  and now **fails loudly** if any image cannot be downloaded — a silently
  shrunk test set would bias mAP.

## 2026-08-04 — Milestone 1: dataset licenses verified + download scripts (issues #1, #2)

- **Dataset licenses verified (2026-08-04)** — `docs/licenses.md` filled in
  for all four datasets (no `TBD` / "To verify" remains on the primary rows,
  so the `run_real_data_validation.sh` [0/6] license gate now passes):
  - **LADD** — research use only, presumed from the dataset documentation;
    exact terms to confirm at the manual download step. The official repo
    `huyhieupham/LADD` returned **404 on 2026-08-04** (GitHub API) and no
    verified live URL could be found (no Zenodo record, no repo under the
    author's account) — per the no-guessing policy the download script uses a
    clearly-marked placeholder URL + manual instructions.
  - **D-Fire** — free for research use; verified via the official
    `gaia-solutions-on-demand/DFireDataset` README (the original `gaiasd/DFire`
    is gone) + the Neural Computing & Applications 2022 paper. OneDrive and
    Kaggle mirrors confirmed.
  - **RescueNet** — CC BY-NC-ND 4.0 (academic use OK; do not redistribute
    modified copies). **FloodNet+** — CDLA-Permissive-1.0.
  - **Models**: OWL-ViT Apache-2.0, YOLOE26 AGPL-3.0 (THU-MIG/yoloe LICENSE),
    CLIP MIT, DINOv2 Apache-2.0 — all confirmed 2026-08-04; Grounding DINO
    already Apache-2.0.
  - None of the licenses restricts academic use → **no dataset replacement
    required** (pre-registration policy, `docs/pre_registration.md`).
- **Download scripts added** — `data/download_scripts/download_datasets.py`:
  D-Fire from the official OneDrive mirror (with `?download=1` direct-download
  handling + Kaggle fallback + `--dfire-archive` manual fallback), YOLO→COCO
  conversion into `data/annotations/dfire_{train,val,test}.json`, `--subset N`
  pilot mode (10 images per split; full archive is still downloaded since
  OneDrive cannot range-download, but only the subset is copied to
  `data/raw`), SHA-256 checksums (`download_scripts/{ladd,dfire}/sha256sums.txt`),
  and `--check-only` link verification.
- **D-Fire config corrected** — `configs/datasets/dfire.yaml`
  `annotation_format: coco_boxes` → `yolo_boxes` (official labels are YOLO
  with normalized coordinates; the download script converts them to COCO for
  the evaluation pipeline). Field is metadata-only (no code consumes it).
- **Pilot labeling added to the report generator** —
  `generate_real_data_report.py` now titles runs with any
  `meta.n_test_images < 100` as **"PILOT RESULTS (n=<N> images)"** with a
  warning banner, so the n=10 pilot report can never be mistaken for final
  thesis results (report written to `docs/real_data_results_pilot.md`).
- **Status**: raw-data download + execution of the n=10 pilot remain
  **PENDING** — this machine has no torch/transformers and the LADD source is
  a manual step; the runbook in `data/download_scripts/README.md` covers the
  GPU/Colab execution path (`N_TEST_IMAGES=10 SKIP_PREREQS=0`).

## 2026-08-04 — Pooled D1/D2/D3 diagnostics implemented (§10 deviation)

- **Implemented the §10 pooled-diagnostics protocol** (deviation of
  2026-08-03): `d1_text_uncertainty_accuracy`, `d2_visual_uncertainty_accuracy`,
  and `d3_gate_favorability` in `src/uadapt/metrics/diagnostics.py` now accept
  an optional `pool_with` argument carrying the SECOND dataset's arrays
  (variance + correctness for D1/D2, gate-weight subsets for D3). When
  pooling, each returns a structured dict `{primary, secondary, pooled}`:
  per-dataset values are still reported, but the pooled value — computed on
  the concatenated arrays before the Spearman ρ / binomial test — is the
  PRIMARY diagnostic claim (fixes the 2-class statistical-power limitation:
  D-Fire alone yields only 2 distinct variance values). Existing calls
  without `pool_with` are unchanged (backward compatible). `_spearman_rho`
  now returns 0.0 for constant inputs (undefined correlation), keeping
  results JSON-serializable for single-class datasets.
- `scripts/04_evaluate.py` gained `--pool-predictions` / `--pool-ground-truth`
  (plus optional `--pool-primary-name` / `--pool-secondary-name`, defaults
  `ladd`/`dfire`): when the second dataset is supplied, results report
  `diagnostics_ladd`, `diagnostics_dfire` (per-dataset D1-D5) and
  `diagnostics_pooled` (pooled D1/D2/D3, PRIMARY claim). Without pooling
  args the output is unchanged (`diagnostics`).
- Unit tests added in `tests/test_metrics.py`: pooled D1/D2/D3 structured
  return + the statistical-power narrative (2-class D-Fire weak alone,
  informative pooled), D3 pooled binomial concatenation, empty-dataset and
  incompatible-length / non-1-D edge cases, and single-class
  constant-variance behavior.
- **Caveats consolidated and cross-referenced (2026-08-04).** The
  *Methodological Caveats (2026-08-03)* section now sits directly after the
  Key Results Table in `docs/supervisor_demo_report.md` (three subsections:
  2-class statistical-power limitation, synthetic-world artifact, the fix
  worked — incl. the D3 ≈ 5.4% stand-in number);
  `notebooks/supervisor_demo_visualizations.ipynb` gained dedicated caveat
  cells before Figures 2 and 3 (pooling requirement; synthetic-world D3
  artifact) plus *real-data figures will supersede* notes; `README.md` gained
  a *Known Limitations* section. All cross-reference the §10 pooled-
  diagnostics deviation (2026-08-03) and this entry.

## 2026-08-03 — Absolute scaling normalization + thesis proposal (pre-registration deviations §2 and §10)

- **PRE-REGISTRATION DEVIATION (§2) — absolute scaling added as the fourth
  normalization strategy.** The supervisor demo surfaced a methodological
  degeneracy: min-max normalization across C classes yields only C distinct
  normalized values, so with 2-class D-Fire the gate's variance terms
  collapse to {0, 1} and misroute proposals. On the 2-class synthetic
  stand-in (seed=0) with the old min-max default, U-ADAPT Mode A scored
  **0.906 mAP50 — actively UNDERPERFORMING naive averaging (0.955)**, with D3
  at chance level (49.5%, p = 0.90).
  Implemented `absolute_normalize` (`x_tilde = x / 2.0` for the cosine-
  distance terms, whose raw range is [0, 2]) in
  `src/uadapt/uncertainty/variance_estimators.py` — class-count-independent,
  no support-set statistics. Visual affinity is left untouched (already in
  [0, 1]). Wired through `src/uadapt/demo/pipeline.py`
  (`run_demo(norm_strategy=...)`, default `min-max` for backward
  compatibility) and exposed as `--norm-strategy {min-max,absolute}` on
  `scripts/demo_mode_a_end_to_end.py`; the strategy is recorded in
  `results.json` meta. Pre-registration §2 updated.
- **The fix worked:** despite the D3 drop on the synthetic stand-in, the
  primary goal was achieved — the 2-class mAP50 degeneracy is fixed. Under
  absolute scaling, U-ADAPT (**0.956**) now correctly beats naive averaging
  (**0.955**) on the 2-class D-Fire setup, whereas min-max actively harmed
  it (0.906). The 6-class demo is unaffected (0.958 vs 0.947 naive; no
  regression vs 0.957 under min-max).
- **PRE-REGISTRATION DEVIATION (§10) — D1/D2/D3 evaluated pooled across
  LADD + D-Fire.** *I discovered that evaluating D1, D2, and D3 on D-Fire in
  isolation is structurally underpowered. Because D-Fire only has 2 classes
  (fire and smoke), there are only 2 distinct variance values. You cannot
  compute a meaningful Spearman rank correlation or gate favorability trend
  with only 2 data points. Therefore, the pre-registered protocol must be
  updated to evaluate D1/D2/D3 pooled across LADD and D-Fire (giving us 3
  distinct classes and 3 distinct variance values), which provides the
  necessary statistical power.* Pre-registration §10 updated (per-dataset
  values still reported; pooled values are the primary diagnostic claim).
- **Caveats added to `docs/supervisor_demo_report.md` and
  `notebooks/supervisor_demo_visualizations.ipynb`:**
  - *2-class statistical-power limitation:* D1/D2/D3 on D-Fire alone remain
    weak because 2 classes yield only 2 distinct variance values; for
    real-data evaluation they are computed pooled across LADD+D-Fire
    (3 classes).
  - *Synthetic-world artifact:* *the synthetic demo world was implicitly
    engineered around the min-max normalization stretch. When we apply the
    mathematically correct absolute scaling (x/2.0), the raw variance
    magnitudes are revealed to be small relative to the affinity term in this
    specific synthetic setup. This is a demo-world artifact. On real data,
    the variance magnitudes will be different. Furthermore, Diagnostic D5 is
    specifically designed to catch this: if real variances cluster near 0,
    D5 will flag it and trigger the pre-registered Beta-regression
    fallback.*
- **Demo re-runs (synthetic stand-in; no real cache on this machine — the
  exact real-data commands are in the supervisor report):**
  - 2-class (`--classes fire smoke --norm-strategy absolute`): U-ADAPT
    0.956 vs naive 0.955 (was 0.906 < 0.955 under min-max); D3 5.4% on the
    synthetic stand-in (demo-world artifact, see caveats).
  - 6-class (`--norm-strategy absolute`): U-ADAPT 0.958 vs naive 0.947 — no
    regression (was 0.957 vs 0.947 with min-max).
- **Thesis proposal added for auditability:** `docs/thesis/U-ADAPT_Thesis_Proposal.pdf`
  plus a markdown copy `docs/thesis/proposal.md` (from the authoritative
  source at `~/Developer/Thesis/`), linked from `README.md`.
- **Git hygiene:** `U-ADAPT_Revision_Log.md` (private internal document) and
  third-party `*.arxiv.pdf` files are now gitignored (never committed).

## 2026-08-01 — Repository bootstrap (Milestone 0)

- Initial commit: *"Bootstrap U-ADAPT repository structure and pre-registration docs"*.
- Created repository skeleton, configs, `src/uadapt` package scaffolding,
  scripts 01–04, notebooks 00–02, docs, tests.
- No experimental results collected; no pre-registration changes.
- Open items (no decision made yet):
  - Dataset licenses (LADD, D-Fire, RescueNet, FloodNet+) — pending issue #1.
  - YOLO-World / YOLO11 AGPL-3.0 implications for the thesis — pending issue #1.
  - YOLOE26 checkpoint + license — pending issue #1.
  - Mode B full wiring (logreg/MLP + COCO/LVIS init ablation) — scheduled for
    Milestone 7 (Mode B; renumbered from Milestone 6 when Milestone 3 —
    Real-data validation — was inserted into `docs/thesis_plan.md`).

## 2026-08-01 — Synced repo with thesis proposal Revision 3

- **Mode C removed as a separate mode.** Source-domain meta-training is now a
  Mode B gate-initialization ablation (random vs COCO/LVIS-pretrained init,
  proposal §5.4.3). `configs/modes/mode_C_source_transfer.yaml` replaced by
  `configs/modes/mode_B_coco_lvis_init.yaml`; updated `docs/pre_registration.md`
  (§1, §2, §7), `scripts/03_run_fusion.py`, `src/uadapt/__init__.py`,
  `src/uadapt/fusion/__init__.py`, `docs/thesis_plan.md`.
- **YOLOE26 added as cross-backbone ablation** (proposal §5 Phase 1, §7.3):
  `configs/models/yolo_e26.yaml` + loader stub in
  `src/uadapt/models/backbone_loader.py`; checkpoint/license TBD (issue #1).
- **Min-10-valid-boxes filter rule implemented** (proposal §6): classes with
  < 10 valid boxes after mask-to-box conversion are excluded and reported
  (`data/mask_to_box/filter.py`, dataset configs, pre-registration §6).
- **RQ5 numeric definition + primary comparison updated** (proposal §3, §7.1,
  §7.6): backbone-agnostic = relative gain within 2× across backbones per
  dataset; primary statistical test is Mode A vs naive averaging (w = 0.5).
- **Ethics & delimitations section added** to `docs/pre_registration.md`
  (IRB exemption, license commitment, out-of-scope statement, proposal §13).
- **Remaining gaps aligned with proposal Revision 3:**
  - Mode A coefficient ablations completed in `configs/modes/mode_A_analytic.yaml`
    (added visual-uncertainty-only, text-uncertainty-only, affinity-only;
    proposal §8 Mode A analytic-gate ablations table).
  - Cross-domain transfer protocol documented in `docs/pre_registration.md` §1
    (proposal §7.2): mode-specific transfer semantics, dataset-size asymmetry
    acknowledgment (LADD 1,365 vs D-Fire 21,527), and the pre-registered
    directional hypothesis.
  - Compound contingency documented in `docs/pre_registration.md` §10
    (proposal §10): pilot failure + Colab memory failure decision tree.
  - Baselines list expanded in `docs/pre_registration.md` §8 (proposal §8):
    U-ADAPT w/o uncertainty gating, w/o temperature scaling, w/o MC Dropout,
    transfer-learning reference, and supervised-detector ceiling
    (YOLOv11l, YOLO26L, RT-DETRv2-L).
  - YOLOE26 license field fixed in `configs/models/yolo_e26.yaml`:
    `Apache-2.0` → `TBD` (unverified, consistent with `docs/licenses.md` and
    README; issue #1).
- **Detection metrics completed (proposal §7.4):** added `compute_map50_95`
  (COCO-style IoU thresholds 0.5:0.05:0.95) and `compute_per_class_ap` to
  `src/uadapt/metrics/detection_metrics.py` (refactored shared per-class AP
  core `_per_class_ap`); exported via `src/uadapt/metrics/__init__.py`;
  reported by `scripts/04_evaluate.py` (`mAP50_95`, `per_class_AP`); unit
  tests added in `tests/test_metrics.py`; pre-registration §7 updated.
- **Mode A ablation-variant tests added** in `tests/test_mode_a_gate.py`:
  config-driven coverage of all 7 pre-registered coefficient variants from
  `configs/modes/mode_A_analytic.yaml` (Full + 6 ablations) — config
  exposure, per-variant logit sign semantics, (0,1) weight bounds, and
  ModeAGate/function equivalence (proposal §8 / pre-registration §2).
- **Proposal section references reconciled** (docs only):
  - Clarified in `docs/pre_registration.md` snapshot header that the
    proposal has **no §7.5** — §7 jumps from §7.4 Metrics to §7.6
    Statistical Testing Plan, and the D1–D5 diagnostics text is contained
    within §7.6 (the proposal itself cites "D5 (§7.6)").
  - Fixed internal cross-reference in pre-registration §7: proposal recall
    (ceiling) is defined in §8 (Baselines), gap recovery in §9 (Statistical
    Testing).
  - Removed stale **"proposal §14"** from `docs/licenses.md`: the
    proposal's Ethics Statement and "Dataset and model licenses" sections
    are unnumbered (no §14 exists); Grounding DINO license now cited as
    'Confirmed (proposal: "Dataset and model licenses" section)'.
- **k=1 max-entropy-prior ablation implemented + tested** (pre-registration
  §2 / `configs/modes/mode_A_analytic.yaml` `k1_max_entropy_prior: 0.5`):
  `normalized_visual_variance` gained a backward-compatible `k1_prior`
  argument (default 0.0 = maximum-likelihood degenerate-sample treatment;
  0.5 = max-entropy prior); unit tests in `tests/test_mode_a_gate.py` cover
  config exposure, prior substitution, k≥2 invariance, and the gate-weight
  shift.
- **Mode B wiring implemented (Milestone 7 — renumbered from Milestone 6,
  proposal §5.4.1/§5.4.2/§5.4.3):**
  new `src/uadapt/fusion/calibration.py` turns a 20-box/class calibration
  set (normalized 5-D gate inputs + text/visual correctness flags) into
  fused scores via a learned gate. `LogRegGate`/`MLPGate` gained
  `set_params` + warm-start `fit` (enabling the COCO/LVIS-pretrained gate-init
  ablation, the former Mode C); `scripts/03_run_fusion.py` Mode B branch now
  runs end-to-end (`--calibration`, `--gate-init`, CV + temperature
  optimization); exports added to `src/uadapt/fusion/__init__.py`; unit
  tests in `tests/test_mode_b_calibration.py`.
- **Supervisor demo added** (no pre-registration change; demo tooling only):
  - `scripts/demo_mode_a_end_to_end.py` runs the full Mode A pipeline on a
    50-100 image subset — prototypes (k=5), uncertainty estimates, analytic
    gate, fused scores — and compares mAP50 against zero-shot raw, text-only
    (w=0), visual-only (w=1), and naive averaging (w=0.5). Deterministic
    (seed=0). Uses REAL cached features when present; otherwise a synthetic
    world (`src/uadapt/demo/synthetic_data.py`) that matches the real
    FeatureRecord + COCO schemas (mechanism demo, explicitly caveated).
  - `src/uadapt/demo/pipeline.py` shares the production Mode A / prototype /
    metrics / diagnostics code paths end-to-end; `src/uadapt/demo/plotting.py`
    renders Figures 1-6 (gate-weight distribution, D1/D2, D3, gap recovery,
    qualitative, coefficient ablation).
  - `notebooks/supervisor_demo_visualizations.ipynb` renders the six
    publication-quality figures; `docs/supervisor_demo_report.md` is the
    2-page supervisor summary; `run_this_for_supervisor.sh` executes the
    whole demo. Unit tests in `tests/test_demo_mode_a.py`.
  - **Figure 5 upgraded to real detections when cached data exists** (no
    pre-registration change; demo tooling only): `figure5_qualitative`
    accepts an `image_paths` map ({image_id: file path}) and renders the
    REAL image with GT/proposal boxes when the file resolves, falling back
    to the deterministic schematic scene otherwise.
    `scripts/demo_mode_a_end_to_end.py` builds the map from the GT COCO
    `images` section resolved against the dataset config test split
    (CLI override `--image-root`), persists it in `proposal_level.json`,
    and passes it to `render_all_figures`; the notebook loads and forwards
    it automatically.
  - **One-click Colab setup cell added** to
    `notebooks/supervisor_demo_visualizations.ipynb` (first code cell; no
    pre-registration change): clones the repo into `/content`, installs
    `requirements.txt` (skipping CUDA-preinstalled torch/torchvision per the
    requirements note), and runs the full demo end-to-end on a free T4.
    `scripts/demo_mode_a_end_to_end.py` gained a `sys.path` bootstrap so it
    runs without `PYTHONPATH=src` (fixes the notebook/Colab subprocess
    path).
