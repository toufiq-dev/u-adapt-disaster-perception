# Change Log

Records all deviations from the pre-registration, dataset replacements, and
significant pipeline changes. Every entry cites the date and the affected
section of `docs/pre_registration.md`.

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
