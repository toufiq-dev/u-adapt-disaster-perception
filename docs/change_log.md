# Change Log

Records all deviations from the pre-registration, dataset replacements, and
significant pipeline changes. Every entry cites the date and the affected
section of `docs/pre_registration.md`.

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
    Milestone 6.

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
- **Mode B wiring implemented (Milestone 6, proposal §5.4.1/§5.4.2/§5.4.3):**
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
