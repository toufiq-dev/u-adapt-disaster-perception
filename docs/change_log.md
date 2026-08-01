# Change Log

Records all deviations from the pre-registration, dataset replacements, and
significant pipeline changes. Every entry cites the date and the affected
section of `docs/pre_registration.md`.

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
