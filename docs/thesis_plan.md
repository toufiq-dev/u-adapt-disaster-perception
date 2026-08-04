# Thesis Plan — 12 Weeks

Mapped from the proposal timeline (§11) into repository milestones M0–M10.
Milestone 3 (*Real-data validation and analysis*) was inserted after the Colab
pilot; the former M3–M9 are renumbered M4–M10.
GitHub milestones group the issues (see issues list); each row lists the
primary deliverables and the issues that track them.

## Overview

| Week | GitHub milestone | Repository milestone | Core focus |
|------|------------------|----------------------|------------|
| 1–2 | Week 1-2 Setup and Pre-registration | M0, M1, M2 | Repo, licenses, datasets, pilot, freeze class list |
| 3–4 | Week 3-4 Core Pipeline | M3, M4, M5 | Real-data validation, baselines, caching, prototypes |
| 5–6 | Week 5-6 Gating Modes | M6, M7 | Mode A, Mode B |
| 7–8 | Week 7-8 Experiments and Diagnostics | M8, M9 | Baselines/ablations, D1–D5, cross-domain |
| 9–12 | Week 9-12 Thesis Writing | M10 | Analysis, figures, tables, writing |

## Milestones

### Milestone 0 — Repository and environment setup (Weeks 1–2)
- Repository bootstrap, `.gitignore`, LICENSE, `pyproject.toml`,
  `requirements.txt`.
- Colab + local environment verification; `pip install -e .`; `pytest` green.
- **Issues:** (bootstrap commit; repo setup tracked via this milestone).

### Milestone 1 — Dataset preparation and license verification (Weeks 1–2)
- **Status: 🟡 in progress — scripts ready; data + license gates pending.**
  `data/download_scripts/` (download scripts + checksums) exist; the remaining
  gates are the raw data (`data/raw/`), the COCO annotations
  (`data/annotations/`), and the verified license rows in `docs/licenses.md`
  (all checked by step [0] of `scripts/run_real_data_validation.sh`).
- Verify academic-use licenses for LADD, D-Fire, RescueNet, FloodNet+.
- Confirm Grounding DINO (Apache-2.0) permits feature extraction and caching;
  confirm OWL-ViT, YOLO-World, YOLO11, CLIP, DINOv2 licenses.
- Record results in `docs/licenses.md`; write download scripts + checksums
  (no raw data in git); freeze class lists after mask-to-box filtering.
- **Issues:** #1 (licenses), #2 (downloads), #3 (mask-to-box filtering).

### Milestone 2 — Pilot experiment and Colab feasibility validation (Week 2)
- **Status: 🟡 in progress — pilot scaffolded, run pending.**
  `notebooks/00_pilot_colab_memory.ipynb` (VRAM/runtime/top-k checks) is ready
  but its decision log is empty until a few real images are available
  (`data/raw/`, Milestone 1 gate).
- Grounding DINO Swin-T on a few images on Colab T4: VRAM, runtime,
  top-k=100 vs top-k=300, feature caching round-trip.
- D1/D2 pilot checks; decide fallback backbone need.
- **Issues:** #5 (Colab pilot).

### Milestone 3 — Real-data validation and analysis (Weeks 3–8; data-ready gated)
- **Status: 🟡 scripts ready — run when the Milestone 1 gates pass.**
- Run the full U-ADAPT pipeline on real LADD + D-Fire with
  `scripts/run_real_data_validation.sh`: feature extraction
  (`01_extract_and_cache.py`, Grounding DINO Swin-T, top-k=100, train+test) →
  prototypes (`02_build_prototypes.py`, k=1/3/5) → Mode A evaluation
  (`demo_mode_a_end_to_end.py` on the real cache) → pooled D1/D2/D3
  (`compute_pooled_diagnostics.py`) → markdown report
  (`generate_real_data_report.py` → `docs/real_data_results.md`).
- D-Fire is evaluated with `--norm-strategy absolute` (2-class variance fix,
  pre-registration deviation 2026-08-03 §2 — min-max collapses the variance
  terms to {0, 1}); LADD uses min-max.
- D1/D2/D3 are the **primary diagnostic claim computed pooled across
  LADD + D-Fire** (3 distinct classes) per pre-registration deviation 2026-08-03
  §10, with per-dataset values still reported; D5 is the pre-registered
  sentinel for variance clustering (Beta-regression fallback).
- **Timeline estimate:** ~2–4 weeks once raw data + annotations are in place
  (feature extraction on 21,527 D-Fire images dominates; run in Colab T4
  batches or on a local GPU).
- **Issues:** real-data run + report (create issue on GitHub).

### Milestone 4 — Baselines and zero-shot evaluation (Weeks 3–4)
- Zero-shot Grounding DINO evaluation on LADD + D-Fire; text-only,
  visual-only, naive-averaging baselines; raw proposal recall ceiling.
- Metrics implementation: mAP50, Gap Recovery, ECE, Brier, uncertainty AUROC.
- **Issues:** #7 (metrics), #8 (baseline evaluation script).

### Milestone 5 — Prototype construction and feature caching (Weeks 3–4)
- `01_extract_and_cache.py` over train/val/test splits (top-k=100).
- Text prototypes (M=20 templates) + visual prototypes (k=1/3/5, outlier
  rejection).
- **Issues:** #4 (feature caching).

### Milestone 6 — Mode A analytic gating implementation (Weeks 5–6)
- Full Mode A wiring: normalized text/visual variance + affinity, fixed
  coefficients α=β=γ=1, T=1.
- Unit tests for the gate; diagnostics D1/D2/D4 on real cached features.
- **Issues:** #6 (Mode A).

### Milestone 7 — Mode B calibration implementation (Weeks 5–6)
- 20-box/class calibration set; logistic-regression gate (primary) + MLP
  (secondary); MC Dropout T=10; temperature scaling on calibration split;
  5-fold CV.
- **Issues:** Mode B implementation (create issue on GitHub).

### Milestone 8 — Baselines, ablations, and diagnostics D1–D5 (Weeks 7–8)
- Coefficient ablations (7 variants), top-k=300 ablation, normalization
  ablations, M=20 vs M=50, k=1 variance prior ablation.
- Full D1–D5 diagnostics; H-fail (D3) analysis; D1/D2/D3 evaluated pooled
  across LADD + D-Fire per the §10 deviation (2026-08-03); statistical tests
  over 10 seeds with BH correction.
- **Issues:** ablations + D1–D5 diagnostics (create issues on GitHub).

### Milestone 9 — Cross-domain transfer experiments (Weeks 7–8)
- RQ3: Mode A primary transfer test (LADD ⇄ D-Fire); Mode B secondary probes,
  including the COCO/LVIS gate-initialization ablation (the former Mode C,
  proposal §5.4.3); RescueNet/FloodNet+ held-out categories.
- **Issues:** cross-domain transfer (create issue on GitHub).

### Milestone 10 — Analysis, figures, tables, and thesis writing (Weeks 9–12)
- Gap-recovery analysis vs pre-registered floors/ceilings; reliability
  figures; tables; statistical tests (§7.6); final draft + revision.
- **Issues:** figures, tables, statistical tests, final draft (create issues
  on GitHub).

## Repository cross-reference

| Stage | Script / module |
|-------|-----------------|
| Phase 1–2 (proposals + features) | `scripts/01_extract_and_cache.py`, `src/uadapt/features/cache_engine.py` |
| Phase 3 (prototypes) | `scripts/02_build_prototypes.py`, `src/uadapt/prototypes/` |
| Phase 4 (gating) | `scripts/03_run_fusion.py`, `src/uadapt/fusion/` (Mode B COCO/LVIS init: `configs/modes/mode_B_coco_lvis_init.yaml`) |
| Phase 5 (calibration + evaluation) | `scripts/04_evaluate.py`, `src/uadapt/metrics/` |
| Real-data validation (M3) | `scripts/run_real_data_validation.sh`, `scripts/compute_pooled_diagnostics.py`, `scripts/generate_real_data_report.py` |
| Pilot | `notebooks/00_pilot_colab_memory.ipynb` |
| Mask→box | `data/mask_to_box/filter.py`, `notebooks/01_mask_to_box_inspection.ipynb` |
| Diagnostics | `notebooks/02_diagnostics_D1_D2.ipynb`, `src/uadapt/metrics/diagnostics.py` |
