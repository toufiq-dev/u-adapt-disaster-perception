# Thesis Plan — 12 Weeks

Mapped from the proposal timeline (§11) into repository milestones M0–M9.
GitHub milestones group the issues (see issues list); each row lists the
primary deliverables and the issues that track them.

## Overview

| Week | GitHub milestone | Repository milestone | Core focus |
|------|------------------|----------------------|------------|
| 1–2 | Week 1-2 Setup and Pre-registration | M0, M1, M2 | Repo, licenses, datasets, pilot, freeze class list |
| 3–4 | Week 3-4 Core Pipeline | M3, M4 | Baselines, caching, prototypes |
| 5–6 | Week 5-6 Gating Modes | M5, M6 | Mode A, Mode B |
| 7–8 | Week 7-8 Experiments and Diagnostics | M7, M8 | Baselines/ablations, D1–D5, cross-domain |
| 9–12 | Week 9-12 Thesis Writing | M9 | Analysis, figures, tables, writing |

## Milestones

### Milestone 0 — Repository and environment setup (Weeks 1–2)
- Repository bootstrap, `.gitignore`, LICENSE, `pyproject.toml`,
  `requirements.txt`.
- Colab + local environment verification; `pip install -e .`; `pytest` green.
- **Issues:** (bootstrap commit; repo setup tracked via this milestone).

### Milestone 1 — Dataset preparation and license verification (Weeks 1–2)
- Verify academic-use licenses for LADD, D-Fire, RescueNet, FloodNet+.
- Confirm Grounding DINO (Apache-2.0) permits feature extraction and caching;
  confirm OWL-ViT, YOLO-World, YOLO11, CLIP, DINOv2 licenses.
- Record results in `docs/licenses.md`; write download scripts + checksums
  (no raw data in git); freeze class lists after mask-to-box filtering.
- **Issues:** #1 (licenses), #2 (downloads), #3 (mask-to-box filtering).

### Milestone 2 — Pilot experiment and Colab feasibility validation (Week 2)
- Grounding DINO Swin-T on a few images on Colab T4: VRAM, runtime,
  top-k=100 vs top-k=300, feature caching round-trip.
- D1/D2 pilot checks; decide fallback backbone need.
- **Issues:** #5 (Colab pilot).

### Milestone 3 — Baselines and zero-shot evaluation (Weeks 3–4)
- Zero-shot Grounding DINO evaluation on LADD + D-Fire; text-only,
  visual-only, naive-averaging baselines; raw proposal recall ceiling.
- Metrics implementation: mAP50, Gap Recovery, ECE, Brier, uncertainty AUROC.
- **Issues:** #7 (metrics), #8 (baseline evaluation script).

### Milestone 4 — Prototype construction and feature caching (Weeks 3–4)
- `01_extract_and_cache.py` over train/val/test splits (top-k=100).
- Text prototypes (M=20 templates) + visual prototypes (k=1/3/5, outlier
  rejection).
- **Issues:** #4 (feature caching).

### Milestone 5 — Mode A analytic gating implementation (Weeks 5–6)
- Full Mode A wiring: normalized text/visual variance + affinity, fixed
  coefficients α=β=γ=1, T=1.
- Unit tests for the gate; diagnostics D1/D2/D4 on real cached features.
- **Issues:** #6 (Mode A).

### Milestone 6 — Mode B calibration implementation (Weeks 5–6)
- 20-box/class calibration set; logistic-regression gate (primary) + MLP
  (secondary); MC Dropout T=10; temperature scaling on calibration split;
  5-fold CV.
- **Issues:** Mode B implementation (create issue on GitHub).

### Milestone 7 — Baselines, ablations, and diagnostics D1–D5 (Weeks 7–8)
- Coefficient ablations (7 variants), top-k=300 ablation, normalization
  ablations, M=20 vs M=50, k=1 variance prior ablation.
- Full D1–D5 diagnostics; H-fail (D3) analysis; statistical tests over 10
  seeds with BH correction.
- **Issues:** ablations + D1–D5 diagnostics (create issues on GitHub).

### Milestone 8 — Cross-domain transfer experiments (Weeks 7–8)
- RQ3: Mode A primary transfer test (LADD ⇄ D-Fire); Mode B secondary probes,
  including the COCO/LVIS gate-initialization ablation (the former Mode C,
  proposal §5.4.3); RescueNet/FloodNet+ held-out categories.
- **Issues:** cross-domain transfer (create issue on GitHub).

### Milestone 9 — Analysis, figures, tables, and thesis writing (Weeks 9–12)
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
| Pilot | `notebooks/00_pilot_colab_memory.ipynb` |
| Mask→box | `data/mask_to_box/filter.py`, `notebooks/01_mask_to_box_inspection.ipynb` |
| Diagnostics | `notebooks/02_diagnostics_D1_D2.ipynb`, `src/uadapt/metrics/diagnostics.py` |
