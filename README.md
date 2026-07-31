# U-ADAPT: Uncertainty-Aware Post-Hoc Adaptation of Open-Vocabulary Detectors for Few-Shot Cross-Domain Disaster Perception

**MSc Thesis implementation repository** — a clean, reproducible research codebase for U-ADAPT, a post-hoc uncertainty-gated fusion module for open-vocabulary object detectors applied to aerial disaster imagery.

## Summary

Open-vocabulary object detectors (Grounding DINO, OWL-ViT, YOLO-World, YOLO11) can localize user-specified concepts from text prompts, but their zero-shot performance degrades under the strong domain shift of aerial disaster scenes. Supervised transfer learning closes much of that gap (e.g., 61.0% → 92.2% mAP50 on LADD; 27.5% → 65.6% on D-Fire) but requires full labels and backbone fine-tuning, which is infeasible under strict few-shot, low-compute constraints. U-ADAPT is a lightweight, post-hoc module that sits on top of a **frozen** open-vocabulary detector and dynamically gates between a **text prototype** and a **visual prototype** (built from k support examples) using modality-specific uncertainty estimates — **with no backbone gradient steps**. Features are extracted once, cached to disk, and re-used across all modes, making the whole pipeline feasible on a single Google Colab T4 GPU.

## Core Research Question

**RQ1 (primary):** Can a lightweight post-hoc adapter improve open-vocabulary detection under 1/3/5-shot cross-domain disaster conditions by dynamically weighting text and visual prompts based on uncertainty — with zero backbone gradient steps?

Supporting questions: **RQ2** how much of the zero-shot-to-transfer gap (LADD +31.2 pp, D-Fire +38.1 pp) can U-ADAPT recover (Gap Recovery)? **RQ3** does the gating mechanism transfer across disaster domains (Mode A as the primary transfer test)? **RQ4** does it improve reliability (ECE, Brier, uncertainty AUROC)? **RQ5** is the relative gain backbone-agnostic?

## Main Modes

| Mode | Extra data beyond k support | Backbone gradients | Status | Description |
|------|----------------------------|-------------------|--------|-------------|
| **A** | None | None | **Primary — strict few-shot, training-free** | Analytic gating rule `w = σ(−α·σ̃²_visual + β·σ̃²_text + γ·ã_visual)` with **fixed** coefficients α = β = γ = 1 and temperature **T = 1**. All inputs are normalized uncertainty proxies derived from frozen feature statistics. |
| **B** | 20 labeled boxes per class (calibration) | None (frozen backbone) | Secondary — lightweight calibration, reported separately | **Logistic-regression gate (6 params, primary claim)** or small MLP (5→128→1, ≈650 params, dropout p=0.3, L2, early stopping) trained on cached features; MC Dropout with T = 10 passes for score variances. |
| **C** | Source-domain episodic simulation (COCO/LVIS) | None (frozen backbone) | **Exploratory** — source-domain transfer | Gating coefficients or MLP pre-trained on a source domain, frozen during target evaluation. If learned coefficients outperform defaults, this suggests domain-invariant uncertainty weighting. |

Mode A results are the headline few-shot claim; Mode B results are reported separately and never conflated with Mode A.

## Primary Datasets

| Dataset | Images | Classes | Role |
|---------|--------|---------|------|
| **LADD** (Large-scale Aerial Disaster Damage) | 1,365 | 1 (pedestrian) | Primary detection benchmark (Search & Rescue) |
| **D-Fire** | 21,527 | 2 (fire, smoke) | Primary detection benchmark |
| **RescueNet** | 4,494 | 10 (segmentation) | Auxiliary — mask→box conversion, novel-category validation |
| **FloodNet+** | 2,289 | 9 (segmentation) | Auxiliary — mask→box conversion, novel-category validation |

See [`docs/datasets.md`](docs/datasets.md) for details and the pre-registered mask-to-box filtering rules (implemented in `data/mask_to_box/filter.py`). **No dataset is stored in this repository** — only download scripts, checksums, and documentation.

## Primary Backbone

- **Grounding DINO Swin-T** (Apache-2.0) — primary backbone for all modes and both primary datasets.

## Fallback Backbones

- **OWL-ViT** (google/owlvit-base-patch32, Apache-2.0)
- **YOLO-World-small** (ultralytics, AGPL-3.0 — see `docs/licenses.md`)
- **YOLO11-small** (ultralytics, AGPL-3.0 — see `docs/licenses.md`)

Fallbacks are used only if Grounding DINO Swin-T exceeds Colab T4 memory/time budgets in the pilot (notebook `00_pilot_colab_memory.ipynb`), and as cross-backbone ablations (RQ5).

## Compute Constraint: Google Colab T4 Feasibility

Every design decision is bounded by single-GPU (16 GB T4) feasibility:

- **Feature caching** — the frozen backbone runs **once per image**; box and image features are cached to disk and never re-extracted (see `src/uadapt/features/cache_engine.py`).
- **Top-k proposal limiting** — candidate proposals are limited to the **top-k = 100** most confident detections per image for primary experiments; **k = 300 is used only as an upper-bound ablation**.
- **MC Dropout T = 10** for Mode B (T = 50 stability check on one subset only).
- Mode A is training-free and requires no MC Dropout, making it the lowest-cost setting.
- Model weights, caches, checkpoints, and Colab outputs are **never** uploaded to the repository (see `.gitignore`).

## Repository Structure

```
u-adapt-disaster-perception/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── datasets/          # LADD, D-Fire, RescueNet, FloodNet+ (classes, splits, mask->box rules)
│   ├── models/            # backbone definitions (GDINO Swin-T, OWL-ViT, YOLO-World, YOLO11)
│   └── modes/             # Mode A / B / C experiment configurations
├── data/
│   ├── README.md
│   ├── download_scripts/  # downloaders + checksums (no raw data in git)
│   └── mask_to_box/       # pre-registered segmentation-mask -> box filtering
├── src/uadapt/
│   ├── models/            # backbone loader (GDINO, OWL-ViT, YOLO)
│   ├── features/          # feature extraction + caching engine
│   ├── prototypes/        # text / visual prototype construction
│   ├── fusion/            # Mode A analytic gate, Mode B logreg / MLP gates
│   ├── uncertainty/       # variance estimators (text, visual, MC Dropout)
│   └── metrics/           # mAP50, Gap Recovery, ECE, Brier, AUROC, diagnostics D1-D5
├── scripts/               # 01 extract+cache, 02 prototypes, 03 fusion, 04 evaluate
├── notebooks/             # Colab pilot, mask->box inspection, diagnostics
├── docs/                  # pre_registration, thesis_plan, datasets, licenses, change_log
└── tests/                 # unit tests
```

## Setup Instructions

*Placeholder — environment bootstrapping will be finalized in Milestone 0 (see [`docs/thesis_plan.md`](docs/thesis_plan.md)).*

```bash
# Python >= 3.10 recommended (Colab ships Python 3.10/3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .           # installs the src/uadapt package

# run tests
pytest
```

> On Colab, PyTorch + CUDA are preinstalled; only install the additional requirements. The `torch`/`torchvision` lines in `requirements.txt` are for local machines.

## Reproducibility Statement

*Placeholder — to be finalized before main experiments. Commitments made so far (see [`docs/pre_registration.md`](docs/pre_registration.md)):*

- Frozen backbones with pinned checkpoints; features extracted once and cached (cache dir lives outside the repository).
- Top-k = 100 proposal limiting for primary experiments; top-k = 300 only as an ablation.
- Mode A is training-free with T = 1 and fixed coefficients α = β = γ = 1; no learned temperature.
- Mask-to-box filtering rules frozen before evaluation; final class list frozen before main experiments.
- Statistical testing pre-registered: paired t-test + Wilcoxon signed-rank over 10 seeds, Benjamini–Hochberg FDR correction, Cohen's d (see `docs/pre_registration.md` §Statistical Testing).
- All dataset/model licenses verified before experiments; any restricted dataset is replaced or dropped and logged in `docs/change_log.md`.

## License

Code in this repository is released under the MIT License (see [LICENSE](LICENSE)). Model and dataset licenses are documented separately in [`docs/licenses.md`](docs/licenses.md).
