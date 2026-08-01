"""Supervisor demo package for U-ADAPT (research-novelty demonstration).

Provides:
  * synthetic_data  — deterministic fake world matching the real pipeline
                      schemas (FeatureRecord + COCO-style GT), used when no
                      backbone cache / annotations exist yet.
  * pipeline        — end-to-end Mode A run: prototypes -> uncertainty ->
                      analytic gating -> fused scores -> mAP50 + D1-D3
                      diagnostics, shared by scripts/demo_mode_a_end_to_end.py,
                      the visualization notebook, and the unit tests.
  * plotting        — the six publication-quality figures used by
                      notebooks/supervisor_demo_visualizations.ipynb.

Everything is seeded (default seed=0) for reproducibility; results on
synthetic data are a MECHANISM demonstration, not a research claim.
"""

from __future__ import annotations

from uadapt.demo.synthetic_data import (
    DEFAULT_CLASSES,
    IMG_SIZE,
    PROFILES,
    SyntheticDataset,
    generate_synthetic_dataset,
)
from uadapt.demo.pipeline import DemoResults, run_demo

__all__ = [
    "DEFAULT_CLASSES",
    "IMG_SIZE",
    "PROFILES",
    "SyntheticDataset",
    "generate_synthetic_dataset",
    "DemoResults",
    "run_demo",
]
