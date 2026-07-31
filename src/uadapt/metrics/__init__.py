"""Evaluation metrics and pre-registered diagnostics D1-D5."""

from .calibration_metrics import brier_score, ece, reliability_table, uncertainty_auroc
from .detection_metrics import (
    compute_ap,
    compute_map50,
    gap_recovery,
    proposal_recall,
)
from .diagnostics import (
    d1_text_uncertainty_accuracy,
    d2_visual_uncertainty_accuracy,
    d3_gate_favorability,
    d4_affinity_diagnostic,
    d5_variance_distribution,
)

__all__ = [
    "brier_score",
    "ece",
    "reliability_table",
    "uncertainty_auroc",
    "compute_ap",
    "compute_map50",
    "gap_recovery",
    "proposal_recall",
    "d1_text_uncertainty_accuracy",
    "d2_visual_uncertainty_accuracy",
    "d3_gate_favorability",
    "d4_affinity_diagnostic",
    "d5_variance_distribution",
]
