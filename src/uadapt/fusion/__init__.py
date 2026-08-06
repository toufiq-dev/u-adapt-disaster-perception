"""Uncertainty-gated fusion modes (A: analytic, B: calibrated).

Mode B optionally supports a COCO/LVIS-pretrained gate initialization
ablation (the former Mode C, proposal §5.4.3).
"""

from .calibration import (
    ModeBResult,
    apply_gate_init,
    build_calibration_matrices,
    build_gate,
    min_max_stats,
    normalize_row,
    optimize_temperature,
    record_gate_input,
    run_mode_b,
)
from .mode_a_analytic import (
    BetaGate,
    ModeAGate,
    analytic_gate_logit,
    beta_regression_gate,
    fuse_scores,
    gate_weight,
)
from .mode_b_logreg import LogRegGate, soft_targets
from .mode_b_mlp import MLPGate

__all__ = [
    "ModeAGate",
    "BetaGate",
    "analytic_gate_logit",
    "beta_regression_gate",
    "gate_weight",
    "fuse_scores",
    "LogRegGate",
    "MLPGate",
    "soft_targets",
    "ModeBResult",
    "apply_gate_init",
    "build_calibration_matrices",
    "build_gate",
    "min_max_stats",
    "normalize_row",
    "optimize_temperature",
    "record_gate_input",
    "run_mode_b",
]
