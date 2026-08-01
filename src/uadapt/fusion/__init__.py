"""Uncertainty-gated fusion modes (A: analytic, B: calibrated).

Mode B optionally supports a COCO/LVIS-pretrained gate initialization
ablation (the former Mode C, proposal §5.4.3).
"""

from .mode_a_analytic import ModeAGate, analytic_gate_logit, fuse_scores, gate_weight
from .mode_b_logreg import LogRegGate
from .mode_b_mlp import MLPGate

__all__ = [
    "ModeAGate",
    "analytic_gate_logit",
    "gate_weight",
    "fuse_scores",
    "LogRegGate",
    "MLPGate",
]
