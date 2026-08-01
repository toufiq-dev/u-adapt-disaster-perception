"""Mode B calibration wiring (proposal §5.4.1/§5.4.2, Milestone 6).

Turns a 20-box-per-class calibration set plus cached test features into fused
detection scores via a learned gate:

    w   = gate.predict(x)               # x = 5-D normalized input
    S_final = (1 - w) * S_text + w * S_visual

Gate input vector (normalized to [0, 1]):

    x = [S_tilde_text, S_tilde_visual, sigma_tilde^2_text,
         sigma_tilde^2_visual, a_tilde_visual]

Calibration-set JSON schema (the ``--calibration`` file)::

    {
      "boxes_per_class": 20,
      "classes": ["pedestrian", "fire", "smoke"],
      "samples": [
        {
          "class": "fire",
          "s_text": 0.42,            # normalized text score
          "s_visual": 0.71,          # normalized visual score
          "sigma2_text": 0.18,       # normalized text uncertainty
          "sigma2_visual": 0.30,     # normalized visual uncertainty
          "a_visual": 0.66,          # normalized visual affinity
          "text_correct": true,      # IoU>=0.5 same-class GT, text-only
          "visual_correct": false    # IoU>=0.5 same-class GT, visual-only
        }
      ]
    }

The pre-registered soft target w* (proposal §5.4.2) is computed from the two
correctness flags and the raw scores, then the configured gate (logistic
regression primary, MLP secondary) is fitted with 5-fold CV on the
calibration set and refit on the full set.

The COCO/LVIS-pretrained gate-init ablation (the former Mode C, proposal
§5.4.3) is supported via ``--gate-init`` JSON with ``{"theta": [...],
"bias": ...}`` for logreg or ``{"w1": ..., "b1": ..., "w2": ..., "b2": ...}``
for MLP; ``set_params`` seeds the fit (warm start) before calibration on the
same target 20-box-per-class set.

Mode B temperature is optimized on the calibration split (NLL), per the
pre-registration (§7): T scales the fused-score logits on the calibration set
and is reported alongside the scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from uadapt.features.cache_engine import FeatureRecord
from uadapt.fusion.mode_a_analytic import fuse_scores
from uadapt.fusion.mode_b_logreg import LogRegGate, soft_targets
from uadapt.fusion.mode_b_mlp import MLPGate
from uadapt.uncertainty.variance_estimators import visual_affinity

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

GATE_INPUT_KEYS = [
    "s_text",
    "s_visual",
    "sigma2_text",
    "sigma2_visual",
    "a_visual",
]

DEFAULT_MINMAX_EPS = 1e-6
DEFAULT_T_MIN = 0.2
DEFAULT_T_MAX = 5.0
DEFAULT_T_STEPS = 97


# ----------------------------------------------------------------------
# Calibration-set -> gate matrices
# ----------------------------------------------------------------------
def build_calibration_matrices(
    samples: Sequence[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the gate input matrix X (N, 5) and pre-registered soft targets.

    Args:
        samples: calibration samples per the module docstring schema.

    Returns:
        (X, y_soft): normalized 5-D inputs and soft targets w* in [0, 1].
    """
    X_rows: List[List[float]] = []
    s_texts: List[float] = []
    s_visuals: List[float] = []
    text_correct: List[bool] = []
    visual_correct: List[bool] = []
    for s in samples:
        X_rows.append([float(s[k]) for k in GATE_INPUT_KEYS])
        s_texts.append(float(s["s_text"]))
        s_visuals.append(float(s["s_visual"]))
        text_correct.append(bool(s["text_correct"]))
        visual_correct.append(bool(s["visual_correct"]))
    X = np.asarray(X_rows, dtype=np.float64)
    y_soft = soft_targets(
        np.asarray(s_texts, dtype=np.float64),
        np.asarray(s_visuals, dtype=np.float64),
        np.asarray(text_correct, dtype=bool),
        np.asarray(visual_correct, dtype=bool),
    )
    return X, y_soft


def min_max_stats(
    X: np.ndarray, eps: float = DEFAULT_MINMAX_EPS
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column min/max of the calibration matrix for normalizing test rows."""
    X = np.asarray(X, dtype=np.float64)
    return (X.min(axis=0), X.max(axis=0))


def normalize_row(
    row: Sequence[float],
    stats: Tuple[np.ndarray, np.ndarray],
    eps: float = DEFAULT_MINMAX_EPS,
) -> np.ndarray:
    """Min-max normalize one 5-D row with calibration-set statistics."""
    vmin, vmax = stats
    row = np.asarray(row, dtype=np.float64)
    denom = (vmax - vmin) + eps
    return np.clip((row - vmin) / denom, 0.0, 1.0)


# ----------------------------------------------------------------------
# Config-driven gate construction + COCO/LVIS init
# ----------------------------------------------------------------------
def build_gate(cfg: Dict) -> LogRegGate | MLPGate:
    """Instantiate the Mode B gate from a mode-B config dict.

    Hyperparameters are read from the type-specific block when present
    (``mlp`` for MLP), falling back to the shared ``calibration`` block and
    finally to the module defaults.
    """
    gate_type = cfg.get("gate", "logistic_regression")
    cal = cfg.get("calibration", {})
    if gate_type == "mlp":
        mlp = cfg.get("mlp", {})
        return MLPGate(
            hidden_dim=int(mlp.get("hidden_dim", 128)),
            dropout_p=float(mlp.get("dropout_p", 0.3)),
            l2=float(mlp.get("l2_weight_decay", cal.get("l2_weight_decay", 1e-4))),
            lr=float(mlp.get("lr", 0.01)),
            epochs=int(mlp.get("epochs", 500)),
            patience=int(
                mlp.get("early_stopping_patience", cal.get("early_stopping_patience", 10))
            ),
            n_folds=int(mlp.get("cv_folds", cal.get("cv_folds", 5))),
        )
    return LogRegGate(
        l2=float(cal.get("l2_weight_decay", 1e-4)),
        epochs=int(cal.get("epochs", 2000)),
        lr=float(cal.get("lr", 0.1)),
        n_folds=int(cal.get("cv_folds", 5)),
    )


def apply_gate_init(gate: LogRegGate | MLPGate, init_payload: Dict) -> LogRegGate | MLPGate:
    """Seed a gate with COCO/LVIS-pretrained weights (former Mode C ablation)."""
    if isinstance(gate, LogRegGate):
        gate.set_params(
            np.asarray(init_payload["theta"], dtype=np.float64),
            float(init_payload["bias"]),
        )
    else:
        gate.set_params(
            np.asarray(init_payload["w1"], dtype=np.float64),
            np.asarray(init_payload["b1"], dtype=np.float64),
            np.asarray(init_payload["w2"], dtype=np.float64),
            float(init_payload["b2"]),
        )
    return gate


# ----------------------------------------------------------------------
# Test-record gate inputs
# ----------------------------------------------------------------------
def record_gate_input(
    rec: FeatureRecord,
    prototype_payload: Dict,
    stats: Tuple[np.ndarray, np.ndarray],
) -> Optional[np.ndarray]:
    """Build the normalized 5-D gate input for one cached test proposal.

    Returns None when the record's class has no visual prototype (the record
    is skipped, mirroring the Mode A flow in scripts/03_run_fusion.py).
    """
    proto = prototype_payload["prototypes"].get(rec.class_name)
    if proto is None:
        return None

    # Text score: cached per-class text similarity for the proposal's class.
    classes = list(rec.classes)
    if rec.class_name in classes:
        idx = classes.index(rec.class_name)
        s_text = float(rec.text_similarities[idx]) if rec.text_similarities.size else 0.0
    else:
        s_text = float(rec.text_similarities.max()) if rec.text_similarities.size else 0.0

    centroid = np.asarray(proto["centroid"], dtype=np.float64)
    aff = visual_affinity(rec.visual_feature, centroid)
    # Visual-only score proxy: the cached record has no dedicated visual-only
    # score, so the prototype affinity (1 + cos)/2 serves as S_visual. This
    # makes s_visual and a_visual collinear for a given box — documented
    # proxy choice; the gate still separates their contributions via the
    # calibration-learned coefficients.
    s_visual = aff
    sigma2_text = float(proto.get("sigma_text", 0.5))  # text prototype not serialized by 02
    sigma2_visual = float(proto.get("sigma_visual", 0.0))
    a_visual = aff

    return normalize_row(
        [s_text, s_visual, sigma2_text, sigma2_visual, a_visual], stats
    )


# ----------------------------------------------------------------------
# Temperature (optimized on the calibration split, pre-registration §7)
# ----------------------------------------------------------------------
def optimize_temperature(
    gate: LogRegGate | MLPGate,
    X_cal: np.ndarray,
    y_soft_cal: np.ndarray,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    n_steps: int = DEFAULT_T_STEPS,
) -> float:
    """Grid-search T minimizing soft-target NLL of the fused-score logits.

    For each candidate T, the fused score is built as
    S_final = (1 - w) S_text + w S_visual with w from the fitted gate, then
    calibrated as sigmoid(logit(S_final) / T); the T minimizing BCE against
    the soft targets (NLL) on the calibration split is returned.
    """
    w = gate.predict(X_cal)
    s_text = X_cal[:, 0]
    s_visual = X_cal[:, 1]
    s_final = np.clip((1.0 - w) * s_text + w * s_visual, 1e-9, 1 - 1e-9)
    logit = np.log(s_final / (1.0 - s_final))

    def _nll(t: float) -> float:
        p = np.clip(_sigmoid(logit / t), 1e-9, 1 - 1e-9)
        return float(
            -np.mean(y_soft_cal * np.log(p) + (1.0 - y_soft_cal) * np.log(1.0 - p))
        )

    # Prefer T=1 (no scaling) unless a candidate strictly improves NLL; this
    # keeps a flat/no-signal calibration set at T=1.
    best_t, best_nll = 1.0, _nll(1.0)
    for t in np.linspace(t_min, t_max, n_steps):
        if abs(t - 1.0) < 1e-9:
            continue
        nll = _nll(t)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return float(best_t)


# ----------------------------------------------------------------------
# Full Mode B pipeline
# ----------------------------------------------------------------------
@dataclass
class ModeBResult:
    """Outcome of a Mode B fusion run (per proposal §5.4.2)."""

    gate: LogRegGate | MLPGate
    temperature: float
    cv_scores: Optional[np.ndarray]
    scores: List[Dict]

    def cv_mean_std(self) -> Optional[Tuple[float, float]]:
        if self.cv_scores is None:
            return None
        return float(self.cv_scores.mean()), float(self.cv_scores.std())


def run_mode_b(
    records: Sequence[FeatureRecord],
    calibration_payload: Dict,
    prototype_payload: Dict,
    cfg: Dict,
    gate_init_payload: Optional[Dict] = None,
) -> ModeBResult:
    """End-to-end Mode B: fit the gate on calibration, fuse test scores.

    Args:
        records: cached test-split feature records.
        calibration_payload: 20-box/class calibration set (module schema).
        prototype_payload: JSON from 02_build_prototypes.py.
        cfg: mode-B config dict (gate, calibration, mlp blocks).
        gate_init_payload: optional COCO/LVIS-pretrained weights (ablation).

    Returns:
        ModeBResult with the fitted gate, optimized temperature, 5-fold CV
        scores, and per-proposal fused score dicts (schema shared with
        04_evaluate.py: image_id, class, score, bbox, gate_weight, affinity).
    """
    samples = calibration_payload["samples"]
    if not samples:
        raise ValueError("calibration set has no samples")

    X_cal, y_soft = build_calibration_matrices(samples)
    stats = min_max_stats(X_cal)
    gate = build_gate(cfg)
    if gate_init_payload is not None:
        apply_gate_init(gate, gate_init_payload)
        logger.info("Mode B gate warm-started from COCO/LVIS-pretrained weights")
    gate.fit_cv(X_cal, y_soft)
    temperature = optimize_temperature(gate, X_cal, y_soft)
    if gate.cv_scores is not None:
        logger.info(
            "Mode B %s fitted: %d calibration samples, T=%.3f, "
            "5-fold CV MSE mean %.4f std %.4f",
            type(gate).__name__,
            len(samples),
            temperature,
            gate.cv_scores.mean(),
            gate.cv_scores.std(),
        )
    else:
        logger.info(
            "Mode B %s fitted: %d calibration samples, T=%.3f",
            type(gate).__name__,
            len(samples),
            temperature,
        )

    scores: List[Dict] = []
    for rec in records:
        x = record_gate_input(rec, prototype_payload, stats)
        if x is None:
            continue
        w = float(gate.predict(x.reshape(1, -1))[0])
        s_text = float(x[0])
        s_visual = float(x[1])
        s_final = fuse_scores(s_text, s_visual, w)
        scores.append(
            {
                "image_id": rec.image_id,
                "class": rec.class_name,
                "score": float(s_final),
                "bbox": rec.bbox.astype(float).tolist(),
                "gate_weight": w,
                "affinity": float(x[4]),
                "s_text": s_text,
                "s_visual": s_visual,
                "temperature": temperature,
            }
        )
    return ModeBResult(
        gate=gate,
        temperature=temperature,
        cv_scores=gate.cv_scores,
        scores=scores,
    )
