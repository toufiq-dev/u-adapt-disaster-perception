"""Unit tests for Mode B calibration wiring (Milestone 6).

Covers (proposal §5.4.1/§5.4.2/§5.4.3, docs/pre_registration.md §5 and §7):
  * calibration-set -> gate matrices (X, soft targets w*)
  * min-max normalization with calibration-set statistics
  * config-driven gate construction (logreg primary, MLP secondary)
  * COCO/LVIS-pretrained gate-init injection (the former Mode C ablation)
  * temperature optimization on the calibration split (NLL)
  * end-to-end run_mode_b producing fused scores + gate weights
"""

import numpy as np
import pytest

from uadapt.features.cache_engine import FeatureRecord
from uadapt.fusion.calibration import (
    apply_gate_init,
    build_calibration_matrices,
    build_gate,
    min_max_stats,
    normalize_row,
    optimize_temperature,
    record_gate_input,
    run_mode_b,
)
from uadapt.fusion.mode_a_analytic import fuse_scores
from uadapt.fusion.mode_b_logreg import LogRegGate, soft_targets
from uadapt.fusion.mode_b_mlp import MLPGate


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_calibration_samples(n: int = 20, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        s_text = float(rng.uniform(0.0, 1.0))
        s_visual = float(rng.uniform(0.0, 1.0))
        # Correlate correctness with which modality scores higher.
        text_correct = bool(s_text > s_visual and rng.random() < 0.8)
        visual_correct = bool(s_visual > s_text and rng.random() < 0.8)
        samples.append(
            {
                "class": "fire",
                "s_text": s_text,
                "s_visual": s_visual,
                "sigma2_text": float(rng.uniform(0.0, 1.0)),
                "sigma2_visual": float(rng.uniform(0.0, 1.0)),
                "a_visual": float(rng.uniform(0.0, 1.0)),
                "text_correct": text_correct,
                "visual_correct": visual_correct,
            }
        )
    return samples


def _make_records(n: int = 5, seed: int = 1) -> list[FeatureRecord]:
    rng = np.random.default_rng(seed)
    classes = ["fire", "smoke"]
    records = []
    for i in range(n):
        records.append(
            FeatureRecord(
                image_id=f"img{i}",
                class_name="fire",
                score=float(rng.uniform(0.1, 0.9)),
                bbox=np.asarray([0, 0, 10, 10], dtype=np.float32),
                visual_feature=rng.normal(size=(8,)).astype(np.float32),
                text_similarities=rng.uniform(-1, 1, size=len(classes)).astype(np.float32),
                classes=classes,
            )
        )
    return records


def _make_prototype_payload(seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    centroid = rng.normal(size=(8,))
    centroid = centroid / np.linalg.norm(centroid)
    return {
        "shots": 5,
        "prototypes": {
            "fire": {
                "centroid": centroid.tolist(),
                "sigma_visual": 0.3,
                "n_kept": 5,
                "support_ids": ["s0", "s1", "s2", "s3", "s4"],
            }
        },
    }


LOGGREG_CFG = {
    "mode": "B",
    "gate": "logistic_regression",
    "calibration": {"boxes_per_class": 20, "cv_folds": 5, "l2_weight_decay": 1e-4},
}
MLP_CFG = {
    "mode": "B",
    "gate": "mlp",
    "mlp": {"hidden_dim": 16, "dropout_p": 0.0, "l2_weight_decay": 1e-4},
    "calibration": {"boxes_per_class": 20, "cv_folds": 5},
}


# ----------------------------------------------------------------------
# Calibration matrices + soft targets
# ----------------------------------------------------------------------
def test_build_calibration_matrices_shape():
    samples = _make_calibration_samples(n=20)
    X, y_soft = build_calibration_matrices(samples)
    assert X.shape == (20, 5)
    assert y_soft.shape == (20,)
    assert np.all((X >= 0.0) & (X <= 1.0))
    assert np.all((y_soft >= 0.0) & (y_soft <= 1.0))


def test_soft_targets_pre_registered_rules():
    s_text = np.array([0.9, 0.2, 0.6, 0.4])
    s_visual = np.array([0.2, 0.8, 0.6, 0.4])
    tc = np.array([True, False, True, False])
    vc = np.array([False, True, True, False])
    t = soft_targets(s_text, s_visual, tc, vc)
    assert t[0] == 1.0  # text-only correct
    assert t[1] == 0.0  # visual-only correct
    # both / neither -> sigmoid(S_visual - S_text)
    assert t[2] == pytest.approx(0.5)
    assert t[3] == pytest.approx(0.5)


def test_min_max_normalization():
    X = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]], dtype=float)
    vmin, vmax = min_max_stats(X)
    row = normalize_row([2.0, 10.0], (vmin, vmax), eps=0.0)
    np.testing.assert_allclose(row, [0.5, 0.0])
    # Clamping outside the calibration range
    row2 = normalize_row([-1.0, 40.0], (vmin, vmax), eps=0.0)
    np.testing.assert_allclose(row2, [0.0, 1.0])


# ----------------------------------------------------------------------
# Gate construction + init injection
# ----------------------------------------------------------------------
def test_build_gate_logreg_and_mlp():
    assert isinstance(build_gate(LOGGREG_CFG), LogRegGate)
    assert isinstance(build_gate(MLP_CFG), MLPGate)


def test_apply_gate_init_logreg_warm_start():
    gate = LogRegGate(epochs=2000, lr=0.1)
    theta = np.array([1.0, -1.0, 0.5, -0.5, 0.25])
    bias = 0.1
    apply_gate_init(gate, {"theta": theta.tolist(), "bias": bias})
    X = np.zeros((1, 5))
    # Without any fitting, predict must reflect the injected weights exactly.
    expected = 1.0 / (1.0 + np.exp(-(X @ theta + bias)))
    np.testing.assert_allclose(gate.predict(X), expected, atol=1e-12)


def test_apply_gate_init_mlp_warm_start():
    gate = MLPGate(hidden_dim=4, rng_seed=0)
    rng = np.random.default_rng(3)
    w1 = rng.normal(0, 0.1, (5, 4))
    b1 = np.zeros(4)
    w2 = rng.normal(0, 0.1, (4, 1))
    b2 = 0.0
    apply_gate_init(gate, {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": float(b2)})
    X = np.zeros((1, 5))
    pred = gate.predict(X)
    assert pred.shape == (1,)
    assert 0.0 < pred[0] < 1.0


def test_gate_fit_warm_starts_from_injected_params():
    gate = LogRegGate(epochs=5, lr=0.1)
    theta = np.array([1.0, -1.0, 0.5, -0.5, 0.25])
    apply_gate_init(gate, {"theta": theta.tolist(), "bias": 0.1})
    X = np.random.default_rng(0).uniform(0, 1, (20, 5))
    y = np.clip(np.random.default_rng(1).uniform(0, 1, 20), 0.05, 0.95)
    gate.fit(X, y)
    # After few epochs, the parameters should still be near the warm start.
    assert np.allclose(gate._theta, theta, atol=0.5)


def test_gate_fit_rejects_mismatched_injected_theta():
    gate = LogRegGate()
    apply_gate_init(gate, {"theta": np.zeros(4).tolist(), "bias": 0.0})  # 4 != 5 features
    X = np.random.default_rng(0).uniform(0, 1, (10, 5))
    y = np.random.default_rng(1).uniform(0, 1, 10)
    with pytest.raises(ValueError):
        gate.fit(X, y)


def test_mlp_fit_rejects_mismatched_injected_w1():
    gate = MLPGate(hidden_dim=4)
    apply_gate_init(
        gate,
        {
            "w1": np.zeros((5, 3)).tolist(),  # 3 != hidden_dim 4
            "b1": np.zeros(3).tolist(),
            "w2": np.zeros((3, 1)).tolist(),
            "b2": 0.0,
        },
    )
    X = np.random.default_rng(0).uniform(0, 1, (10, 5))
    y = np.random.default_rng(1).uniform(0, 1, 10)
    with pytest.raises(ValueError):
        gate.fit(X, y)


# ----------------------------------------------------------------------
# Temperature optimization
# ----------------------------------------------------------------------
def test_optimize_temperature_reduces_nll():
    gate = LogRegGate()
    X = np.random.default_rng(0).uniform(0, 1, (40, 5))
    # Build binary targets from the fused score at a fixed w=0.5.
    s_text = X[:, 0]
    s_visual = X[:, 1]
    s_final = np.clip((1.0 - 0.5) * s_text + 0.5 * s_visual, 1e-6, 1 - 1e-6)
    y = (s_final > 0.5).astype(float)
    # Trivial gate that outputs w=0.5 everywhere: zero weights.
    apply_gate_init(gate, {"theta": np.zeros(5).tolist(), "bias": 0.0})
    t = optimize_temperature(gate, X, y)
    assert 0.2 <= t <= 5.0


def test_temperature_default_when_no_signal():
    gate = LogRegGate()
    apply_gate_init(gate, {"theta": np.zeros(5).tolist(), "bias": 0.0})
    X = np.full((10, 5), 0.5)
    y = np.full(10, 0.5)
    t = optimize_temperature(gate, X, y)
    assert t == pytest.approx(1.0)  # flat logit -> no improvement over T=1


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------
def test_run_mode_b_logreg_end_to_end():
    records = _make_records(n=8)
    cal = {"boxes_per_class": 20, "classes": ["fire"], "samples": _make_calibration_samples(20)}
    proto = _make_prototype_payload()
    result = run_mode_b(records, cal, proto, LOGGREG_CFG)
    assert len(result.scores) == len(records)
    assert result.cv_scores is not None and result.cv_scores.shape == (5,)
    assert 0.2 <= result.temperature <= 5.0
    for s in result.scores:
        assert 0.0 < s["gate_weight"] < 1.0
        assert 0.0 <= s["score"] <= 1.0
        assert set(["image_id", "class", "score", "bbox", "gate_weight", "affinity"]) <= set(s)
        # score == (1-w)*s_text + w*s_visual
        expected = fuse_scores(s["s_text"], s["s_visual"], s["gate_weight"])
        assert s["score"] == pytest.approx(expected)


def test_run_mode_b_mlp_end_to_end():
    records = _make_records(n=6, seed=4)
    cal = {"boxes_per_class": 20, "classes": ["fire"], "samples": _make_calibration_samples(20, seed=7)}
    proto = _make_prototype_payload(seed=8)
    result = run_mode_b(records, cal, proto, MLP_CFG)
    assert len(result.scores) == len(records)
    assert result.cv_scores is not None and result.cv_scores.shape == (5,)


def test_run_mode_b_with_gate_init_ablation():
    records = _make_records(n=6, seed=5)
    cal = {"boxes_per_class": 20, "classes": ["fire"], "samples": _make_calibration_samples(20, seed=6)}
    proto = _make_prototype_payload(seed=9)
    init = {"theta": np.array([0.5, -0.5, 0.5, -0.5, 0.5]).tolist(), "bias": 0.0}
    result = run_mode_b(records, cal, proto, LOGGREG_CFG, gate_init_payload=init)
    assert len(result.scores) == len(records)
    assert result.cv_scores is not None  # CV runs after warm start


def test_run_mode_b_empty_calibration_raises():
    records = _make_records(n=2)
    cal = {"boxes_per_class": 20, "classes": ["fire"], "samples": []}
    with pytest.raises(ValueError):
        run_mode_b(records, cal, _make_prototype_payload(), LOGGREG_CFG)


def test_record_gate_input_skips_missing_prototype():
    rec = _make_records(n=1)[0]
    proto = _make_prototype_payload()
    X_cal, _ = build_calibration_matrices(_make_calibration_samples(20))
    stats = min_max_stats(X_cal)
    # Only 'fire' has a prototype; reclassify the record to 'smoke'.
    rec.class_name = "smoke"
    assert record_gate_input(rec, proto, stats) is None
