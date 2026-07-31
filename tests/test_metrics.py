"""Unit tests for metrics (mAP50, Gap Recovery, ECE, Brier, AUROC, D1-D5)."""

import numpy as np
import pytest

from uadapt.metrics.calibration_metrics import (
    brier_score,
    ece,
    reliability_table,
    uncertainty_auroc,
)
from uadapt.metrics.detection_metrics import (
    compute_ap,
    compute_map50,
    gap_recovery,
    proposal_recall,
)
from uadapt.metrics.diagnostics import (
    d1_text_uncertainty_accuracy,
    d2_visual_uncertainty_accuracy,
    d3_gate_favorability,
    d4_affinity_diagnostic,
    d5_variance_distribution,
)


# ----------------------------------------------------------------------
# Detection metrics
# ----------------------------------------------------------------------
def test_compute_ap_perfect():
    # One TP at recall 1.0 -> AP = 1.0
    ap = compute_ap(np.array([1.0]), np.array([True]), np.array([False]))
    assert ap == pytest.approx(1.0)


def test_compute_ap_no_positives():
    assert compute_ap(np.array([0.5]), np.array([False]), np.array([True])) == 0.0


def test_compute_map50_simple():
    # One GT box, perfect prediction -> mAP50 = 1.0
    preds = [
        {"image_id": "img0", "class": "fire", "score": 0.9, "bbox": [0, 0, 10, 10]}
    ]
    gts = [{"image_id": "img0", "class": "fire", "bbox": [0, 0, 10, 10]}]
    assert compute_map50(preds, gts) == pytest.approx(1.0)


def test_compute_map50_missed_detection():
    preds = []  # no predictions -> 0.0
    gts = [{"image_id": "img0", "class": "fire", "bbox": [0, 0, 10, 10]}]
    assert compute_map50(preds, gts) == 0.0


def test_proposal_recall_ceiling():
    preds = [
        {"image_id": "img0", "class": "x", "score": 0.1, "bbox": [0, 0, 10, 10]},
        {"image_id": "img1", "class": "y", "score": 0.1, "bbox": [100, 100, 110, 110]},
    ]
    gts = [
        {"image_id": "img0", "class": "x", "bbox": [0, 0, 10, 10]},
        {"image_id": "img1", "class": "y", "bbox": [0, 0, 5, 5]},  # uncovered
    ]
    assert proposal_recall(preds, gts) == pytest.approx(0.5)


def test_gap_recovery():
    # Adapted halfway between floor and ceiling -> 0.5
    assert gap_recovery(70.0, 60.0, 80.0) == pytest.approx(0.5)
    # Negative gap recovery is reported as-is (pre-registered)
    assert gap_recovery(50.0, 60.0, 80.0) == pytest.approx(-0.5)
    # Degenerate floor == ceiling -> 0.0
    assert gap_recovery(60.0, 60.0, 60.0) == 0.0


# ----------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------
def test_ece_perfect_calibration():
    # Bins with accuracy == mean confidence -> ECE = 0
    conf = np.array([0.2] * 5 + [0.8] * 5)
    correct = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])  # 1/5 and 4/5 accurate
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_miscalibrated():
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 1, 0, 0])  # 50% accuracy at 90% confidence
    assert ece(conf, correct) > 0.3


def test_brier_score():
    conf = np.array([1.0, 0.0])
    correct = np.array([1, 0])
    assert brier_score(conf, correct) == pytest.approx(0.0)
    conf2 = np.array([0.5, 0.5])
    assert brier_score(conf2, correct) == pytest.approx(0.25)


def test_reliability_table_shape():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 1000)
    correct = conf > 0.5
    bc, ba, bn = reliability_table(conf, correct, n_bins=10)
    assert bc.shape == ba.shape == bn.shape == (10,)
    assert bn.sum() == pytest.approx(1000)


def test_uncertainty_auroc_perfect_discrimination():
    # Uncertain predictions are exactly the wrong ones -> AUC = 1
    unc = np.array([0.9, 0.8, 0.1, 0.2])
    correct = np.array([False, False, True, True])
    assert uncertainty_auroc(unc, correct) == pytest.approx(1.0)


def test_uncertainty_auroc_chance():
    # Identical distributions for errors and correct -> AUC = 0.5 (all ties)
    unc = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    correct = np.array([True, True, True, False, False, False])
    assert uncertainty_auroc(unc, correct) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# Diagnostics D1-D5
# ----------------------------------------------------------------------
def test_d1_text_uncertainty_accuracy():
    rng = np.random.default_rng(0)
    n = 2000
    v = rng.uniform(0, 1, n)
    correct = (rng.uniform(0, 1, n) > 0.2 + 0.6 * v)  # errors grow with variance
    res = d1_text_uncertainty_accuracy(v, correct)
    assert res.summary["spearman_rho"] > 0.3


def test_d2_visual_uncertainty_accuracy():
    rng = np.random.default_rng(0)
    n = 2000
    v = rng.uniform(0, 1, n)
    correct = (rng.uniform(0, 1, n) > 0.2 + 0.6 * v)
    res = d2_visual_uncertainty_accuracy(v, correct)
    assert res.summary["spearman_rho"] > 0.3


def test_d3_gate_favorability():
    # Gate almost always points at the better modality
    res = d3_gate_favorability(
        w_text_better=np.full(50, 0.1),  # text better -> low w
        w_visual_better=np.full(50, 0.9),  # visual better -> high w
    )
    assert res.summary["favorability_fraction"] == pytest.approx(1.0)
    assert res.summary["binomial_pvalue"] < 0.05


def test_d4_affinity_diagnostic():
    # Affinity positively correlated with delta w -> model validated
    rng = np.random.default_rng(0)
    aff = np.linspace(0, 1, 500)
    delta = 0.8 * aff - 0.4 + rng.normal(0, 0.05, 500)
    w_full = delta + 0.5
    w_g0 = np.full(500, 0.5)
    res = d4_affinity_diagnostic(w_full, w_g0, aff)
    assert res.summary["affinity_delta_spearman"] > 0.9


def test_d5_variance_distribution_flag_and_ok():
    # Concentrated near boundaries -> flagged (Taylor expansion invalid)
    boundary = np.concatenate([np.full(200, 0.05), np.full(200, 0.95)])
    res = d5_variance_distribution(boundary, boundary)
    assert "FLAGGED" in (res.flag or "")

    # Beta(5,5) concentrates in (0.25, 0.75) -> expansion valid
    rng = np.random.default_rng(0)
    ok = rng.beta(5, 5, size=400)
    res2 = d5_variance_distribution(ok, ok)
    assert res2.flag is None or "ok" in res2.flag
