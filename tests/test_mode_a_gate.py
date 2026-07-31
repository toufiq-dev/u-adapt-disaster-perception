"""Unit tests for Mode A analytic gating (training-free, T=1).

Pre-registration guarantees under test (docs/pre_registration.md §2):
  * w = sigma(-alpha*v_visual + beta*v_text + gamma*a_visual)
  * default coefficients alpha = beta = gamma = 1 (fixed, not learned)
  * temperature T = 1 (ModeAGate rejects any other temperature)
  * pure function: no trainable state
  * k=1 visual variance is zero
  * fuse: S_final = (1 - w) S_text + w S_visual
"""

import numpy as np
import pytest

from uadapt.fusion.mode_a_analytic import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_TEMPERATURE,
    ModeAGate,
    analytic_gate_logit,
    fuse_scores,
    gate_weight,
)
from uadapt.uncertainty.variance_estimators import normalized_visual_variance


def test_default_coefficients_and_temperature():
    assert DEFAULT_ALPHA == DEFAULT_BETA == DEFAULT_GAMMA == 1.0
    assert DEFAULT_TEMPERATURE == 1.0


def test_gate_is_training_free_and_stateless():
    gate = ModeAGate()
    w1 = gate.weight(0.3, 0.6, 0.8)
    w2 = gate.weight(0.3, 0.6, 0.8)
    assert w1 == w2  # deterministic, no state
    with pytest.raises(ValueError):
        ModeAGate(temperature=1.2)  # Mode A must use T=1


def test_logit_sign_conventions():
    # High visual uncertainty -> lower logit (negative alpha term)
    logit_hi_vis = analytic_gate_logit(0.2, 0.9, 0.5)
    logit_lo_vis = analytic_gate_logit(0.2, 0.1, 0.5)
    assert logit_hi_vis < logit_lo_vis

    # High text uncertainty -> higher logit (positive beta term)
    logit_hi_text = analytic_gate_logit(0.9, 0.2, 0.5)
    logit_lo_text = analytic_gate_logit(0.1, 0.2, 0.5)
    assert logit_hi_text > logit_lo_text

    # High affinity -> higher logit (positive gamma term)
    logit_hi_aff = analytic_gate_logit(0.2, 0.2, 0.9)
    logit_lo_aff = analytic_gate_logit(0.2, 0.2, 0.1)
    assert logit_hi_aff > logit_lo_aff


def test_gate_weight_bounds_and_special_cases():
    # Equal variances, zero affinity -> w = sigma(0) = 0.5 (T-Rex2 recovery)
    w = gate_weight(0.0, 0.0, 0.0)
    assert w == pytest.approx(0.5)

    # Strongly visual regime: low visual var, high text var, high affinity
    w_vis = gate_weight(1.0, 0.0, 1.0)
    assert w_vis > 0.5 and w_vis < 1.0

    # Strongly text regime: high visual var, low text var, low affinity
    w_text = gate_weight(0.0, 1.0, 0.0)
    assert w_text < 0.5 and w_text > 0.0

    # Sigmoid bounds strictly inside (0, 1) over the full input range
    for w in [
        gate_weight(0.0, 0.0, 0.0),
        gate_weight(1.0, 1.0, 1.0),
        gate_weight(0.5, 0.5, 0.5),
    ]:
        assert 0.0 < w < 1.0


def test_fuse_scores():
    assert fuse_scores(1.0, 0.0, 0.0) == pytest.approx(1.0)  # all text
    assert fuse_scores(1.0, 0.0, 1.0) == pytest.approx(0.0)  # all visual
    assert fuse_scores(0.6, 0.4, 0.5) == pytest.approx(0.5)  # naive avg


def test_k1_visual_variance_is_zero():
    # k=1 support: single exemplar -> sigma^2_visual = 0 (pre-registered)
    single = np.random.default_rng(0).normal(size=(1, 64))
    assert normalized_visual_variance(single) == 0.0
    # k=3 support: dispersion exists
    three = np.random.default_rng(0).normal(size=(3, 64))
    assert normalized_visual_variance(three) > 0.0


def test_predict_batch_matches_scalar():
    gate = ModeAGate()
    inputs = [
        {"norm_text_variance": 0.2, "norm_visual_variance": 0.6, "norm_affinity": 0.8},
        {"norm_text_variance": 0.9, "norm_visual_variance": 0.1, "norm_affinity": 0.2},
    ]
    batch = gate.predict_batch(inputs)
    scalar = [gate.weight(i["norm_text_variance"], i["norm_visual_variance"], i["norm_affinity"]) for i in inputs]
    np.testing.assert_allclose(batch, scalar, atol=1e-12)
