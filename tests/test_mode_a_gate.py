"""Unit tests for Mode A analytic gating (training-free, T=1).

Pre-registration guarantees under test (docs/pre_registration.md §2):
  * w = sigma(-alpha*v_visual + beta*v_text + gamma*a_visual)
  * default coefficients alpha = beta = gamma = 1 (fixed, not learned)
  * temperature T = 1 (ModeAGate rejects any other temperature)
  * pure function: no trainable state
  * k=1 visual variance is zero
  * fuse: S_final = (1 - w) S_text + w S_visual

The 7 pre-registered coefficient variants (proposal §8 / pre-registration §2
and configs/modes/mode_A_analytic.yaml) are validated config-driven below:
Full (default) plus six ablations (no-visual-uncertainty, no-text-
uncertainty, no-affinity, visual-only, text-only, affinity-only).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from uadapt.fusion.mode_a_analytic import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_TEMPERATURE,
    BetaGate,
    ModeAGate,
    analytic_gate_logit,
    beta_regression_gate,
    fuse_scores,
    gate_weight,
)
from uadapt.uncertainty.variance_estimators import normalized_visual_variance

# Mode A config that defines the pre-registered ablation variants.
MODE_A_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "modes" / "mode_A_analytic.yaml"
)

# The 7 pre-registered coefficient variants (proposal §8): Full (default) plus
# the six coefficient ablations. Keys match the config's ablation names; the
# values are the exact (alpha, beta, gamma) triplets from the pre-registration.
COEFFICIENT_VARIANTS = {
    "full": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    "no_visual_uncertainty": {"alpha": 0.0, "beta": 1.0, "gamma": 1.0},
    "no_text_uncertainty": {"alpha": 1.0, "beta": 0.0, "gamma": 1.0},
    "no_affinity": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
    "visual_uncertainty_only": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "text_uncertainty_only": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0},
    "affinity_only": {"alpha": 0.0, "beta": 0.0, "gamma": 1.0},
}


def _load_mode_a_config() -> dict:
    with open(MODE_A_CONFIG) as fh:
        return yaml.safe_load(fh)


def _as_float_coeffs(cfg_coeffs: dict) -> dict:
    return {k: float(v) for k, v in cfg_coeffs.items()}


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


def test_k1_max_entropy_prior_ablation_config():
    """The config must define the k=1 ablations: default zero and the
    max-entropy-prior ablation value 0.5 (pre-registration §2)."""
    cfg = _load_mode_a_config()
    ablations = cfg["ablations"]
    assert ablations["k1_visual_variance"] == "zero"
    assert ablations["k1_max_entropy_prior"] == 0.5


def test_k1_max_entropy_prior_substitutes_zero():
    # Default: k=1 -> 0.0 (maximum-likelihood degenerate-sample treatment)
    single = np.random.default_rng(0).normal(size=(1, 64))
    assert normalized_visual_variance(single) == 0.0
    # Ablation: k=1 with max-entropy prior -> 0.5
    assert normalized_visual_variance(single, k1_prior=0.5) == pytest.approx(0.5)
    # An explicit zero prior is identical to the default
    assert normalized_visual_variance(single, k1_prior=0.0) == 0.0


def test_k1_prior_does_not_affect_k_ge_2():
    # k>=2 has real dispersion; the k1 prior must not change the estimate
    three = np.random.default_rng(0).normal(size=(3, 64))
    assert normalized_visual_variance(three, k1_prior=0.5) == pytest.approx(
        normalized_visual_variance(three)
    )
    assert normalized_visual_variance(three, k1_prior=0.5) > 0.0


def test_k1_prior_shifts_gate_weight():
    # Higher visual variance (0.5 prior vs 0.0 default) lowers the gate's
    # weight on the visual branch (alpha term is negative).
    gate = ModeAGate()
    w_default = gate.weight(0.2, 0.0, 0.5)
    w_prior = gate.weight(0.2, 0.5, 0.5)
    assert w_prior < w_default


def test_predict_batch_matches_scalar():
    gate = ModeAGate()
    inputs = [
        {"norm_text_variance": 0.2, "norm_visual_variance": 0.6, "norm_affinity": 0.8},
        {"norm_text_variance": 0.9, "norm_visual_variance": 0.1, "norm_affinity": 0.2},
    ]
    batch = gate.predict_batch(inputs)
    scalar = [gate.weight(i["norm_text_variance"], i["norm_visual_variance"], i["norm_affinity"]) for i in inputs]
    np.testing.assert_allclose(batch, scalar, atol=1e-12)


# ----------------------------------------------------------------------
# Mode A coefficient ablations (config-driven; proposal §8)
# ----------------------------------------------------------------------
def test_mode_a_config_exposes_all_seven_variants():
    """The config must define Full (top-level coefficients) plus the 6
    coefficient ablations with the pre-registered (alpha, beta, gamma)."""
    cfg = _load_mode_a_config()

    # Full variant = top-level coefficients (alpha = beta = gamma = 1)
    assert _as_float_coeffs(cfg["coefficients"]) == COEFFICIENT_VARIANTS["full"]

    # Six ablations, exactly the pre-registered names and triplets
    ablations = cfg["ablations"]["coefficient_ablations"]
    expected = {k: v for k, v in COEFFICIENT_VARIANTS.items() if k != "full"}
    assert set(ablations) == set(expected)
    for name, coeffs in expected.items():
        assert _as_float_coeffs(ablations[name]) == coeffs


@pytest.mark.parametrize("variant", sorted(COEFFICIENT_VARIANTS))
def test_ablation_logit_signs_match_coefficients(variant):
    """Each variant's logit follows the pre-registered sign convention, and
    coefficients set to 0 have exactly no influence on the logit."""
    c = COEFFICIENT_VARIANTS[variant]
    base = dict(norm_text_variance=0.5, norm_visual_variance=0.5, norm_affinity=0.5)
    lo = analytic_gate_logit(**base, **c)

    # alpha (visual uncertainty): negative term when active, else no effect
    hi_vis = analytic_gate_logit(
        norm_text_variance=0.5, norm_visual_variance=0.9, norm_affinity=0.5, **c
    )
    if c["alpha"] > 0:
        assert hi_vis < lo
    else:
        assert hi_vis == pytest.approx(lo)

    # beta (text uncertainty): positive term when active, else no effect
    hi_text = analytic_gate_logit(
        norm_text_variance=0.9, norm_visual_variance=0.5, norm_affinity=0.5, **c
    )
    if c["beta"] > 0:
        assert hi_text > lo
    else:
        assert hi_text == pytest.approx(lo)

    # gamma (affinity): positive term when active, else no effect
    hi_aff = analytic_gate_logit(
        norm_text_variance=0.5, norm_visual_variance=0.5, norm_affinity=0.9, **c
    )
    if c["gamma"] > 0:
        assert hi_aff > lo
    else:
        assert hi_aff == pytest.approx(lo)


@pytest.mark.parametrize("variant", sorted(COEFFICIENT_VARIANTS))
def test_ablation_variant_gate_weight_in_unit_interval(variant):
    """Every variant produces a valid sigmoid weight in (0, 1)."""
    c = COEFFICIENT_VARIANTS[variant]
    gate = ModeAGate(alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"])
    for inputs in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, 0.6, 0.8)]:
        assert 0.0 < gate.weight(*inputs) < 1.0


@pytest.mark.parametrize("variant", sorted(COEFFICIENT_VARIANTS))
def test_ablation_variant_gate_matches_function(variant):
    """ModeAGate instantiated with a variant's coefficients agrees with the
    module-level gate_weight for the same coefficients."""
    c = COEFFICIENT_VARIANTS[variant]
    gate = ModeAGate(alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"])
    w_gate = gate.weight(0.2, 0.7, 0.6)
    w_fn = gate_weight(0.2, 0.7, 0.6, alpha=c["alpha"], beta=c["beta"], gamma=c["gamma"])
    assert w_gate == pytest.approx(w_fn)


# ----------------------------------------------------------------------
# Beta-regression fallback gate (pre-registered D5 contingency, §10)
# ----------------------------------------------------------------------
# Inputs spanning the D5 boundary regime: exact 0.0/1.0 variances and all
# combinations with the affinity term.
BETA_BOUNDARY_INPUTS = [
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.5, 0.5, 0.5),
    (1.0, 1.0, 0.5),
    (0.0, 0.0, 0.5),
]


def test_beta_regression_gate_valid_probabilities_at_boundaries():
    """The Beta gate must output a valid [0, 1] weight for every boundary
    combination — including exact 0.0/1.0 variances (the D5-flagged regime
    where the Taylor/logit approximation is invalid)."""
    for tv, vv, aff in BETA_BOUNDARY_INPUTS:
        w = beta_regression_gate(tv, vv, aff)
        assert np.isfinite(w)
        assert 0.0 <= w <= 1.0


def test_beta_regression_gate_exact_boundary_no_nan():
    """Exact extreme inputs must never produce NaN/Inf (no divide-by-zero:
    the precision denominator >= 1 always)."""
    for tv, vv in ((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
        for aff in (0.0, 0.5, 1.0):
            w = beta_regression_gate(tv, vv, aff)
            assert np.isfinite(w)
            assert 0.0 <= w <= 1.0


def test_beta_regression_gate_monotone_in_affinity():
    """Higher affinity must never lower the gate's visual weight."""
    ws = [beta_regression_gate(0.2, 0.2, a) for a in np.linspace(0.0, 1.0, 11)]
    assert np.all(np.diff(ws) >= -1e-12)


def test_beta_regression_gate_monotone_in_visual_variance():
    """Higher visual variance lowers the visual weight (negative alpha term)."""
    ws = [beta_regression_gate(0.2, v, 0.5) for v in np.linspace(0.0, 1.0, 11)]
    assert np.all(np.diff(ws) <= 1e-12)


def test_beta_regression_gate_monotone_in_text_variance():
    """Higher text variance raises the visual weight (positive beta term)."""
    ws = [beta_regression_gate(t, 0.2, 0.5) for t in np.linspace(0.0, 1.0, 11)]
    assert np.all(np.diff(ws) >= -1e-12)


def test_beta_regression_gate_recovers_analytic_in_high_precision_limit():
    """As precision_max -> inf (reliable, low-variance inputs) the Beta gate
    must converge to the pre-registered analytic gate."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        tv, vv, aff = rng.uniform(0, 1, 3)
        wb = beta_regression_gate(tv, vv, aff, precision_max=1e6)
        wa = gate_weight(tv, vv, aff)
        assert wb == pytest.approx(wa, abs=1e-4)


def test_beta_regression_gate_softens_commitment_vs_analytic():
    """In the D5 boundary regime the Beta gate is pulled toward the neutral
    0.5 prior — it never commits MORE strongly than the analytic gate."""
    for tv, vv, aff in [(0.9, 0.9, 0.9), (0.1, 0.1, 0.1), (1.0, 1.0, 1.0),
                        (0.0, 0.0, 0.0)]:
        wb = beta_regression_gate(tv, vv, aff)
        wa = gate_weight(tv, vv, aff)
        assert abs(wb - 0.5) <= abs(wa - 0.5) + 1e-9


def test_beta_gate_class_matches_function():
    gate = BetaGate()
    for tv, vv, aff in [(0.2, 0.6, 0.8), (0.9, 0.1, 0.2),
                        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]:
        assert gate.weight(tv, vv, aff) == pytest.approx(
            beta_regression_gate(tv, vv, aff)
        )
        assert 0.0 <= gate.weight(tv, vv, aff) <= 1.0


def test_beta_gate_fused_score():
    gate = BetaGate()
    w = gate.weight(0.2, 0.3, 0.7)
    fused = gate.fused_score(0.4, 0.8, 0.2, 0.3, 0.7)
    assert fused == pytest.approx(fuse_scores(0.4, 0.8, w))


def test_beta_gate_predict_batch_matches_scalar():
    gate = BetaGate()
    inputs = [
        {"norm_text_variance": 0.2, "norm_visual_variance": 0.6, "norm_affinity": 0.8},
        {"norm_text_variance": 0.9, "norm_visual_variance": 0.1, "norm_affinity": 0.2},
        {"norm_text_variance": 1.0, "norm_visual_variance": 1.0, "norm_affinity": 1.0},
    ]
    batch = gate.predict_batch(inputs)
    scalar = [
        gate.weight(i["norm_text_variance"], i["norm_visual_variance"], i["norm_affinity"])
        for i in inputs
    ]
    np.testing.assert_allclose(batch, scalar, atol=1e-12)
    assert np.all((batch >= 0.0) & (batch <= 1.0))
