"""Unit tests for the per-proposal real-data uncertainty estimators.

These estimators (change_log.md 2026-08-05) give D1/D2 continuous,
per-proposal input on the real-cache path — replacing the class-constant 0.5
placeholder (and the C-distinct-value class-level terms) that produced the
pooled D1/D2/D3 = 0.000 on the n=10 pilot.
"""

from __future__ import annotations

import numpy as np
import pytest

from uadapt.uncertainty.variance_estimators import (
    proposal_text_variance,
    proposal_visual_variance,
)


# ----------------------------------------------------------------------
# proposal_text_variance (normalized class-similarity entropy)
# ----------------------------------------------------------------------
def test_text_variance_flat_distribution_is_maximally_uncertain():
    # Perfectly flat similarities -> the model cannot discriminate classes.
    assert proposal_text_variance(np.ones(3)) == pytest.approx(1.0, abs=1e-9)
    # Near-flat is near-maximum.
    assert proposal_text_variance(np.array([0.34, 0.33, 0.33])) > 0.99


def test_text_variance_one_hot_is_zero():
    # One dominant class -> confident text assignment -> no uncertainty.
    assert proposal_text_variance(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_text_variance_is_continuous_and_monotone():
    # As the dominant class's similarity grows (others fixed tiny), the
    # normalized entropy strictly decreases -> a continuous per-proposal
    # signal (the whole point vs. C distinct class-level values).
    dom = np.linspace(0.1, 0.9, 100)
    sims = np.column_stack([dom, np.full(100, 0.01), np.full(100, 0.01)])
    vals = np.asarray([proposal_text_variance(s) for s in sims])
    assert vals[0] > vals[-1]
    assert np.all(np.diff(vals) < 0)
    assert np.ptp(vals) > 0.2
    assert np.all((vals >= 0.0) & (vals <= 1.0))


def test_text_variance_normalized_into_unit_interval():
    # Any (positive) similarity vector maps into [0, 1].
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.uniform(0.01, 1.0, size=5)
        v = proposal_text_variance(s)
        assert 0.0 <= v <= 1.0


def test_text_variance_edge_cases():
    assert proposal_text_variance(np.array([0.7])) == 0.0       # single class
    assert proposal_text_variance(None) == pytest.approx(0.5)    # no signal
    assert proposal_text_variance(np.array([])) == pytest.approx(0.5)
    assert proposal_text_variance(np.zeros(3)) == pytest.approx(1.0)  # all zero


# ----------------------------------------------------------------------
# proposal_visual_variance (box-to-support mean 1 - cos)
# ----------------------------------------------------------------------
def test_visual_variance_box_at_support_is_zero():
    # Box identical to EVERY support -> mean (1 - cos) = 0.
    sup = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))
    assert proposal_visual_variance(sup[0], sup) == pytest.approx(0.0, abs=1e-9)


def test_visual_variance_orthogonal_box_is_one():
    sup = np.array([[1.0, 0.0, 0.0]])
    assert proposal_visual_variance(np.array([0.0, 1.0, 0.0]), sup) == pytest.approx(1.0)


def test_visual_variance_opposite_box_is_two():
    sup = np.array([[1.0, 0.0, 0.0]])
    assert proposal_visual_variance(np.array([-1.0, 0.0, 0.0]), sup) == pytest.approx(2.0)


def test_visual_variance_continuous_over_proposals():
    # Box rotating from the support direction to orthogonal -> distance grows
    # monotonically from 0 to 1 (a continuous per-proposal signal).
    sup = np.array([[1.0, 0.0, 0.0]])
    boxes = np.stack(
        [np.array([np.cos(t), np.sin(t), 0.0]) for t in np.linspace(0.0, np.pi / 2, 40)]
    )
    vals = np.asarray([proposal_visual_variance(b, sup) for b in boxes])
    assert vals[0] == pytest.approx(0.0, abs=1e-9)
    assert vals[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(vals) > 0)


def test_visual_variance_empty_support_uses_k1_prior():
    # Degenerate record (no support features): the configured prior applies.
    assert proposal_visual_variance(np.ones(3), np.empty((0, 3))) == 0.0
    assert proposal_visual_variance(np.ones(3), np.empty((0, 3)), k1_prior=0.5) == 0.5
    # Missing box feature likewise.
    assert proposal_visual_variance(None, np.eye(3), k1_prior=0.25) == 0.25


def test_visual_variance_k1_is_well_defined():
    # Unlike the class-level sigma_visual (dispersion across k supports, 0 for
    # k=1), the per-proposal distance to a single support is an observed value.
    sup = np.array([[1.0, 0.0, 0.0]])
    assert proposal_visual_variance(np.array([0.6, 0.8, 0.0]), sup) == pytest.approx(
        1.0 - 0.6, abs=1e-9
    )
