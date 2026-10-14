"""Tests for the retained negative results and gates of MOSAIC."""

from __future__ import annotations

import pytest

from src.shared.python.estimation.mosaic.negative_results import (
    verify_massless_collapse_under_resolved_basis,
    verify_acceleration_noise_without_input_smoothness,
    verify_scale_collapse_without_anchor,
)

pytestmark = pytest.mark.unit


def test_massless_collapse_under_resolved_basis_is_retained() -> None:
    # A basis with too few coefficients (e.g. 16) fails kinematic whiteness / triggers massless collapse
    report = verify_massless_collapse_under_resolved_basis()
    assert report.collapsed_to_massless is True or report.whiteness_gate_passed is False


def test_acceleration_noise_without_input_smoothness_is_retained() -> None:
    # Fitting with sigma_tau=0.05 and no input smoothness leads to high torque noise (>1.0 N*m RMS error)
    report = verify_acceleration_noise_without_input_smoothness()
    assert report.torque_noise_rms_error > 1.0


def test_scale_collapse_without_anchor_is_retained() -> None:
    # Omitting the anchor leads to scale collapse
    report = verify_scale_collapse_without_anchor()
    assert report.scale_collapsed is True
