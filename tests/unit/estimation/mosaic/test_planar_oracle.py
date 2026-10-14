"""Tests for the planar fixture as a reusable MOSAIC conformance oracle."""

from __future__ import annotations

import pytest

from src.shared.python.estimation.mosaic.planar_oracle import (
    PlanarConformanceOracle,
    ConformanceReport,
)

pytestmark = pytest.mark.unit


def test_planar_oracle_reproduces_epic_reference_within_tolerances() -> None:
    oracle = PlanarConformanceOracle()
    report = oracle.run_conformance()

    assert isinstance(report, ConformanceReport)
    # Frozen tolerances verification
    assert report.geometry_error_m < 1e-4  # < 0.1 mm
    assert report.kinematic_rms_max_rad < 1e-3  # kinematics agreement
    assert report.torque_rms_ratio <= 0.25  # ~20%
    assert report.structural_rank == 9
    assert report.unobservable_parameters == ("m_0",)
    assert report.open_loop_accepted is True
    assert report.open_loop_rms_max_rad <= 0.035
    assert report.all_gates_passed is True
