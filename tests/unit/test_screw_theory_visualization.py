"""Tests for screw_theory.visualization and screw_theory.kinematics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.screw_theory.kinematics import ScrewAxis, compute_screw_endpoints
from src.shared.python.screw_theory.visualization import (
    compute_screw_endpoints as viz_compute,
)


def _make_screw(
    direction=(0.0, 0.0, 1.0), point=(0.0, 0.0, 0.0), pitch=0.0
) -> ScrewAxis:
    return ScrewAxis(
        axis_direction=np.array(direction),
        axis_point=np.array(point),
        pitch=pitch,
        angular_magnitude=1.0,
        linear_magnitude=0.0,
        is_singular=False,
    )


class TestScrewAxis:
    def test_screw_theory_visualization_construction(self) -> None:
        sa = _make_screw()
        assert sa is not None

    def test_axis_direction(self) -> None:
        sa = _make_screw(direction=(1.0, 0.0, 0.0))
        assert sa.axis_direction[0] == pytest.approx(1.0)

    def test_pitch(self) -> None:
        sa = _make_screw(pitch=0.5)
        assert sa.pitch == pytest.approx(0.5)


class TestComputeScrewEndpoints:
    def test_screw_theory_visualization_returns_two_arrays(self) -> None:
        sa = _make_screw()
        start, end = compute_screw_endpoints(sa, length=1.0)
        assert isinstance(start, np.ndarray)
        assert isinstance(end, np.ndarray)

    def test_endpoints_length_correct(self) -> None:
        sa = _make_screw()
        start, end = compute_screw_endpoints(sa, length=1.0)
        dist = np.linalg.norm(end - start)
        assert dist == pytest.approx(1.0, rel=1e-5)

    def test_z_axis_endpoints(self) -> None:
        sa = _make_screw(direction=(0.0, 0.0, 1.0))
        start, end = compute_screw_endpoints(sa, length=0.5)
        assert start[2] == pytest.approx(-0.25, rel=1e-5)
        assert end[2] == pytest.approx(0.25, rel=1e-5)

    def test_viz_module_reexports_function(self) -> None:
        sa = _make_screw()
        result = viz_compute(sa, length=1.0)
        assert len(result) == 2
