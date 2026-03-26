"""Tests for OpenSimScrewKinematics (Guideline C3 - Required).

OpenSim is rarely installed in CI, so most tests use mocked bindings.
The availability guard and ``ImportError`` path are always tested.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.opensim_screw_kinematics import (
    OpenSimScrewKinematics,
)
from src.shared.python.screw_theory import ScrewAxis, Twist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vec3(x: float, y: float, z: float) -> MagicMock:
    """Return a Vec3-like mock with indexed access."""
    v = MagicMock()
    v.__getitem__ = lambda self, i: [x, y, z][i]
    return v


def _make_body_mock(ang: tuple, lin: tuple, pos: tuple) -> MagicMock:
    body = MagicMock()
    body.getAngularVelocityInGround.return_value = _make_vec3(*ang)
    body.getLinearVelocityInGround.return_value = _make_vec3(*lin)
    body.getPositionInGround.return_value = _make_vec3(*pos)
    return body


def _make_model_mock(body_name: str = "radius") -> MagicMock:
    body = _make_body_mock((0.0, 0.0, 1.5), (0.2, 0.1, 0.0), (0.5, 0.0, 0.3))
    body_set = MagicMock()
    body_set.getIndex.return_value = 0
    body_set.get.return_value = body

    model = MagicMock()
    model.getBodySet.return_value = body_set
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_raises_import_error_when_opensim_unavailable(self) -> None:
        with (
            patch(
                "src.engines.physics_engines.opensim.python"
                ".opensim_screw_kinematics.OPENSIM_AVAILABLE",
                False,
            ),
            pytest.raises(ImportError, match="opensim is not installed"),
        ):
            OpenSimScrewKinematics(MagicMock(), MagicMock())


class TestWithMockedOpenSim:
    """Tests using fully mocked OpenSim bindings."""

    @pytest.fixture
    def sk(self) -> OpenSimScrewKinematics:
        """Return analyzer bypassing the OPENSIM_AVAILABLE guard."""
        model = _make_model_mock()
        state = MagicMock()
        with patch(
            "src.engines.physics_engines.opensim.python"
            ".opensim_screw_kinematics.OPENSIM_AVAILABLE",
            True,
        ):
            analyzer = OpenSimScrewKinematics.__new__(OpenSimScrewKinematics)
            analyzer.model = model
            analyzer.state = state
        return analyzer

    def test_compute_twist_returns_twist_instance(self, sk: OpenSimScrewKinematics) -> None:
        mock_opensim = MagicMock()
        with patch.dict("sys.modules", {"opensim": mock_opensim}):
            twist = sk.compute_twist("radius")
        assert isinstance(twist, Twist)
        assert twist.body_name == "radius"

    def test_compute_twist_angular_velocity(self, sk: OpenSimScrewKinematics) -> None:
        mock_opensim = MagicMock()
        with patch.dict("sys.modules", {"opensim": mock_opensim}):
            twist = sk.compute_twist("radius")
        assert abs(twist.angular[2] - 1.5) < 1e-9  # z-component from mock

    def test_compute_twist_unknown_body_raises(self, sk: OpenSimScrewKinematics) -> None:
        sk.model.getBodySet().getIndex.return_value = -1
        mock_opensim = MagicMock()
        with (
            patch.dict("sys.modules", {"opensim": mock_opensim}),
            pytest.raises(ValueError, match="not found"),
        ):
            sk.compute_twist("ghost_body")

    def test_compute_screw_axis_returns_screw_axis(self, sk: OpenSimScrewKinematics) -> None:
        twist = Twist(
            angular=np.array([0.0, 0.0, 2.0]),
            linear=np.array([0.3, 0.0, 0.0]),
            body_name="radius",
            reference_point=np.array([0.5, 0.0, 0.0]),
        )
        screw = sk.compute_screw_axis(twist)
        assert isinstance(screw, ScrewAxis)

    def test_analyze_key_points_skips_missing_bodies(self, sk: OpenSimScrewKinematics) -> None:
        def index_side_effect(name: str) -> int:
            return 0 if name == "radius" else -1

        sk.model.getBodySet().getIndex.side_effect = index_side_effect

        mock_opensim = MagicMock()
        with patch.dict("sys.modules", {"opensim": mock_opensim}):
            results = sk.analyze_key_points(["radius", "missing_body"])

        assert "radius" in results
        assert "missing_body" not in results

    def test_visualize_screw_axis_segment_length(self, sk: OpenSimScrewKinematics) -> None:
        screw = ScrewAxis(
            axis_direction=np.array([0.0, 0.0, 1.0]),
            axis_point=np.zeros(3),
            pitch=0.0,
            angular_magnitude=2.0,
            linear_magnitude=0.0,
            is_singular=False,
        )
        start, end = sk.visualize_screw_axis(screw, length=0.6)
        assert abs(np.linalg.norm(end - start) - 0.6) < 1e-9
