"""Tests for PinocchioScrewKinematics (Guideline C3 - Required).

Pinocchio may not be installed in the test environment, so tests that
require the library are marked with ``pytest.importorskip``.  The
availability guard and ``ImportError`` path are always tested.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.engines.physics_engines.pinocchio.python.pinocchio_screw_kinematics import (
    PinocchioScrewKinematics,
)
from src.shared.python.screw_theory import ScrewAxis, Twist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(nq: int = 3, nv: int = 3, nframes: int = 3) -> MagicMock:
    model = MagicMock()
    model.nq = nq
    model.nv = nv
    model.nframes = nframes
    model.getFrameId.return_value = 1  # valid frame index < nframes
    return model


def _make_mock_data(frame_pos: np.ndarray | None = None) -> MagicMock:
    if frame_pos is None:
        frame_pos = np.array([0.1, 0.2, 0.3])
    data = MagicMock()
    frame_mock = MagicMock()
    frame_mock.translation = frame_pos
    data.oMf = {1: frame_mock}
    return data


# ---------------------------------------------------------------------------
# Tests that do NOT require the real pinocchio package
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Test behaviour when pinocchio is not installed."""

    def test_raises_import_error_when_pinocchio_unavailable(self) -> None:
        with (
            patch(
                "src.engines.physics_engines.pinocchio.python"
                ".pinocchio_screw_kinematics.PINOCCHIO_AVAILABLE",
                False,
            ),
            pytest.raises(ImportError, match="pinocchio is not installed"),
        ):
            PinocchioScrewKinematics(MagicMock(), MagicMock())


class TestWithMockedPinocchio:
    """Test PinocchioScrewKinematics using fully mocked pinocchio bindings."""

    @pytest.fixture
    def sk(self) -> PinocchioScrewKinematics:
        """Return analyzer with mocked pinocchio available."""
        mock_model = _make_mock_model()
        mock_data = _make_mock_data()
        with patch(
            "src.engines.physics_engines.pinocchio.python"
            ".pinocchio_screw_kinematics.PINOCCHIO_AVAILABLE",
            True,
        ):
            analyzer = PinocchioScrewKinematics.__new__(PinocchioScrewKinematics)
            analyzer.model = mock_model
            analyzer.data = mock_data
        return analyzer

    def test_compute_twist_returns_twist(self, sk: PinocchioScrewKinematics) -> None:
        q = np.zeros(3)
        v = np.zeros(3)

        mock_pin = MagicMock()
        mock_pin.ReferenceFrame.LOCAL_WORLD_ALIGNED = 0
        # Jacobian returns a 6×3 matrix with first row = [0,0,1] (z-rotation)
        mock_J = np.zeros((6, 3))
        mock_J[2, 0] = 1.0  # angular z from joint 0
        mock_pin.getFrameJacobian.return_value = mock_J

        with patch(
            "src.engines.physics_engines.pinocchio.python"
            ".pinocchio_screw_kinematics.pin",
            mock_pin,
            create=True,
        ):
            twist = sk.compute_twist(q, v, "some_frame")

        assert isinstance(twist, Twist)
        assert twist.body_name == "some_frame"
        assert twist.angular.shape == (3,)
        assert twist.linear.shape == (3,)

    def test_compute_twist_raises_for_unknown_frame(
        self, sk: PinocchioScrewKinematics
    ) -> None:
        # Make getFrameId return an index >= nframes → unknown frame
        sk.model.getFrameId.return_value = 999  # >= nframes=3

        mock_pin = MagicMock()
        mock_pin.ReferenceFrame.LOCAL_WORLD_ALIGNED = 0

        with (
            patch(
                "src.engines.physics_engines.pinocchio.python"
                ".pinocchio_screw_kinematics.pin",
                mock_pin,
                create=True,
            ),
            pytest.raises(ValueError, match="not found"),
        ):
            sk.compute_twist(np.zeros(3), np.zeros(3), "ghost_frame")

    def test_compute_screw_axis_returns_screw_axis(
        self, sk: PinocchioScrewKinematics
    ) -> None:
        twist = Twist(
            angular=np.array([0.0, 0.0, 2.0]),
            linear=np.array([0.1, 0.2, 0.0]),
            body_name="end_effector",
            reference_point=np.array([0.5, 0.0, 0.0]),
        )
        screw = sk.compute_screw_axis(twist)
        assert isinstance(screw, ScrewAxis)
        assert not screw.is_singular

    def test_analyze_key_points_skips_missing_frames(
        self, sk: PinocchioScrewKinematics
    ) -> None:
        q = np.zeros(3)
        v = np.zeros(3)

        # Frame "good" exists, "bad" does not
        def frame_id_side_effect(name: str) -> int:
            return 1 if name == "good" else 999

        sk.model.getFrameId.side_effect = frame_id_side_effect

        mock_pin = MagicMock()
        mock_pin.ReferenceFrame.LOCAL_WORLD_ALIGNED = 0
        mock_J = np.zeros((6, 3))
        mock_pin.getFrameJacobian.return_value = mock_J

        with patch(
            "src.engines.physics_engines.pinocchio.python"
            ".pinocchio_screw_kinematics.pin",
            mock_pin,
            create=True,
        ):
            results = sk.analyze_key_points(q, v, ["good", "bad"])

        assert "good" in results
        assert "bad" not in results

    def test_visualize_screw_axis_segment_length(
        self, sk: PinocchioScrewKinematics
    ) -> None:
        screw = ScrewAxis(
            axis_direction=np.array([0.0, 0.0, 1.0]),
            axis_point=np.array([0.0, 0.0, 0.0]),
            pitch=0.0,
            angular_magnitude=1.0,
            linear_magnitude=0.0,
            is_singular=False,
        )
        start, end = sk.visualize_screw_axis(screw, length=0.4)
        assert abs(np.linalg.norm(end - start) - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# Integration tests (require real pinocchio)
# ---------------------------------------------------------------------------


def _real_pinocchio_available() -> bool:
    """Return True only when the real pinocchio C extension is importable.

    ``importlib.util.find_spec`` can raise ``ValueError`` when ``pinocchio``
    is in ``sys.modules`` as a ``MagicMock`` (e.g. from another worker's
    patch context) because ``mock.__spec__`` is not a real ``ModuleSpec``.

    We also guard against the case where another xdist worker has patched
    ``pinocchio`` into ``sys.modules`` as a ``MagicMock``: the import would
    succeed but the object is not the real C extension.
    """
    try:
        import pinocchio as _pin  # noqa: F401

        return not isinstance(_pin, MagicMock)
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.mark.skipif(
    not _real_pinocchio_available(),
    reason="pinocchio not installed",
)
class TestWithRealPinocchio:
    """Integration tests that require the real pinocchio package."""

    @pytest.fixture
    def double_pendulum(self):
        """Simple 2-DOF planar double pendulum model in pinocchio."""
        pin = pytest.importorskip("pinocchio")

        model = pin.Model()

        # Add revolute joint 1 at origin
        joint_placement = pin.SE3.Identity()
        joint_model = pin.JointModelRZ()
        body_inertia = pin.Inertia(1.0, np.array([0.0, 0.0, -0.5]), np.eye(3) * 0.1)
        link1_id = model.addJoint(0, joint_model, joint_placement, "joint1")
        model.appendBodyToJoint(link1_id, body_inertia, pin.SE3.Identity())
        model.addFrame(
            pin.Frame(
                "link1_tip",
                link1_id,
                0,
                pin.SE3(np.eye(3), np.array([0, 0, -1.0])),
                pin.FrameType.OP_FRAME,
            )
        )

        # Add revolute joint 2 at tip of link 1
        j2_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, -1.0]))
        link2_id = model.addJoint(link1_id, pin.JointModelRZ(), j2_placement, "joint2")
        model.appendBodyToJoint(link2_id, body_inertia, pin.SE3.Identity())
        model.addFrame(
            pin.Frame(
                "link2_tip",
                link2_id,
                0,
                pin.SE3(np.eye(3), np.array([0, 0, -1.0])),
                pin.FrameType.OP_FRAME,
            )
        )

        data = model.createData()
        return model, data

    def test_integration_twist_at_rest_is_zero(self, double_pendulum) -> None:
        pin = pytest.importorskip("pinocchio")
        model, data = double_pendulum
        sk = PinocchioScrewKinematics(model, data)
        q = pin.neutral(model)
        v = np.zeros(model.nv)
        twist = sk.compute_twist(q, v, "link1_tip")
        assert np.allclose(twist.angular, 0, atol=1e-10)
        assert np.allclose(twist.linear, 0, atol=1e-10)
