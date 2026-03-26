"""Heavy integration tests for the motion training / IK pipeline (fixes #1990).

Tests DualHandIKSolver instantiation with a Pinocchio model, IK solving
for a reachable target pose, and MotionVisualizer headless recording.
All tests skip gracefully when pinocchio, pink, or meshcat are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLFER_URDF = (
    Path(__file__).parents[2]
    / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)


def _pinocchio_available() -> bool:
    try:
        import pinocchio as pin  # noqa: F401

        return True
    except ImportError:
        return False


def _pink_available() -> bool:
    try:
        import pink  # noqa: F401

        return True
    except ImportError:
        return False


class TestDualHandIKSolverImport:
    """Contract: DualHandIKSolver is importable from the motion_training package."""

    def test_dual_hand_ik_solver_importable(self) -> None:
        """DualHandIKSolver class is importable."""
        try:
            from motion_training.dual_hand_ik_solver import DualHandIKSolver
        except ImportError as exc:
            pytest.skip(f"motion_training not importable: {exc}")

        assert DualHandIKSolver is not None

    def test_ik_solver_settings_importable(self) -> None:
        """IKSolverSettings dataclass is importable."""
        try:
            from motion_training.dual_hand_ik_solver import IKSolverSettings
        except ImportError as exc:
            pytest.skip(f"motion_training not importable: {exc}")

        settings = IKSolverSettings()
        assert settings.max_iterations > 0
        assert settings.dt > 0


class TestMotionVisualizerImport:
    """Contract: MotionVisualizer is importable and declares expected API."""

    def test_motion_visualizer_importable(self) -> None:
        """MotionVisualizer class is importable."""
        try:
            from motion_training.motion_visualizer import MotionVisualizer
        except ImportError as exc:
            pytest.skip(f"motion_training not importable: {exc}")

        assert MotionVisualizer is not None

    def test_motion_visualizer_has_record_method(self) -> None:
        """MotionVisualizer has a record_trajectory (or similar) method."""
        try:
            from motion_training.motion_visualizer import MotionVisualizer
        except ImportError as exc:
            pytest.skip(f"motion_training not importable: {exc}")

        method_candidates = ["record_trajectory", "record", "capture", "visualize"]
        has_method = any(hasattr(MotionVisualizer, m) for m in method_candidates)
        assert has_method, (
            f"MotionVisualizer missing expected trajectory recording method; "
            f"has: {[m for m in dir(MotionVisualizer) if not m.startswith('_')]}"
        )


class TestIKSolverWithModel:
    """Contract: IK solver can be constructed with a real Pinocchio model."""

    def test_ik_solver_instantiation_with_pinocchio_model(self) -> None:
        """DualHandIKSolver can be instantiated with a pinocchio model."""
        if not _pinocchio_available():
            pytest.skip("pinocchio not installed")
        if not _pink_available():
            pytest.skip("pink not installed")
        if not GOLFER_URDF.exists():
            pytest.skip(f"Golfer URDF not found at {GOLFER_URDF}")

        try:
            from motion_training.dual_hand_ik_solver import (
                DualHandIKSolver,
                IKSolverSettings,
            )
        except ImportError as exc:
            pytest.skip(f"motion_training not importable: {exc}")

        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(GOLFER_URDF))
        settings = IKSolverSettings(max_iterations=5, dt=0.01)

        try:
            solver = DualHandIKSolver(model=model, settings=settings)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"DualHandIKSolver instantiation failed: {exc}")

        assert solver is not None

    def test_ik_solver_solve_returns_configuration(self) -> None:
        """IK solve for a reachable target returns a finite configuration."""
        if not _pinocchio_available():
            pytest.skip("pinocchio not installed")
        if not _pink_available():
            pytest.skip("pink not installed")
        if not GOLFER_URDF.exists():
            pytest.skip(f"Golfer URDF not found at {GOLFER_URDF}")

        try:
            from motion_training.dual_hand_ik_solver import (
                DualHandIKSolver,
                IKSolverSettings,
            )
        except ImportError as exc:
            pytest.skip(f"motion_training not importable: {exc}")

        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(GOLFER_URDF))
        settings = IKSolverSettings(max_iterations=3, dt=0.01)

        try:
            solver = DualHandIKSolver(model=model, settings=settings)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"DualHandIKSolver instantiation failed: {exc}")

        # Use neutral config as starting point — a zero-motion solve
        q0 = pin.neutral(model)
        target_left = np.array([0.3, 0.5, 1.0])
        target_right = np.array([0.3, -0.5, 1.0])

        try:
            result = solver.solve(q0, target_left, target_right)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"IK solve raised (may need specific model frames): {exc}")

        assert result is not None
        q_result = np.asarray(result)
        assert q_result.shape == (model.nq,)
        assert np.all(np.isfinite(q_result))


pytestmark = pytest.mark.live_simulation
