"""Unit tests for OpenSim forward-kinematics extraction (issue #4116).

This test module validates the `fk.py` module's interface and basic functionality.
Full acceptance tests (RMSE ≤ 5 mm vs Simscape reference) are in
`test_opensim_fk_matches_simscape.py` once the dependencies (#4110, #4114) are
resolved and the model is available.

Per issue #4116 acceptance criteria:
    - Grip RMSE vs Simscape ≤ 5 mm at address, top-of-backswing, impact.
    - Clubhead RMSE vs Simscape ≤ 5 mm at same poses.
    - Vectorised path ≥ 10× faster than per-step loop.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit]

# Skip these tests if OpenSim is unavailable
pytest.importorskip(
    "opensim",
    minversion=None,
    reason="OpenSim not installed; install with: pip install opensim",
)


class TestFKModule:
    """Smoke tests for the fk.py module interface."""

    def test_fk_module_imports(self) -> None:
        """Verify fk.py is importable."""
        from src.engines.physics_engines.opensim.python.opensim_golf import fk

        assert fk is not None
        assert hasattr(fk, "compute_grip")
        assert hasattr(fk, "compute_clubhead")
        assert hasattr(fk, "compute_skeleton_fk")

    def test_compute_grip_signature(self) -> None:
        """Verify compute_grip has the correct signature."""
        import inspect

        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            compute_grip,
        )

        sig = inspect.signature(compute_grip)
        params = list(sig.parameters.keys())
        assert params == [
            "model",
            "state",
        ], f"Expected ['model', 'state'], got {params}"

    def test_compute_clubhead_signature(self) -> None:
        """Verify compute_clubhead has the correct signature."""
        import inspect

        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            compute_clubhead,
        )

        sig = inspect.signature(compute_clubhead)
        params = list(sig.parameters.keys())
        assert params == [
            "model",
            "state",
        ], f"Expected ['model', 'state'], got {params}"

    def test_compute_skeleton_fk_signature(self) -> None:
        """Verify compute_skeleton_fk has the correct signature."""
        import inspect

        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            compute_skeleton_fk,
        )

        sig = inspect.signature(compute_skeleton_fk)
        params = list(sig.parameters.keys())
        assert params == [
            "model",
            "states",
        ], f"Expected ['model', 'states'], got {params}"

    def test_rotmat_to_quat_utility(self) -> None:
        """Verify the rotation-matrix-to-quaternion utility works."""
        import numpy as np
        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            _rotmat_to_quat,
        )

        # Create a mock rotation matrix (identity).
        class MockRot:
            def get(self, i: int, j: int) -> float:
                return 1.0 if i == j else 0.0

        rot = MockRot()
        q = _rotmat_to_quat(rot)

        # Identity rotation should give [1, 0, 0, 0].
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q, expected, atol=1e-6)

    def test_average_quaternions_utility(self) -> None:
        """Verify the quaternion averaging utility preserves unit norm."""
        import numpy as np
        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            _average_quaternions,
        )

        # Two identical quaternions should average to themselves.
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_avg = _average_quaternions(q, q)

        np.testing.assert_allclose(q_avg, q, atol=1e-6)
        # Verify unit norm.
        assert np.abs(np.linalg.norm(q_avg) - 1.0) < 1e-6


class TestFKErrorHandling:
    """Test error handling and edge cases in fk.py."""

    def test_compute_skeleton_fk_rejects_empty_states(self) -> None:
        """Empty state sequence should raise ValueError."""
        import opensim as osim
        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            compute_skeleton_fk,
        )

        model = osim.Model()
        with pytest.raises(ValueError, match="empty"):
            compute_skeleton_fk(model, [])

    def test_compute_skeleton_fk_rejects_wrong_type(self) -> None:
        """Non-list/array states should raise TypeError."""
        import opensim as osim
        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            compute_skeleton_fk,
        )

        model = osim.Model()
        with pytest.raises(TypeError):
            compute_skeleton_fk(model, "not a list")  # type: ignore[arg-type]

    def test_compute_skeleton_fk_array_not_implemented(self) -> None:
        """Array-based state trajectories raise NotImplementedError pending #4110, #4114."""
        import numpy as np
        import opensim as osim
        from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
            compute_skeleton_fk,
        )

        model = osim.Model()
        trajectory = np.zeros((10, 23))  # Dummy trajectory
        with pytest.raises(NotImplementedError, match="Array-based.*not yet supported"):
            compute_skeleton_fk(model, trajectory)
