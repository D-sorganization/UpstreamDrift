"""
Calibration tests for the angle-of-repose experiment.

Resolves issue #5677 (audit-followup: AoR formula state).

Current state (as of PR #5682):
- The `atan(mu)` closed-form mock has been REPLACED with a proper
  mock formula (``20.0 + friction * 24.0``) and a real MuJoCo
  hopper experiment path.
- ``AngleOfReposeExperiment`` lives at
  ``src/bunkershot3d/calibration/angle_of_repose.py`` (not at the
  top-level ``src/bunkershot3d/angle_of_repose.py`` path the issue
  originally referenced; the canonical location is under calibration/).
- The MuJoCo path is present but requires the optional ``mujoco``
  dependency.  Tests that exercise the real path are marked
  ``@pytest.mark.slow`` and are skipped when mujoco is absent.

DbC postconditions documented via assertions:
- mock formula output is deterministic and within physical range
- backend validation raises ``BackendNotImplementedError`` for unknown backends
- calibrate() returns a dict containing ``friction_coefficient``
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.calibration.angle_of_repose import (
    AngleOfReposeExperiment,
    compute_angle_of_repose,
)
from bunkershot3d.exceptions import BackendNotImplementedError

# ---------------------------------------------------------------------------
# Mock-path tests (no mujoco required)
# ---------------------------------------------------------------------------


class TestMockFormula:
    """The mock formula returns deterministic values within physical range."""

    def test_known_friction_returns_expected_angle(self) -> None:
        """DbC postcondition: mock angle matches formula 20 + 24*mu."""
        exp = AngleOfReposeExperiment(backend="mock")
        angle = exp.run_simulation({"friction_coefficient": 0.5})
        expected = 20.0 + 0.5 * 24.0
        assert math.isclose(angle, expected, abs_tol=1e-9)

    @pytest.mark.parametrize(
        "friction,expected",
        [
            (0.0, 20.0),
            (0.25, 26.0),
            (0.5, 32.0),
            (1.0, 44.0),
        ],
    )
    def test_mock_formula_parametrized(self, friction: float, expected: float) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        angle = exp.run_simulation({"friction_coefficient": friction})
        assert math.isclose(angle, expected, abs_tol=1e-9)

    def test_mock_angle_within_physical_range(self) -> None:
        """DbC postcondition: angle of repose must be in [0, 90) degrees."""
        for friction in [0.0, 0.3, 0.5, 0.9]:
            exp = AngleOfReposeExperiment(backend="mock")
            angle = exp.run_simulation({"friction_coefficient": friction})
            assert (
                0.0 <= angle < 90.0
            ), f"Angle {angle} out of physical range for mu={friction}"

    def test_use_mock_override_forces_mock_path(self) -> None:
        """use_mock=True forces mock path regardless of backend string."""
        exp = AngleOfReposeExperiment(backend="mpm", use_mock=True)
        assert exp._use_mock is True
        angle = exp.run_simulation({"friction_coefficient": 0.5})
        assert math.isclose(angle, 32.0, abs_tol=1e-9)


class TestBackendValidation:
    """Backend validation raises BackendNotImplementedError for unsupported values."""

    def test_unsupported_backend_raises(self) -> None:
        """DbC precondition: backend must be one of mock/mpm/mujoco."""
        with pytest.raises(BackendNotImplementedError):
            AngleOfReposeExperiment(backend="liggghts")

    def test_unknown_backend_name_raises(self) -> None:
        with pytest.raises(BackendNotImplementedError):
            AngleOfReposeExperiment(backend="dem_plus_plus")


class TestCalibration:
    """Calibration finds friction that minimises residual vs target angle."""

    def test_calibrate_returns_friction_dict(self) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        result = exp.calibrate()
        assert "friction_coefficient" in result
        assert 0.0 <= result["friction_coefficient"] <= 1.0

    def test_calibrate_target_angle_attribute(self) -> None:
        exp = AngleOfReposeExperiment(backend="mock")
        assert hasattr(exp, "target_angle")
        assert exp.target_angle == 32.0


class TestConvenienceFunction:
    """compute_angle_of_repose convenience wrapper behaves correctly."""

    def test_compute_angle_mock(self) -> None:
        angle = compute_angle_of_repose(friction=0.5, backend="mpm", use_mock=True)
        assert math.isclose(angle, 32.0, abs_tol=1e-9)

    def test_compute_angle_explicit_mock_backend(self) -> None:
        angle = compute_angle_of_repose(friction=0.3, backend="mock")
        assert math.isclose(angle, 20.0 + 0.3 * 24.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Real MuJoCo path (slow, optional)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMujocoPhysicalBounds:
    """Real MuJoCo hopper returns an angle within physical bounds [5, 50] deg."""

    def test_mujoco_angle_within_bounds(self) -> None:
        mujoco = pytest.importorskip("mujoco")  # noqa: F841
        from bunkershot3d.calibration.angle_of_repose import _mujoco_angle_of_repose

        angle = _mujoco_angle_of_repose(friction=0.5, n_grains=80, settle_steps=1500)
        assert (
            5.0 <= angle <= 50.0
        ), f"Angle {angle:.1f} out of physical range [5, 50] deg"
