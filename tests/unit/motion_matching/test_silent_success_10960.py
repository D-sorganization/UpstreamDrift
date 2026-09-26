"""Unit tests for issue #10960 slice P2-b: Fabricated evidence audit.

Verifies:
1. Forced SciPy failure in kinematic smoothing reports smoothing_applied=False (or raises).
2. Forced energy evaluation errors in Drake simulate produce NaN (not 0.0) and solver_status='partial'.
3. Missing grip/clubhead poses in Drake simulate produce NaN (not zeros) and solver_status='partial'.
4. Clean Drake simulate run without errors reports solver_status='success'.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)
from src.shared.python.motion_matching.kinematic_smoothing import (
    SmoothedTrajectory,
    smooth_kinematic_trajectory,
)

pytestmark = [pytest.mark.unit]


_PYDRAKE_KEYS = [
    "pydrake",
    "pydrake.all",
    "pydrake.multibody",
    "pydrake.multibody.parsing",
    "pydrake.multibody.plant",
    "pydrake.multibody.tree",
    "pydrake.systems",
    "pydrake.systems.analysis",
    "pydrake.systems.framework",
    "pydrake.systems.primitives",
    "pydrake.math",
]


class _FakeLeafSystem:
    """Minimal LeafSystem stand-in for mocked Drake tests."""

    def __init__(self) -> None:
        self._ports: list[Any] = []

    def DeclareVectorOutputPort(
        self,
        name: str,
        model_vector: Any,
        calc_callback: Any,
    ) -> Any:
        port = MagicMock(name=f"OutputPort[{name}]")
        self._ports.append(port)
        return port

    def get_output_port(self, idx: int) -> Any:
        return self._ports[idx] if self._ports else MagicMock()


class _MockQuaternion:
    def w(self) -> float:
        return 1.0

    def x(self) -> float:
        return 0.0

    def y(self) -> float:
        return 0.0

    def z(self) -> float:
        return 0.0


class _MockRotation:
    def ToQuaternion(self) -> _MockQuaternion:
        return _MockQuaternion()


class _MockTransform:
    def translation(self) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=np.float64)

    def rotation(self) -> _MockRotation:
        return _MockRotation()


def _create_mocked_pydrake_env(
    *,
    has_bodies: bool = True,
    energy_error: bool = False,
) -> tuple[dict[str, MagicMock], MagicMock]:
    """Create a mock pydrake environment for testing simulate_with_coefficients."""
    mocks: dict[str, MagicMock] = {key: MagicMock() for key in _PYDRAKE_KEYS}

    n_q = 25
    n_v = 25
    n_actuators = 19

    plant = MagicMock(name="MultibodyPlant")
    plant.num_positions.return_value = n_q
    plant.num_velocities.return_value = n_v
    plant.num_actuators.return_value = n_actuators
    plant.num_multibody_states.return_value = n_q + n_v
    plant.GetPositions.return_value = np.zeros(n_q)
    plant.GetVelocities.return_value = np.zeros(n_v)

    if has_bodies:
        plant.HasBodyNamed.return_value = True
        body_mock = MagicMock(name="Body")
        plant.GetBodyByName.return_value = body_mock
        plant.CalcRelativeTransform.return_value = _MockTransform()
    else:
        plant.HasBodyNamed.return_value = False

    if energy_error:
        plant.CalcKineticEnergy.side_effect = ValueError(
            "kinetic energy calculation failed"
        )
        plant.CalcPotentialEnergy.side_effect = TypeError(
            "potential energy calculation failed"
        )
    else:
        plant.CalcKineticEnergy.return_value = 15.5
        plant.CalcPotentialEnergy.return_value = 3.2

    scene_graph = MagicMock(name="SceneGraph")
    mocks["pydrake.multibody.plant"].AddMultibodyPlantSceneGraph = MagicMock(
        return_value=(plant, scene_graph)
    )

    framework_mod = mocks["pydrake.systems.framework"]
    framework_mod.LeafSystem = _FakeLeafSystem
    framework_mod.BasicVector = MagicMock(side_effect=lambda n: MagicMock())
    framework_mod.DiagramBuilder = MagicMock(return_value=MagicMock())

    sim_instance = MagicMock(name="Simulator")
    sim_context = MagicMock(name="DiagramContext")
    sim_instance.get_context.return_value = sim_context
    mocks["pydrake.systems.analysis"].Simulator = MagicMock(return_value=sim_instance)
    mocks["pydrake.systems.primitives"].VectorLogSink = MagicMock(
        return_value=MagicMock()
    )

    plant_ctx = MagicMock(name="PlantContext")
    plant.GetMyMutableContextFromRoot.return_value = plant_ctx
    plant.GetMyContextFromRoot.return_value = plant_ctx

    builder_instance = framework_mod.DiagramBuilder.return_value
    builder_instance.Build.return_value = MagicMock(name="Diagram")
    builder_instance.Build.return_value.CreateDefaultContext.return_value = sim_context

    return mocks, plant


def test_smoothing_forced_scipy_failure_reports_not_applied() -> None:
    """Forced SciPy failure yields smoothing_applied=False (or raises), not silent success."""
    time_s = np.linspace(0.0, 0.5, 50)
    q = np.sin(2.0 * np.pi * 2.0 * time_s)

    with patch(
        "scipy.signal.sosfiltfilt", side_effect=ValueError("SOS filtering failed")
    ):
        res = smooth_kinematic_trajectory(time_s, q, cutoff_hz=10.0)
        assert isinstance(res, SmoothedTrajectory)
        assert res.smoothing_applied is False
        # Unsmoothed data returned rather than false smoothed claims
        np.testing.assert_allclose(res.q, q)


def test_smoothing_clean_run_reports_applied() -> None:
    """Normal execution with functional SciPy yields smoothing_applied=True."""
    time_s = np.linspace(0.0, 0.5, 50)
    q = np.sin(2.0 * np.pi * 2.0 * time_s)

    res = smooth_kinematic_trajectory(time_s, q, cutoff_hz=10.0)
    assert isinstance(res, SmoothedTrajectory)
    assert res.smoothing_applied is True
    # Can still unpack as a 3-tuple (q, v, a) for backward compatibility
    q_s, v_s, a_s = res
    assert q_s.shape == q.shape
    assert v_s.shape == q.shape
    assert a_s.shape == q.shape


def test_drake_simulate_clean_run_reports_success() -> None:
    """A clean rollout with valid poses and energies reports solver_status='success'."""
    mocks, _ = _create_mocked_pydrake_env(has_bodies=True, energy_error=False)
    with patch.dict(sys.modules, mocks):
        n_joints = 19
        theta = np.zeros(n_joints * COEFFS_PER_JOINT)
        out = simulate_with_coefficients(
            theta,
            options=SimOptions(
                simulation_time_s=0.01, sample_rate_hz=1000.0, compute_energy=True
            ),
        )

        assert isinstance(out, SimOut)
        assert out.solver_status == "success"
        assert np.all(np.isfinite(out.kinetic_energy))
        assert np.all(np.isfinite(out.potential_energy))
        assert np.all(np.isfinite(out.grip))
        assert np.all(np.isfinite(out.clubhead))


def test_drake_simulate_forced_energy_error_yields_nan_and_partial() -> None:
    """A forced energy calculation failure yields NaN (not 0.0) and solver_status='partial'."""
    mocks, _ = _create_mocked_pydrake_env(has_bodies=True, energy_error=True)
    with patch.dict(sys.modules, mocks):
        n_joints = 19
        theta = np.zeros(n_joints * COEFFS_PER_JOINT)
        out = simulate_with_coefficients(
            theta,
            options=SimOptions(
                simulation_time_s=0.01, sample_rate_hz=1000.0, compute_energy=True
            ),
        )

        assert isinstance(out, SimOut)
        assert out.solver_status == "partial"
        assert np.all(np.isnan(out.kinetic_energy))
        assert np.all(np.isnan(out.potential_energy))


def test_drake_simulate_missing_grip_club_poses_yield_nan_and_partial() -> None:
    """Missing grip/clubhead bodies in plant yield NaN poses (not zeros) and solver_status='partial'."""
    mocks, _ = _create_mocked_pydrake_env(has_bodies=False, energy_error=False)
    with patch.dict(sys.modules, mocks):
        n_joints = 19
        theta = np.zeros(n_joints * COEFFS_PER_JOINT)
        out = simulate_with_coefficients(
            theta,
            options=SimOptions(
                simulation_time_s=0.01, sample_rate_hz=1000.0, compute_energy=True
            ),
        )

        assert isinstance(out, SimOut)
        assert out.solver_status == "partial"
        assert np.all(np.isnan(out.grip))
        assert np.all(np.isnan(out.grip_quat))
        assert np.all(np.isnan(out.clubhead))
        assert np.all(np.isnan(out.club_quat))


def test_simout_missing_grip_and_energy_defaults_to_nan() -> None:
    """SimOut constructor defaults missing grip/club poses and energy to NaN, not zeros."""
    n = 10
    time = np.linspace(0.0, 0.1, n)
    out = SimOut(time=time, solver_status="partial")
    assert np.all(np.isnan(out.grip))
    assert np.all(np.isnan(out.grip_quat))
    assert np.all(np.isnan(out.clubhead))
    assert np.all(np.isnan(out.club_quat))
    assert np.all(np.isnan(out.kinetic_energy))
    assert np.all(np.isnan(out.potential_energy))
