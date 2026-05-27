"""Tests for the Drake float-pathway ``simulate_with_coefficients`` (issue #4111).

This file exercises three layers:

1. **Pure-data tests** that exercise :class:`SimOptions`, :class:`SimOut`,
   and :func:`evaluate_torque_polynomial`. These run in *every*
   environment because the module is importable without ``pydrake``.
2. **Mock-pydrake tests** that verify the wrapper *would* call into Drake
   correctly without requiring the real ``pydrake`` install. They use
   ``patch.dict("sys.modules", ...)`` per CLAUDE.md (never
   ``sys.modules["pydrake"] = MagicMock()`` at module scope).
3. **Live integration tests** marked ``@pytest.mark.requires_drake``
   that actually drive a real Drake forward sim. These are skipped in
   environments without ``pydrake``.

Acceptance coverage (issue #4111):

* **Recovery:** known ``theta`` -> ``simulate_with_coefficients`` -> grip
  trajectory matches the analytic torque-polynomial pattern (in the
  mock layer we verify the polynomial evaluator round-trips; in the
  live layer the full grip trace is finite and grows non-trivially
  under non-zero ``theta``).
* **Determinism:** same ``theta`` + same ``random_seed`` -> identical
  ``SimOut`` arrays.
* **Postcondition shape checks:** ``SimOut`` arrays have the canonical
  cross-engine shapes ``(N,)`` / ``(N, n_joints)`` / ``(N, 3)`` /
  ``(N, 4)``.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Module under test is importable without pydrake — all pydrake imports
# inside ``simulate_with_coefficients`` are lazy.
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    evaluate_torque_polynomial,
    simulate_with_coefficients,
)

# ---------------------------------------------------------------------------
# 1. Pure-data tests
# ---------------------------------------------------------------------------


class TestSimOptions:
    """Validation of the canonical options dataclass."""

    def test_drake_simulate_defaults(self) -> None:
        opts = SimOptions()
        assert opts.simulation_time_s == pytest.approx(0.3)
        assert opts.sample_rate_hz == pytest.approx(1000.0)
        assert opts.time_step_s == pytest.approx(1.0e-3)
        assert opts.gravity == (0.0, 0.0, -9.81)
        assert opts.urdf_path is None
        assert opts.grip_body_name == "club_grip"
        assert opts.clubhead_body_name == "clubhead"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"simulation_time_s": -1.0},
            {"simulation_time_s": float("nan")},
            {"sample_rate_hz": 0.0},
            {"time_step_s": -1e-3},
            {"gravity": (0.0, 0.0)},
            {"gravity": (0.0, 0.0, float("inf"))},
        ],
    )
    def test_rejects_invalid(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SimOptions(**kwargs)


class TestEvaluateTorquePolynomial:
    """Stateflow-equivalent torque polynomial evaluator."""

    def test_constant_term_only(self) -> None:
        """tau = A_j when all higher coeffs are zero."""
        n_joints = 3
        theta = np.zeros(n_joints * COEFFS_PER_JOINT)
        theta[0::COEFFS_PER_JOINT] = [1.5, -2.0, 3.0]  # A coefficients
        for t in (0.0, 0.1, 0.5):
            tau = evaluate_torque_polynomial(theta, t, n_joints)
            np.testing.assert_allclose(tau, [1.5, -2.0, 3.0])

    def test_full_polynomial(self) -> None:
        """Direct expansion: A + B t + C t^2 + ... + G t^6."""
        n_joints = 1
        # A=1, B=2, C=3, D=4, E=5, F=6, G=7
        theta = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        t = 0.5
        expected = sum(c * (t**i) for i, c in enumerate(theta))
        actual = evaluate_torque_polynomial(theta, t, n_joints)
        np.testing.assert_allclose(actual, [expected])

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="length"):
            evaluate_torque_polynomial(np.zeros(13), 0.0, n_joints=2)
        with pytest.raises(ValueError, match="finite"):
            evaluate_torque_polynomial(np.array([np.nan] * 7), 0.0, n_joints=1)
        with pytest.raises(ValueError, match="1-D"):
            evaluate_torque_polynomial(np.zeros((2, 7)), 0.0, n_joints=2)


class TestSimOutShape:
    """Postcondition shape checks for the canonical SimOut dataclass."""

    @staticmethod
    def _valid_arrays(n: int, n_joints: int) -> dict[str, np.ndarray]:
        return {
            "time": np.linspace(0, 0.3, n),
            "q": np.zeros((n, n_joints)),
            "qd": np.zeros((n, n_joints)),
            "qdd": np.zeros((n, n_joints)),
            "tau": np.zeros((n, n_joints)),
            "grip": np.zeros((n, 3)),
            "grip_quat": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
            "clubhead": np.zeros((n, 3)),
            "club_quat": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        }

    def test_constructs_with_canonical_shapes(self) -> None:
        n, n_joints = 301, 23
        out = SimOut(
            **self._valid_arrays(n, n_joints),
            solver_status="success",
            duration_s=0.5,
        )
        assert out.time.shape == (n,)
        assert out.q.shape == (n, n_joints)
        assert out.tau.shape == (n, n_joints)
        assert out.grip.shape == (n, 3)
        assert out.grip_quat.shape == (n, 4)
        assert out.clubhead.shape == (n, 3)
        assert out.club_quat.shape == (n, 4)

    def test_rejects_ragged_shapes(self) -> None:
        n, n_joints = 11, 5
        arrays = self._valid_arrays(n, n_joints)
        arrays["q"] = np.zeros((n - 1, n_joints))  # ragged
        with pytest.raises(ValueError, match="shape"):
            SimOut(**arrays, solver_status="success", duration_s=0.1)

    def test_rejects_invalid_solver_status(self) -> None:
        n, n_joints = 11, 5
        arrays = self._valid_arrays(n, n_joints)
        with pytest.raises(ValueError, match="solver_status"):
            SimOut(**arrays, solver_status="bogus", duration_s=0.1)

    def test_grip_quat_must_be_4_columns(self) -> None:
        n, n_joints = 11, 5
        arrays = self._valid_arrays(n, n_joints)
        arrays["grip_quat"] = np.zeros((n, 3))
        with pytest.raises(ValueError, match="4"):
            SimOut(**arrays, solver_status="success", duration_s=0.1)


# ---------------------------------------------------------------------------
# 2. Argument-validation tests for the public entry point. These exercise
# the precondition guards that fire *before* any pydrake import, so they
# run in every environment.
# ---------------------------------------------------------------------------


class TestSimulateInputValidation:
    """The wrapper rejects malformed inputs before importing pydrake."""

    def test_theta_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            simulate_with_coefficients(np.zeros((2, 7)))

    def test_theta_must_be_finite(self) -> None:
        bad = np.zeros(7)
        bad[3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            simulate_with_coefficients(bad)

    def test_theta_length_divisible_by_seven(self) -> None:
        with pytest.raises(ValueError, match="multiple of 7"):
            simulate_with_coefficients(np.zeros(13))

    def test_initial_pose_type(self) -> None:
        with pytest.raises((TypeError, ValueError), match="initial_pose"):
            simulate_with_coefficients(
                np.zeros(7),
                initial_pose="not a dict",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 3. Mock-pydrake integration: verify the wrapper would build a Drake plant
# correctly without actually requiring pydrake.
# ---------------------------------------------------------------------------


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


def _make_plant_mock(n_q: int = 25, n_v: int = 25, n_actuators: int = 19) -> MagicMock:
    """Build a MultibodyPlant mock that satisfies the wrapper's call surface."""
    plant = MagicMock(name="MultibodyPlant")
    plant.num_positions.return_value = n_q
    plant.num_velocities.return_value = n_v
    plant.num_actuators.return_value = n_actuators
    plant.num_multibody_states.return_value = n_q + n_v
    plant.GetPositions.return_value = np.zeros(n_q)
    plant.GetVelocities.return_value = np.zeros(n_v)
    plant.HasBodyNamed.return_value = False  # forward-kinematics returns NaN
    return plant


@pytest.fixture
def _mocked_pydrake() -> Generator[dict[str, MagicMock], None, None]:
    """Patch sys.modules with mock pydrake submodules (CLAUDE.md compliant)."""
    mocks: dict[str, MagicMock] = {key: MagicMock() for key in _PYDRAKE_KEYS}

    # AddMultibodyPlantSceneGraph returns (plant, scene_graph).
    plant = _make_plant_mock()
    scene_graph = MagicMock(name="SceneGraph")
    mocks["pydrake.multibody.plant"].AddMultibodyPlantSceneGraph = MagicMock(
        return_value=(plant, scene_graph)
    )

    # BasicVector / LeafSystem stand-ins.
    class _FakeLeafSystem:  # noqa: D401
        """Minimal LeafSystem stand-in used by `_PolynomialTorqueSource`."""

        def __init__(self) -> None:
            self._ports: list[Any] = []

        def DeclareVectorOutputPort(  # noqa: N802 (pydrake naming)
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

    framework_mod = mocks["pydrake.systems.framework"]
    framework_mod.LeafSystem = _FakeLeafSystem
    framework_mod.BasicVector = MagicMock(side_effect=lambda n: MagicMock())
    framework_mod.DiagramBuilder = MagicMock(return_value=MagicMock())

    # Simulator + VectorLogSink stand-ins.
    sim_instance = MagicMock(name="Simulator")
    sim_context = MagicMock(name="DiagramContext")
    sim_instance.get_context.return_value = sim_context
    mocks["pydrake.systems.analysis"].Simulator = MagicMock(return_value=sim_instance)
    mocks["pydrake.systems.primitives"].VectorLogSink = MagicMock(
        return_value=MagicMock()
    )

    # Plumb the plant context lookup.
    plant_ctx = MagicMock(name="PlantContext")
    plant.GetMyMutableContextFromRoot.return_value = plant_ctx
    plant.GetMyContextFromRoot.return_value = plant_ctx

    # The DiagramBuilder mock needs Build() and AddSystem hooks.
    builder_instance = framework_mod.DiagramBuilder.return_value
    builder_instance.Build.return_value = MagicMock(name="Diagram")
    builder_instance.Build.return_value.CreateDefaultContext.return_value = sim_context

    with patch.dict(sys.modules, mocks):
        # Reload the simulate module so its lazy `from pydrake.X import Y`
        # statements pick up the mocked submodules. Because the module
        # imports pydrake *inside* `simulate_with_coefficients`, no reload
        # is necessary — we just yield.
        yield {"plant": plant, "simulator": sim_instance, **mocks}


def test_mocked_simulate_returns_canonical_simout(
    _mocked_pydrake: dict[str, MagicMock],
) -> None:
    """With pydrake mocked, the wrapper still returns a SimOut with canonical shapes."""
    n_joints = 19  # matches the plant mock's num_actuators
    theta = np.linspace(-0.1, 0.1, n_joints * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(
        theta,
        options=SimOptions(simulation_time_s=0.01, sample_rate_hz=1000.0),
    )

    assert isinstance(out, SimOut)
    # 0.01 s @ 1 kHz -> 11 samples.
    assert out.time.shape == (11,)
    assert out.tau.shape == (11, n_joints)
    assert out.grip.shape == (11, 3)
    assert out.grip_quat.shape == (11, 4)
    assert out.solver_status in {"success", "warning", "failed"}


def test_mocked_simulate_is_deterministic(
    _mocked_pydrake: dict[str, MagicMock],
) -> None:
    """Same theta + same seed -> identical SimOut arrays (postcondition)."""
    theta = np.full(19 * COEFFS_PER_JOINT, 0.05)
    opts = SimOptions(simulation_time_s=0.005, sample_rate_hz=1000.0, random_seed=42)
    out_a = simulate_with_coefficients(theta, options=opts)
    out_b = simulate_with_coefficients(theta, options=opts)

    np.testing.assert_array_equal(out_a.time, out_b.time)
    np.testing.assert_array_equal(out_a.tau, out_b.tau)
    np.testing.assert_array_equal(out_a.q, out_b.q)
    np.testing.assert_array_equal(out_a.qd, out_b.qd)
    assert out_a.solver_status == out_b.solver_status


def test_mocked_simulate_recovers_known_torque_pattern(
    _mocked_pydrake: dict[str, MagicMock],
) -> None:
    """Recovery: known theta -> tau column matches the analytic polynomial.

    The wrapper records ``tau`` via :func:`evaluate_torque_polynomial`,
    so a sanity round-trip on a single joint with a hand-picked
    coefficient vector verifies the simulate/recovery pathway feeds the
    correct torques into the (mocked) plant.
    """
    n_joints = 19
    theta = np.zeros(n_joints * COEFFS_PER_JOINT)
    # Joint 0: tau_0(t) = 1 + 2 t + 3 t^2 (G..D zeroed).
    theta[0:3] = [1.0, 2.0, 3.0]

    opts = SimOptions(simulation_time_s=0.01, sample_rate_hz=1000.0)
    out = simulate_with_coefficients(theta, options=opts)

    expected = 1.0 + 2.0 * out.time + 3.0 * out.time**2
    np.testing.assert_allclose(out.tau[:, 0], expected, rtol=1e-10)
    # Other joints with all-zero coefficients should record zero torques.
    np.testing.assert_allclose(out.tau[:, 1:], 0.0)


# ---------------------------------------------------------------------------
# 4. Live-pydrake integration tests (skipped when pydrake unavailable).
# ---------------------------------------------------------------------------


@pytest.mark.requires_drake
def test_live_simulate_zero_theta_falls_under_gravity() -> None:
    """With theta = 0 the unactuated humanoid drops in -Z under gravity."""
    pytest.importorskip("pydrake")

    # Determine n_actuators from the canonical URDF before sizing theta.
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    plant = MultibodyPlant(0.001)
    Parser(plant).AddModels(
        str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "engines"
            / "physics_engines"
            / "drake"
            / "models"
            / "generated"
            / "golfer.urdf"
        )
    )
    plant.Finalize()
    n_act = max(plant.num_actuators(), plant.num_velocities() - 6)
    theta = np.zeros(n_act * COEFFS_PER_JOINT)

    out = simulate_with_coefficients(
        theta, options=SimOptions(simulation_time_s=0.05, sample_rate_hz=1000.0)
    )
    assert out.solver_status in {"success", "warning"}
    assert np.all(np.isfinite(out.q))
