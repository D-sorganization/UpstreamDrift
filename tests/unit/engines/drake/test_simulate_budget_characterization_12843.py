"""Characterization tests pinning Drake ``SimOut`` / ``simulate_with_coefficients``.

Written before the issue #12843 budget refactor so that the dataclass
conversion of ``SimOut``, the parameter-object conversion of
``_record_rollout`` and the decomposition of ``simulate_with_coefficients``
provably keep behaviour identical. pydrake is mocked (``patch.dict`` on
``sys.modules``) so these run headless on any host.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_matching import (
    simulate as sim_mod,
)
from src.shared.python.core.contracts.exceptions import PreconditionError
from src.engines.physics_engines.drake.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)

_FIELD_ORDER = [
    "time",
    "q",
    "qd",
    "qdd",
    "tau",
    "grip",
    "grip_quat",
    "clubhead",
    "club_quat",
    "solver_status",
    "duration_s",
    "kinetic_energy",
    "potential_energy",
    "meta",
]


def _eye_series(n: int) -> np.ndarray:
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


# --------------------------------------------------------------------------- #
# SimOut
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_simout_is_dataclass_with_canonical_field_order() -> None:
    assert dataclasses.is_dataclass(SimOut)
    assert [f.name for f in dataclasses.fields(SimOut)] == _FIELD_ORDER


@pytest.mark.unit
def test_simout_time_only_defaults() -> None:
    out = SimOut(time=np.linspace(0.0, 1.0, 5))
    assert out.q.shape == (5, 0)
    assert out.qd.shape == (5, 0)
    assert out.qdd.shape == (5, 0)
    assert out.tau.shape == (5, 0)
    assert out.grip.shape == (5, 3) and np.all(np.isnan(out.grip))
    assert out.clubhead.shape == (5, 3) and np.all(np.isnan(out.clubhead))
    assert out.grip_quat.shape == (5, 4) and np.all(np.isnan(out.grip_quat))
    assert out.club_quat.shape == (5, 4) and np.all(np.isnan(out.club_quat))
    assert out.kinetic_energy.shape == (5,) and np.all(np.isnan(out.kinetic_energy))
    assert out.potential_energy.shape == (5,)
    assert out.solver_status == "success"
    assert out.duration_s == 0.0
    assert out.meta == {}


@pytest.mark.unit
def test_simout_qdd_tau_default_to_nv_columns() -> None:
    out = SimOut(time=np.zeros(4), qd=np.zeros((4, 3)))
    assert out.qdd.shape == (4, 3)
    assert out.tau.shape == (4, 3)


@pytest.mark.unit
def test_simout_requires_time_or_t() -> None:
    with pytest.raises(ValueError, match="Either 'time' or 't' must be provided"):
        SimOut()


@pytest.mark.unit
def test_simout_time_must_be_1d() -> None:
    with pytest.raises(ValueError, match="SimOut.time must be 1-D"):
        SimOut(time=np.zeros((2, 2)))


@pytest.mark.unit
def test_simout_time_wins_over_t_alias() -> None:
    assert SimOut(time=np.arange(3.0), t=np.arange(7.0)).time.shape == (3,)


@pytest.mark.unit
def test_simout_positional_order_and_float_duration() -> None:
    n = 3
    t = np.arange(float(n))
    q = np.ones((n, 2))
    out = SimOut(
        t,
        q,
        q,
        q,
        q,
        np.zeros((n, 3)),
        np.zeros((n, 4)),
        np.ones((n, 3)),
        np.ones((n, 4)),
        "warning",
        2,
        np.ones(n),
        np.full(n, 2.0),
        {"k": 1},
    )
    assert out.solver_status == "warning"
    assert out.duration_s == 2.0 and isinstance(out.duration_s, float)
    assert np.all(out.clubhead == 1.0)
    assert np.all(out.potential_energy == 2.0)
    assert out.meta == {"k": 1}


@pytest.mark.unit
def test_simout_aliases_and_rotation_to_quaternion() -> None:
    n = 4
    out = SimOut(
        t=np.arange(float(n)),
        grip_position=np.ones((n, 3)),
        grip_rotation=_eye_series(n),
        clubhead_position=np.full((n, 3), 2.0),
        clubhead_rotation=_eye_series(n),
    )
    assert np.all(out.grip == 1.0) and np.all(out.clubhead == 2.0)
    np.testing.assert_allclose(out.grip_quat, np.tile([1.0, 0, 0, 0], (n, 1)))
    np.testing.assert_allclose(out.club_quat, np.tile([1.0, 0, 0, 0], (n, 1)))
    np.testing.assert_allclose(out.grip_rotation, _eye_series(n), atol=1e-12)
    np.testing.assert_allclose(out.clubhead_rotation, _eye_series(n), atol=1e-12)
    assert out.t is out.time
    assert out.grip_position is out.grip
    assert out.clubhead_position is out.clubhead


@pytest.mark.unit
def test_simout_meta_beats_metadata_alias_and_metadata_property() -> None:
    meta = {"a": 1}
    out = SimOut(time=np.zeros(1), meta=meta, metadata={"b": 2})
    assert out.meta is meta and out.metadata is meta
    only_alias = {"b": 2}
    assert SimOut(time=np.zeros(1), metadata=only_alias).meta is only_alias


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"q": np.zeros((2, 1))}, r"SimOut.q must have shape \(N=3"),
        ({"qd": np.zeros(3)}, r"SimOut.qd must have shape \(N=3"),
        ({"grip": np.zeros((3, 2))}, r"SimOut.grip must have shape \(N, 3\)"),
        ({"club_quat": np.zeros((3, 3))}, r"SimOut.club_quat must have shape"),
        ({"solver_status": "bogus"}, "SimOut.solver_status must be"),
        ({"kinetic_energy": np.zeros(2)}, r"SimOut.kinetic_energy must have shape"),
        ({"potential_energy": np.zeros((3, 1))}, r"SimOut.potential_energy must"),
    ],
)
def test_simout_validation_errors(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SimOut(time=np.zeros(3), **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("status", ["success", "warning", "failed", "partial"])
def test_simout_accepts_every_canonical_status(status: str) -> None:
    assert SimOut(time=np.zeros(2), solver_status=status).solver_status == status


@pytest.mark.unit
def test_simout_frozen_unknown_kwarg_and_meta_not_shared() -> None:
    a = SimOut(time=np.zeros(2))
    b = SimOut(time=np.zeros(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.solver_status = "failed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        SimOut(time=np.zeros(2), bogus=1)  # type: ignore[call-arg]
    a.meta["x"] = 1
    assert b.meta == {}


# --------------------------------------------------------------------------- #
# Mocked pydrake harness
# --------------------------------------------------------------------------- #

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
    def __init__(self) -> None:
        self._ports: list[Any] = []

    def DeclareVectorOutputPort(self, name: str, model: Any, cb: Any) -> Any:
        self.calc = cb
        port = MagicMock(name=f"port[{name}]")
        self._ports.append(port)
        return port

    def get_output_port(self, idx: int) -> Any:
        return self._ports[idx]


class _Quat:
    def w(self) -> float:
        return 1.0

    def x(self) -> float:
        return 0.5

    def y(self) -> float:
        return 0.25

    def z(self) -> float:
        return 0.125


class _Rot:
    def ToQuaternion(self) -> _Quat:
        return _Quat()


class _Xf:
    def translation(self) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def rotation(self) -> _Rot:
        return _Rot()


_N_ACT = 4
_N_Q = 5
_N_V = 5


def _make_env(
    *, advance_error_at: float | None = None, actuators: int = _N_ACT
) -> tuple[dict[str, MagicMock], MagicMock, MagicMock]:
    mocks = {key: MagicMock() for key in _PYDRAKE_KEYS}
    plant = MagicMock(name="plant")
    plant.num_positions.return_value = _N_Q
    plant.num_velocities.return_value = _N_V
    plant.num_actuators.return_value = actuators
    plant.num_multibody_states.return_value = _N_Q + _N_V
    clock = {"k": 0}

    def _velocities(_ctx: Any) -> np.ndarray:
        clock["k"] += 1
        return np.full(_N_V, float(clock["k"]) ** 2)

    plant.GetPositions.return_value = np.arange(_N_Q, dtype=float)
    plant.GetVelocities.side_effect = _velocities
    plant.HasBodyNamed.return_value = True
    plant.CalcRelativeTransform.return_value = _Xf()
    plant.CalcKineticEnergy.return_value = 4.0
    plant.CalcPotentialEnergy.return_value = 1.5
    mocks["pydrake.multibody.plant"].AddMultibodyPlantSceneGraph = MagicMock(
        return_value=(plant, MagicMock())
    )
    fw = mocks["pydrake.systems.framework"]
    fw.LeafSystem = _FakeLeafSystem
    fw.BasicVector = MagicMock(side_effect=lambda n: MagicMock())
    builder = MagicMock(name="builder")
    fw.DiagramBuilder = MagicMock(return_value=builder)
    simulator = MagicMock(name="simulator")
    if advance_error_at is not None:

        def _advance(t: float) -> None:
            if t >= advance_error_at:
                raise RuntimeError("diverged")

        simulator.AdvanceTo.side_effect = _advance
    mocks["pydrake.systems.analysis"].Simulator = MagicMock(return_value=simulator)
    mocks["pydrake.systems.primitives"].VectorLogSink = MagicMock(
        return_value=MagicMock()
    )
    plant.simulator = simulator  # convenience for assertions
    plant.builder = builder
    return mocks, plant, simulator


def _theta(n_act: int = _N_ACT) -> np.ndarray:
    return np.random.default_rng(12843).normal(0.0, 2.0, n_act * COEFFS_PER_JOINT)


def _opts(**kw: Any) -> SimOptions:
    base: dict[str, Any] = {"simulation_time_s": 0.004, "sample_rate_hz": 1000.0}
    base.update(kw)
    return SimOptions(**base)


@pytest.fixture
def _no_urdf_load():  # type: ignore[no-untyped-def]
    with patch.object(sim_mod, "load_humanoid_into_plant") as loader:
        yield loader


@pytest.mark.unit
def test_mocked_simulate_golden_outputs(_no_urdf_load: MagicMock) -> None:
    mocks, plant, simulator = _make_env()
    theta = _theta()
    with patch.dict(sys.modules, mocks):
        out = simulate_with_coefficients(theta, _opts())
    n = 5
    np.testing.assert_allclose(out.time, np.arange(n) * 1e-3)
    assert out.solver_status == "success"
    assert out.q.shape == (n, _N_Q) and out.qd.shape == (n, _N_V)
    np.testing.assert_allclose(out.q, np.tile(np.arange(_N_Q, dtype=float), (n, 1)))
    # velocity log is k**2 for k = 1..5 (one GetVelocities per output sample)
    v = np.array([k**2 for k in range(1, n + 1)], dtype=float)
    np.testing.assert_allclose(out.qd[:, 0], v)
    qdd = np.zeros((n, _N_V))
    qdd[1:-1, 0] = (v[2:] - v[:-2]) / 2e-3
    qdd[0, 0] = (v[1] - v[0]) / 1e-3
    qdd[-1, 0] = (v[-1] - v[-2]) / 1e-3
    np.testing.assert_allclose(out.qdd[:, 0], qdd[:, 0])
    coeffs = theta.reshape(_N_ACT, COEFFS_PER_JOINT)
    for i, t in enumerate(out.time):
        expected = coeffs @ np.array([t**k for k in range(COEFFS_PER_JOINT)])
        np.testing.assert_allclose(out.tau[i], expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(out.grip, np.tile([0.1, 0.2, 0.3], (n, 1)))
    np.testing.assert_allclose(out.clubhead, np.tile([0.1, 0.2, 0.3], (n, 1)))
    np.testing.assert_allclose(out.grip_quat, np.tile([1.0, 0.5, 0.25, 0.125], (n, 1)))
    assert np.all(out.kinetic_energy == 4.0)
    assert np.all(out.potential_energy == 1.5)
    assert set(out.meta) == {
        "n_actuators",
        "num_positions",
        "num_velocities",
        "urdf_path",
        "basis",
    }
    assert out.meta["n_actuators"] == _N_ACT
    assert out.meta["num_positions"] == _N_Q
    assert out.meta["basis"] == "power"
    assert out.duration_s > 0.0
    simulator.set_publish_every_time_step.assert_called_once_with(False)
    simulator.Initialize.assert_called_once()
    assert simulator.AdvanceTo.call_count == n - 1


@pytest.mark.unit
def test_mocked_simulate_wiring_gravity_urdf_and_actuation(
    _no_urdf_load: MagicMock,
) -> None:
    mocks, plant, _sim = _make_env()
    with patch.dict(sys.modules, mocks):
        simulate_with_coefficients(
            _theta(), _opts(gravity=(0.0, 0.0, -1.0), urdf_path="x.urdf")
        )
    plant.mutable_gravity_field().set_gravity_vector.assert_called_once()
    np.testing.assert_allclose(
        plant.mutable_gravity_field().set_gravity_vector.call_args[0][0],
        [0.0, 0.0, -1.0],
    )
    assert str(_no_urdf_load.call_args[0][1]) == "x.urdf"
    plant.Finalize.assert_called_once()
    plant.builder.AddSystem.assert_called()
    plant.builder.Connect.assert_any_call(
        plant.builder.AddSystem.call_args_list[0][0][0].get_output_port(0),
        plant.get_actuation_input_port(),
    )
    plant.builder.Build.assert_called_once()


@pytest.mark.unit
def test_mocked_simulate_energy_off_leaves_nan_and_bernstein_tau(
    _no_urdf_load: MagicMock,
) -> None:
    mocks, _plant, _sim = _make_env()
    theta = _theta()
    with patch.dict(sys.modules, mocks):
        out = simulate_with_coefficients(
            theta, _opts(basis="bernstein", compute_energy=False, compute_qdd=False)
        )
    assert np.all(np.isnan(out.kinetic_energy))
    assert np.all(np.isnan(out.potential_energy))
    assert np.all(out.qdd == 0.0)
    assert out.meta["basis"] == "bernstein"
    expected = sim_mod.evaluate_bernstein_torque(
        theta.reshape(_N_ACT, COEFFS_PER_JOINT), float(out.time[2]), T_s=0.004, t0=0.0
    )
    np.testing.assert_allclose(out.tau[2], expected)


@pytest.mark.unit
def test_mocked_simulate_initial_pose_applied_and_validated(
    _no_urdf_load: MagicMock,
) -> None:
    mocks, plant, _sim = _make_env()
    q0 = np.linspace(0.0, 1.0, _N_Q)
    v0 = np.linspace(1.0, 2.0, _N_V)
    with patch.dict(sys.modules, mocks):
        simulate_with_coefficients(_theta(), _opts(), {"q": q0, "v": v0})
        np.testing.assert_allclose(plant.SetPositions.call_args[0][1], q0)
        np.testing.assert_allclose(plant.SetVelocities.call_args[0][1], v0)
        with pytest.raises(ValueError, match=r"initial_pose\['q'\] must have shape"):
            simulate_with_coefficients(_theta(), _opts(), {"q": np.zeros(_N_Q + 1)})
        with pytest.raises(ValueError, match=r"initial_pose\['v'\] must have shape"):
            simulate_with_coefficients(_theta(), _opts(), {"v": np.zeros(_N_V + 1)})


@pytest.mark.unit
def test_mocked_simulate_theta_actuator_mismatch(_no_urdf_load: MagicMock) -> None:
    mocks, _plant, _sim = _make_env()
    with patch.dict(sys.modules, mocks):
        with pytest.raises(ValueError, match="theta is sized for 2 joint"):
            simulate_with_coefficients(_theta(2), _opts())


@pytest.mark.unit
def test_mocked_simulate_advance_failure_reports_failed(
    _no_urdf_load: MagicMock,
) -> None:
    mocks, _plant, _sim = _make_env(advance_error_at=0.002)
    with patch.dict(sys.modules, mocks):
        out = simulate_with_coefficients(_theta(), _opts())
    assert out.solver_status == "failed"
    assert out.meta["error"] == repr(RuntimeError("diverged"))
    assert np.all(np.isnan(out.q[2:]))
    assert np.all(np.isfinite(out.q[:2]))


@pytest.mark.unit
def test_simulate_argument_validation_errors() -> None:
    with pytest.raises(ValueError, match="theta must be 1-D"):
        simulate_with_coefficients(np.zeros((7, 2)))
    with pytest.raises(ValueError, match="positive multiple of 7"):
        simulate_with_coefficients(np.zeros(0))
    # The decorator preconditions fire before the body runs.
    with pytest.raises(PreconditionError, match="multiple of 7"):
        simulate_with_coefficients(np.zeros(8))
    with pytest.raises(PreconditionError, match="theta must be finite"):
        simulate_with_coefficients(np.array([np.nan] * 7))
    with pytest.raises(PreconditionError, match="initial_pose type must be a dict"):
        simulate_with_coefficients(np.zeros(7), initial_pose=[1])  # type: ignore[arg-type]


@pytest.mark.unit
def test_simulate_without_pydrake_raises_import_error() -> None:
    blocked = dict.fromkeys(_PYDRAKE_KEYS)
    with patch.dict(sys.modules, blocked):
        with pytest.raises(ImportError, match="Drake runtime"):
            simulate_with_coefficients(np.zeros(7))


# --------------------------------------------------------------------------- #
# _record_rollout (parameter-object refactor target)
# --------------------------------------------------------------------------- #


def _logs(n: int, n_act: int) -> Any:
    nan = np.nan
    return sim_mod._RolloutLogs(
        q=np.full((n, _N_Q), nan),
        v=np.full((n, _N_V), nan),
        tau=np.full((n, n_act), nan),
        grip=np.full((n, 3), nan),
        grip_quat=np.full((n, 4), nan),
        clubhead=np.full((n, 3), nan),
        club_quat=np.full((n, 4), nan),
        kinetic_energy=np.full(n, nan),
        potential_energy=np.full(n, nan),
    )


def _roll(plant: Any, simulator: Any, grid: np.ndarray, logs: Any, **kw: Any) -> Any:
    return sim_mod._record_rollout(
        simulator=simulator,
        plant=plant,
        grid=grid,
        theta=kw.pop("theta", _theta()),
        n_actuators=kw.pop("n_actuators", _N_ACT),
        grip_body_name=kw.pop("grip_body_name", "club_grip"),
        clubhead_body_name=kw.pop("clubhead_body_name", "clubhead"),
        logs=logs,
        **kw,
    )


@pytest.mark.unit
def test_rollout_records_logs_for_clean_run() -> None:
    _mocks, plant, simulator = _make_env()
    grid = np.array([0.0, 0.001, 0.002])
    logs = _logs(3, _N_ACT)
    theta = _theta()
    result = _roll(plant, simulator, grid, logs, theta=theta)
    assert result.solver_status == "success" and result.error is None
    assert np.all(logs.kinetic_energy == 4.0)
    assert np.all(logs.potential_energy == 1.5)
    np.testing.assert_allclose(logs.grip[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(logs.club_quat[1], [1.0, 0.5, 0.25, 0.125])
    coeffs = theta.reshape(_N_ACT, COEFFS_PER_JOINT)
    np.testing.assert_allclose(logs.tau[0], coeffs[:, 0])
    np.testing.assert_allclose(logs.tau[2], coeffs @ [0.002**k for k in range(7)])
    assert simulator.AdvanceTo.call_count == 2


@pytest.mark.unit
def test_rollout_t0_and_bernstein_offsets() -> None:
    _mocks, plant, simulator = _make_env()
    theta = _theta()
    logs = _logs(2, _N_ACT)
    grid = np.array([0.0, 0.5])
    _roll(plant, simulator, grid, logs, theta=theta, t0=0.25)
    coeffs = theta.reshape(_N_ACT, COEFFS_PER_JOINT)
    np.testing.assert_allclose(logs.tau[1], coeffs @ [0.25**k for k in range(7)])
    logs_b = _logs(2, _N_ACT)
    _roll(plant, simulator, grid, logs_b, theta=theta, basis="bernstein", T_s=2.0)
    np.testing.assert_allclose(
        logs_b.tau[1], sim_mod.evaluate_bernstein_torque(coeffs, 0.5, T_s=2.0, t0=0.0)
    )


@pytest.mark.unit
def test_rollout_failure_returns_failed_and_stops() -> None:
    _mocks, plant, simulator = _make_env(advance_error_at=0.001)
    logs = _logs(3, _N_ACT)
    result = _roll(plant, simulator, np.array([0.0, 0.001, 0.002]), logs)
    assert result.solver_status == "failed"
    assert isinstance(result.error, RuntimeError)
    assert np.all(np.isnan(logs.q[1:]))


@pytest.mark.unit
def test_rollout_partial_when_energy_or_pose_missing() -> None:
    _mocks, plant, simulator = _make_env()
    plant.CalcKineticEnergy.return_value = float("inf")
    logs = _logs(2, _N_ACT)
    res = _roll(plant, simulator, np.array([0.0, 0.001]), logs)
    assert res.solver_status == "partial"
    assert np.all(np.isnan(logs.kinetic_energy))
    assert np.all(logs.potential_energy == 1.5)

    _m2, plant2, sim2 = _make_env()
    plant2.CalcPotentialEnergy.side_effect = TypeError("x")
    logs2 = _logs(2, _N_ACT)
    res2 = _roll(plant2, sim2, np.array([0.0, 0.001]), logs2)
    assert res2.solver_status == "partial"
    assert np.all(np.isnan(logs2.potential_energy))

    _m3, plant3, sim3 = _make_env()
    plant3.HasBodyNamed.return_value = False
    logs3 = _logs(2, _N_ACT)
    res3 = _roll(plant3, sim3, np.array([0.0, 0.001]), logs3)
    assert res3.solver_status == "partial"
    assert np.all(np.isnan(logs3.grip))


@pytest.mark.unit
def test_rollout_energy_disabled_and_zero_actuators() -> None:
    _mocks, plant, simulator = _make_env()
    logs = _logs(2, 0)
    res = _roll(
        plant,
        simulator,
        np.array([0.0, 0.001]),
        logs,
        theta=np.zeros(0),
        n_actuators=0,
        compute_energy=False,
    )
    assert res.solver_status == "success"
    assert np.all(np.isnan(logs.kinetic_energy))
    assert logs.tau.shape == (2, 0)
