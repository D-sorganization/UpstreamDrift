"""Characterization tests pinning Pinocchio ``SimOut`` / ``simulate_with_coefficients``.

Written before the issue #12843 budget refactor so that the dataclass
conversion of ``SimOut`` and the decomposition of
``simulate_with_coefficients`` provably keep behaviour identical.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.motion_matching import (
    simulate as sim_mod,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    is_pinocchio_available,
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


@pytest.mark.unit
def test_simout_is_dataclass_with_canonical_field_order() -> None:
    assert dataclasses.is_dataclass(SimOut)
    assert [f.name for f in dataclasses.fields(SimOut)] == _FIELD_ORDER


@pytest.mark.unit
def test_simout_time_only_fills_zero_defaults() -> None:
    out = SimOut(time=np.linspace(0.0, 1.0, 5))
    assert out.q.shape == (5, 0)
    assert out.qd.shape == (5, 0)
    assert out.qdd.shape == (5, 0)
    assert out.tau.shape == (5, 0)
    for arr, cols in ((out.grip, 3), (out.clubhead, 3)):
        assert arr.shape == (5, cols)
        assert np.all(arr == 0.0)
    for arr in (out.grip_quat, out.club_quat):
        assert arr.shape == (5, 4)
        assert np.all(arr == 0.0)
    assert out.kinetic_energy.shape == (5,)
    assert out.potential_energy.shape == (5,)
    assert np.all(out.kinetic_energy == 0.0)
    assert out.solver_status == "success"
    assert out.duration_s == 0.0
    assert out.meta == {}


@pytest.mark.unit
def test_simout_qdd_and_tau_default_to_nv_columns() -> None:
    out = SimOut(time=np.zeros(4), qd=np.zeros((4, 3)))
    assert out.qdd.shape == (4, 3)
    assert out.tau.shape == (4, 3)


@pytest.mark.unit
def test_simout_requires_time_or_t() -> None:
    with pytest.raises(ValueError, match="Either 'time' or 't' must be provided"):
        SimOut()


@pytest.mark.unit
def test_simout_time_wins_over_t_alias() -> None:
    out = SimOut(time=np.arange(3.0), t=np.arange(7.0))
    assert out.time.shape == (3,)


@pytest.mark.unit
def test_simout_accepts_positional_arguments_in_field_order() -> None:
    n = 3
    t = np.arange(float(n))
    q = np.ones((n, 2))
    out = SimOut(t, q)
    assert np.array_equal(out.time, t)
    assert np.array_equal(out.q, q)
    out2 = SimOut(
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
    assert out2.solver_status == "warning"
    assert out2.duration_s == 2.0
    assert isinstance(out2.duration_s, float)
    assert np.all(out2.clubhead == 1.0)
    assert np.all(out2.potential_energy == 2.0)
    assert out2.meta == {"k": 1}


@pytest.mark.unit
def test_simout_legacy_aliases_and_rotation_to_quaternion() -> None:
    n = 4
    out = SimOut(
        t=np.arange(float(n)),
        grip_position=np.ones((n, 3)),
        grip_rotation=_eye_series(n),
        clubhead_position=np.full((n, 3), 2.0),
        clubhead_rotation=_eye_series(n),
    )
    assert np.all(out.grip == 1.0)
    assert np.all(out.clubhead == 2.0)
    np.testing.assert_allclose(out.grip_quat, np.tile([1.0, 0, 0, 0], (n, 1)))
    np.testing.assert_allclose(out.club_quat, np.tile([1.0, 0, 0, 0], (n, 1)))
    np.testing.assert_allclose(out.grip_rotation, _eye_series(n), atol=1e-12)
    np.testing.assert_allclose(out.clubhead_rotation, _eye_series(n), atol=1e-12)
    assert out.t is out.time
    assert out.grip_position is out.grip
    assert out.clubhead_position is out.clubhead


@pytest.mark.unit
def test_simout_canonical_field_beats_alias() -> None:
    n = 2
    out = SimOut(
        time=np.zeros(n),
        grip=np.ones((n, 3)),
        grip_position=np.full((n, 3), 9.0),
        grip_quat=np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)),
        grip_rotation=_eye_series(n),
    )
    assert np.all(out.grip == 1.0)
    assert out.grip_quat[0, 1] == 1.0


@pytest.mark.unit
def test_simout_is_frozen_and_meta_not_shared() -> None:
    a = SimOut(time=np.zeros(2))
    b = SimOut(time=np.zeros(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.solver_status = "failed"  # type: ignore[misc]
    a.meta["x"] = 1
    assert b.meta == {}


@pytest.mark.unit
def test_simout_coerces_to_float64_arrays() -> None:
    out = SimOut(time=[0, 1, 2], q=[[1], [2], [3]])
    assert out.time.dtype == np.float64
    assert out.q.dtype == np.float64


@pytest.mark.unit
def test_simout_rejects_unknown_keyword() -> None:
    with pytest.raises(TypeError):
        SimOut(time=np.zeros(2), bogus=1)  # type: ignore[call-arg]


@pytest.mark.unit
def test_simout_meta_dict_is_kept_by_identity() -> None:
    meta = {"a": 1}
    assert SimOut(time=np.zeros(1), meta=meta).meta is meta


# --------------------------------------------------------------------------- #
# simulate_with_coefficients (real pinocchio only)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_simulate_without_pinocchio_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sim_mod, "is_pinocchio_available", lambda: False)
    with pytest.raises(ImportError, match="Pinocchio C\\+\\+ runtime"):
        simulate_with_coefficients(np.zeros(7))


def _golfer_nv() -> int:
    import pinocchio as pin

    return int(pin.buildModelFromUrdf(str(sim_mod._DEFAULT_URDF)).nv)


def _theta(nv: int) -> np.ndarray:
    rng = np.random.default_rng(12843)
    return rng.normal(0.0, 3.0, nv * COEFFS_PER_JOINT)


@pytest.fixture
def _real_pin() -> None:
    if not is_pinocchio_available():
        pytest.skip("Pinocchio C++ runtime not installed")
    if not sim_mod._DEFAULT_URDF.exists():
        pytest.skip("golfer.urdf missing")


@pytest.mark.unit
@pytest.mark.requires_pinocchio
@pytest.mark.parametrize("basis", ["power", "bernstein"])
def test_simulate_golden_summary(_real_pin: None, basis: str) -> None:
    nv = _golfer_nv()
    opts = SimOptions(t_final=0.02, dt=0.001, basis=basis)  # type: ignore[arg-type]
    out = simulate_with_coefficients(_theta(nv), opts)
    assert isinstance(out, SimOut)
    assert out.solver_status == "success"
    assert out.time.shape == (21,)
    assert out.q.shape[0] == out.qd.shape[0] == out.qdd.shape[0] == 21
    assert out.tau.shape == (21, nv)
    got = {
        "q_sum": float(out.q.sum()),
        "qd_sum": float(out.qd.sum()),
        "qdd_sum": float(out.qdd.sum()),
        "tau_sum": float(out.tau.sum()),
        "grip_sum": float(out.grip.sum()),
        "clubhead_sum": float(out.clubhead.sum()),
        "grip_quat_sum": float(out.grip_quat.sum()),
        "club_quat_sum": float(out.club_quat.sum()),
        "ke_sum": float(out.kinetic_energy.sum()),
        "pe_sum": float(out.potential_energy.sum()),
    }
    expected = _GOLDEN[basis]
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], value, rtol=1e-6, atol=1e-9, err_msg=key)


@pytest.mark.unit
@pytest.mark.requires_pinocchio
def test_simulate_meta_keys_and_values(_real_pin: None) -> None:
    nv = _golfer_nv()
    out = simulate_with_coefficients(
        _theta(nv), SimOptions(t_final=0.01, dt=0.001, compute_energy=False)
    )
    assert set(out.meta) == {
        "n_joints",
        "model_nq",
        "model_nv",
        "grip_frame",
        "clubhead_frame",
        "grip_frame_id",
        "clubhead_frame_id",
        "urdf_path",
        "integrator",
        "dt",
        "t_final",
        "n_steps",
        "wallclock_integrate_s",
        "wallclock_fk_s",
        "wallclock_total_s",
        "compute_energy",
        "basis",
    }
    assert out.meta["n_joints"] == nv
    assert out.meta["n_steps"] == 10
    assert out.meta["compute_energy"] is False
    assert out.meta["basis"] == "power"
    assert np.all(out.kinetic_energy == 0.0)
    assert np.all(out.potential_energy == 0.0)
    assert out.duration_s == pytest.approx(out.meta["wallclock_total_s"])


@pytest.mark.unit
@pytest.mark.requires_pinocchio
def test_simulate_initial_pose_is_honoured_and_validated(_real_pin: None) -> None:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(sim_mod._DEFAULT_URDF))
    nv, nq = int(model.nv), int(model.nq)
    theta = np.zeros(nv * COEFFS_PER_JOINT)
    q0 = np.asarray(pin.neutral(model)).copy()
    q0[-1] += 0.0
    out = simulate_with_coefficients(
        theta, SimOptions(t_final=0.002, dt=0.001), {"q": q0, "qd": np.zeros(nv)}
    )
    np.testing.assert_allclose(out.q[0], q0)
    with pytest.raises(ValueError, match="initial_pose\\['q'\\] has shape"):
        simulate_with_coefficients(
            theta, SimOptions(t_final=0.002, dt=0.001), {"q": np.zeros(nq + 1)}
        )
    with pytest.raises(ValueError, match="initial_pose\\['qd'\\] has shape"):
        simulate_with_coefficients(
            theta, SimOptions(t_final=0.002, dt=0.001), {"qd": np.zeros(nv + 1)}
        )


@pytest.mark.unit
@pytest.mark.requires_pinocchio
def test_simulate_rejects_bad_theta_and_divergence(_real_pin: None) -> None:
    nv = _golfer_nv()
    with pytest.raises(ValueError):
        simulate_with_coefficients(np.zeros(7 * (nv + 1)), SimOptions(t_final=0.01))
    huge = np.full(nv * COEFFS_PER_JOINT, 1e300)
    with pytest.raises(ValueError, match="Integration diverged|finite"):
        simulate_with_coefficients(huge, SimOptions(t_final=0.05, dt=0.01))


@pytest.mark.unit
@pytest.mark.requires_pinocchio
def test_simulate_gravity_and_t0_change_result(_real_pin: None) -> None:
    nv = _golfer_nv()
    theta = np.zeros(nv * COEFFS_PER_JOINT)
    a = simulate_with_coefficients(theta, SimOptions(t_final=0.02, dt=0.001))
    b = simulate_with_coefficients(
        theta, SimOptions(t_final=0.02, dt=0.001, gravity=np.zeros(3))
    )
    assert not np.allclose(a.qd[-1], b.qd[-1])
    assert np.allclose(b.qd, 0.0)


_GOLDEN: dict[str, dict[str, float]] = {
    # Recorded on unmodified origin/main (c5a1d31930) with pinocchio 4.1.0.
    "power": {
        "q_sum": -44.81623224437393,
        "qd_sum": -6595.147383488165,
        "qdd_sum": -661059.1398965712,
        "tau_sum": -83.89463940148049,
        "grip_sum": 10.714037413134687,
        "clubhead_sum": -0.8111410533190451,
        "grip_quat_sum": 21.091033341017155,
        "club_quat_sum": 21.091033341017155,
        "ke_sum": 346.06234863446116,
        "pe_sum": 4192.313712105286,
    },
    "bernstein": {
        "q_sum": -20.808041097213586,
        "qd_sum": -2548.6774342044355,
        "qdd_sum": -211183.39136123648,
        "tau_sum": 29.581477552238127,
        "grip_sum": 10.712356491256847,
        "clubhead_sum": -0.8277193494985552,
        "grip_quat_sum": 20.997000059247593,
        "club_quat_sum": 20.997000059247593,
        "ke_sum": 41.968552925911084,
        "pe_sum": 4192.469210600218,
    },
}
