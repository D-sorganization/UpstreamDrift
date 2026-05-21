"""Tests for src.engines.simscape.adapter.SimscapeAdapter.

Covers the skeleton-mode behaviour (no MATLAB), MATLAB-mocked code paths
via patched module imports, lifecycle guards, cache, and error wrapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.simscape._errors import (
    SimscapeModelNotFoundError,
    SimscapeNotInstalledError,
    SimscapeSimulationError,
    SimscapeStateError,
)
from src.shared.python.core.contracts.exceptions import PreconditionError
from src.engines.simscape._lifecycle import AdapterState
from src.engines.simscape._output import SimscapeOutput
from src.engines.simscape.adapter import (
    _GOLFSWING3D_JOINT_NAMES,
    SimscapeAdapter,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint


@pytest.fixture
def slx_pair(tmp_path: Path) -> Path:
    """Create a fake .slx + sibling metadata mat file."""
    slx = tmp_path / "GolfSwing3D_Kinetic.slx"
    slx.write_bytes(b"PK\x03\x04")  # zip header magic for fun
    (tmp_path / "PolynomialInputValues.mat").write_bytes(b"\x00")
    return slx


@pytest.fixture(autouse=True)
def _force_no_matlab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: act as if MATLAB is not available."""
    monkeypatch.setenv("UD_SIMSCAPE_FORCE_NO_MATLAB", "1")


def test_init_defaults() -> None:
    a = SimscapeAdapter()
    assert a.name == "simscape_3d"
    assert a.model_name == ""
    assert a.model_loaded is False
    assert a.joint_names == ()
    a.close()


def test_init_invalid_rng_seed() -> None:
    with pytest.raises(PreconditionError):
        SimscapeAdapter(rng_seed=-1)


def test_init_invalid_cache_capacity() -> None:
    with pytest.raises(PreconditionError):
        SimscapeAdapter(cache_max_entries=-5)


def test_dof_requires_loaded_state(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    with pytest.raises(SimscapeStateError):
        _ = a.dof
    a.close()


def test_load_from_path_skeleton_mode(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    assert a.model_name == "GolfSwing3D_Kinetic"
    assert a.model_loaded is True
    assert a.dof == len(_GOLFSWING3D_JOINT_NAMES)
    assert a.joint_names == _GOLFSWING3D_JOINT_NAMES
    a.close()


def test_load_from_path_rejects_non_slx(tmp_path: Path) -> None:
    a = SimscapeAdapter()
    bogus = tmp_path / "model.mdl"
    bogus.write_text("x")
    with pytest.raises(ValueError, match=".slx"):
        a.load_from_path(str(bogus))


def test_load_from_path_missing_file(tmp_path: Path) -> None:
    a = SimscapeAdapter()
    with pytest.raises(SimscapeModelNotFoundError):
        a.load_from_path(str(tmp_path / "nope.slx"))


def test_load_from_path_missing_metadata(tmp_path: Path) -> None:
    a = SimscapeAdapter()
    slx = tmp_path / "model.slx"
    slx.write_bytes(b"x")
    with pytest.raises(SimscapeModelNotFoundError, match="metadata"):
        a.load_from_path(str(slx))


def test_load_from_path_empty_string() -> None:
    a = SimscapeAdapter()
    with pytest.raises(PreconditionError):
        a.load_from_path("")


def test_load_from_string_not_supported() -> None:
    a = SimscapeAdapter()
    with pytest.raises(NotImplementedError):
        a.load_from_string("blob")


def test_state_summary(slx_pair: Path) -> None:
    a = SimscapeAdapter(rng_seed=7, cache_max_entries=8)
    a.load_from_path(str(slx_pair))
    snap = a.state_summary()
    assert snap["name"] == "simscape_3d"
    assert snap["model_loaded"] is True
    assert snap["lifecycle"] == "loaded"
    assert snap["dof"] == len(_GOLFSWING3D_JOINT_NAMES)
    assert snap["rng_seed"] == 7
    assert snap["cache_max_entries"] == 8
    a.close()


def test_repr_does_not_leak_path(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    r = repr(a)
    assert str(slx_pair) not in r
    assert "GolfSwing3D_Kinetic" in r
    assert "simscape_3d" in r
    a.close()


def test_repr_unloaded() -> None:
    a = SimscapeAdapter()
    assert "<unloaded>" in repr(a)
    a.close()


def test_reset_requires_loaded() -> None:
    a = SimscapeAdapter()
    with pytest.raises(SimscapeStateError):
        a.reset()


def test_reset_clears_sim_time(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    a._sim_time = 1.5
    a._control = np.zeros(3)
    a.reset()
    assert a.get_time() == 0.0
    a.close()


def test_step_without_matlab_raises_not_installed(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(SimscapeNotInstalledError):
        a.step()
    a.close()


def test_step_bad_dt(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(PreconditionError):
        a.step(dt=-0.1)
    a.close()


def test_step_with_mocked_engine(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    a._engine = eng
    a.step(dt=0.01)
    eng.set_param.assert_called_once()
    eng.sim.assert_called_once()
    # second call
    a.step()
    assert eng.set_param.call_count == 2
    assert a.get_time() > 0
    a.close()


def test_step_wraps_matlab_failure(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.sim.side_effect = RuntimeError("step boom")
    a._engine = eng
    with pytest.raises(SimscapeSimulationError, match="step boom"):
        a.step()
    a.close()


def test_forward_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError, match="deferred"):
        a.forward()
    a.close()


def test_get_state_skeleton(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    q, v = a.get_state()
    assert q.shape == (a.dof,)
    assert v.shape == (a.dof,)
    assert np.all(q == 0.0)
    a.close()


def test_get_state_with_engine(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.eval.return_value = "OP"
    eng.getfield.side_effect = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    a._engine = eng
    q, v = a.get_state()
    assert np.array_equal(q, [1.0, 2.0])
    assert np.array_equal(v, [3.0, 4.0])
    a.close()


def test_get_state_wraps_matlab_error(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.eval.side_effect = RuntimeError("getstate boom")
    a._engine = eng
    with pytest.raises(SimscapeSimulationError, match="getstate boom"):
        a.get_state()
    a.close()


def test_set_state_requires_engine(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    dof = a.dof
    with pytest.raises(SimscapeNotInstalledError):
        a.set_state(np.zeros(dof), np.zeros(dof))
    a.close()


def test_set_state_bad_shape(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(ValueError):
        a.set_state(np.zeros(3), np.zeros(3))
    a.close()


def test_set_state_with_mocked_engine(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    a._engine = eng
    dof = a.dof
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="DBL")
    with patch.dict(sys.modules, {"matlab": fake_matlab}):
        a.set_state(np.zeros(dof), np.zeros(dof))
    assert eng.set_param.called
    assert eng.assignin.called
    assert eng.eval.called
    a.close()


def test_set_state_wraps_matlab_failure(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.set_param.side_effect = RuntimeError("setstate boom")
    a._engine = eng
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="DBL")
    dof = a.dof
    with (
        patch.dict(sys.modules, {"matlab": fake_matlab}),
        pytest.raises(SimscapeSimulationError, match="setstate boom"),
    ):
        a.set_state(np.zeros(dof), np.zeros(dof))
    a.close()


def test_set_control_stores_copy(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    u = np.array([1.0, 2.0, 3.0])
    a.set_control(u)
    u[0] = 99.0
    assert a._control is not None
    assert a._control[0] == 1.0
    a.close()


def test_get_joint_names_unloaded() -> None:
    a = SimscapeAdapter()
    assert a.get_joint_names() == []
    a.close()


def test_get_full_state(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    full = a.get_full_state()
    assert set(full.keys()) == {"q", "v", "t", "M"}
    assert full["M"] is None
    a.close()


def test_get_capabilities(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    caps = a.get_capabilities()
    assert caps.engine_name == "simscape_3d"
    a.close()


@pytest.mark.parametrize(
    "method,args",
    [
        ("compute_mass_matrix", ()),
        ("compute_bias_forces", ()),
        ("compute_gravity_forces", ()),
        ("compute_drift_acceleration", ()),
        ("get_link_masses", ()),
        ("get_joint_damping", ()),
    ],
)
def test_deferred_array_methods(slx_pair: Path, method: str, args: tuple) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError, match="deferred"):
        getattr(a, method)(*args)
    a.close()


def test_compute_inverse_dynamics_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.compute_inverse_dynamics(np.zeros(a.dof))
    a.close()


def test_compute_control_acceleration_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.compute_control_acceleration(np.zeros(a.dof))
    a.close()


def test_compute_jacobian_returns_none(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    assert a.compute_jacobian("Hip") is None
    a.close()


def test_compute_contact_forces_zero(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    f = a.compute_contact_forces()
    assert f.shape == (3,)
    assert np.all(f == 0.0)
    a.close()


def test_set_shaft_returns_false() -> None:
    a = SimscapeAdapter()
    assert a.set_shaft_properties(1.0, np.ones(3), np.ones(3)) is False
    assert a.get_shaft_state() is None
    a.close()


def test_compute_ztcf_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.compute_ztcf(np.zeros(a.dof), np.zeros(a.dof))
    a.close()


def test_compute_zvcf_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.compute_zvcf(np.zeros(a.dof))
    a.close()


def test_recordable_methods_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.get_time_series("q")
    with pytest.raises(NotImplementedError):
        a.get_induced_acceleration_series("Hip")
    with pytest.raises(NotImplementedError):
        a.set_analysis_config({"log_q": True})
    a.close()


def test_set_link_masses_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.set_link_masses(np.ones(a.dof))
    a.close()


def test_set_joint_damping_deferred(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(NotImplementedError):
        a.set_joint_damping(np.ones(a.dof))
    a.close()


def _valid_output(n: int = 3, n_joints: int = 16) -> SimscapeOutput:
    return SimscapeOutput(
        time=np.linspace(0, 0.2, n),
        q=np.zeros((n, n_joints)),
        qd=np.zeros((n, n_joints)),
        qdd=np.zeros((n, n_joints)),
        tau=np.zeros((n, n_joints)),
        omega=np.zeros((n, n_joints)),
        r_butt=np.zeros((n, 3)),
        r_clubhead=np.zeros((n, 3)),
        q_club=np.tile([1.0, 0, 0, 0], (n, 1)),
        v_clubhead=np.zeros((n, 3)),
    )


def test_simulate_with_coefficients_caches(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    out = _valid_output(n_joints=a.dof)
    with patch.object(a, "_simulate_uncached", return_value=out) as mock_sim:
        coeffs = np.arange(a.dof * 7, dtype=np.float64)
        r1 = a.simulate_with_coefficients(coeffs)
        r2 = a.simulate_with_coefficients(coeffs)
        assert r1 is out
        assert r2 is out
        mock_sim.assert_called_once()  # second call is a cache hit
    a.close()


def test_simulate_with_coefficients_cache_disabled(slx_pair: Path) -> None:
    a = SimscapeAdapter(cache_enabled=False)
    a.load_from_path(str(slx_pair))
    out = _valid_output(n_joints=a.dof)
    with patch.object(a, "_simulate_uncached", return_value=out) as mock_sim:
        coeffs = np.zeros(a.dof * 7, dtype=np.float64)
        a.simulate_with_coefficients(coeffs)
        a.simulate_with_coefficients(coeffs)
        assert mock_sim.call_count == 2
    a.close()


def test_simulate_with_coefficients_bad_shape(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    with pytest.raises(PreconditionError):
        a.simulate_with_coefficients(np.zeros(0))
    with pytest.raises(PreconditionError):
        a.simulate_with_coefficients(np.zeros(3))  # not multiple of 7
    a.close()


def test_simulate_uncached_no_matlab_raises(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    coeffs = np.zeros(a.dof * 7, dtype=np.float64)
    with pytest.raises(SimscapeNotInstalledError):
        a._simulate_uncached(coeffs)
    a.close()


def test_simulate_uncached_with_mocked_engine(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.eval.return_value = "SIM_INPUT"
    eng.setVariable.return_value = "SIM_INPUT2"

    n = 3
    n_joints = a.dof
    logsout = {
        "time": np.linspace(0, 0.2, n),
        "q": np.zeros((n, n_joints)),
        "qd": np.zeros((n, n_joints)),
        "qdd": np.zeros((n, n_joints)),
        "tau": np.zeros((n, n_joints)),
        "omega": np.zeros((n, n_joints)),
        "r_butt": np.zeros((n, 3)),
        "r_clubhead": np.zeros((n, 3)),
        "q_club": np.tile([1.0, 0, 0, 0], (n, 1)),
        "v_clubhead": np.zeros((n, 3)),
    }
    eng.simulate_with_coefficients.return_value = logsout
    a._engine = eng
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="DBL")
    coeffs = np.zeros(n_joints * 7, dtype=np.float64)
    with patch.dict(sys.modules, {"matlab": fake_matlab}):
        result = a._simulate_uncached(coeffs)
    assert isinstance(result, SimscapeOutput)
    eng.simulate_with_coefficients.assert_called_once()
    a.close()


def test_simulate_uncached_wraps_matlab_failure(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.eval.return_value = "SIM_INPUT"
    eng.setVariable.return_value = "SIM_INPUT2"
    eng.simulate_with_coefficients.side_effect = RuntimeError("sim boom")
    a._engine = eng
    fake_matlab = MagicMock()
    fake_matlab.double = MagicMock(return_value="DBL")
    coeffs = np.zeros(a.dof * 7, dtype=np.float64)
    with (
        patch.dict(sys.modules, {"matlab": fake_matlab}),
        pytest.raises(SimscapeSimulationError, match="sim boom"),
    ):
        a._simulate_uncached(coeffs)
    a.close()


def test_engine_type(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    assert isinstance(a.engine_type, str)
    a.close()


def test_save_and_restore_checkpoint(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    a._sim_time = 0.123
    cp = a.save_checkpoint()
    assert isinstance(cp, StateCheckpoint)
    assert cp.timestamp == pytest.approx(0.123)
    a._sim_time = 0.0
    a.restore_checkpoint(cp)
    assert a.get_time() == pytest.approx(0.123)
    a.close()


def test_restore_checkpoint_engine_type_mismatch(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    cp = a.save_checkpoint()
    bogus = StateCheckpoint.create(
        engine_type="totally_other",
        engine_state={},
        q=np.zeros(a.dof),
        v=np.zeros(a.dof),
        timestamp=0.0,
    )
    with pytest.raises(ValueError, match="engine_type"):
        a.restore_checkpoint(bogus)
    # And original checkpoint still restores cleanly
    a.restore_checkpoint(cp)
    a.close()


def test_close_idempotent(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    a.close()
    assert a._lifecycle.state is AdapterState.STOPPED
    a.close()  # no error


def test_close_with_engine_calls_close_system(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    a._engine = eng
    a.close()
    eng.close_system.assert_called_once()


def test_close_swallows_close_system_errors(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    a.load_from_path(str(slx_pair))
    eng = MagicMock()
    eng.close_system.side_effect = RuntimeError("close boom")
    a._engine = eng
    a.close()  # must not raise


def test_context_manager_closes(slx_pair: Path) -> None:
    with SimscapeAdapter() as a:
        a.load_from_path(str(slx_pair))
    assert a._lifecycle.state is AdapterState.STOPPED


def test_load_from_path_matlab_branch(slx_pair: Path) -> None:
    """Exercise the MATLAB-available branch of load_from_path."""
    fake_eng = MagicMock()
    fake_eng.version.return_value = "9.13.0 (R2022b)"
    fake_eng.getPolynomialParameterInfo.return_value = {
        "JointNames": ["Hip", "Spine"],
        "NumJoints": 2,
    }
    from src.engines.simscape import _engine_pool

    _engine_pool._engine = None  # type: ignore[attr-defined]
    with (
        patch(
            "src.engines.simscape._engine_pool.is_matlab_available", return_value=True
        ),
        patch(
            "src.engines.simscape._engine_pool.get_shared_engine", return_value=fake_eng
        ),
    ):
        a = SimscapeAdapter()
        a.load_from_path(str(slx_pair))
        assert a.joint_names == ("Hip", "Spine")
        assert a.dof == 2
        assert a._matlab_version.startswith("9.13")
        a.close()
    _engine_pool._engine = None  # type: ignore[attr-defined]


def test_load_matlab_model_wraps_failure(slx_pair: Path) -> None:
    """Direct test of _load_matlab_model error path."""
    a = SimscapeAdapter()
    a._model_name = "GolfSwing3D_Kinetic"
    eng = MagicMock()
    eng.load_system.side_effect = RuntimeError("load boom")
    a._engine = eng
    with pytest.raises(SimscapeSimulationError, match="load boom"):
        a._load_matlab_model(slx_pair)


def test_fetch_joint_metadata_wraps_failure(slx_pair: Path) -> None:
    a = SimscapeAdapter()
    eng = MagicMock()
    eng.getPolynomialParameterInfo.side_effect = RuntimeError("meta boom")
    a._engine = eng
    with pytest.raises(SimscapeSimulationError, match="meta boom"):
        a._fetch_joint_metadata()


def test_fetch_joint_metadata_malformed_struct() -> None:
    a = SimscapeAdapter()
    eng = MagicMock()
    eng.getPolynomialParameterInfo.return_value = {"unrelated": 1}
    a._engine = eng
    with pytest.raises(SimscapeSimulationError, match="unexpected"):
        a._fetch_joint_metadata()
