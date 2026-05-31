"""Unit tests for the MJX/JAX differentiable batched backend."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.simulation_backends import (
    BackendCapabilityError,
    BackendNotAvailableError,
    GolfModelParams,
    has_mjx,
    make_backend,
)
from src.shared.python.simulation_backends.mjx_backend import MJXBackend

pytestmark = pytest.mark.unit


@dataclass
class _FakeOpt:
    timestep: float


class _FakeCpuModel:
    nq = 2
    nv = 2
    nu = 2

    def __init__(self) -> None:
        self.opt = _FakeOpt(timestep=0.01)


@dataclass(frozen=True)
class _FakeMxmModel:
    opt: _FakeOpt
    nq: int = 2
    nv: int = 2
    nu: int = 2


@dataclass(frozen=True)
class _FakeData:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    qacc: np.ndarray
    time: np.ndarray

    def replace(self, **kwargs: object) -> _FakeData:
        values = {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "ctrl": self.ctrl,
            "qacc": self.qacc,
            "time": self.time,
        }
        values.update(kwargs)
        return _FakeData(**values)


def _make_fake_jax() -> types.ModuleType:
    jax = types.ModuleType("jax")
    jnp = types.ModuleType("jax.numpy")
    for name in (
        "array",
        "asarray",
        "concatenate",
        "empty",
        "isfinite",
        "stack",
        "zeros",
        "zeros_like",
    ):
        setattr(jnp, name, getattr(np, name))
    jnp.float64 = np.float64

    def jacrev(func):
        def _jacobian(x):
            x_arr = np.asarray(x, dtype=np.float64)
            base = np.asarray(func(x_arr), dtype=np.float64).reshape(-1)
            jac = np.empty((base.size, x_arr.size), dtype=np.float64)
            eps = 1e-6
            for index in range(x_arr.size):
                plus = x_arr.copy()
                minus = x_arr.copy()
                plus[index] += eps
                minus[index] -= eps
                jac[:, index] = (
                    np.asarray(func(plus), dtype=np.float64).reshape(-1)
                    - np.asarray(func(minus), dtype=np.float64).reshape(-1)
                ) / (2.0 * eps)
            return jac

        return _jacobian

    jax.numpy = jnp
    jax.jacrev = jacrev
    return jax


def _make_fake_mujoco() -> types.ModuleType:
    mujoco = types.ModuleType("mujoco")

    class MjModel:
        @staticmethod
        def from_xml_string(xml: str) -> _FakeCpuModel:
            assert "golf_double_pendulum" in xml
            return _FakeCpuModel()

    mujoco.MjModel = MjModel
    return mujoco


def _make_fake_mjx() -> types.ModuleType:
    mjx = types.ModuleType("mujoco.mjx")

    def put_model(cpu_model: _FakeCpuModel) -> _FakeMxmModel:
        return _FakeMxmModel(opt=cpu_model.opt)

    def make_data(model: _FakeMxmModel) -> _FakeData:
        return _FakeData(
            qpos=np.zeros(model.nq),
            qvel=np.zeros(model.nv),
            ctrl=np.zeros(model.nu),
            qacc=np.zeros(model.nv),
            time=np.zeros(()),
        )

    def _as_batch(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        return np.atleast_2d(arr)

    def forward(model: _FakeMxmModel, data: _FakeData) -> _FakeData:
        qacc = np.asarray(data.ctrl, dtype=np.float64)
        return data.replace(qacc=qacc)

    def step(model: _FakeMxmModel, data: _FakeData) -> _FakeData:
        dt = float(model.opt.timestep)
        qpos = _as_batch(data.qpos)
        qvel = _as_batch(data.qvel)
        ctrl = _as_batch(data.ctrl)
        next_qvel = qvel + ctrl * dt
        next_qpos = qpos + next_qvel * dt
        time = np.asarray(data.time, dtype=np.float64) + dt
        if np.asarray(data.qpos).ndim == 1:
            next_qpos = next_qpos[0]
            next_qvel = next_qvel[0]
        return data.replace(qpos=next_qpos, qvel=next_qvel, qacc=ctrl, time=time)

    mjx.put_model = put_model
    mjx.make_data = make_data
    mjx.forward = forward
    mjx.step = step
    return mjx


@pytest.fixture
def mjx_backend(monkeypatch: pytest.MonkeyPatch) -> MJXBackend:
    import src.shared.python.simulation_backends.capabilities as caps

    fake_jax = _make_fake_jax()
    fake_mujoco = _make_fake_mujoco()
    fake_mjx = _make_fake_mjx()
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jax.numpy)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco.mjx", fake_mjx)
    monkeypatch.setattr(caps, "require_mjx", lambda: None)
    return MJXBackend(GolfModelParams.default(), dt=0.01)


def test_module_imports_without_jax_or_mjx() -> None:
    spec = importlib.util.find_spec("src.shared.python.simulation_backends.mjx_backend")
    assert spec is not None
    assert MJXBackend.__name__ == "MJXBackend"


def test_describe_capabilities_flags() -> None:
    caps = MJXBackend.describe_capabilities()
    assert caps.name == "mjx"
    assert caps.device == "jax"
    assert caps.supports_batched is True
    assert caps.is_differentiable is True
    assert caps.provides_dynamics is False


@pytest.mark.skipif(has_mjx(), reason="MJX/JAX stack installed")
def test_make_backend_without_mjx_raises() -> None:
    with pytest.raises(BackendNotAvailableError):
        make_backend("mjx", GolfModelParams.default())


@pytest.mark.skipif(has_mjx(), reason="MJX/JAX stack installed")
def test_direct_construction_without_mjx_raises() -> None:
    with pytest.raises(BackendNotAvailableError):
        MJXBackend(GolfModelParams.default())


def test_no_top_level_jax_or_mjx_imports() -> None:
    source = importlib.util.find_spec(
        "src.shared.python.simulation_backends.mjx_backend"
    ).origin
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    forbidden = ("import jax", "import mujoco", "from mujoco import mjx")
    offending = []
    for raw_line in text.splitlines():
        if raw_line != raw_line.lstrip():
            continue
        stripped = raw_line.strip()
        if any(
            stripped == token or stripped.startswith(token + " ") for token in forbidden
        ):
            offending.append(raw_line)
    assert not offending


def test_mocked_mjx_rollout_batch_is_batched_and_finite(
    mjx_backend: MJXBackend,
) -> None:
    controls = np.ones((3, 4, 2), dtype=np.float64)
    batch = mjx_backend.rollout_batch(controls, horizon=4, dt=0.02, num_envs=3)

    assert batch.backend == "mjx"
    assert batch.num_envs == 3
    assert batch.num_steps == 5
    assert batch.q.shape == (3, 5, 2)
    assert batch.v.shape == (3, 5, 2)
    assert batch.u is not None
    assert batch.u.shape == (3, 5, 2)
    assert np.all(np.isfinite(batch.q))
    assert batch.meta["mjcf_source"] == "GolfModelParams"
    assert batch.meta["differentiable"] is True


def test_mocked_mjx_forward_dynamics_uses_mjx_forward(
    mjx_backend: MJXBackend,
) -> None:
    qacc = mjx_backend.forward_dynamics([0.1, -0.2], [0.0, 0.0], [0.5, -0.25])

    np.testing.assert_allclose(qacc, [0.5, -0.25])


def test_mocked_mjx_rollout_supports_shared_controls(
    mjx_backend: MJXBackend,
) -> None:
    controls = np.ones((3, 2), dtype=np.float64)
    trace = mjx_backend.rollout(controls, horizon=3, dt=0.01)

    assert trace.backend == "mjx"
    assert trace.num_steps == 4
    assert trace.u is not None
    assert trace.u.shape == (4, 2)


def test_mocked_mjx_final_state_control_jacobian_shape(
    mjx_backend: MJXBackend,
) -> None:
    controls = np.ones((2, 3, 2), dtype=np.float64)

    jacobian = mjx_backend.final_state_control_jacobian(
        controls,
        horizon=3,
        dt=0.01,
        num_envs=2,
    )

    assert jacobian.shape == (8, 12)
    assert np.all(np.isfinite(jacobian))


def test_mjx_dynamics_primitives_are_not_claimed(
    mjx_backend: MJXBackend,
) -> None:
    with pytest.raises(BackendCapabilityError, match="does not expose"):
        mjx_backend.mass_matrix(np.zeros(2))
    with pytest.raises(BackendCapabilityError, match="does not expose"):
        mjx_backend.bias_forces(np.zeros(2), np.zeros(2))
