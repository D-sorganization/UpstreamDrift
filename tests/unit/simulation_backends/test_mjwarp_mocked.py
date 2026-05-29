"""Coverage for the MJWarp GPU backend host-side logic via mocked warp stack.

The real ``warp`` / ``mujoco_warp`` GPU stack is not installed on CI/dev boxes,
so the GPU code paths of :mod:`simulation_backends.mjwarp_backend` would
otherwise be untestable. Here we inject *fake* ``warp`` and ``mujoco_warp``
modules into ``sys.modules`` (via ``monkeypatch.setitem``, which auto-reverts —
no module-level ``MagicMock`` pollution) and bypass the ``require_warp`` gate.

This exercises the genuine host-side orchestration — control broadcasting,
``(N, T, 2)`` history assembly, the synchronize-before-read ordering, and the
device read/write plumbing — against a deterministic fake device. The GPU
*numerics* are validated separately by the ``@requires_gpu`` test in
``test_mjwarp_backend.py`` on a real CUDA runner.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.shared.python.simulation_backends import GolfModelParams
from src.shared.python.simulation_backends.exceptions import BackendCapabilityError

pytestmark = pytest.mark.unit


class _FakeArray:
    """Minimal stand-in for a ``warp`` device array (only ``.numpy()`` used)."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)

    def numpy(self) -> np.ndarray:
        return self._data


def _make_fake_warp() -> types.ModuleType:
    """Build a fake ``warp`` module covering the API the backend touches."""
    wp = types.ModuleType("warp")
    wp.float32 = np.float32
    wp.sync_calls = 0

    def synchronize() -> None:
        wp.sync_calls += 1

    def array(data: object, dtype: object = None) -> _FakeArray:
        arr = np.asarray(data)
        if dtype is np.float32:
            arr = arr.astype(np.float32)
        return _FakeArray(arr)

    wp.init = lambda: None
    wp.synchronize = synchronize
    wp.get_cuda_device_count = lambda: 1
    wp.array = array
    return wp


def _make_fake_mujoco_warp(wp: types.ModuleType) -> types.ModuleType:
    """Build a fake ``mujoco_warp`` module with a no-op (shape-preserving) step."""
    mjw = types.ModuleType("mujoco_warp")

    def put_model(cpu_model: object) -> object:
        return types.SimpleNamespace(
            opt=types.SimpleNamespace(timestep=cpu_model.opt.timestep)
        )

    def make_data(model: object, nworld: int = 1) -> object:
        return types.SimpleNamespace(
            qpos=wp.array(np.zeros((nworld, 2))),
            qvel=wp.array(np.zeros((nworld, 2))),
            ctrl=wp.array(np.zeros((nworld, 2))),
            qacc=wp.array(np.zeros((nworld, 2))),
            time=wp.array(np.zeros((nworld,))),
        )

    mjw.put_model = put_model
    mjw.make_data = make_data
    mjw.step = lambda model, data: None
    mjw.forward = lambda model, data: None
    return mjw


@pytest.fixture
def warp_backend(monkeypatch: pytest.MonkeyPatch):
    """Construct an MJWarpBackend with the warp stack faked out."""
    import src.shared.python.simulation_backends.capabilities as caps
    from src.shared.python.simulation_backends.mjwarp_backend import MJWarpBackend

    wp = _make_fake_warp()
    mjw = _make_fake_mujoco_warp(wp)
    monkeypatch.setitem(sys.modules, "warp", wp)
    monkeypatch.setitem(sys.modules, "mujoco_warp", mjw)
    monkeypatch.setattr(caps, "require_warp", lambda: None)
    return MJWarpBackend(GolfModelParams.default(), dt=0.01)


def test_capabilities_flags(warp_backend) -> None:
    caps = warp_backend.capabilities
    assert caps.name == "mjwarp"
    assert caps.device == "cuda"
    assert caps.supports_batched is True
    assert caps.is_differentiable is False
    assert caps.provides_dynamics is False


def test_single_env_cycle(warp_backend) -> None:
    from src.shared.python.simulation_backends import SimState

    warp_backend.reset(SimState(q=[0.1, -0.2], v=[0.0, 0.0], time=0.0))
    warp_backend.set_control(np.array([0.5, -0.3]))
    warp_backend.step()
    warp_backend.step(dt=0.02)
    state = warp_backend.get_state()
    assert state.q.shape == (2,)
    assert state.v.shape == (2,)
    assert isinstance(warp_backend.get_time(), float)


def test_set_control_wrong_length_raises(warp_backend) -> None:
    with pytest.raises(ValueError, match="control must have length"):
        warp_backend.set_control(np.array([1.0]))


def test_step_rejects_nonpositive_dt(warp_backend) -> None:
    with pytest.raises(ValueError):
        warp_backend.step(dt=0.0)


def test_forward_dynamics_shapes(warp_backend) -> None:
    qacc = warp_backend.forward_dynamics([0.1, 0.2], [0.0, 0.0])
    assert qacc.shape == (2,)
    qacc2 = warp_backend.forward_dynamics([0.1, 0.2], [0.1, 0.1], u=[0.2, -0.1])
    assert qacc2.shape == (2,)


def test_forward_dynamics_bad_lengths_raise(warp_backend) -> None:
    with pytest.raises(ValueError):
        warp_backend.forward_dynamics([0.1], [0.0, 0.0])
    with pytest.raises(ValueError):
        warp_backend.forward_dynamics([0.1, 0.2], [0.0])
    with pytest.raises(ValueError):
        warp_backend.forward_dynamics([0.1, 0.2], [0.0, 0.0], u=[1.0])


def test_rollout_single_env(warp_backend) -> None:
    trace = warp_backend.rollout(controls=None, horizon=5, dt=0.01)
    assert trace.num_steps == 6
    assert trace.q.shape == (6, 2)
    assert trace.backend == "mjwarp"


def test_rollout_batch_passive(warp_backend) -> None:
    batch = warp_backend.rollout_batch(controls=None, horizon=4, dt=0.01, num_envs=8)
    assert batch.num_envs == 8
    assert batch.num_steps == 5
    assert batch.q.shape == (8, 5, 2)
    assert batch.u is None


def test_rollout_batch_shared_controls(warp_backend) -> None:
    shared = np.ones((4, 2))
    batch = warp_backend.rollout_batch(shared, horizon=4, dt=0.01, num_envs=3)
    assert batch.q.shape == (3, 5, 2)
    assert batch.u is not None
    assert batch.u.shape == (3, 4, 2)


def test_rollout_batch_per_env_controls(warp_backend) -> None:
    per_env = np.ones((3, 4, 2))
    batch = warp_backend.rollout_batch(per_env, horizon=4, dt=0.01, num_envs=3)
    assert batch.u.shape == (3, 4, 2)


def test_rollout_batch_bad_control_shape_raises(warp_backend) -> None:
    with pytest.raises(ValueError, match="controls must be"):
        warp_backend.rollout_batch(np.ones((9, 9)), horizon=4, dt=0.01, num_envs=3)


@pytest.mark.parametrize(
    ("horizon", "dt", "num_envs"),
    [(0, 0.01, 1), (4, 0.0, 1), (4, 0.01, 0)],
)
def test_rollout_batch_precondition_violations(
    warp_backend, horizon: int, dt: float, num_envs: int
) -> None:
    with pytest.raises(ValueError):
        warp_backend.rollout_batch(None, horizon, dt, num_envs)


def test_dynamics_primitives_unsupported(warp_backend) -> None:
    with pytest.raises(BackendCapabilityError, match="does not expose dynamics"):
        warp_backend.mass_matrix(np.zeros(2))
    with pytest.raises(BackendCapabilityError, match="does not expose dynamics"):
        warp_backend.bias_forces(np.zeros(2), np.zeros(2))


# --- capabilities.py warp-present branches (mocked) ------------------------- #


def test_warp_device_available_true(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.shared.python.simulation_backends.capabilities as caps

    caps.warp_device_available.cache_clear()
    monkeypatch.setattr(caps, "has_warp", lambda: True)
    fake_wp = types.ModuleType("warp")
    fake_wp.init = lambda: None
    fake_wp.get_cuda_device_count = lambda: 2
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    try:
        assert caps.warp_device_available() is True
    finally:
        caps.warp_device_available.cache_clear()


def test_warp_device_available_handles_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.simulation_backends.capabilities as caps

    def _boom() -> int:
        raise RuntimeError("no CUDA driver")

    caps.warp_device_available.cache_clear()
    monkeypatch.setattr(caps, "has_warp", lambda: True)
    fake_wp = types.ModuleType("warp")
    fake_wp.init = lambda: None
    fake_wp.get_cuda_device_count = _boom
    monkeypatch.setitem(sys.modules, "warp", fake_wp)
    try:
        assert caps.warp_device_available() is False
    finally:
        caps.warp_device_available.cache_clear()


def test_warp_device_available_false_without_warp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.simulation_backends.capabilities as caps

    caps.warp_device_available.cache_clear()
    monkeypatch.setattr(caps, "has_warp", lambda: False)
    try:
        assert caps.warp_device_available() is False
    finally:
        caps.warp_device_available.cache_clear()


def test_can_import_branches() -> None:
    from src.shared.python.simulation_backends.capabilities import _can_import

    assert _can_import("os") is True
    assert _can_import("definitely_not_a_real_module_xyz_42") is False
