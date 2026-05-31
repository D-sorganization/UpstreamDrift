"""Unit tests for batched-rollout orchestration and memory budgeting (M6.2).

These tests exercise the device-agnostic batching glue with **no GPU**: chunk
planning and byte estimation are pure arithmetic, the CPU fallback loops a real
single-env backend, and ``run_batched`` is driven against an in-process fake
:class:`BatchedBackend` so the chunk-plan-then-concatenate path is verified
without any optional dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends.batched import (
    CPU_BATCH_BACKEND_NAME,
    cpu_batch_rollout,
    estimate_trace_bytes,
    plan_chunks,
    run_batched,
    run_estimation_windows_batched,
)
from src.shared.python.simulation_backends.factory import make_backend
from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.protocol import BatchTrace

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)
_NQ = 2  # double-pendulum: [theta1, theta2]


class _FakeBatchedBackend:
    """In-process :class:`BatchedBackend` returning zeros sized to the request.

    Records every ``rollout_batch`` call so tests can assert on the chunk sizes
    requested by :func:`run_batched`.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def rollout_batch(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
        num_envs: int,
    ) -> BatchTrace:
        self.calls.append(num_envs)
        t = np.arange(horizon + 1, dtype=float) * dt
        q = np.zeros((num_envs, horizon + 1, _NQ), dtype=float)
        v = np.zeros((num_envs, horizon + 1, _NQ), dtype=float)
        return BatchTrace(t=t, q=q, v=v, dt=dt, backend="fake")


# --------------------------------------------------------------------------- #
# plan_chunks                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("num_envs", "max_batch", "expected"),
    [
        (1000, 256, [(0, 256), (256, 512), (512, 768), (768, 1000)]),
        (256, 256, [(0, 256)]),
        (5, 2, [(0, 2), (2, 4), (4, 5)]),
        (1, 1, [(0, 1)]),
        (3, 10, [(0, 3)]),
    ],
)
def test_plan_chunks_partitions(
    num_envs: int, max_batch: int, expected: list[tuple[int, int]]
) -> None:
    assert plan_chunks(num_envs, max_batch) == expected


def test_plan_chunks_is_a_contiguous_cover() -> None:
    chunks = plan_chunks(1000, 256)
    covered = [i for start, stop in chunks for i in range(start, stop)]
    assert covered == list(range(1000))
    assert all(stop - start <= 256 for start, stop in chunks)


@pytest.mark.parametrize("bad_num_envs", [0, -1, -100])
def test_plan_chunks_rejects_nonpositive_num_envs(bad_num_envs: int) -> None:
    with pytest.raises(ValueError, match="num_envs must be > 0"):
        plan_chunks(bad_num_envs, 2)


@pytest.mark.parametrize("bad_max_batch", [0, -1])
def test_plan_chunks_rejects_nonpositive_max_batch(bad_max_batch: int) -> None:
    with pytest.raises(ValueError, match="max_batch must be > 0"):
        plan_chunks(8, bad_max_batch)


def test_plan_chunks_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        plan_chunks(8.0, 2)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# estimate_trace_bytes                                                        #
# --------------------------------------------------------------------------- #
def test_estimate_trace_bytes_formula() -> None:
    # q + v: 2 arrays * N * (H+1) * dim * dtype_bytes
    assert estimate_trace_bytes(4, 10, 2, dtype_bytes=4) == 2 * 4 * 11 * 2 * 4


def test_estimate_trace_bytes_controls_add_one_array() -> None:
    without = estimate_trace_bytes(4, 10, 2)
    with_u = estimate_trace_bytes(4, 10, 2, include_controls=True)
    assert with_u == without + (without // 2)  # third array of equal size


def test_estimate_trace_bytes_monotonic_in_num_envs() -> None:
    base = estimate_trace_bytes(4, 10, 2)
    assert estimate_trace_bytes(5, 10, 2) > base


def test_estimate_trace_bytes_monotonic_in_horizon() -> None:
    base = estimate_trace_bytes(4, 10, 2)
    assert estimate_trace_bytes(4, 11, 2) > base


def test_estimate_trace_bytes_monotonic_in_dtype_bytes() -> None:
    assert estimate_trace_bytes(4, 10, 2, dtype_bytes=8) > estimate_trace_bytes(
        4, 10, 2, dtype_bytes=4
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_envs": 0, "horizon": 1, "state_dim": 2}, "num_envs must be > 0"),
        ({"num_envs": 1, "horizon": -1, "state_dim": 2}, "horizon must be >= 0"),
        ({"num_envs": 1, "horizon": 1, "state_dim": 0}, "state_dim must be > 0"),
    ],
)
def test_estimate_trace_bytes_rejects_bad_args(
    kwargs: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        estimate_trace_bytes(**kwargs)


# --------------------------------------------------------------------------- #
# cpu_batch_rollout                                                           #
# --------------------------------------------------------------------------- #
def test_cpu_batch_rollout_passive_with_explicit_num_envs() -> None:
    params = GolfModelParams.default()
    batch = cpu_batch_rollout(
        lambda _i: make_backend("ode", params),
        controls_batch=None,
        num_envs=4,
        horizon=10,
        dt=0.01,
    )
    assert batch.num_envs == 4
    assert batch.num_steps == 11
    assert batch.q.shape == (4, 11, _NQ)
    assert batch.backend == CPU_BATCH_BACKEND_NAME
    assert batch.u is None


def test_cpu_batch_rollout_infers_num_envs_from_controls() -> None:
    params = GolfModelParams.default()
    controls = _RNG.normal(scale=0.1, size=(3, 10, _NQ))
    batch = cpu_batch_rollout(
        lambda _i: make_backend("ode", params),
        controls_batch=controls,
        horizon=10,
        dt=0.01,
    )
    assert batch.num_envs == 3
    assert batch.num_steps == 11
    assert batch.u is not None
    # A Trace records its control history aligned to the horizon+1 sample times
    # (Trace.__post_init__ enforces u.shape[0] == len(t)), so the stacked batch
    # control array spans horizon+1, not horizon.
    assert batch.u.shape == (3, batch.num_steps, _NQ)


def test_cpu_batch_rollout_per_env_states_match_single_rollout() -> None:
    params = GolfModelParams.default()
    batch = cpu_batch_rollout(
        lambda _i: make_backend("ode", params),
        controls_batch=None,
        num_envs=2,
        horizon=5,
        dt=0.01,
    )
    # All envs use the identical backend + passive control, so env 0 == env 1.
    np.testing.assert_allclose(batch.env(0).q, batch.env(1).q)


def test_cpu_batch_rollout_requires_num_envs_when_passive() -> None:
    params = GolfModelParams.default()
    with pytest.raises(ValueError, match="num_envs is required"):
        cpu_batch_rollout(
            lambda _i: make_backend("ode", params),
            controls_batch=None,
            horizon=5,
            dt=0.01,
        )


def test_cpu_batch_rollout_rejects_contradictory_num_envs() -> None:
    params = GolfModelParams.default()
    controls = _RNG.normal(size=(3, 5, _NQ))
    with pytest.raises(ValueError, match="contradicts"):
        cpu_batch_rollout(
            lambda _i: make_backend("ode", params),
            controls_batch=controls,
            num_envs=4,
            horizon=5,
            dt=0.01,
        )


def test_cpu_batch_rollout_rejects_non_callable_factory() -> None:
    with pytest.raises(TypeError, match="callable"):
        cpu_batch_rollout(
            object(),  # type: ignore[arg-type]
            controls_batch=None,
            num_envs=2,
            horizon=5,
            dt=0.01,
        )


# --------------------------------------------------------------------------- #
# run_batched                                                                 #
# --------------------------------------------------------------------------- #
def test_run_batched_chunks_and_concatenates() -> None:
    backend = _FakeBatchedBackend()
    batch = run_batched(
        backend,
        controls=None,
        horizon=10,
        dt=0.01,
        num_envs=700,
        max_batch=256,
    )
    assert batch.num_envs == 700
    assert batch.num_steps == 11
    assert batch.q.shape == (700, 11, _NQ)
    # 700 -> 256 + 256 + 188
    assert backend.calls == [256, 256, 188]


def test_run_batched_single_launch_when_max_batch_none() -> None:
    backend = _FakeBatchedBackend()
    batch = run_batched(
        backend,
        controls=None,
        horizon=4,
        dt=0.01,
        num_envs=50,
        max_batch=None,
    )
    assert batch.num_envs == 50
    assert backend.calls == [50]


def test_run_batched_slices_per_env_controls() -> None:
    class _ControlRecordingBackend:
        def __init__(self) -> None:
            self.control_rows: list[int] = []

        def rollout_batch(
            self,
            controls: np.ndarray | None,
            horizon: int,
            dt: float,
            num_envs: int,
        ) -> BatchTrace:
            assert controls is not None
            self.control_rows.append(controls.shape[0])
            t = np.arange(horizon + 1, dtype=float) * dt
            zeros = np.zeros((num_envs, horizon + 1, _NQ), dtype=float)
            return BatchTrace(t=t, q=zeros, v=zeros.copy(), dt=dt, backend="rec")

    backend = _ControlRecordingBackend()
    controls = _RNG.normal(size=(5, 3, _NQ))
    batch = run_batched(
        backend,
        controls=controls,
        horizon=3,
        dt=0.01,
        num_envs=5,
        max_batch=2,
    )
    assert batch.num_envs == 5
    # Per-env controls sliced to chunk sizes 2 + 2 + 1.
    assert backend.control_rows == [2, 2, 1]


def test_run_batched_rejects_bad_per_env_controls() -> None:
    backend = _FakeBatchedBackend()
    controls = _RNG.normal(size=(4, 3, _NQ))  # 4 envs, but request 5
    with pytest.raises(ValueError, match="per-env controls"):
        run_batched(
            backend,
            controls=controls,
            horizon=3,
            dt=0.01,
            num_envs=5,
            max_batch=2,
        )


def test_run_batched_rejects_backend_without_rollout_batch() -> None:
    with pytest.raises(TypeError, match="rollout_batch"):
        run_batched(
            object(),  # type: ignore[arg-type]
            controls=None,
            horizon=3,
            dt=0.01,
            num_envs=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0, "num_envs": 4, "dt": 0.01}, "horizon must be > 0"),
        ({"horizon": 3, "num_envs": 0, "dt": 0.01}, "num_envs must be > 0"),
        ({"horizon": 3, "num_envs": 4, "dt": 0.0}, "dt must be > 0"),
    ],
)
def test_run_batched_rejects_bad_args(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_batched(_FakeBatchedBackend(), controls=None, max_batch=2, **kwargs)


# --------------------------------------------------------------------------- #
# Estimation trial-window batching                                            #
# --------------------------------------------------------------------------- #
def test_run_estimation_windows_batched_flattens_trial_windows() -> None:
    backend = _FakeBatchedBackend()
    controls = _RNG.normal(size=(2, 3, 4, _NQ))

    batch = run_estimation_windows_batched(
        backend,
        controls_windows=controls,
        dt=0.01,
        max_batch=4,
    )

    assert batch.num_envs == 6
    assert batch.num_steps == 5
    assert batch.u is None
    assert backend.calls == [4, 2]
    assert batch.meta["layout"] == "trial_window"
    assert batch.meta["num_trials"] == 2
    assert batch.meta["num_windows"] == 3
    assert batch.meta["control_dim"] == _NQ


def test_run_estimation_windows_batched_rejects_bad_controls() -> None:
    with pytest.raises(ValueError, match="rank-4"):
        run_estimation_windows_batched(
            _FakeBatchedBackend(),
            controls_windows=np.zeros((2, 4, _NQ)),
            dt=0.01,
        )
