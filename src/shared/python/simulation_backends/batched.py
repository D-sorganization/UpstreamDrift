"""Batched-rollout orchestration and memory budgeting (epic task M6.2).

GPU backends (``mjwarp``) evaluate thousands of rollouts in parallel, but a
single device cannot always hold every environment at once. This module sits
*above* the backend Protocols and provides the device-agnostic glue:

* :func:`plan_chunks` — split ``num_envs`` into contiguous spans no larger than a
  per-launch ``max_batch`` budget.
* :func:`estimate_trace_bytes` — predict the host/device memory a
  :class:`~simulation_backends.protocol.BatchTrace` will occupy, so callers can
  pick a ``max_batch`` that fits.
* :func:`run_batched` — drive a :class:`BatchedBackend` over those chunks and
  stitch the per-chunk :class:`BatchTrace` results back into one.
* :func:`cpu_batch_rollout` — a CPU fallback that loops single-env
  :meth:`SimulationBackend.rollout` calls and stacks them, so the batching
  contract is exercisable on a machine with **no GPU**.

Everything here is pure orchestration: no GPU import, no MuJoCo import. The
backend is injected, which keeps the module testable with an in-process fake
(see ``tests/unit/simulation_backends/test_batched.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.logging_pkg.logging_config import get_logger

from .protocol import BatchTrace

if TYPE_CHECKING:
    from .protocol import BatchedBackend, SimulationBackend, Trace

logger = get_logger(__name__)

#: Backend name stamped onto traces produced by :func:`cpu_batch_rollout`.
CPU_BATCH_BACKEND_NAME = "cpu-batch"


def plan_chunks(num_envs: int, max_batch: int) -> list[tuple[int, int]]:
    """Partition ``[0, num_envs)`` into contiguous spans of at most ``max_batch``.

    The spans tile the index range with no gaps and no overlap; only the final
    span may be shorter than ``max_batch`` (when ``num_envs`` is not a multiple
    of ``max_batch``).

    Args:
        num_envs: Total number of environments to cover (``> 0``).
        max_batch: Maximum environments per chunk / device launch (``> 0``).

    Returns:
        A list of ``(start, stop)`` half-open index pairs. Concatenating
        ``range(start, stop)`` over the list reproduces ``range(num_envs)``.

    Raises:
        TypeError: If ``num_envs`` or ``max_batch`` is not an ``int``.
        ValueError: If ``num_envs <= 0`` or ``max_batch <= 0``.

    Postconditions:
        * Spans are contiguous: ``chunks[0][0] == 0`` and
          ``chunks[i][0] == chunks[i - 1][1]`` and ``chunks[-1][1] == num_envs``.
        * Every span width is in ``[1, max_batch]``.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError(f"num_envs must be an int, got {type(num_envs).__name__}")
    if isinstance(max_batch, bool) or not isinstance(max_batch, int):
        raise TypeError(f"max_batch must be an int, got {type(max_batch).__name__}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")
    if max_batch <= 0:
        raise ValueError(f"max_batch must be > 0, got {max_batch}")

    chunks = [
        (start, min(start + max_batch, num_envs))
        for start in range(0, num_envs, max_batch)
    ]

    ensure(chunks[0][0] == 0, "first chunk must start at 0", value=chunks)
    ensure(chunks[-1][1] == num_envs, "last chunk must reach num_envs", value=chunks)
    return chunks


def estimate_trace_bytes(
    num_envs: int,
    horizon: int,
    state_dim: int,
    *,
    dtype_bytes: int = 4,
    include_controls: bool = False,
) -> int:
    """Estimate the memory footprint of a :class:`BatchTrace`'s arrays.

    A batch trace stores ``q`` and ``v`` each of shape
    ``(num_envs, horizon + 1, state_dim)`` and — optionally — a control array
    ``u`` of the same shape. The shared time vector ``t`` is negligible and is
    ignored. The returned value is monotonically non-decreasing in every
    argument (more envs, more steps, more coordinates, wider dtype, or adding
    controls can only grow it).

    Args:
        num_envs: Number of parallel environments ``N`` (``> 0``).
        horizon: Number of integration steps; the trace stores ``horizon + 1``
            samples (``>= 0``).
        state_dim: Per-array trailing dimension (``nq``/``nv``/``nu``) (``> 0``).
        dtype_bytes: Bytes per element (``4`` for float32, ``8`` for float64);
            (``> 0``).
        include_controls: Whether to count a third array ``u`` of equal size.

    Returns:
        The total number of bytes the trace arrays occupy.

    Raises:
        TypeError: If any argument is not an ``int``.
        ValueError: If ``num_envs``/``state_dim``/``dtype_bytes`` is not ``> 0``
            or ``horizon`` is negative.

    Postconditions:
        Result ``>= 0`` and monotonic in each argument.
    """
    _require_int(num_envs, "num_envs")
    _require_int(horizon, "horizon")
    _require_int(state_dim, "state_dim")
    _require_int(dtype_bytes, "dtype_bytes")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if state_dim <= 0:
        raise ValueError(f"state_dim must be > 0, got {state_dim}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be > 0, got {dtype_bytes}")

    elements_per_array = num_envs * (horizon + 1) * state_dim
    num_arrays = 3 if include_controls else 2  # q + v (+ u)
    total = num_arrays * elements_per_array * dtype_bytes

    ensure(total >= 0, "estimated bytes must be non-negative", value=total)
    return int(total)


def run_batched(
    backend: BatchedBackend,
    controls: np.ndarray | None,
    horizon: int,
    dt: float,
    num_envs: int,
    max_batch: int | None = None,
) -> BatchTrace:
    """Evaluate ``num_envs`` rollouts on ``backend``, chunked to ``max_batch``.

    When ``max_batch`` is ``None`` the backend handles the whole batch in a
    single :meth:`BatchedBackend.rollout_batch` launch. Otherwise the envs are
    split via :func:`plan_chunks`, one ``rollout_batch`` is issued per chunk, and
    the per-chunk :class:`BatchTrace` objects are concatenated along the env axis
    into one trace sharing the same ``t``/``dt``/``backend``.

    Per-env (rank-3) controls are sliced to match each chunk; a shared
    ``(horizon, nu)`` history or ``None`` is forwarded unchanged to every launch.

    Args:
        backend: A :class:`BatchedBackend` (must expose ``rollout_batch``).
        controls: ``None`` (passive), shared ``(horizon, nu)``, or per-env
            ``(num_envs, horizon, nu)``.
        horizon: Number of integration steps (``> 0``).
        dt: Integration step size [s] (``> 0``).
        num_envs: Number of parallel environments (``> 0``).
        max_batch: Per-launch env budget, or ``None`` for one launch.

    Returns:
        A single :class:`BatchTrace` with ``num_envs`` environments and
        ``horizon + 1`` samples.

    Raises:
        TypeError: If ``backend`` lacks ``rollout_batch`` or args are mistyped.
        ValueError: If ``horizon``/``num_envs`` is not ``> 0``, ``dt`` is not
            ``> 0``, ``max_batch`` is not ``> 0`` (when given), or per-env
            ``controls`` disagree with ``num_envs``.

    Postconditions:
        ``result.num_envs == num_envs`` and ``result.num_steps == horizon + 1``.
    """
    if not callable(getattr(backend, "rollout_batch", None)):
        raise TypeError("backend must implement BatchedBackend.rollout_batch")
    _require_int(horizon, "horizon")
    _require_int(num_envs, "num_envs")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")
    if not dt > 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    per_env_controls = _validate_batch_controls(controls, num_envs)

    if max_batch is None:
        result = backend.rollout_batch(controls, horizon, dt, num_envs)
        return _check_batch_result(result, num_envs, horizon)

    chunks = plan_chunks(num_envs, max_batch)
    chunk_traces = [
        backend.rollout_batch(
            _slice_controls(controls, start, stop, per_env_controls),
            horizon,
            dt,
            stop - start,
        )
        for start, stop in chunks
    ]
    result = _concatenate_batch_traces(chunk_traces)
    return _check_batch_result(result, num_envs, horizon)


def cpu_batch_rollout(
    make_backend_fn: Callable[[int], SimulationBackend],
    controls_batch: np.ndarray | None,
    horizon: int,
    dt: float,
    num_envs: int | None = None,
) -> BatchTrace:
    """Emulate a batched rollout on the CPU by looping single-env backends.

    For each environment ``i`` a fresh backend is built via ``make_backend_fn(i)``
    and rolled out for ``horizon`` steps; the resulting single-env
    :class:`Trace` objects are stacked into one :class:`BatchTrace`. This is the
    GPU-free reference path that makes the batching contract testable anywhere.

    Args:
        make_backend_fn: Factory mapping an env index to a fresh
            :class:`SimulationBackend`.
        controls_batch: Per-env controls ``(num_envs, horizon, nu)``, or ``None``
            for a passive batch (then ``num_envs`` must be supplied).
        horizon: Number of integration steps (``> 0``).
        dt: Integration step size [s] (``> 0``).
        num_envs: Required only when ``controls_batch`` is ``None``; otherwise
            inferred from ``len(controls_batch)`` and, if given, must match it.

    Returns:
        A :class:`BatchTrace` of shape ``(N, horizon + 1, nq)`` with backend
        name :data:`CPU_BATCH_BACKEND_NAME`.

    Raises:
        TypeError: If ``make_backend_fn`` is not callable or args are mistyped.
        ValueError: If ``horizon``/``dt`` is not ``> 0``, the env count is
            undetermined or non-positive, or ``num_envs`` contradicts
            ``len(controls_batch)``.

    Postconditions:
        ``result.num_envs == N`` and ``result.num_steps == horizon + 1``.
    """
    if not callable(make_backend_fn):
        raise TypeError("make_backend_fn must be callable")
    _require_int(horizon, "horizon")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if not dt > 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    resolved_controls = (
        None if controls_batch is None else np.asarray(controls_batch, dtype=float)
    )
    n_envs = _resolve_cpu_num_envs(resolved_controls, num_envs)

    per_env_traces = [
        make_backend_fn(i).rollout(
            None if resolved_controls is None else resolved_controls[i],
            horizon,
            dt,
        )
        for i in range(n_envs)
    ]
    result = _stack_traces(per_env_traces, dt)
    return _check_batch_result(result, n_envs, horizon)


def _require_int(value: object, name: str) -> None:
    """Raise ``TypeError`` unless ``value`` is a non-bool ``int`` (DbC guard)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def _is_per_env_controls(controls: np.ndarray | None) -> bool:
    """Return whether ``controls`` is a rank-3 per-env ``(N, horizon, nu)`` array."""
    return controls is not None and np.asarray(controls).ndim == 3


def _validate_batch_controls(
    controls: np.ndarray | None, num_envs: int
) -> np.ndarray | None:
    """Return per-env controls as an array, or ``None`` if shared/passive.

    Raises:
        ValueError: If per-env controls' leading axis disagrees with ``num_envs``.
    """
    if not _is_per_env_controls(controls):
        return None
    arr = np.asarray(controls, dtype=float)
    if arr.shape[0] != num_envs:
        raise ValueError(
            "per-env controls leading axis must equal num_envs; "
            f"got {arr.shape[0]} vs {num_envs}"
        )
    return arr


def _slice_controls(
    controls: np.ndarray | None,
    start: int,
    stop: int,
    per_env_controls: np.ndarray | None,
) -> np.ndarray | None:
    """Return the controls for env span ``[start, stop)``.

    Per-env (rank-3) controls are sliced along the env axis; a shared
    ``(horizon, nu)`` history or ``None`` passes through unchanged.
    """
    if per_env_controls is not None:
        return per_env_controls[start:stop]
    return controls


def _resolve_cpu_num_envs(
    controls_batch: np.ndarray | None, num_envs: int | None
) -> int:
    """Resolve the env count for :func:`cpu_batch_rollout`.

    Raises:
        TypeError: If a supplied ``num_envs`` is not an ``int``.
        ValueError: If the count cannot be determined, is non-positive, or
            contradicts ``len(controls_batch)``.
    """
    if num_envs is not None:
        _require_int(num_envs, "num_envs")
    if controls_batch is None:
        if num_envs is None:
            raise ValueError("num_envs is required when controls_batch is None")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        return num_envs

    inferred = int(controls_batch.shape[0])
    if num_envs is not None and num_envs != inferred:
        raise ValueError(
            f"num_envs ({num_envs}) contradicts len(controls_batch) ({inferred})"
        )
    if inferred <= 0:
        raise ValueError(f"controls_batch must have >= 1 env, got {inferred}")
    return inferred


def _concatenate_batch_traces(traces: list[BatchTrace]) -> BatchTrace:
    """Concatenate chunk batch traces along the env axis into one trace.

    All chunks share the same ``t``/``dt``/``backend``; ``u`` is concatenated
    only when present on every chunk (mixed presence is a programming error).

    Raises:
        ValueError: If ``traces`` is empty.
    """
    require(len(traces) > 0, "cannot concatenate an empty list of traces")
    head = traces[0]
    q = np.concatenate([tr.q for tr in traces], axis=0)
    v = np.concatenate([tr.v for tr in traces], axis=0)
    u = _concatenate_controls(traces)
    return BatchTrace(
        t=head.t,
        q=q,
        v=v,
        u=u,
        dt=head.dt,
        backend=head.backend,
        meta=dict(head.meta),
    )


def _concatenate_controls(traces: list[BatchTrace]) -> np.ndarray | None:
    """Concatenate per-chunk ``u`` arrays along the env axis, or return ``None``.

    Returns ``None`` when no chunk carries controls. If some — but not all —
    chunks carry controls the batch is inconsistent and an error is raised.

    Raises:
        ValueError: On mixed control presence across chunks.
    """
    present = [tr.u is not None for tr in traces]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("inconsistent control presence across chunk traces")
    return np.concatenate([tr.u for tr in traces], axis=0)


def _stack_traces(traces: list[Trace], dt: float) -> BatchTrace:
    """Stack single-env traces into a :class:`BatchTrace` (new leading axis).

    Args:
        traces: Per-env single rollouts; all must share the same time axis.
        dt: Integration step recorded on the batch trace.

    Raises:
        ValueError: If ``traces`` is empty or the traces disagree on step count.
    """
    require(len(traces) > 0, "cannot stack an empty list of traces")
    num_steps = traces[0].num_steps
    require(
        all(tr.num_steps == num_steps for tr in traces),
        "all per-env traces must share the same number of steps",
    )
    q = np.stack([tr.q for tr in traces], axis=0)
    v = np.stack([tr.v for tr in traces], axis=0)
    u = _stack_controls(traces)
    return BatchTrace(
        t=traces[0].t,
        q=q,
        v=v,
        u=u,
        dt=dt,
        backend=CPU_BATCH_BACKEND_NAME,
    )


def _stack_controls(traces: list[Trace]) -> np.ndarray | None:
    """Stack per-env ``u`` arrays on a new leading axis, or return ``None``.

    Raises:
        ValueError: On mixed control presence across envs.
    """
    present = [tr.u is not None for tr in traces]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("inconsistent control presence across per-env traces")
    return np.stack([tr.u for tr in traces], axis=0)


def _check_batch_result(result: BatchTrace, num_envs: int, horizon: int) -> BatchTrace:
    """Assert a backend/assembled batch trace matches the requested shape.

    Args:
        result: The :class:`BatchTrace` to validate.
        num_envs: Expected environment count.
        horizon: Expected step count is ``horizon + 1``.

    Returns:
        ``result`` unchanged (for fluent use).

    Raises:
        TypeError: If ``result`` is not a :class:`BatchTrace`.
    """
    if not isinstance(result, BatchTrace):
        raise TypeError(f"expected a BatchTrace, got {type(result).__name__}")
    ensure(
        result.num_envs == num_envs,
        f"batch trace has {result.num_envs} envs, expected {num_envs}",
        value=result.num_envs,
    )
    ensure(
        result.num_steps == horizon + 1,
        f"batch trace has {result.num_steps} steps, expected {horizon + 1}",
        value=result.num_steps,
    )
    return result
