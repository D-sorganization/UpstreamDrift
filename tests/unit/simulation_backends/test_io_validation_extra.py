"""Supplementary unit tests across the analysis/orchestration helpers.

Targets the remaining uncovered error/edge branches in:

trace_io:
    * the ``path`` type guard (non-str / non-PathLike) and the ``trace`` type
      guard in :func:`write_trace`;
    * the HDF5 byte-string decode path in the attribute reader;
    * the unrecognised ``kind`` discriminator in :func:`read_trace`;
    * reading a missing file and writing into a non-existent directory;
    * a :class:`BatchTrace` round-trip with a non-``None`` control history.

validation:
    * the energy-conservation horizon guard;
    * the trajectory sample-count mismatch guard;
    * the ``(2, 2)`` / ``(2,)`` shape guards on a provider's ``mass_matrix`` /
      ``bias_forces`` via the public cross-validate entry points;
    * a deliberately mismatched provider pair reporting ``passed=False``.

batched:
    * ``plan_chunks`` rejecting a non-int / bool ``max_batch``;
    * ``estimate_trace_bytes`` rejecting ``dtype_bytes <= 0`` and a bool arg,
      and counting a third array when ``include_controls=True``;
    * ``cpu_batch_rollout`` horizon/dt/num_envs guards, an explicit ``num_envs``
      with a per-env ``controls_batch``, the empty-controls-batch guard, and the
      inconsistent-control-presence guard;
    * ``run_batched`` rejecting a non-:class:`BatchTrace` result and an
      inconsistent control presence across chunks, plus ``max_batch`` smaller
      than ``num_envs`` reconstructing the full batch.

ztcf_zvcf:
    * the finite/length preconditions of ``ztcf_acceleration`` /
      ``zvcf_acceleration`` and the shape of ``evaluate_ztcf_along_trajectory``.

All RNG is seeded (``np.random.default_rng(0)``); no optional dependency is
required (every backend is an in-process fake or the pure-Python ``ode``).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends import batched as batched_mod
from src.shared.python.simulation_backends import validation as validation_mod
from src.shared.python.simulation_backends.batched import (
    cpu_batch_rollout,
    estimate_trace_bytes,
    plan_chunks,
    run_batched,
)
from src.shared.python.simulation_backends.protocol import (
    BatchTrace,
    SimState,
    Trace,
)
from src.shared.python.simulation_backends.trace_io import read_trace, write_trace
from src.shared.python.simulation_backends.validation import (
    cross_validate_bias,
    cross_validate_mass_matrix,
    cross_validate_trajectory,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    evaluate_ztcf_along_trajectory,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)
_NQ = 2


# --------------------------------------------------------------------------- #
# Lightweight Protocol-satisfying fakes (no optional dependency).
# --------------------------------------------------------------------------- #
class _GoodProvider:
    """A minimal, well-shaped :class:`DynamicsProvider`."""

    def __init__(self, skew: float = 0.0) -> None:
        self.skew = float(skew)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        del q
        return np.array([[2.0, 0.5], [0.5, 1.0]]) + self.skew

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        del q, v
        return np.array([0.1, -0.2]) + self.skew


class _BadMassProvider(_GoodProvider):
    """A provider whose ``mass_matrix`` returns the wrong shape."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        del q
        return np.zeros((3, 3))


class _BadBiasProvider(_GoodProvider):
    """A provider whose ``bias_forces`` returns the wrong length."""

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        del q, v
        return np.zeros(3)


class _ShortRolloutBackend:
    """A backend whose rollout emits a fixed (configurable) sample count."""

    def __init__(self, num_samples: int) -> None:
        self._n = int(num_samples)

    def reset(self, state: SimState | None = None) -> None:
        del state

    def rollout(self, controls: np.ndarray | None, horizon: int, dt: float) -> Trace:
        del controls, horizon
        n = self._n
        t = np.arange(n, dtype=float) * dt
        q = np.zeros((n, _NQ))
        v = np.zeros((n, _NQ))
        return Trace(t=t, q=q, v=v, u=None, dt=dt, backend="short")


# --------------------------------------------------------------------------- #
# trace_io: type guards
# --------------------------------------------------------------------------- #
def test_write_trace_rejects_non_path_type() -> None:
    """A non-str / non-PathLike path raises TypeError."""
    trace = Trace(t=[0.0], q=np.zeros((1, _NQ)), v=np.zeros((1, _NQ)))
    with pytest.raises(TypeError, match="str or os.PathLike"):
        write_trace(trace, 123)  # type: ignore[arg-type]


def test_write_trace_rejects_non_trace_object(tmp_path) -> None:
    """A non-Trace / non-BatchTrace ``trace`` argument raises TypeError."""
    with pytest.raises(TypeError, match="Trace or BatchTrace"):
        write_trace(object(), tmp_path / "bad.h5")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# trace_io: filesystem error paths
# --------------------------------------------------------------------------- #
def test_read_trace_missing_file_raises(tmp_path) -> None:
    """Reading a non-existent file surfaces an OSError."""
    with pytest.raises(OSError):
        read_trace(tmp_path / "does_not_exist.h5")


def test_write_trace_into_missing_directory_raises(tmp_path) -> None:
    """Writing into a non-existent directory surfaces an OSError."""
    trace = Trace(t=[0.0], q=np.zeros((1, _NQ)), v=np.zeros((1, _NQ)))
    with pytest.raises(OSError):
        write_trace(trace, tmp_path / "no_such_dir" / "x.h5")


# --------------------------------------------------------------------------- #
# trace_io: attribute decode + unrecognised kind
# --------------------------------------------------------------------------- #
def test_read_decodes_byte_string_attributes(tmp_path) -> None:
    """A ``bytes`` HDF5 attribute is decoded to ``str`` on read."""
    trace = Trace(
        t=[0.0, 0.01],
        q=np.zeros((2, _NQ)),
        v=np.zeros((2, _NQ)),
        dt=0.01,
        backend="ode",
    )
    path = tmp_path / "bytes.h5"
    write_trace(trace, path)
    # Rewrite the backend attribute as a raw byte string (h5py would otherwise
    # store a native str); this drives the decode branch of the reader.
    with h5py.File(path, "r+") as handle:
        handle.attrs["backend"] = np.bytes_(b"ode-bytes")

    loaded = read_trace(path)
    assert isinstance(loaded.backend, str)
    assert loaded.backend == "ode-bytes"


def test_read_rejects_unrecognised_kind(tmp_path) -> None:
    """An unknown ``kind`` discriminator raises ValueError."""
    trace = Trace(
        t=[0.0, 0.01],
        q=np.zeros((2, _NQ)),
        v=np.zeros((2, _NQ)),
        dt=0.01,
        backend="ode",
    )
    path = tmp_path / "kind.h5"
    write_trace(trace, path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["kind"] = "bogus-kind"

    with pytest.raises(ValueError, match="unrecognised trace kind"):
        read_trace(path)


# --------------------------------------------------------------------------- #
# trace_io: BatchTrace round-trip WITH controls
# --------------------------------------------------------------------------- #
def test_batch_trace_with_controls_round_trips(tmp_path) -> None:
    """A BatchTrace whose ``u`` is non-None round-trips losslessly."""
    num_envs, horizon = 3, 6
    t = np.arange(horizon + 1, dtype=float) * 0.02
    q = _RNG.standard_normal((num_envs, horizon + 1, _NQ))
    v = _RNG.standard_normal((num_envs, horizon + 1, _NQ))
    u = _RNG.standard_normal((num_envs, horizon + 1, _NQ))
    trace = BatchTrace(t=t, q=q, v=v, u=u, dt=0.02, backend="mjwarp")
    path = tmp_path / "batch_u.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert isinstance(loaded, BatchTrace)
    assert loaded.u is not None
    np.testing.assert_allclose(loaded.u, u)
    np.testing.assert_allclose(loaded.q, q)
    np.testing.assert_allclose(loaded.v, v)


# --------------------------------------------------------------------------- #
# validation: precondition / shape guards
# --------------------------------------------------------------------------- #
def test_cross_validate_mass_matrix_empty_raises() -> None:
    """An empty ``q_samples`` iterable violates the precondition."""
    with pytest.raises(ValueError, match="non-empty"):
        cross_validate_mass_matrix(_GoodProvider(), _GoodProvider(), iter(()))


def test_cross_validate_bias_empty_raises() -> None:
    """An empty ``states`` iterable violates the precondition."""
    with pytest.raises(ValueError, match="non-empty"):
        cross_validate_bias(_GoodProvider(), _GoodProvider(), iter(()))


def test_cross_validate_mass_matrix_bad_provider_shape_raises() -> None:
    """A provider returning a non-(2, 2) mass matrix is rejected."""
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        cross_validate_mass_matrix(_BadMassProvider(), _GoodProvider(), [np.zeros(2)])


def test_cross_validate_bias_bad_provider_shape_raises() -> None:
    """A provider returning a non-length-2 bias vector is rejected."""
    with pytest.raises(ValueError, match="2 entries"):
        cross_validate_bias(
            _BadBiasProvider(), _GoodProvider(), [(np.zeros(2), np.zeros(2))]
        )


def test_cross_validate_mass_matrix_mismatch_reports_not_passed() -> None:
    """A skewed provider pair yields a report with ``passed=False``."""
    report = cross_validate_mass_matrix(
        _GoodProvider(), _GoodProvider(skew=1.0), [np.zeros(2), np.array([0.1, 0.2])]
    )
    assert report.passed is False
    assert report.max_abs_error == pytest.approx(1.0, rel=1e-6)


def test_cross_validate_trajectory_sample_count_mismatch_raises() -> None:
    """Backends whose rollouts disagree on sample count are rejected."""
    with pytest.raises(ValueError, match="trajectory length mismatch"):
        cross_validate_trajectory(
            _ShortRolloutBackend(6),
            _ShortRolloutBackend(8),
            None,
            horizon=10,
            dt=0.01,
        )


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_check_energy_conservation_rejects_nonpositive_horizon(
    bad_horizon: int,
) -> None:
    """The energy-conservation check rejects a non-positive horizon."""
    backend = make_backend(
        "ode",
        GolfModelParams.default().model_copy(
            update={
                "gravity_enabled": False,
                "damping_shoulder": 0.0,
                "damping_wrist": 0.0,
            }
        ),
    )
    with pytest.raises(ValueError, match="horizon must be > 0"):
        validation_mod.check_energy_conservation(
            backend,
            backend,
            SimState(q=[0.1, -0.1], v=[1.0, -0.5]),
            horizon=bad_horizon,
            dt=0.01,
        )


# --------------------------------------------------------------------------- #
# batched: plan_chunks / estimate_trace_bytes guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_max_batch", [2.0, True])
def test_plan_chunks_rejects_non_int_max_batch(bad_max_batch: object) -> None:
    """A float / bool ``max_batch`` is a type violation."""
    with pytest.raises(TypeError, match="max_batch must be an int"):
        plan_chunks(8, bad_max_batch)  # type: ignore[arg-type]


def test_estimate_trace_bytes_rejects_nonpositive_dtype_bytes() -> None:
    """A non-positive ``dtype_bytes`` is rejected."""
    with pytest.raises(ValueError, match="dtype_bytes must be > 0"):
        estimate_trace_bytes(4, 10, 2, dtype_bytes=0)


def test_estimate_trace_bytes_rejects_bool_argument() -> None:
    """A bool passed where an int is required is rejected (DbC guard)."""
    with pytest.raises(TypeError, match="must be an int"):
        estimate_trace_bytes(4, True, 2)  # type: ignore[arg-type]


def test_estimate_trace_bytes_with_controls_counts_third_array() -> None:
    """``include_controls=True`` adds a third equal-size array to the estimate."""
    base = estimate_trace_bytes(4, 10, 2, dtype_bytes=8)
    with_u = estimate_trace_bytes(4, 10, 2, dtype_bytes=8, include_controls=True)
    # Two arrays -> three arrays is a 1.5x growth.
    assert with_u == base + base // 2
    assert with_u == 3 * 4 * 11 * 2 * 8


# --------------------------------------------------------------------------- #
# batched: cpu_batch_rollout guards and control paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0, "dt": 0.01, "num_envs": 2}, "horizon must be > 0"),
        ({"horizon": 5, "dt": 0.0, "num_envs": 2}, "dt must be > 0"),
        ({"horizon": 5, "dt": 0.01, "num_envs": 0}, "num_envs must be > 0"),
    ],
)
def test_cpu_batch_rollout_rejects_bad_args(
    kwargs: dict[str, float], match: str
) -> None:
    """Non-positive horizon/dt/num_envs each violate a precondition."""
    params = GolfModelParams.default()
    with pytest.raises(ValueError, match=match):
        cpu_batch_rollout(
            lambda _i: make_backend("ode", params),
            controls_batch=None,
            **kwargs,
        )


def test_cpu_batch_rollout_rejects_empty_controls_batch() -> None:
    """A zero-environment ``controls_batch`` is rejected."""
    params = GolfModelParams.default()
    with pytest.raises(ValueError, match=">= 1 env"):
        cpu_batch_rollout(
            lambda _i: make_backend("ode", params),
            controls_batch=np.zeros((0, 5, _NQ)),
            horizon=5,
            dt=0.01,
        )


def test_cpu_batch_rollout_explicit_num_envs_with_controls() -> None:
    """An explicit ``num_envs`` matching a per-env controls batch is accepted."""
    params = GolfModelParams.default()
    controls = _RNG.normal(scale=0.1, size=(2, 5, _NQ))
    batch = cpu_batch_rollout(
        lambda _i: make_backend("ode", params),
        controls_batch=controls,
        horizon=5,
        dt=0.01,
        num_envs=2,
    )
    assert batch.num_envs == 2
    assert batch.num_steps == 6
    assert batch.u is not None
    assert batch.u.shape == (2, batch.num_steps, _NQ)


def test_cpu_batch_rollout_inconsistent_control_presence_raises() -> None:
    """Per-env traces with mixed control presence are rejected."""

    class _MixedBackend:
        def __init__(self, idx: int) -> None:
            self.idx = idx

        def rollout(
            self, controls: np.ndarray | None, horizon: int, dt: float
        ) -> Trace:
            del controls
            t = np.arange(horizon + 1, dtype=float) * dt
            q = np.zeros((horizon + 1, _NQ))
            v = np.zeros((horizon + 1, _NQ))
            # Env 0 records a control history; env 1 does not -> inconsistent.
            u = np.zeros((horizon + 1, _NQ)) if self.idx == 0 else None
            return Trace(t=t, q=q, v=v, u=u, dt=dt, backend="mixed")

    with pytest.raises(ValueError, match="inconsistent control presence"):
        cpu_batch_rollout(
            lambda i: _MixedBackend(i),
            controls_batch=None,
            horizon=3,
            dt=0.01,
            num_envs=2,
        )


# --------------------------------------------------------------------------- #
# batched: run_batched guards and chunk reconstruction
# --------------------------------------------------------------------------- #
class _FakeBatched:
    """In-process :class:`BatchedBackend` returning zero-filled batch traces."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def rollout_batch(
        self, controls: np.ndarray | None, horizon: int, dt: float, num_envs: int
    ) -> BatchTrace:
        del controls
        self.calls.append(num_envs)
        t = np.arange(horizon + 1, dtype=float) * dt
        q = np.zeros((num_envs, horizon + 1, _NQ))
        v = np.zeros((num_envs, horizon + 1, _NQ))
        return BatchTrace(t=t, q=q, v=v, dt=dt, backend="fake")


def test_run_batched_max_batch_smaller_than_num_envs_reconstructs_full() -> None:
    """``max_batch < num_envs`` chunks and concatenates back to the full batch."""
    backend = _FakeBatched()
    batch = run_batched(
        backend, controls=None, horizon=4, dt=0.01, num_envs=5, max_batch=2
    )
    assert batch.num_envs == 5
    assert batch.num_steps == 5
    assert batch.q.shape == (5, 5, _NQ)
    # 5 -> 2 + 2 + 1
    assert backend.calls == [2, 2, 1]


def test_run_batched_concatenates_chunk_controls_when_all_present() -> None:
    """When every chunk carries ``u`` the controls are concatenated, not dropped."""

    class _AllControlsBackend:
        def rollout_batch(
            self, controls: np.ndarray | None, horizon: int, dt: float, num_envs: int
        ) -> BatchTrace:
            del controls
            t = np.arange(horizon + 1, dtype=float) * dt
            q = np.zeros((num_envs, horizon + 1, _NQ))
            v = np.zeros((num_envs, horizon + 1, _NQ))
            u = np.zeros((num_envs, horizon + 1, _NQ))  # every chunk has controls
            return BatchTrace(t=t, q=q, v=v, u=u, dt=dt, backend="all-u")

    batch = run_batched(
        _AllControlsBackend(),
        controls=None,
        horizon=3,
        dt=0.01,
        num_envs=5,
        max_batch=2,
    )
    assert batch.num_envs == 5
    assert batch.u is not None
    assert batch.u.shape == (5, batch.num_steps, _NQ)


def test_run_batched_rejects_non_batch_trace_result() -> None:
    """A backend returning a non-BatchTrace result is rejected."""

    class _BadResultBackend:
        def rollout_batch(
            self, controls: np.ndarray | None, horizon: int, dt: float, num_envs: int
        ) -> object:
            del controls, horizon, dt, num_envs
            return None

    with pytest.raises(TypeError, match="expected a BatchTrace"):
        run_batched(_BadResultBackend(), controls=None, horizon=3, dt=0.01, num_envs=2)


def test_run_batched_inconsistent_chunk_control_presence_raises() -> None:
    """Chunks with mixed control presence cannot be concatenated."""

    class _MixedChunkBackend:
        def __init__(self) -> None:
            self.i = 0

        def rollout_batch(
            self, controls: np.ndarray | None, horizon: int, dt: float, num_envs: int
        ) -> BatchTrace:
            del controls
            t = np.arange(horizon + 1, dtype=float) * dt
            q = np.zeros((num_envs, horizon + 1, _NQ))
            v = np.zeros((num_envs, horizon + 1, _NQ))
            # First chunk carries controls; later chunks do not -> inconsistent.
            u = np.zeros((num_envs, horizon + 1, _NQ)) if self.i == 0 else None
            self.i += 1
            return BatchTrace(t=t, q=q, v=v, u=u, dt=dt, backend="mixed")

    with pytest.raises(ValueError, match="inconsistent control presence"):
        run_batched(
            _MixedChunkBackend(),
            controls=None,
            horizon=3,
            dt=0.01,
            num_envs=4,
            max_batch=2,
        )


# --------------------------------------------------------------------------- #
# ztcf_zvcf: finite / length preconditions and trajectory shape
# --------------------------------------------------------------------------- #
def test_ztcf_acceleration_rejects_wrong_length_q() -> None:
    """A ``q``/``v`` length mismatch violates the ZTCF precondition."""
    provider = _GoodProvider()
    with pytest.raises(ValueError):
        ztcf_acceleration(provider, np.zeros(3), np.zeros(2))


def test_ztcf_acceleration_rejects_non_finite() -> None:
    """A non-finite ``q`` entry violates the ZTCF finite-input precondition."""
    provider = _GoodProvider()
    with pytest.raises(ValueError, match="finite"):
        ztcf_acceleration(provider, np.array([np.inf, 0.0]), np.zeros(2))


def test_zvcf_acceleration_rejects_wrong_length_q() -> None:
    """A ``q``/``tau`` length mismatch violates the ZVCF precondition."""
    provider = _GoodProvider()
    with pytest.raises(ValueError):
        zvcf_acceleration(provider, np.zeros(3), np.zeros(2))


def test_zvcf_acceleration_rejects_non_finite_tau() -> None:
    """A non-finite ``tau`` entry violates the ZVCF finite-input precondition."""
    provider = _GoodProvider()
    with pytest.raises(ValueError, match="finite"):
        zvcf_acceleration(provider, np.zeros(2), np.array([np.nan, 1.0]))


def test_evaluate_ztcf_along_trajectory_shape() -> None:
    """The pointwise trajectory evaluation preserves the ``(T, n)`` shape."""
    provider = _GoodProvider()
    horizon = 7
    q_traj = _RNG.uniform(-1.0, 1.0, (horizon, _NQ))
    v_traj = _RNG.uniform(-1.0, 1.0, (horizon, _NQ))
    out = evaluate_ztcf_along_trajectory(provider, q_traj, v_traj)
    assert out.shape == (horizon, _NQ)
    assert np.all(np.isfinite(out))


def test_batched_module_exposes_cpu_backend_name() -> None:
    """Smoke check the module-level constant is importable (LOD-friendly)."""
    assert isinstance(batched_mod.CPU_BATCH_BACKEND_NAME, str)
    assert batched_mod.CPU_BATCH_BACKEND_NAME
