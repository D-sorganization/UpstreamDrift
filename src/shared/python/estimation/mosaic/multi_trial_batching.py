"""Vectorised multi-trial batching, time-parallel Riccati, and runtime receipts (MOSAIC-14).

This module provides:
1. Batched block-arrow Schur complement solves across K trials that couple only
   through shared parameters (geometry and inertia).
2. Time-parallel Riccati recursion using associative Redheffer star products over
   sub-horizons for feedback gains K_t.
3. Structured runtime receipts capturing hardware, threads, library versions,
   per-iteration wall times, and scaling across trial counts.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.local_policy import LinearizedDynamics

FloatArray: TypeAlias = npt.NDArray[np.float64]


class BlockArrowSolution(NamedTuple):
    """Solution of a block-arrow linear system decomposed by Schur complement."""

    delta_c: FloatArray
    delta_pi: FloatArray

    @property
    def full(self) -> FloatArray:
        """Concatenated parameter updates [delta_c_0, ..., delta_c_{K-1}, delta_pi]."""
        return np.concatenate(
            [np.asarray(self.delta_c).ravel(), np.asarray(self.delta_pi).ravel()]
        )


@dataclass(frozen=True)
class BlockArrowSystem:
    """Explicit block components of a multi-trial block-arrow Hessian and RHS."""

    h_kk: FloatArray
    h_kpi: FloatArray
    h_pipi: FloatArray
    g_k: FloatArray
    g_pi: FloatArray

    def __post_init__(self) -> None:
        require(
            self.h_kk.ndim == 3 and self.h_kk.shape[1] == self.h_kk.shape[2],
            "h_kk must have shape (K, d_c, d_c)",
            self.h_kk.shape,
        )
        k, d_c, _ = self.h_kk.shape
        require(
            self.h_kpi.ndim == 3
            and self.h_kpi.shape[0] == k
            and self.h_kpi.shape[1] == d_c,
            "h_kpi must have shape (K, d_c, d_pi)",
            self.h_kpi.shape,
        )
        d_pi = self.h_kpi.shape[2]
        require(
            self.h_pipi.shape == (d_pi, d_pi),
            "h_pipi must have shape (d_pi, d_pi)",
            self.h_pipi.shape,
        )
        require(
            self.g_k.shape == (k, d_c),
            "g_k must have shape (K, d_c)",
            self.g_k.shape,
        )
        require(
            self.g_pi.shape == (d_pi,),
            "g_pi must have shape (d_pi,)",
            self.g_pi.shape,
        )


def extract_block_arrow(
    hessian: FloatArray,
    gradient: FloatArray,
    trial_dims: Sequence[int],
    shared_dim: int,
) -> BlockArrowSystem:
    """Extract block-arrow components from a monolithic Hessian and RHS.

    Precondition: all trial dimensions must be equal for vectorized batching.
    """
    require(hessian.ndim == 2 and hessian.shape[0] == hessian.shape[1], "H square")
    require(gradient.ndim == 1 and gradient.size == hessian.shape[0], "g matches H")
    require(len(trial_dims) >= 1, "at least one trial required")
    d_c = trial_dims[0]
    require(
        all(dim == d_c for dim in trial_dims),
        "vectorized batching requires uniform trial kinematic dimensions",
    )
    k = len(trial_dims)
    total_expected = k * d_c + shared_dim
    require(
        hessian.shape[0] == total_expected,
        f"Hessian dimension {hessian.shape[0]} != expected {total_expected}",
    )

    h_kk = np.empty((k, d_c, d_c), dtype=np.float64)
    h_kpi = np.empty((k, d_c, shared_dim), dtype=np.float64)
    g_k = np.empty((k, d_c), dtype=np.float64)

    shared_slice = slice(k * d_c, total_expected)
    for i in range(k):
        trial_slice = slice(i * d_c, (i + 1) * d_c)
        h_kk[i] = hessian[trial_slice, trial_slice]
        h_kpi[i] = hessian[trial_slice, shared_slice]
        g_k[i] = gradient[trial_slice]

    h_pipi = np.asarray(hessian[shared_slice, shared_slice], dtype=np.float64)
    g_pi = np.asarray(gradient[shared_slice], dtype=np.float64)

    return BlockArrowSystem(
        h_kk=h_kk,
        h_kpi=h_kpi,
        h_pipi=h_pipi,
        g_k=g_k,
        g_pi=g_pi,
    )


def assemble_block_arrow(
    h_kk: FloatArray,
    h_kpi: FloatArray,
    h_pipi: FloatArray,
    g_k: FloatArray,
    g_pi: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Assemble monolithic (H, g) from block-arrow components."""
    k, d_c, _ = h_kk.shape
    d_pi = h_pipi.shape[0]
    total_dim = k * d_c + d_pi

    h_mono = np.zeros((total_dim, total_dim), dtype=np.float64)
    g_mono = np.zeros(total_dim, dtype=np.float64)
    shared_slice = slice(k * d_c, total_dim)

    for i in range(k):
        trial_slice = slice(i * d_c, (i + 1) * d_c)
        h_mono[trial_slice, trial_slice] = h_kk[i]
        h_mono[trial_slice, shared_slice] = h_kpi[i]
        h_mono[shared_slice, trial_slice] = h_kpi[i].T
        g_mono[trial_slice] = g_k[i]

    h_mono[shared_slice, shared_slice] = h_pipi
    g_mono[shared_slice] = g_pi
    return h_mono, g_mono


def solve_block_arrow_schur(
    h_kk: FloatArray,
    h_kpi: FloatArray,
    h_pipi: FloatArray,
    g_k: FloatArray,
    g_pi: FloatArray,
) -> BlockArrowSolution:
    """Solve the multi-trial block-arrow system via vectorized Schur reduction.

    Reduced shared system:
        H_{pi, pi, red} = H_{pi, pi} - sum_{k=1}^K H_{k, pi}^T H_{kk}^{-1} H_{k, pi}
        g_{pi, red}     = g_pi      - sum_{k=1}^K H_{k, pi}^T H_{kk}^{-1} g_k

    Back-substitution per trial:
        delta_c_k = H_{kk}^{-1} (g_k - H_{k, pi} delta_pi)
    """
    _validate_block_arrow_inputs(h_kk, h_kpi, h_pipi, g_k, g_pi)

    # Vectorized solves over the leading trial dimension
    x_kpi = np.linalg.solve(h_kk, h_kpi)
    y_k = np.linalg.solve(h_kk, g_k[..., None]).squeeze(-1)

    # Schur complement accumulation across trials
    sum_h = np.einsum("kci,kcj->ij", h_kpi, x_kpi)
    sum_g = np.einsum("kci,kc->i", h_kpi, y_k)

    h_schur = 0.5 * (h_pipi - sum_h + (h_pipi - sum_h).T)
    g_schur = g_pi - sum_g

    delta_pi = np.asarray(np.linalg.solve(h_schur, g_schur), dtype=np.float64)
    delta_c = np.asarray(
        y_k - np.einsum("kcp,p->kc", x_kpi, delta_pi), dtype=np.float64
    )

    ensure(bool(np.all(np.isfinite(delta_pi))), "delta_pi must be finite")
    ensure(bool(np.all(np.isfinite(delta_c))), "delta_c must be finite")
    return BlockArrowSolution(delta_c=delta_c, delta_pi=delta_pi)


def _validate_block_arrow_inputs(
    h_kk: FloatArray,
    h_kpi: FloatArray,
    h_pipi: FloatArray,
    g_k: FloatArray,
    g_pi: FloatArray,
) -> tuple[int, int, int]:
    """Validate shapes, compatibility and finiteness of block-arrow system components."""
    require(h_kk.ndim == 3, "h_kk must be (K, d_c, d_c)")
    require(h_kk.shape[1] == h_kk.shape[2], "h_kk slices must be square")
    require(h_kpi.ndim == 3, "h_kpi must be (K, d_c, d_pi)")
    require(h_pipi.ndim == 2, "h_pipi must be (d_pi, d_pi)")
    require(h_pipi.shape[0] == h_pipi.shape[1], "h_pipi must be square")
    require(g_k.ndim == 2, "g_k must be (K, d_c)")
    require(g_pi.ndim == 1, "g_pi must be (d_pi,)")

    k, d_c, _ = h_kk.shape
    d_pi = h_pipi.shape[0]

    require(h_kpi.shape[0] == k and h_kpi.shape[1] == d_c, "h_kpi trial dimensions")
    require(h_kpi.shape[2] == d_pi, "h_kpi shared dimension matches h_pipi")
    require(g_k.shape[0] == k and g_k.shape[1] == d_c, "g_k shape matches h_kk")
    require(g_pi.shape[0] == d_pi, "g_pi shape matches h_pipi")

    require(bool(np.all(np.isfinite(h_kk))), "h_kk must be finite")
    require(bool(np.all(np.isfinite(h_kpi))), "h_kpi must be finite")
    require(bool(np.all(np.isfinite(h_pipi))), "h_pipi must be finite")
    require(bool(np.all(np.isfinite(g_k))), "g_k must be finite")
    require(bool(np.all(np.isfinite(g_pi))), "g_pi must be finite")

    return k, d_c, d_pi


def sequential_block_arrow_schur(
    h_kk: FloatArray,
    h_kpi: FloatArray,
    h_pipi: FloatArray,
    g_k: FloatArray,
    g_pi: FloatArray,
) -> BlockArrowSolution:
    """Sequential (unvectorized loop) multi-trial block-arrow Schur solve reference."""
    k, _, _ = _validate_block_arrow_inputs(h_kk, h_kpi, h_pipi, g_k, g_pi)

    sum_h = np.zeros_like(h_pipi)
    sum_g = np.zeros_like(g_pi)
    x_list: list[FloatArray] = []
    y_list: list[FloatArray] = []
    for i in range(k):
        x_i = np.linalg.solve(h_kk[i], h_kpi[i])
        y_i = np.linalg.solve(h_kk[i], g_k[i])
        sum_h += h_kpi[i].T @ x_i
        sum_g += h_kpi[i].T @ y_i
        x_list.append(x_i)
        y_list.append(y_i)

    h_schur = 0.5 * (h_pipi - sum_h + (h_pipi - sum_h).T)
    g_schur = g_pi - sum_g
    delta_pi = np.asarray(np.linalg.solve(h_schur, g_schur), dtype=np.float64)
    delta_c = np.asarray(
        np.stack(
            [y_i - x_i @ delta_pi for x_i, y_i in zip(x_list, y_list, strict=True)]
        ),
        dtype=np.float64,
    )

    ensure(bool(np.all(np.isfinite(delta_pi))), "delta_pi must be finite")
    ensure(bool(np.all(np.isfinite(delta_c))), "delta_c must be finite")
    return BlockArrowSolution(delta_c=delta_c, delta_pi=delta_pi)


def _star_compose(
    elem1: tuple[FloatArray, FloatArray, FloatArray],
    elem2: tuple[FloatArray, FloatArray, FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Redheffer star product composition of two Riccati scattering operators."""
    a1, s1, q1 = elem1
    a2, s2, q2 = elem2
    nx = a1.shape[0]
    eye_nx = np.eye(nx, dtype=np.float64)

    m = eye_nx + s1 @ q2
    x = np.linalg.solve(m, a1)
    y = np.linalg.solve(m, s1)

    a12 = a2 @ x
    s12 = 0.5 * (s2 + a2 @ y @ a2.T + (s2 + a2 @ y @ a2.T).T)
    q12 = 0.5 * (q1 + a1.T @ q2 @ x + (q1 + a1.T @ q2 @ x).T)
    return a12, s12, q12


def time_parallel_riccati(
    linearization: LinearizedDynamics,
    state_weight: FloatArray,
    input_weight: FloatArray,
    terminal_weight: FloatArray | None = None,
    chunk_size: int | None = None,
) -> FloatArray:
    """Time-parallel Riccati recursion for time-varying linear feedback gains K_t.

    Solves for K_t across the full horizon using an associative prefix/suffix scan
    over local Redheffer scattering operators in O(log T) parallel stages.
    """
    n_steps, n_x, n_u = linearization.b.shape
    require(n_steps >= 1, "at least one step required", n_steps)
    require(
        state_weight.shape == (n_x, n_x),
        "state_weight must be (nx, nx)",
        state_weight.shape,
    )
    require(
        input_weight.shape == (n_u, n_u),
        "input_weight must be (nu, nu)",
        input_weight.shape,
    )
    if terminal_weight is not None:
        require(
            terminal_weight.shape == (n_x, n_x),
            "terminal_weight must be (nx, nx)",
            terminal_weight.shape,
        )
    require(bool(np.all(np.isfinite(state_weight))), "state_weight finite")
    require(bool(np.all(np.isfinite(input_weight))), "input_weight finite")

    q_mat = 0.5 * (state_weight + state_weight.T)
    r_mat = 0.5 * (input_weight + input_weight.T)
    v_term = (
        q_mat
        if terminal_weight is None
        else 0.5 * (terminal_weight + terminal_weight.T)
    )

    # 1. Build local scattering elements (A_t, S_t, Q_t)
    elements: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for t in range(n_steps):
        a_t = linearization.a[t]
        b_t = linearization.b[t]
        s_t = b_t @ np.linalg.solve(r_mat, b_t.T)
        s_t = 0.5 * (s_t + s_t.T)
        elements.append((a_t, s_t, q_mat.copy()))

    # 2. Parallel associative suffix scan:
    # Suffixes[t] composes interval [t, n_steps)
    suffixes = list(elements)
    step = 1
    while step < n_steps:
        new_suffixes = list(suffixes)
        for t in range(n_steps - step):
            new_suffixes[t] = _star_compose(suffixes[t], suffixes[t + step])
        suffixes = new_suffixes
        step *= 2

    # 3. Compute cost-to-go matrices V_{t+1} from suffixes
    v_matrices = np.empty((n_steps + 1, n_x, n_x), dtype=np.float64)
    v_matrices[-1] = v_term
    eye_nx = np.eye(n_x, dtype=np.float64)

    for t in range(n_steps - 1):
        at_term, st_term, qt_term = suffixes[t + 1]
        m_t = eye_nx + st_term @ v_term
        z_t = np.linalg.solve(m_t, at_term)
        vt1 = qt_term + at_term.T @ v_term @ z_t
        v_matrices[t + 1] = 0.5 * (vt1 + vt1.T)

    # 4. Extract feedback gains K_t in parallel
    gains = np.empty((n_steps, n_u, n_x), dtype=np.float64)
    for t in range(n_steps):
        a_t = linearization.a[t]
        b_t = linearization.b[t]
        vt1 = v_matrices[t + 1]
        gain = np.linalg.solve(r_mat + b_t.T @ vt1 @ b_t, b_t.T @ vt1 @ a_t)
        gains[t] = gain

    ensure(bool(np.all(np.isfinite(gains))), "computed gains must be finite")
    return gains


@dataclass(frozen=True)
class RuntimeReceipt:
    """Receipt documenting environment configuration, execution timings, and scaling."""

    hardware: dict[str, Any]
    threads: dict[str, str]
    library_versions: dict[str, str]
    wall_time_per_iteration_s: float
    trial_scaling: dict[int, float]
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to JSON-serialisable dictionary."""
        return asdict(self)


def generate_runtime_receipt(
    wall_time_per_iteration_s: float,
    trial_scaling: dict[int, float],
) -> RuntimeReceipt:
    """Collect hardware, threads, library versions, and timings into an audit receipt."""
    require(
        wall_time_per_iteration_s >= 0.0,
        "wall_time_per_iteration_s must be non-negative",
        wall_time_per_iteration_s,
    )

    import scipy

    hw_info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }

    thread_env: dict[str, str] = {
        k: os.environ.get(k, "<unset>")
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }

    lib_versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    return RuntimeReceipt(
        hardware=hw_info,
        threads=thread_env,
        library_versions=lib_versions,
        wall_time_per_iteration_s=float(wall_time_per_iteration_s),
        trial_scaling={int(k): float(v) for k, v in trial_scaling.items()},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
