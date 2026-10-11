"""Arrival factors for moving-horizon estimation (#11545).

Square-root Gaussian arrival factors on a window's first knot state, their
rank-revealing marginalisation, and the information-form elimination of a
solved window's leading knots used by
:class:`~src.shared.python.estimation.moving_horizon.MovingHorizonEstimator`.
See that module's docstring for the window cost and propagation equations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    SplineTrajectoryEvaluation,
)

RowsFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    """Return a read-only float64 numpy array copy."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class ArrivalUpdate:
    """Tentative arrival factor for a window, committed only on acceptance.

    ``failure`` is non-empty when the propagation linearisation was
    non-finite; the window is then rejected and committed state is unchanged.
    ``discard_reason`` is non-empty when information is lost on commit.
    """

    factor: ArrivalFactor | None
    anchor: int
    marginalized: tuple[tuple[int, float], ...] = ()
    failure: str = ""
    discard_reason: str = ""


@dataclass(frozen=True)
class ArrivalFactor:
    """Square-root quadratic arrival factor in tangent coordinates about a reference.

    Cost: 0.5 * || R * (x - x_ref) - r ||^2
    """

    reference_state: np.ndarray
    sqrt_information: np.ndarray
    residual_offset: np.ndarray
    rank: int
    linearization_point: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ref = np.asarray(self.reference_state, dtype=np.float64)
        sqrt_info = np.asarray(self.sqrt_information, dtype=np.float64)
        offset = np.asarray(self.residual_offset, dtype=np.float64)
        require(ref.ndim == 1, "reference_state must be a 1D vector")
        require(sqrt_info.ndim == 2, "sqrt_information must be a 2D matrix")
        require(offset.ndim == 1, "residual_offset must be a 1D vector")
        require(
            sqrt_info.shape[0] == offset.size,
            "sqrt_information rows must match residual_offset size",
        )
        require(
            sqrt_info.shape[1] == ref.size,
            "sqrt_information columns must match reference_state size",
        )
        require(0 <= self.rank <= sqrt_info.shape[1], "rank out of valid bounds")
        object.__setattr__(self, "reference_state", _make_readonly(ref))
        object.__setattr__(self, "sqrt_information", _make_readonly(sqrt_info))
        object.__setattr__(self, "residual_offset", _make_readonly(offset))
        if self.linearization_point is not None:
            lin = np.asarray(self.linearization_point, dtype=np.float64)
            require(lin.shape == ref.shape, "linearization_point shape mismatch")
            object.__setattr__(self, "linearization_point", _make_readonly(lin))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_gaussian_prior(
        cls,
        mean: np.ndarray,
        information: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArrivalFactor:
        """Build the factor of a Gaussian prior ``N(mean, information^-1)``.

        Preconditions: ``mean`` is a finite 1D vector of size ``n`` and
        ``information`` is a finite, symmetric, positive-definite ``(n, n)``
        matrix (checked by Cholesky factorisation).

        Postcondition: ``R^T R == information`` with ``R`` upper triangular,
        ``residual_offset == 0`` and ``rank == n``, so
        ``evaluate_cost(x) == 0.5 (x - mean)^T information (x - mean)``.
        """
        mu = np.asarray(mean, dtype=np.float64)
        info = np.asarray(information, dtype=np.float64)
        require(mu.ndim == 1 and mu.size > 0, "mean must be a non-empty 1D vector")
        require(bool(np.all(np.isfinite(mu))), "mean must be finite")
        require(
            info.shape == (mu.size, mu.size),
            f"information must have shape {(mu.size, mu.size)}",
        )
        require(bool(np.all(np.isfinite(info))), "information must be finite")
        require(
            bool(np.allclose(info, info.T, rtol=1e-12, atol=1e-12)),
            "information must be symmetric",
        )
        try:
            lower = np.linalg.cholesky(info)
        except np.linalg.LinAlgError:
            lower = None
        require(lower is not None, "information must be positive definite")
        assert lower is not None  # narrowed by the precondition above
        return cls(
            reference_state=mu,
            sqrt_information=lower.T,
            residual_offset=np.zeros(mu.size),
            rank=mu.size,
            metadata=dict(metadata or {}),
        )

    @property
    def is_rank_deficient(self) -> bool:
        """Whether the information matrix has revealed rank deficiency."""
        return self.rank < self.sqrt_information.shape[1]

    def evaluate_residual(self, state: np.ndarray) -> np.ndarray:
        """Compute arrival residual R * (state - ref) - r."""
        x = np.asarray(state, dtype=np.float64)
        require(x.shape == self.reference_state.shape, "state shape mismatch")
        delta_x = x - self.reference_state
        return self.sqrt_information @ delta_x - self.residual_offset

    def evaluate_cost(self, state: np.ndarray) -> float:
        """Compute quadratic cost 0.5 * || R * (state - ref) - r ||^2."""
        res = self.evaluate_residual(state)
        return 0.5 * float(np.sum(res**2))

    def evaluate_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian with respect to tangent perturbation."""
        x = np.asarray(state, dtype=np.float64)
        require(x.shape == self.reference_state.shape, "state shape mismatch")
        return np.array(self.sqrt_information, copy=True)


def marginalize_arrival_factor(
    A0: np.ndarray,
    A1: np.ndarray,
    b: np.ndarray,
    reference_state_k1: np.ndarray,
    gauge_policy: str = "retain_rank_deficiency",
    metadata: Mapping[str, Any] | None = None,
    rank_tol: float = 1e-10,
) -> ArrivalFactor:
    """Perform rank-revealing marginalization of A0 delta x0 + A1 delta x1 ~ b.

    Eliminates delta x0 and forms an updated square-root arrival factor on delta x1.
    If A0 is singular/unsupported, rank deficiency is retained without diagonal jitter.
    """
    mat_a0 = np.asarray(A0, dtype=np.float64)
    mat_a1 = np.asarray(A1, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)
    require(mat_a0.ndim == 2 and mat_a1.ndim == 2, "A0 and A1 must be 2D matrices")
    require(mat_a0.shape[0] == mat_a1.shape[0] == vec_b.size, "row dimension mismatch")
    n0 = mat_a0.shape[1]
    n1 = mat_a1.shape[1]

    # Full SVD on A0: A0 = U @ diag(S) @ Vt
    U, S, _ = np.linalg.svd(mat_a0, full_matrices=True)
    tol = max(rank_tol, S[0] * max(mat_a0.shape) * 1e-12) if S.size > 0 else rank_tol
    rank_a0 = int(np.sum(tol < S))

    if rank_a0 < n0 and gauge_policy == "fail_closed":
        raise PreconditionError(
            f"Singular/rank-deficient marginalization (rank {rank_a0} < {n0}) under fail_closed gauge policy."
        )

    # Rotate system by U^T:
    # First rank_a0 rows depend on x0 in observed subspace.
    # Remaining M - rank_a0 rows are completely decoupled from x0!
    U2 = U[:, rank_a0:]
    R_rem = U2.T @ mat_a1
    b_rem = U2.T @ vec_b

    # SVD on R_rem to reveal rank and obtain canonical square-root form without jitter
    if R_rem.shape[0] > 0 and R_rem.shape[1] > 0:
        U_rem, S_rem, Vt_rem = np.linalg.svd(R_rem, full_matrices=False)
        tol_rem = (
            max(rank_tol, S_rem[0] * max(R_rem.shape) * 1e-12)
            if S_rem.size > 0
            else rank_tol
        )
        rank_1 = int(np.sum(S_rem > tol_rem))
        # Zero out below-threshold singular values strictly without jitter
        S_clean = np.where(S_rem > tol_rem, S_rem, 0.0)
        # Pad or form square (n1, n1) matrix
        k_modes = len(S_clean)
        R_sq = np.zeros((n1, n1), dtype=np.float64)
        R_sq[:k_modes, :] = np.diag(S_clean) @ Vt_rem
        r_sq = np.zeros(n1, dtype=np.float64)
        r_sq[:k_modes] = U_rem.T @ b_rem
    else:
        R_sq = np.zeros((n1, n1), dtype=np.float64)
        r_sq = np.zeros(n1, dtype=np.float64)
        rank_1 = 0

    meta = dict(metadata or {})
    meta["gauge_policy"] = gauge_policy
    meta["eliminated_dim"] = n0
    meta["eliminated_rank"] = rank_a0

    return ArrivalFactor(
        reference_state=np.asarray(reference_state_k1, dtype=np.float64),
        sqrt_information=R_sq,
        residual_offset=r_sq,
        rank=rank_1,
        linearization_point=np.asarray(reference_state_k1, dtype=np.float64),
        metadata=meta,
    )


class AccumulationGuard:
    """Guard against double-counted measurements across receding window advances."""

    def __init__(self) -> None:
        self._marginalized_indices: set[int] = set()
        self._marginalized_timestamps: set[float] = set()

    def record_marginalized(self, sample_index: int, timestamp: float) -> None:
        """Record a sample that has been marginalized into the arrival factor."""
        require(
            isinstance(sample_index, int) and sample_index >= 0, "invalid sample_index"
        )
        require(np.isfinite(timestamp), "invalid timestamp")
        self._marginalized_indices.add(sample_index)
        self._marginalized_timestamps.add(float(timestamp))

    def is_marginalized(self, sample_index: int) -> bool:
        """Check if sample_index has been marginalized into arrival factor."""
        return sample_index in self._marginalized_indices

    def validate_sample(self, sample_index: int, timestamp: float) -> None:
        """Ensure sample has not been previously marginalized into the arrival factor."""
        if sample_index in self._marginalized_indices:
            raise PreconditionError(
                f"Sample index {sample_index} at t={timestamp} is already marginalized into the arrival factor; double-counting prevented."
            )


def knot_state_columns(n_knots: int, n_dof: int, knot: int) -> np.ndarray:
    """Decision columns of knot ``knot``'s state ``(q, v)`` in spline order."""
    q_cols = knot * n_dof + np.arange(n_dof)
    return np.concatenate([q_cols, n_knots * n_dof + q_cols])


def arrival_rows(
    arrival: ArrivalFactor, evaluation: SplineTrajectoryEvaluation, sample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Arrival residual and coefficient Jacobian at evaluation sample ``sample``.

    The sample must coincide with a knot so ``(q, v)`` there is the knot state.
    """
    state = np.concatenate([evaluation.q[sample], evaluation.v[sample]])
    basis = np.vstack([evaluation.q_basis[sample], evaluation.v_basis[sample]])
    return arrival.evaluate_residual(state), arrival.sqrt_information @ basis


def forward_difference(
    fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, f0: np.ndarray
) -> np.ndarray:
    jac = np.zeros((f0.size, x.size))
    for col in range(x.size):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(x[col])))
        probe = np.array(x, dtype=float, copy=True)
        probe[col] += step
        jac[:, col] = (fn(probe) - f0) / step
    return jac


def marginalize_window_prefix(
    traj: CubicHermiteSplineTrajectory,
    coeffs: np.ndarray,
    arrival: ArrivalFactor | None,
    knot_span: tuple[int, int],
    rows_fn: RowsFn,
) -> ArrivalFactor | None:
    """Eliminate knots ``[a, f)`` of a solved window onto knot ``f``.

    The rows that leave the problem are those evaluated on samples ``a..f``
    (including transitions into ``f``) minus those of sample ``f`` alone,
    which the next window evaluates again, plus the arrival rows on knot
    ``a``. Linearised at ``coeffs`` they give the information
    ``H = J_a^T J_a - J_f^T J_f`` and gradient ``g``; a square root
    ``A^T A = H``, ``A^T b = -g`` is marginalised by
    :func:`marginalize_arrival_factor`. Returns ``None`` when the
    linearisation is non-finite.

    Preconditions: residual rows are per-sample or couple consecutive
    samples only, so ``H`` is positive semidefinite and touches no knot
    outside ``a..f``; anything else would drop or double count information
    and is rejected.
    """
    start, stop = knot_span
    n_knots, n_dof = traj.n_knots, traj.n_dof
    value, jac = rows_fn(coeffs, traj.knot_times[start : stop + 1])
    boundary_value, boundary_jac = rows_fn(coeffs, traj.knot_times[stop : stop + 1])
    if arrival is not None:
        evaluation = traj.evaluate(coeffs, traj.knot_times[start : start + 1])
        arrival_value, arrival_jac = arrival_rows(arrival, evaluation, 0)
        value = np.concatenate([value, arrival_value])
        jac = np.vstack([jac, arrival_jac])
    arrays = (value, jac, boundary_value, boundary_jac)
    if not all(bool(np.all(np.isfinite(arr))) for arr in arrays):
        return None
    info = jac.T @ jac - boundary_jac.T @ boundary_jac
    info = 0.5 * (info + info.T)
    grad = jac.T @ value - boundary_jac.T @ boundary_value
    eliminated = np.concatenate(
        [knot_state_columns(n_knots, n_dof, k) for k in range(start, stop)]
    )
    kept = knot_state_columns(n_knots, n_dof, stop)
    cols = np.concatenate([eliminated, kept])
    other = np.setdiff1d(np.arange(traj.coefficient_size), cols)
    scale = max(1.0, float(np.max(np.abs(info))))
    require(
        other.size == 0 or float(np.max(np.abs(info[other]))) <= 1e-9 * scale,
        "marginalised residual rows must touch only the dropped knots and the "
        "new first knot (per-sample or consecutive-sample residuals); otherwise "
        "information is dropped or double counted",
    )
    eigvals, eigvecs = np.linalg.eigh(info[np.ix_(cols, cols)])
    require(
        float(eigvals.min()) >= -1e-9 * scale,
        "boundary-sample rows must be a subset of the span rows (information "
        "of the marginalised rows is not positive semidefinite)",
    )
    keep = eigvals > 1e-12 * scale * cols.size
    root = np.sqrt(eigvals[keep])
    sqrt_rows = root[:, None] * eigvecs[:, keep].T
    target = -(eigvecs[:, keep].T @ grad[cols]) / root
    n_elim = eliminated.size
    return marginalize_arrival_factor(
        A0=sqrt_rows[:, :n_elim],
        A1=sqrt_rows[:, n_elim:],
        b=target,
        reference_state_k1=coeffs[kept],
        metadata={"eliminated_knots": stop - start},
    )
