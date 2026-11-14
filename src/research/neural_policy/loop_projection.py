"""Closed grip-loop constraint projection of policy outputs (#14500).

Projects raw generalized force outputs onto ker(J_loop(q0)) using
reused loop_nullspace machinery: P(q0) = I - J_loop(q0)^+ J_loop(q0).
Diagnostic projection only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import require
from src.shared.python.loop_nullspace.tangent_linearization import tangent_projector


def _check_arr(x: Any, ndim: int, name: str) -> None:
    ok = isinstance(x, np.ndarray) and x.ndim == ndim and bool(np.all(np.isfinite(x)))
    require(ok, f"{name} must be {ndim}D finite ndarray")


@dataclass(frozen=True)
class LoopProjectionRecord:
    """Diagnostic run record for closed-loop constraint projection (#14500)."""

    violation_before: float
    violation_after: float
    reduction_ratio: float
    is_valid: bool
    q0_norm: float
    diagnostic: str = "np07_loop_nullspace_projection"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_loop_nullspace_projector(
    jacobian: NDArray[np.floating], rcond: float = 1e-12
) -> NDArray[np.float64]:
    """Orthogonal projector onto ker(J_loop) reusing loop_nullspace."""
    _check_arr(jacobian, 2, "jacobian")
    require(rcond > 0.0 and math.isfinite(rcond), "rcond must be positive", rcond)
    proj = tangent_projector(jacobian)
    return np.asarray(0.5 * (proj + proj.T), dtype=np.float64)


@dataclass(frozen=True)
class FoldedLoopProjection:
    """Weight-folded projector valid only at a single reference q0 (#14500)."""

    weight_projected: NDArray[np.float64]
    bias_projected: NDArray[np.float64] | None
    q0: NDArray[np.float64]
    tolerance: float

    def is_valid_at(self, q: NDArray[np.floating]) -> bool:
        return bool(
            isinstance(q, np.ndarray)
            and q.shape == self.q0.shape
            and np.all(np.isfinite(q))
            and np.linalg.norm(q - self.q0) <= self.tolerance
        )

    def forward(
        self, h: NDArray[np.floating], q: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        """Evaluate folded linear output layer W_proj @ h + b_proj."""
        _check_arr(h, 1, "h")
        require(
            h.shape[0] == self.weight_projected.shape[1],
            "h dimension mismatch",
            (h.shape[0], self.weight_projected.shape[1]),
        )
        if q is not None:
            require(
                isinstance(q, np.ndarray) and q.ndim == 1 and self.is_valid_at(q),
                "stale Jacobian",
            )
        out = self.weight_projected @ h
        return np.asarray(
            out if self.bias_projected is None else out + self.bias_projected,
            dtype=np.float64,
        )


class LoopConstraintProjector:
    """Closed grip-loop nullspace projector for policy output (#14500)."""

    def __init__(
        self,
        q0: NDArray[np.floating],
        jacobian: NDArray[np.floating],
        tol: float = 0.05,
        rcond: float = 1e-12,
    ) -> None:
        _check_arr(q0, 1, "q0")
        _check_arr(jacobian, 2, "jacobian")
        require(
            jacobian.shape[1] == q0.shape[0],
            "jacobian columns must match q0 dimension",
            (jacobian.shape[1], q0.shape[0]),
        )
        require(tol > 0.0 and math.isfinite(tol), "tol must be positive", tol)
        self.q0 = np.asarray(q0, dtype=np.float64)
        self.jacobian = np.asarray(jacobian, dtype=np.float64)
        self.tolerance, self._dim = float(tol), int(q0.shape[0])
        self.projector_matrix = compute_loop_nullspace_projector(
            self.jacobian, rcond=rcond
        )

    def is_valid_at(self, q: NDArray[np.floating]) -> bool:
        return bool(
            isinstance(q, np.ndarray)
            and q.shape == (self._dim,)
            and np.all(np.isfinite(q))
            and np.linalg.norm(q - self.q0) <= self.tolerance
        )

    def project(
        self, u: NDArray[np.floating], q: NDArray[np.floating] | None = None
    ) -> NDArray[np.float64]:
        """Project raw output u onto ker(J_loop(q0))."""
        _check_arr(u, 1, "u")
        require(
            u.shape[0] == self._dim, "u dimension mismatch", (u.shape[0], self._dim)
        )
        if q is not None:
            _check_arr(q, 1, "q")
            require(q.shape[0] == self._dim and self.is_valid_at(q), "stale Jacobian")
        return np.asarray(self.projector_matrix @ u, dtype=np.float64)

    def evaluate_diagnostic(
        self, u: NDArray[np.floating], q: NDArray[np.floating] | None = None
    ) -> tuple[NDArray[np.float64], LoopProjectionRecord]:
        """Evaluate projection and return projected output with diagnostic record."""
        u_p = self.project(u, q=q)
        vb = float(np.linalg.norm(self.jacobian @ u))
        va = float(np.linalg.norm(self.jacobian @ u_p))
        return u_p, LoopProjectionRecord(
            violation_before=vb,
            violation_after=va,
            reduction_ratio=float((vb + 1e-12) / (va + 1e-12)),
            is_valid=self.is_valid_at(q) if q is not None else True,
            q0_norm=float(np.linalg.norm(self.q0)),
        )

    def fold_weights(
        self,
        w_out: NDArray[np.floating],
        b_out: NDArray[np.floating] | None = None,
    ) -> FoldedLoopProjection:
        """Fold projection into final linear layer weights: W_proj = P @ W_out."""
        _check_arr(w_out, 2, "w_out")
        require(
            w_out.shape[0] == self._dim,
            "w_out dimension mismatch",
            (w_out.shape[0], self._dim),
        )
        b_p = None
        if b_out is not None:
            _check_arr(b_out, 1, "b_out")
            require(
                b_out.shape[0] == self._dim,
                "b_out dimension mismatch",
                (b_out.shape[0], self._dim),
            )
            b_p = np.asarray(self.projector_matrix @ b_out, dtype=np.float64)
        return FoldedLoopProjection(
            weight_projected=np.asarray(
                self.projector_matrix @ w_out, dtype=np.float64
            ),
            bias_projected=b_p,
            q0=self.q0,
            tolerance=self.tolerance,
        )
