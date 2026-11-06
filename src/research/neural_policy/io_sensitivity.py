"""Policy input-output sensitivity (dtau/dq, dtau/dqdot) (#14502)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import require

NAMED_SWING_EVENTS: tuple[str, ...] = ("address", "top", "mid_downswing", "impact")


@dataclass(frozen=True)
class CompactMLP:
    """Compact neural policy with analytic sensitivity."""

    weights: tuple[NDArray[np.floating], ...]
    biases: tuple[NDArray[np.floating], ...]
    activation: str = "tanh"

    def __call__(self, q: Any, qdot: Any) -> NDArray[np.floating]:
        x, last = np.hstack([q, qdot], dtype=float), len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = w @ x + b
            x = np.tanh(z) if (i < last and self.activation == "tanh") else z
        return x

    def analytic_jacobian(self, q: Any, qdot: Any) -> tuple[NDArray, NDArray]:
        x, n, last = np.hstack([q, qdot], dtype=float), len(q), len(self.weights) - 1
        j = np.eye(len(x), dtype=float)
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z, act = w @ x + b, (i < last and self.activation == "tanh")
            x = np.tanh(z) if act else z
            j = ((1.0 - x**2)[:, None] * (w @ j)) if act else (w @ j)
        return j[:, :n], j[:, n:]


@dataclass(frozen=True)
class SensitivityReport:
    """Input-output sensitivity report at a stated operating point."""

    event: str
    operating_point: dict[str, list[float]]
    s_q: list[list[float]]
    s_qd: list[list[float]]
    metrics_q: dict[str, Any]
    metrics_qd: dict[str, Any]
    off_manifold: bool = False
    disclaimer: str = (
        "Sensitivities at stated operating point do not generalize off-manifold; "
        "symmetry and positive-definiteness are not guaranteed."
    )


def check_matrix_properties(s: NDArray[np.floating]) -> dict[str, Any]:
    """Report symmetry and positive-definiteness (neither guaranteed)."""
    a, sq = np.asarray(s, float), s.ndim == 2 and s.shape[0] == s.shape[1]
    eig = float(np.min(np.linalg.eigvalsh((a + a.T) / 2.0))) if sq else None
    return {
        "frobenius_norm": float(np.linalg.norm(a)),
        "asymmetry_norm": float(np.linalg.norm(a - a.T)) if sq else None,
        "symmetric_min_eigenvalue": eig,
        "is_positive_definite": bool(eig is not None and eig > 0.0),
        "guaranteed": False,
    }


def compute_finite_difference_jacobian(
    policy: Any, q: Any, qdot: Any, h: float = 1e-6
) -> tuple[NDArray, NDArray]:
    """Compute local sensitivity via central finite differences."""
    qa, qda = np.asarray(q, float), np.asarray(qdot, float)

    def _col(v: NDArray[np.floating], i: int, is_q: bool) -> NDArray[np.floating]:
        e = np.zeros_like(v)
        e[i] = h
        f1 = policy(v + e, qda) if is_q else policy(qa, v + e)
        f2 = policy(v - e, qda) if is_q else policy(qa, v - e)
        return (f1 - f2) / (2.0 * h)

    return (
        np.column_stack([_col(qa, i, True) for i in range(len(qa))]),
        np.column_stack([_col(qda, i, False) for i in range(len(qda))]),
    )


def compute_io_sensitivity(
    policy: Any,
    q: Any,
    qdot: Any,
    *,
    event: str = "custom",
    envelope: Any = None,
    strict: bool = False,
    step: float = 1e-6,
) -> SensitivityReport:
    """Compute local sensitivity S_q and S_qd at a stated operating point."""
    q_arr, qd_arr = np.asarray(q, float), np.asarray(qdot, float)
    ok = bool(np.all(np.isfinite(q_arr) & np.isfinite(qd_arr)))
    require(ok, "finite inputs", None)
    off = False
    if envelope is not None:
        lo, hi = np.asarray(envelope[0], float), np.asarray(envelope[1], float)
        x = np.concatenate([q_arr, qd_arr])
        out = bool(not np.all((x >= lo) & (x <= hi)))
        require(not (out and strict), "outside training envelope", None)
        off = out
    fd = compute_finite_difference_jacobian
    fn = getattr(policy, "analytic_jacobian", None)
    s_q, s_qd = fn(q_arr, qd_arr) if fn else fd(policy, q_arr, qd_arr, h=step)
    tau = np.asarray(policy(q_arr, qd_arr), float)
    op = {"q": q_arr.tolist(), "qdot": qd_arr.tolist(), "tau": tau.tolist()}
    cq = check_matrix_properties
    return SensitivityReport(
        event, op, s_q.tolist(), s_qd.tolist(), cq(s_q), cq(s_qd), off
    )
