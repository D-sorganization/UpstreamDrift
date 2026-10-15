"""Null-space analysis of a dynamics equality (NSX-03 split from force_nullspace)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg

from ._common import Array, _array


@dataclass(frozen=True)
class NullSpaceAnalysis:
    """Diagnostic numerical metrics of an equality null-space decomposition."""

    rank: int
    nullity: int
    singular_values: Array
    equality_residual: float
    null_residual: float
    condition_number: float


@dataclass(frozen=True)
class ForceNullSpace:
    """Scaled SVD or QR representation x = particular + basis @ z of A x = b.

    Use from_balance to construct. Row scales have the units of b; variable
    scales have the units of x. Rank is numerical and may change with contact
    mode. Physical derivatives must never be computed from z differences.
    """

    matrix: Array
    rhs: Array
    row_scale: Array
    particular: Array
    basis: Array
    rank: int
    variable_scale: Array

    def analyze(self) -> NullSpaceAnalysis:
        """Compute rank, nullity, residuals, singular values, and condition number."""
        scaled = self.matrix * self.variable_scale[None, :] / self.row_scale[:, None]
        s = np.linalg.svd(scaled, compute_uv=False)
        cond = (
            float(s[0] / s[self.rank - 1])
            if self.rank > 0 and s[self.rank - 1] > 0
            else float("inf")
        )
        eq_res = float(
            np.max(np.abs((self.matrix @ self.particular - self.rhs) / self.row_scale))
        )
        null_res = (
            float(np.max(np.abs((self.matrix @ self.basis) / self.row_scale[:, None])))
            if self.basis.size > 0
            else 0.0
        )
        s_arr = np.array(s, dtype=float, copy=True)
        s_arr.setflags(write=False)
        return NullSpaceAnalysis(
            rank=self.rank,
            nullity=int(self.basis.shape[1]),
            singular_values=s_arr,
            equality_residual=eq_res,
            null_residual=null_res,
            condition_number=cond,
        )

    def validate_null_space(self, tolerance: float = 1e-8) -> bool:
        """Verify that equality residual and null basis residual satisfy tolerance."""
        analysis = self.analyze()
        return bool(
            analysis.equality_residual <= tolerance
            and analysis.null_residual <= tolerance
        )

    @classmethod
    def from_balance(
        cls,
        matrix: Array,
        rhs: Array,
        *,
        variable_scale: Array | None = None,
        row_scale: Array | None = None,
        rank_rtol: float = 1e-12,
        balance_tolerance: float = 1e-9,
        method: Literal["svd", "qr"] = "svd",
    ) -> ForceNullSpace:
        """Reject inconsistent/nonfinite equations; return owned read-only arrays.

        balance_tolerance is a dimensionless per-row residual after division
        by row_scale, NOT a combined norm of forces and moments.
        method chooses between scaled SVD or column-pivoted QR decomposition.
        """
        if method not in ("svd", "qr"):
            raise ValueError(f"Unknown decomposition method: {method!r}")
        a = _array(matrix, 2, "matrix")
        b = _array(rhs, 1, "rhs")
        m, n = a.shape
        if m == 0 or n == 0 or b.shape != (m,):
            raise ValueError("balance dimensions must be nonempty and compatible")
        sx = _array(
            np.ones(n) if variable_scale is None else variable_scale,
            1,
            "variable_scale",
        )
        sr = _array(np.ones(m) if row_scale is None else row_scale, 1, "row_scale")
        if sx.shape != (n,) or sr.shape != (m,) or np.any(sx <= 0) or np.any(sr <= 0):
            raise ValueError("scales must be positive and match balance dimensions")
        if (
            not 0 < rank_rtol < 1
            or not np.isfinite(balance_tolerance)
            or balance_tolerance <= 0
        ):
            raise ValueError(
                "rank_rtol and balance_tolerance must be positive and finite"
            )
        scaled = a * sx[None, :] / sr[:, None]
        if not np.isfinite(scaled).all():
            raise ValueError("scaled balance must be finite")

        if method == "svd":
            u, singular, vt = np.linalg.svd(scaled, full_matrices=True)
            rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
            y = vt[:rank].T @ ((u[:, :rank].T @ (b / sr)) / singular[:rank])
            particular = sx * y
            basis = sx[:, None] * vt[rank:].T
        else:
            # method == "qr"
            q, r, p = scipy.linalg.qr(scaled.T, pivoting=True)
            diag_r = np.abs(np.diag(r))
            rank = (
                int(np.count_nonzero(diag_r > rank_rtol * diag_r[0]))
                if diag_r.size
                else 0
            )
            q1 = q[:, :rank]
            q2 = q[:, rank:]
            r1 = r[:rank, :rank]
            b_scaled = b / sr
            if rank > 0:
                y = scipy.linalg.solve_triangular(r1.T, b_scaled[p[:rank]], lower=True)
                particular = sx * (q1 @ y)
            else:
                particular = np.zeros(n)
            basis = sx[:, None] * q2

        if np.max(np.abs((a @ particular - b) / sr)) > balance_tolerance:
            raise ValueError("inconsistent dynamics equality at requested tolerance")
        if basis.size and np.max(np.abs((a @ basis) / sr[:, None])) > balance_tolerance:
            raise ValueError("numerical null basis fails dynamics tolerance")
        return cls(
            a,
            b,
            sr,
            _array(particular, 1, "particular"),
            _array(basis, 2, "basis"),
            rank,
            sx,
        )
