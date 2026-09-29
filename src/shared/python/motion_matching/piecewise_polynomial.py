"""Piecewise polynomial torque representations and multi-phase stitching (#9921).

Provides representation, evaluation, and stitching for segmented swing phases:
- Backswing phase: [0, t_top]
- Downswing phase: [t_top, t_final]
Enforces C^0 and C^1 boundary continuity at the transition, supports
evaluation on arbitrary time grids, and computes global least-squares projections.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .polynomial_torque import COEFFS_PER_JOINT, POLY_DEGREE

Array: TypeAlias = NDArray[np.float64]


def _bernstein_basis_matrix(degree: int, s: Array) -> Array:
    """Compute Bernstein basis values B_{k, degree}(s) for s in [0, 1].

    Returns array of shape (len(s), degree + 1).
    """
    s_arr = np.asarray(s, dtype=np.float64)
    n_pts = s_arr.size
    basis = np.zeros((n_pts, degree + 1), dtype=np.float64)
    s_clamped = np.clip(s_arr, 0.0, 1.0)
    for k in range(degree + 1):
        basis[:, k] = (
            comb(degree, k) * (s_clamped**k) * ((1.0 - s_clamped) ** (degree - k))
        )
    return basis


def bernstein_to_power_matrix(degree: int) -> Array:
    """Matrix M of shape (degree + 1, degree + 1) mapping control points to power coefficients.

    p = c @ M.T where p are coefficients of s^j, j=0..degree.
    """
    mat = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for k in range(degree + 1):
        for j in range(k, degree + 1):
            mat[k, j] = comb(degree, k) * comb(degree - k, j - k) * ((-1) ** (j - k))
    return mat


@dataclass(frozen=True)
class PolynomialSegment:
    """A single continuous polynomial segment on time interval [start_s, end_s].

    Parameters:
    - start_s: Beginning of interval (seconds).
    - end_s: End of interval (seconds), strictly greater than start_s.
    - coefficients: Array of shape (n_channels, degree + 1).
      If is_bernstein is True, coefficients are Bernstein control points c_k.
      If is_bernstein is False, coefficients are ascending powers of local time:
      sum_{k=0}^degree c_k * ((t - start_s) / duration_s)^k.
    - is_bernstein: Whether coefficients represent Bernstein control points.
    """

    start_s: float
    end_s: float
    coefficients: Array
    is_bernstein: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_s) or not np.isfinite(self.end_s):
            raise ValueError("start_s and end_s must be finite")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be strictly greater than start_s")
        coeffs = np.asarray(self.coefficients, dtype=np.float64)
        if coeffs.ndim != 2 or coeffs.shape[0] == 0 or coeffs.shape[1] == 0:
            raise ValueError("coefficients must have shape (n_channels, degree + 1)")
        if not np.isfinite(coeffs).all():
            raise ValueError("coefficients must be finite")
        object.__setattr__(self, "coefficients", coeffs)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def n_channels(self) -> int:
        return self.coefficients.shape[0]

    @property
    def degree(self) -> int:
        return self.coefficients.shape[1] - 1

    def evaluate(self, t: float) -> Array:
        """Evaluate torque vector at scalar time t in [start_s, end_s]."""
        s = (t - self.start_s) / self.duration_s
        s = float(np.clip(s, 0.0, 1.0))
        if self.is_bernstein:
            basis = _bernstein_basis_matrix(self.degree, np.array([s]))[0]
            return np.dot(self.coefficients, basis)
        # Power basis in normalized local time s
        powers = s ** np.arange(self.degree + 1)
        return np.dot(self.coefficients, powers)

    def evaluate_grid(self, time_grid: Array) -> Array:
        """Evaluate torque matrix across time grid.

        Returns array of shape (len(time_grid), n_channels).
        """
        times = np.asarray(time_grid, dtype=np.float64)
        s = (times - self.start_s) / self.duration_s
        s = np.clip(s, 0.0, 1.0)
        if self.is_bernstein:
            basis = _bernstein_basis_matrix(self.degree, s)
            return np.dot(basis, self.coefficients.T)
        powers = s[:, None] ** np.arange(self.degree + 1)[None, :]
        return np.dot(powers, self.coefficients.T)

    def evaluate_derivative(self, t: float) -> Array:
        """Evaluate first time derivative dtau/dt at scalar time t."""
        s = float(np.clip((t - self.start_s) / self.duration_s, 0.0, 1.0))
        degree = self.degree
        dur = self.duration_s
        if degree == 0:
            return np.zeros(self.n_channels, dtype=np.float64)

        if self.is_bernstein:
            # Derivative of degree d Bernstein polynomial is d/dur * sum_{k=0}^{d-1} (c_{k+1} - c_k) B_{k, d-1}(s)
            delta_c = self.coefficients[:, 1:] - self.coefficients[:, :-1]
            basis = _bernstein_basis_matrix(degree - 1, np.array([s]))[0]
            return (degree / dur) * np.dot(delta_c, basis)

        # Power basis: d/dt sum_{k=0}^d p_k s^k = 1/dur * sum_{k=1}^d k p_k s^{k-1}
        k_vec = np.arange(1, degree + 1, dtype=np.float64)
        powers = s ** np.arange(degree)
        return (1.0 / dur) * np.dot(self.coefficients[:, 1:] * k_vec, powers)


@dataclass(frozen=True)
class PiecewisePolynomialTorque:
    """Composite piecewise polynomial torque over multiple contiguous segments."""

    segments: tuple[PolynomialSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("PiecewisePolynomialTorque requires at least one segment")
        n_channels = self.segments[0].n_channels
        for i, seg in enumerate(self.segments):
            if seg.n_channels != n_channels:
                raise ValueError(
                    f"Segment {i} has {seg.n_channels} channels; expected {n_channels}"
                )
            if i > 0:
                prev = self.segments[i - 1]
                if abs(seg.start_s - prev.end_s) > 1e-9:
                    raise ValueError(
                        f"Discontinuous time interval between segment {i - 1} "
                        f"([{prev.start_s}, {prev.end_s}]) and segment {i} "
                        f"([{seg.start_s}, {seg.end_s}])"
                    )

    @property
    def start_s(self) -> float:
        return self.segments[0].start_s

    @property
    def end_s(self) -> float:
        return self.segments[-1].end_s

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def n_channels(self) -> int:
        return self.segments[0].n_channels

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def breakpoints(self) -> tuple[float, ...]:
        pts = [self.segments[0].start_s]
        for seg in self.segments:
            pts.append(seg.end_s)
        return tuple(pts)

    def evaluate(self, t: float) -> Array:
        """Evaluate torque vector at scalar time t."""
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        # Clamp t to active range
        t_clamped = float(np.clip(t, self.start_s, self.end_s))
        for seg in self.segments:
            if seg.start_s <= t_clamped <= seg.end_s:
                return seg.evaluate(t_clamped)
        return self.segments[-1].evaluate(t_clamped)

    def evaluate_grid(self, time_grid: Array) -> Array:
        """Evaluate torque matrix across time grid.

        Returns array of shape (len(time_grid), n_channels).
        """
        times = np.asarray(time_grid, dtype=np.float64)
        out = np.zeros((len(times), self.n_channels), dtype=np.float64)
        for seg in self.segments:
            mask = (times >= seg.start_s) & (times <= seg.end_s)
            if np.any(mask):
                out[mask] = seg.evaluate_grid(times[mask])
        # Handle points slightly before start or after end
        before = times < self.start_s
        if np.any(before):
            out[before] = self.segments[0].evaluate(self.start_s)
        after = times > self.end_s
        if np.any(after):
            out[after] = self.segments[-1].evaluate(self.end_s)
        return out

    def check_continuity(self, *, tol: float = 1e-4) -> dict[str, Any]:
        """Check C^0 and C^1 continuity across all internal breakpoints.

        Returns dictionary with maximum jumps and boolean passes.
        """
        c0_jumps: list[float] = []
        c1_jumps: list[float] = []
        for i in range(len(self.segments) - 1):
            t_trans = self.segments[i].end_s
            val_left = self.segments[i].evaluate(t_trans)
            val_right = self.segments[i + 1].evaluate(t_trans)
            c0_jump = float(np.max(np.abs(val_right - val_left)))
            c0_jumps.append(c0_jump)

            dval_left = self.segments[i].evaluate_derivative(t_trans)
            dval_right = self.segments[i + 1].evaluate_derivative(t_trans)
            c1_jump = float(np.max(np.abs(dval_right - dval_left)))
            c1_jumps.append(c1_jump)

        max_c0 = max(c0_jumps) if c0_jumps else 0.0
        max_c1 = max(c1_jumps) if c1_jumps else 0.0
        return {
            "c0_continuous": max_c0 <= tol,
            "max_c0_jump": max_c0,
            "c0_jumps": c0_jumps,
            "c1_continuous": max_c1 <= tol,
            "max_c1_jump": max_c1,
            "c1_jumps": c1_jumps,
        }

    def project_to_global_polynomial(
        self,
        *,
        degree: int = 6,
        num_samples: int = 501,
        pin_endpoints: bool = True,
    ) -> Array:
        """Fit a single global polynomial of specified degree across [start_s, end_s].

        Returns array of shape (n_channels, degree + 1) in normalized ascending power basis:
        tau(s) = sum_{k=0}^degree c_k * s^k, where s = (t - start_s) / duration_s in [0, 1].
        """
        s_grid = np.linspace(0.0, 1.0, num_samples, dtype=np.float64)
        t_grid = np.asarray(self.start_s + s_grid * self.duration_s, dtype=np.float64)
        y_target = self.evaluate_grid(t_grid)  # (N, n_channels)

        # Vandermonde matrix for s in [0, 1]
        vandermonde = (
            s_grid[:, None] ** np.arange(degree + 1)[None, :]
        )  # (N, degree + 1)

        if not pin_endpoints or degree < 2:
            coeffs, _, _, _ = np.linalg.lstsq(vandermonde, y_target, rcond=None)
            return np.asarray(coeffs.T, dtype=np.float64)

        # Constrained least squares:
        # Enforce exact match at s=0: c_0 = y_target[0]
        # Enforce exact match at s=1: sum_{k=0}^d c_k = y_target[-1] => sum_{k=1}^d c_k = y_target[-1] - c_0
        n_ch = self.n_channels
        result = np.zeros((n_ch, degree + 1), dtype=np.float64)
        c0 = y_target[0]  # (n_ch,)
        result[:, 0] = c0

        # For s in (0, 1), y_target - c0 = sum_{k=1}^d c_k * s^k
        # With constraint at s=1: c_d = (y_target[-1] - c0) - sum_{k=1}^{d-1} c_k
        # Substitute into Vandermonde:
        # sum_{k=1}^{d-1} c_k (s^k - s^d) + s^d (y_target[-1] - c0)
        y_end_delta = y_target[-1] - c0  # (n_ch,)
        s_d = s_grid**degree
        v_reduced = np.zeros((num_samples, degree - 1), dtype=np.float64)
        for k in range(1, degree):
            v_reduced[:, k - 1] = s_grid**k - s_d

        rhs = (y_target - c0[None, :]) - s_d[:, None] * y_end_delta[None, :]
        sol, _, _, _ = np.linalg.lstsq(v_reduced, rhs, rcond=None)  # (degree - 1, n_ch)
        intermediate = sol.T  # (n_ch, degree - 1)
        result[:, 1:degree] = intermediate
        result[:, degree] = y_end_delta - np.sum(intermediate, axis=1)

        return result


def stitch_two_phase_trajectories(
    backswing: PolynomialSegment,
    downswing: PolynomialSegment,
    *,
    enforce_c0: bool = True,
    enforce_c1: bool = False,
) -> PiecewisePolynomialTorque:
    """Stitch backswing and downswing polynomial segments into a composite trajectory.

    If enforce_c0 is True, sets the initial value of downswing to match backswing terminal torque.
    If enforce_c1 is True, also adjusts the downswing initial derivative to match backswing terminal derivative.
    """
    if backswing.n_channels != downswing.n_channels:
        raise ValueError(
            "Backswing and downswing must have the same number of channels"
        )
    if abs(backswing.end_s - downswing.start_s) > 1e-6:
        raise ValueError(
            f"Backswing ends at {backswing.end_s}s but downswing starts at {downswing.start_s}s"
        )

    t_trans = backswing.end_s
    tau_trans = backswing.evaluate(t_trans)

    down_coeffs = np.array(downswing.coefficients, copy=True)
    deg_down = downswing.degree
    dur_down = downswing.duration_s

    if enforce_c0:
        if downswing.is_bernstein:
            # In Bernstein basis, tau(0) = c_0
            down_coeffs[:, 0] = tau_trans
        else:
            # In power basis, tau(0) = p_0
            down_coeffs[:, 0] = tau_trans

    if enforce_c1 and deg_down >= 1:
        dtau_trans = backswing.evaluate_derivative(t_trans)
        if downswing.is_bernstein:
            # dtau(0)/dt = (deg / dur) * (c_1 - c_0) => c_1 = c_0 + (dur / deg) * dtau_trans
            c0 = down_coeffs[:, 0]
            down_coeffs[:, 1] = c0 + (dur_down / deg_down) * dtau_trans
        else:
            # dtau(0)/dt = (1 / dur) * p_1 => p_1 = dur * dtau_trans
            down_coeffs[:, 1] = dur_down * dtau_trans

    adjusted_downswing = PolynomialSegment(
        start_s=downswing.start_s,
        end_s=downswing.end_s,
        coefficients=down_coeffs,
        is_bernstein=downswing.is_bernstein,
    )

    return PiecewisePolynomialTorque(segments=(backswing, adjusted_downswing))
