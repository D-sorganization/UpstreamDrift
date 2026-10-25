"""Local solution-manifold continuation with rank gates (NSE-UD-2, #13462).

For a smooth residual ``F: R^n -> R^m`` the post-fit solution set ``F(u) = 0``
is nonlinear in general (Hill-type activation-to-force maps, parameter-coupled
balance).  This module moves along it from one admitted solution ``u_0`` by a
predictor-corrector step and reports, at every point, the rank of the Jacobian
that justified the move.

What is guaranteed
------------------
* **Local only.**  At a point where ``J = DF(u)`` has constant rank ``r`` in a
  neighbourhood, the implicit function theorem makes ``{F = 0}`` a smooth
  manifold of dimension ``n - r`` there, with tangent space ``ker J``.  The
  chart and the step are valid only inside such a neighbourhood.
* **Rank gate.**  The step raises :class:`ManifoldSingularityError` when the
  rank at the corrected point (or at any corrector iterate) differs from the
  rank of the starting chart, when the corrector's linear system becomes
  inconsistent, or when the new chart is ill-conditioned (condition number of
  the scaled Jacobian above ``1 / tol``).  The theorem does not apply across a
  rank change: ``F(u) = u1 * u2`` has rank 1 on the axes away from the origin
  and rank 0 at the origin, where the zero set is a cross, not a manifold.
* **No coverage claim.**  Continuation does not certify that a sequence of
  steps visits (or even approaches) every point of a connected component, that
  two reached points lie on the same component, or that no component was
  skipped by a step.  The step length is not adapted to curvature.
* **Orders.**  The tangent predictor leaves a residual ``O(h^2)`` in the step
  length ``h`` (no higher-order claim is made).  The corrector is a
  minimum-norm Newton iteration, which converges locally and quadratically
  where the rank is constant.
* **The correction is not the nearest-point projection.**  Each Newton update
  is the *minimum-norm solution of the linearised equation* in the declared
  scaled norm, ``du = argmin ||S^-1 du||  s.t.  J du = -F``; the endpoint is
  generally not the point of ``{F = 0}`` nearest to the predicted point.

Scaling convention
------------------
``variable_scale`` ``s`` (positive, one entry per unknown, in the units of
``u``) defines the norm ``||x||_S = ||S^-1 x||_2``, ``S = diag(s)``.
:class:`~src.shared.python.motion_matching.force_nullspace.space.ForceNullSpace`
already carries ``s`` in its basis: its columns are ``N = S V_2`` where ``V_2``
has orthonormal columns.  Hence the tangent basis stored here satisfies

``N^T S^-2 N = I``  (orthonormal *in the declared scaling*), and ``J N = 0``

in raw coordinates; ``N`` is not orthonormal in the raw Euclidean inner
product unless ``s`` is constant ``1``.  The scale is applied exactly once
(inside ``ForceNullSpace``) and ``s`` is stored on the chart so a step never
needs it again from the caller.  ``singular_values`` are those of ``J S``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import (
    Array,
    check_shape,
    readonly_float,
)
from src.shared.python.motion_matching.force_nullspace.space import ForceNullSpace

__all__ = [
    "ContinuationStep",
    "ManifoldChart",
    "ManifoldConvergenceError",
    "ManifoldSingularityError",
    "local_chart",
    "predictor_corrector_step",
]

ResidualFn = Callable[[Array], ArrayLike]
JacobianFn = Callable[[Array], ArrayLike]

_MIN_BALANCE_TOL = 1e-9
"""Floor of ``ForceNullSpace``'s per-row consistency check (its default)."""

_TANGENT_FLOOR = 1e-12
"""Relative size below which a projected direction counts as zero."""


class ManifoldSingularityError(RuntimeError):
    """The rank (or conditioning) gate failed: the constant-rank IFT does not apply."""


class ManifoldConvergenceError(RuntimeError):
    """The corrector did not reach ``||F|| <= tol`` within ``max_iter`` updates."""


@dataclass(frozen=True)
class ManifoldChart:
    """Tangent data of ``{F = 0}`` at one admitted point (valid locally only).

    Attributes:
        u_base: The point ``u`` (read-only, length ``n``).
        tangent_basis: ``ForceNullSpace`` basis of ``ker J`` (``n x (n - rank)``,
            read-only); ``N^T diag(variable_scale)^-2 N = I`` and ``J N = 0``.
        singular_values: Singular values of ``J diag(variable_scale)``.
        rank: Numerical rank of the scaled Jacobian (relative tolerance
            ``tol`` of the call that built the chart).
        condition_number: ``sigma_1 / sigma_rank`` (``inf`` if ``rank == 0``).
        variable_scale: The declared scaling ``s`` the basis already carries.

    Preconditions (``ValueError``): finite arrays of consistent shapes,
    positive ``variable_scale``, ``0 <= rank <= n``.
    """

    u_base: Array
    tangent_basis: Array
    singular_values: Array
    rank: int
    condition_number: float
    variable_scale: Array

    def __post_init__(self) -> None:
        u = readonly_float(self.u_base, "u_base")
        scale = readonly_float(self.variable_scale, "variable_scale")
        basis = readonly_float(self.tangent_basis, "tangent_basis")
        values = readonly_float(self.singular_values, "singular_values")
        if u.ndim != 1 or u.size == 0:
            raise ValueError("u_base must be a non-empty 1-D array")
        check_shape(scale, u.shape, "variable_scale")
        if np.any(scale <= 0):
            raise ValueError("variable_scale must be positive")
        if basis.ndim != 2 or basis.shape[0] != u.size:
            raise ValueError("tangent_basis must have shape (n, k)")
        if not 0 <= self.rank <= u.size:
            raise ValueError("rank must lie in [0, n]")
        object.__setattr__(self, "u_base", u)
        object.__setattr__(self, "variable_scale", scale)
        object.__setattr__(self, "tangent_basis", basis)
        object.__setattr__(self, "singular_values", values)
        object.__setattr__(self, "condition_number", float(self.condition_number))


@dataclass(frozen=True)
class ContinuationStep:
    """Result of one predictor-corrector step.

    Attributes:
        chart: Chart at the corrected point (``||F|| <= tol``, same rank).
        iterations: Number of corrector (Newton) updates performed.
        residual_history: ``||F||_2`` at the predicted point and after each
            update; ``len == iterations + 1``.
    """

    chart: ManifoldChart
    iterations: int
    residual_history: tuple[float, ...]


def _check_callable(fn: object, name: str) -> None:
    if not callable(fn):
        raise TypeError(f"{name} must be callable, got {type(fn).__name__}")


def _check_tol(tol: float) -> float:
    value = float(tol)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"tol must be finite and in (0, 1), got {tol!r}")
    return value


def _check_point(u: ArrayLike, name: str) -> Array:
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite non-empty 1-D array")
    return arr


def _check_scale(scale: ArrayLike, n: int) -> Array:
    arr = np.asarray(scale, dtype=np.float64)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"variable_scale must be positive and finite, shape ({n},)")
    return arr


def _evaluate(
    residual_fn: ResidualFn, jacobian_fn: JacobianFn, u: Array
) -> tuple[Array, Array]:
    """Residual (length ``m``) and Jacobian (``m x n``) at ``u``, validated."""
    f = np.asarray(residual_fn(u), dtype=np.float64)
    if f.ndim != 1 or f.size == 0 or not np.all(np.isfinite(f)):
        raise ValueError("residual_fn must return a finite non-empty 1-D array")
    jac = np.asarray(jacobian_fn(u), dtype=np.float64)
    if jac.shape != (f.size, u.size) or not np.all(np.isfinite(jac)):
        raise ValueError(
            f"jacobian_fn must return a finite array of shape {(f.size, u.size)}, "
            f"got {jac.shape}"
        )
    return f, jac


def _decompose(jac: Array, rhs: Array, scale: Array, tol: float) -> ForceNullSpace:
    """``J x = rhs`` on ``ForceNullSpace``: minimum scaled-norm ``particular``
    and the scaled-orthonormal kernel ``basis`` (the only SVD in this module)."""
    return ForceNullSpace.from_balance(
        jac,
        rhs,
        variable_scale=scale,
        rank_rtol=tol,
        balance_tolerance=max(tol, _MIN_BALANCE_TOL),
    )


def local_chart(
    residual_fn: ResidualFn,
    jacobian_fn: JacobianFn,
    u_base: ArrayLike,
    *,
    variable_scale: ArrayLike,
    tol: float,
) -> ManifoldChart:
    """Tangent chart of ``{F = 0}`` at the admitted point ``u_base``.

    The tangent basis is the ``ForceNullSpace`` kernel of ``J(u_base)`` under
    ``variable_scale`` (no direct ``pinv``/``null_space``); see the module
    docstring for the exact orthonormality statement.  ``tol`` is both the
    residual tolerance for admission and the relative rank tolerance.

    Preconditions (``TypeError`` / ``ValueError``): callable ``residual_fn``
    and ``jacobian_fn``; finite 1-D ``u_base``; positive finite
    ``variable_scale`` of the same length; ``0 < tol < 1``; finite ``F`` of
    length ``m`` and ``J`` of shape ``(m, n)``; ``||F(u_base)||_2 <= tol``
    (``u_base`` must already lie on the manifold).

    Postconditions: ``J N = 0`` and ``N^T S^-2 N = I`` to roundoff;
    ``rank + tangent_basis.shape[1] == n``.  A rank below the expected one
    (for example ``rank == 0`` at a crossing) is *reported*, not repaired; the
    chart is a tangent space only where the rank is locally constant.
    """
    _check_callable(residual_fn, "residual_fn")
    _check_callable(jacobian_fn, "jacobian_fn")
    tol = _check_tol(tol)
    u = _check_point(u_base, "u_base")
    scale = _check_scale(variable_scale, u.size)
    f, jac = _evaluate(residual_fn, jacobian_fn, u)
    if float(np.linalg.norm(f)) > tol:
        raise ValueError(
            "u_base must lie on the manifold: "
            f"||F(u_base)|| = {float(np.linalg.norm(f)):.3e} > tol = {tol:.3e}"
        )
    fns = _decompose(jac, np.zeros(f.size), scale, tol)
    report = fns.analyze()
    return ManifoldChart(
        u_base=u,
        tangent_basis=fns.basis,
        singular_values=report.singular_values,
        rank=fns.rank,
        condition_number=report.condition_number,
        variable_scale=scale,
    )


def _predict(chart: ManifoldChart, direction: ArrayLike, step: float) -> Array:
    """Tangent predictor ``u + step * N z / ||z||`` (``||.||`` in the scaled norm)."""
    d = np.asarray(direction, dtype=np.float64)
    n = chart.u_base.size
    if d.shape != (n,) or not np.all(np.isfinite(d)):
        raise ValueError(f"direction must be a finite array of shape ({n},)")
    scaled_basis = chart.tangent_basis / chart.variable_scale[:, None]
    scaled_dir = d / chart.variable_scale
    z = scaled_basis.T @ scaled_dir
    length = float(np.linalg.norm(z))
    if length <= _TANGENT_FLOOR * max(float(np.linalg.norm(scaled_dir)), 1e-300):
        raise ValueError("direction has no component in the tangent space of the chart")
    return chart.u_base + step * (chart.tangent_basis @ z) / length


def _correct(
    chart: ManifoldChart,
    start: Array,
    residual_fn: ResidualFn,
    jacobian_fn: JacobianFn,
    tol: float,
    max_iter: int,
) -> tuple[Array, tuple[float, ...]]:
    """Minimum-norm Newton iteration from ``start``; rank-gated at each iterate."""
    u = start
    f = np.asarray(residual_fn(u), dtype=np.float64)
    history = [float(np.linalg.norm(f))]
    for _ in range(max_iter):
        if history[-1] <= tol:
            break
        f, jac = _evaluate(residual_fn, jacobian_fn, u)
        try:
            fns = _decompose(jac, -f, chart.variable_scale, tol)
        except ValueError as exc:
            raise ManifoldSingularityError(
                f"corrector system is inconsistent (rank-deficient Jacobian): {exc}"
            ) from exc
        if fns.rank != chart.rank:
            raise ManifoldSingularityError(
                f"rank changed from {chart.rank} to {fns.rank} during correction"
            )
        u = u + fns.particular
        f = np.asarray(residual_fn(u), dtype=np.float64)
        history.append(float(np.linalg.norm(f)))
    if not history[-1] <= tol:
        raise ManifoldConvergenceError(
            f"corrector did not reach tol={tol:.3e} within max_iter={max_iter}; "
            f"last residual {history[-1]:.3e}"
        )
    return u, tuple(history)


def predictor_corrector_step(
    chart: ManifoldChart,
    direction: ArrayLike,
    step: float,
    residual_fn: ResidualFn,
    jacobian_fn: JacobianFn,
    *,
    tol: float,
    max_iter: int,
) -> ContinuationStep:
    """One tangent-predictor / minimum-norm-Newton-corrector step along ``F = 0``.

    ``direction`` (ambient, length ``n``) is projected onto the chart's tangent
    space in the scaled inner product; the predictor moves ``step`` along it,
    measured in the declared scaled norm ``||S^-1 .||``.  The projected
    direction fixes the orientation; the sign of the chart's basis does not.
    The tangent residual is ``O(step^2)``; the corrector converges locally and
    quadratically while the rank is constant.  The minimum-norm correction is
    not the nearest-point projection onto ``{F = 0}``.  Coverage of a
    connected component is not certified.

    Preconditions (``TypeError`` / ``ValueError``): ``chart`` a
    :class:`ManifoldChart`; callable ``residual_fn``/``jacobian_fn``; finite
    ``direction`` of length ``n`` with a non-zero tangent component; finite
    ``step > 0``; ``0 < tol < 1``; integer ``max_iter >= 1``.

    Raises:
        ManifoldSingularityError: The rank at any corrector iterate or at the
            new point differs from ``chart.rank``, the linearised system is
            inconsistent, or the new chart's condition number exceeds
            ``1 / tol``.
        ManifoldConvergenceError: ``||F|| > tol`` after ``max_iter`` updates.

    Postconditions: ``||F(chart_new.u_base)||_2 <= tol``; ``chart_new.rank ==
    chart.rank``; the new tangent basis is orthonormal in the declared scaling;
    ``len(residual_history) == iterations + 1``.
    """
    if not isinstance(chart, ManifoldChart):
        raise TypeError(f"chart must be a ManifoldChart, got {type(chart).__name__}")
    _check_callable(residual_fn, "residual_fn")
    _check_callable(jacobian_fn, "jacobian_fn")
    tol = _check_tol(tol)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter!r}")
    step_len = float(step)
    if not np.isfinite(step_len) or step_len <= 0.0:
        raise ValueError(f"step must be finite and > 0, got {step!r}")
    predicted = _predict(chart, direction, step_len)
    corrected, history = _correct(
        chart, predicted, residual_fn, jacobian_fn, tol, max_iter
    )
    new = local_chart(
        residual_fn,
        jacobian_fn,
        corrected,
        variable_scale=chart.variable_scale,
        tol=tol,
    )
    if new.rank != chart.rank:
        raise ManifoldSingularityError(
            f"rank changed from {chart.rank} to {new.rank} at the corrected point"
        )
    if new.rank > 0 and new.condition_number > 1.0 / tol:
        raise ManifoldSingularityError(
            f"corrected point is ill-conditioned (cond = {new.condition_number:.3e})"
        )
    return ContinuationStep(new, len(history) - 1, history)
