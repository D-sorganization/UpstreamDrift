"""Hybrid Option 2 -> Option 1 surrogate-then-polish handoff (#4000 / #031).

Implements ``fit_swing_hybrid``: run the trained NN surrogate inversion
(:func:`fit_swing_via_surrogate`, #029) to get a warm-start coefficient
vector, then hand that warm start to a polish solver - in the canonical
production path that polish solver is MATLAB's ``fit_swing_fmincon``
(issue #024) called via the matlab_bridge (#4006/4007).

Because the MATLAB bridge is a separate concern and not always available
in unit-test environments, the polish step is dependency-injected: the
caller passes any callable that accepts ``(target, theta_warm)`` and
returns a polish-result mapping. A sentinel ``"none"`` solver label can
also be used to skip the polish entirely.

Public API:
    HybridOptions    -- frozen dataclass of hybrid hyperparameters.
    HybridFitResult  -- return bundle with both phases recorded.
    fit_swing_hybrid -- main entry point.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.club_target import ClubTarget

from .surrogate import FitResult, InvertOptions, SwingSurrogate, fit_swing_via_surrogate

__all__ = [
    "HybridFitResult",
    "HybridOptions",
    "PolishCallable",
    "fit_swing_hybrid",
]

logger = get_logger(__name__)

PolishSolver = Literal["fmincon", "none"]
PolishCallable = Callable[[ClubTarget, np.ndarray], dict[str, Any]]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridOptions:
    """Hyperparameters for :func:`fit_swing_hybrid`.

    Attributes:
        invert: Surrogate-inversion options forwarded to
            :func:`fit_swing_via_surrogate`.
        polish_solver: ``"fmincon"`` (default) to call the polish callable,
            or ``"none"`` to return the surrogate warm start directly.
        skip_polish_tol: When the surrogate's reported ``final_loss`` is
            already at or below this value, the polish stage is skipped
            even if ``polish_solver == "fmincon"``. Default ``-inf`` =
            never skip.
    """

    invert: InvertOptions = field(default_factory=InvertOptions)
    polish_solver: PolishSolver = "fmincon"
    skip_polish_tol: float = float("-inf")

    def __post_init__(self) -> None:
        """Validate options at construction time (DbC)."""
        if self.polish_solver not in ("fmincon", "none"):
            raise ValueError(
                f"polish_solver must be 'fmincon' or 'none', got {self.polish_solver!r}"
            )
        if not isinstance(self.invert, InvertOptions):
            raise TypeError(
                f"invert must be an InvertOptions instance, got {type(self.invert)!r}"
            )
        if not (
            np.isfinite(self.skip_polish_tol) or self.skip_polish_tol == float("-inf")
        ):
            raise ValueError(
                f"skip_polish_tol must be finite or -inf, got {self.skip_polish_tol}"
            )


@dataclass
class HybridFitResult:
    """Return bundle for :func:`fit_swing_hybrid`.

    Attributes:
        coefficients: Final coefficient vector (warm start if polish was
            skipped, polished otherwise).
        final_loss: Loss of the final coefficients. When polish ran, this
            is the polish solver's reported ``final_rmse_m`` (squared if
            the solver returned an RMSE in meters); when polish was
            skipped it is the surrogate's ``final_loss``.
        solver: Always ``"surrogate+fmincon"`` when polish ran,
            ``"surrogate"`` when it was skipped.
        surrogate_phase: The full :class:`FitResult` from the surrogate
            inversion.
        polish_phase: The polish solver's returned mapping, or ``None``
            when polish was skipped.
        duration_s: Wall-clock duration of the entire hybrid call.
    """

    coefficients: np.ndarray
    final_loss: float
    solver: str
    surrogate_phase: FitResult
    polish_phase: dict[str, Any] | None
    duration_s: float

    @property
    def method(self) -> str:
        """Alias for ``solver`` matching the canonical FitResult schema."""
        return self.solver

    @property
    def theta_optimal(self) -> np.ndarray:
        """Alias for ``coefficients`` matching the canonical FitResult schema."""
        return self.coefficients


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_args(
    target: ClubTarget,
    surrogate: SwingSurrogate,
    *,
    options: HybridOptions | None = None,
    polish_fn: PolishCallable | None = None,
) -> bool:
    """Precondition predicate."""
    return (
        isinstance(target, ClubTarget)
        and isinstance(surrogate, SwingSurrogate)
        and (options is None or isinstance(options, HybridOptions))
    )


def _check_result(result: HybridFitResult) -> bool:
    """Postcondition: coefficients finite, loss finite & non-negative."""
    coeffs = np.asarray(result.coefficients)
    return bool(
        coeffs.ndim == 1
        and np.all(np.isfinite(coeffs))
        and np.isfinite(result.final_loss)
        and result.final_loss >= 0.0
        and result.solver in ("surrogate+fmincon", "surrogate")
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@precondition(_check_args, "target, surrogate, and options must have valid types")
@postcondition(_check_result, "result coefficients finite, loss >= 0")
def fit_swing_hybrid(
    target: ClubTarget,
    surrogate: SwingSurrogate,
    *,
    options: HybridOptions | None = None,
    polish_fn: PolishCallable | None = None,
) -> HybridFitResult:
    """Run surrogate inversion then polish.

    Args:
        target: Validated :class:`ClubTarget` to fit.
        surrogate: Trained :class:`SwingSurrogate` for the warm start.
        options: :class:`HybridOptions`. Defaults to
            ``HybridOptions()`` (fmincon polish, never skip).
        polish_fn: Callable invoked as ``polish_fn(target, theta_warm)``;
            must return a mapping that at minimum exposes
            ``"coefficients"`` (a 1-D ndarray) and ``"final_rmse_m"``
            (a non-negative float). Required when
            ``options.polish_solver == "fmincon"``.

    Returns:
        :class:`HybridFitResult` recording both phases.

    Raises:
        ValueError: If ``polish_solver == "fmincon"`` but no
            ``polish_fn`` is supplied, or if the polish callable returns
            a malformed mapping.
    """
    opts = options if options is not None else HybridOptions()

    t_start = _time.perf_counter()

    # ---- 1. Surrogate warm start ------------------------------------------
    surrogate_phase = fit_swing_via_surrogate(target, surrogate, opts.invert)
    theta_warm = np.asarray(surrogate_phase.coefficients, dtype=np.float64).reshape(-1)

    # ---- 2. Decide whether to polish --------------------------------------
    skip_for_tol = (
        np.isfinite(opts.skip_polish_tol)
        and surrogate_phase.final_loss <= opts.skip_polish_tol
    )
    will_polish = opts.polish_solver == "fmincon" and not skip_for_tol

    if not will_polish:
        logger.debug(
            "fit_swing_hybrid: polish skipped (solver=%s, skip_for_tol=%s)",
            opts.polish_solver,
            skip_for_tol,
        )
        return HybridFitResult(
            coefficients=theta_warm,
            final_loss=float(surrogate_phase.final_loss),
            solver="surrogate",
            surrogate_phase=surrogate_phase,
            polish_phase=None,
            duration_s=_time.perf_counter() - t_start,
        )

    # ---- 3. Polish --------------------------------------------------------
    if polish_fn is None:
        raise ValueError(
            "fit_swing_hybrid: polish_fn is required when "
            "options.polish_solver == 'fmincon'"
        )
    polish_out = polish_fn(target, theta_warm)
    if not isinstance(polish_out, dict) or "coefficients" not in polish_out:
        raise ValueError(
            "polish_fn must return a mapping with at least 'coefficients'; "
            f"got {type(polish_out).__name__}"
        )
    polished = np.asarray(polish_out["coefficients"], dtype=np.float64).reshape(-1)
    final_rmse = float(polish_out.get("final_rmse_m", float("nan")))
    final_loss = (
        final_rmse if np.isfinite(final_rmse) else float(surrogate_phase.final_loss)
    )

    logger.debug(
        "fit_swing_hybrid: surrogate_loss=%.6e -> polished_rmse_m=%.6e",
        surrogate_phase.final_loss,
        final_rmse,
    )

    return HybridFitResult(
        coefficients=polished,
        final_loss=final_loss,
        solver="surrogate+fmincon",
        surrogate_phase=surrogate_phase,
        polish_phase=polish_out,
        duration_s=_time.perf_counter() - t_start,
    )
