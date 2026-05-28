"""Recovery harness for the Pinocchio motion-matching oracle (issue #4121).

This module builds the *executable* TDD acceptance gate: it generates K
random ``theta`` vectors within the canonical bounds, synthesises a
:class:`ClubTarget` via :func:`synthesize_target_from_coefficients`,
hands the target to the optimiser ``fit_swing_pinocchio``, and asserts
the recovered coefficient vector matches the ground truth within
``||theta_recovered - theta_truth||_inf < 1e-3`` (per the parity spec).

The harness is import-time decoupled from the optimiser: the
``fit_swing`` callable is injected (defaulting to a lazy import of
``fit_swing_pinocchio``). Until ``PIN-FIT-DRIVER`` lands, callers can
pass a stub or simply skip the test that exercises the harness.

CLAUDE.md gotcha (echoed): never call ``pin.computeTotalEnergy``. The
upstream simulator handles energy correctly; this harness does not
touch energy at all.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.motion_matching.club_target import ClubTarget

from .simulate import COEFFS_PER_JOINT
from .synthesize import SynthesizeOptions, synthesize_target_from_coefficients

logger = logging.getLogger(__name__)

# Per-coefficient bounds matching ``generateRandomCoefficients.m`` (and
# the MATLAB Simscape oracle): |A|,|B| <= 1000; |C|,|D| <= 500;
# |E|,|F| <= 100; |G| <= 25. These cap the random sampling so the
# resulting trajectories stay inside the simulator's well-posed regime.
DEFAULT_THETA_BOUNDS: tuple[float, ...] = (
    1000.0,
    1000.0,
    500.0,
    500.0,
    100.0,
    100.0,
    25.0,
)

# Default recovery tolerance, per PINOCCHIO_PARITY_SPEC.md sec 5.3.
DEFAULT_RECOVERY_TOL: float = 1.0e-3

# Default sample count -- small enough to keep heavy_integration runs
# bounded, large enough to catch flake-y optimisers.
DEFAULT_NUM_SAMPLES: int = 5

# A ``fit_swing`` callable consumes a target + (optionally) opts and
# returns ``(theta_recovered, info_dict)``. We keep the contract loose
# so the harness can wrap any of the engine optimisers as they land.
FitSwingFn = Callable[
    [ClubTarget, dict[str, Any]],
    tuple[npt.NDArray[np.float64], dict[str, Any]],
]


@dataclass(frozen=True)
class RecoveryHarnessOptions:
    """Knobs for the recovery sweep.

    Attributes:
        num_samples: K -- how many random theta vectors to fit.
        n_joints: Number of actuated joints. ``None`` -> infer from the
            URDF the first time the simulator is invoked.
        seed: RNG seed for reproducibility. ``None`` -> non-deterministic
            (``np.random.default_rng()`` with OS entropy).
        bounds: Per-coefficient amplitude bounds (length 7).
        bounds_scale: Multiplier applied to ``bounds`` before sampling.
            Useful to shrink the test envelope so optimisers without
            warm starts can still recover.
        tolerance: Recovery tolerance on ``||theta_rec - theta_truth||_inf``.
        synthesize_options: Forwarded to
            :func:`synthesize_target_from_coefficients`.
        fit_swing_kwargs: Extra kwargs forwarded to ``fit_swing``.
    """

    num_samples: int = DEFAULT_NUM_SAMPLES
    n_joints: int | None = None
    seed: int | None = 0
    bounds: tuple[float, ...] = DEFAULT_THETA_BOUNDS
    bounds_scale: float = 0.1
    tolerance: float = DEFAULT_RECOVERY_TOL
    synthesize_options: SynthesizeOptions = field(default_factory=SynthesizeOptions)
    fit_swing_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            msg = f"num_samples must be > 0, got {self.num_samples!r}"
            raise ValueError(msg)
        if self.tolerance <= 0:
            msg = f"tolerance must be > 0, got {self.tolerance!r}"
            raise ValueError(msg)
        if len(self.bounds) != COEFFS_PER_JOINT:
            msg = (
                f"bounds must have length {COEFFS_PER_JOINT}; got "
                f"{len(self.bounds)} entries"
            )
            raise ValueError(msg)
        if not (self.bounds_scale > 0):
            msg = f"bounds_scale must be > 0, got {self.bounds_scale!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoveryTrial:
    """Per-sample diagnostics for a single recovery attempt."""

    index: int
    theta_truth: npt.NDArray[np.float64]
    theta_recovered: npt.NDArray[np.float64] | None
    residual_inf: float
    success: bool
    wallclock_s: float
    error: str | None


@dataclass(frozen=True)
class RecoverySummary:
    """Aggregate recovery statistics over a sweep."""

    num_samples: int
    num_success: int
    success_rate: float
    median_residual_inf: float
    max_residual_inf: float
    wallclock_min_s: float
    wallclock_median_s: float
    wallclock_max_s: float
    tolerance: float
    trials: tuple[RecoveryTrial, ...]


def _resolve_n_joints(opts: RecoveryHarnessOptions) -> int:
    """Resolve ``n_joints`` from explicit override or by loading the URDF."""
    if opts.n_joints is not None:
        if opts.n_joints <= 0:
            raise ValueError(f"n_joints must be > 0, got {opts.n_joints!r}")
        return int(opts.n_joints)
    # Defer to the simulator's lazy URDF cache so callers without
    # pinocchio fail at the simulate call rather than at import.
    from .simulate import (
        _get_cached_model,  # noqa: PLC0415  -- private helper, lives next door
        _resolve_urdf_path,  # noqa: PLC0415
    )

    sim_options = opts.synthesize_options.sim_options
    urdf_override = sim_options.urdf_path if sim_options is not None else None
    model = _get_cached_model(_resolve_urdf_path(urdf_override))
    return int(model.nv)


def sample_random_theta(
    n_joints: int,
    rng: np.random.Generator,
    bounds: tuple[float, ...] = DEFAULT_THETA_BOUNDS,
    scale: float = 0.1,
) -> npt.NDArray[np.float64]:
    """Draw a random ``theta`` vector inside ``scale * bounds``.

    Returns a flat ``float64`` array of shape ``(n_joints * 7,)`` with
    layout ``theta.reshape(n_joints, 7)[j, k] == a_{j, k}``.
    """
    if n_joints <= 0:
        raise ValueError(f"n_joints must be > 0, got {n_joints!r}")
    if len(bounds) != COEFFS_PER_JOINT:
        msg = f"bounds must have length {COEFFS_PER_JOINT}; got {len(bounds)}"
        raise ValueError(msg)
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale!r}")
    bounds_arr = np.asarray(bounds, dtype=np.float64) * float(scale)
    # Uniform in [-b, b] per coefficient, broadcast across joints.
    raw = rng.uniform(-1.0, 1.0, size=(n_joints, COEFFS_PER_JOINT))
    return (raw * bounds_arr[None, :]).reshape(-1)


def _default_fit_swing() -> FitSwingFn:
    """Lazily import the canonical ``fit_swing_pinocchio`` driver.

    Raises ``NotImplementedError`` until ``PIN-FIT-DRIVER`` lands; this
    keeps the harness usable as soon as the optimiser appears, with no
    further wiring on the test side.
    """
    try:
        from .fit_swing import (
            fit_swing_pinocchio,  # type: ignore[attr-defined]  # noqa: PLC0415
        )
    except ImportError as exc:  # pragma: no cover - exercised once optimiser exists
        msg = (
            "fit_swing_pinocchio is not yet implemented (issue PIN-FIT-DRIVER). "
            "Inject a custom ``fit_swing`` into ``run_recovery_sweep`` to "
            "exercise the harness against a stub or experimental optimiser."
        )
        raise NotImplementedError(msg) from exc
    return fit_swing_pinocchio  # type: ignore[return-value]


def run_recovery_sweep(
    options: RecoveryHarnessOptions | None = None,
    fit_swing: FitSwingFn | None = None,
) -> RecoverySummary:
    """Run K synthesis -> fit -> verify trials and return aggregate stats.

    Args:
        options: Harness options. ``None`` -> defaults (K=5, seed=0,
            10% of canonical bounds, 1e-3 tolerance).
        fit_swing: Optimiser callable. ``None`` -> lazy import of
            ``fit_swing_pinocchio`` (raises ``NotImplementedError``
            until PIN-FIT-DRIVER lands).

    Returns:
        :class:`RecoverySummary` with success rate, residual statistics,
        and per-trial diagnostics.
    """
    opts = options if options is not None else RecoveryHarnessOptions()
    fit_fn = fit_swing if fit_swing is not None else _default_fit_swing()

    rng = np.random.default_rng(opts.seed)
    n_joints = _resolve_n_joints(opts)
    trials: list[RecoveryTrial] = []

    for i in range(opts.num_samples):
        theta_truth = sample_random_theta(
            n_joints, rng, bounds=opts.bounds, scale=opts.bounds_scale
        )
        wall_start = time.perf_counter()
        try:
            target = synthesize_target_from_coefficients(
                theta_truth, options=opts.synthesize_options
            )
            theta_recovered, _info = fit_fn(target, dict(opts.fit_swing_kwargs))
            recovered_arr = np.asarray(theta_recovered, dtype=np.float64).reshape(-1)
            if recovered_arr.shape != theta_truth.shape:
                msg = (
                    f"fit_swing returned theta of shape {recovered_arr.shape}; "
                    f"expected {theta_truth.shape}"
                )
                raise ValueError(msg)
            residual_inf = float(np.max(np.abs(recovered_arr - theta_truth)))
            success = residual_inf < opts.tolerance
            err_msg: str | None = None
        except Exception as exc:  # noqa: BLE001  -- harness reports any failure
            recovered_arr = None
            residual_inf = float("inf")
            success = False
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Recovery trial %d failed: %s", i, err_msg)
        wall = time.perf_counter() - wall_start

        trials.append(
            RecoveryTrial(
                index=i,
                theta_truth=theta_truth,
                theta_recovered=recovered_arr,
                residual_inf=residual_inf,
                success=success,
                wallclock_s=wall,
                error=err_msg,
            )
        )

    residuals = [t.residual_inf for t in trials if np.isfinite(t.residual_inf)]
    walls = [t.wallclock_s for t in trials]
    n_success = sum(1 for t in trials if t.success)

    return RecoverySummary(
        num_samples=opts.num_samples,
        num_success=n_success,
        success_rate=n_success / opts.num_samples,
        median_residual_inf=(
            float(statistics.median(residuals)) if residuals else float("inf")
        ),
        max_residual_inf=float(max(residuals)) if residuals else float("inf"),
        wallclock_min_s=float(min(walls)) if walls else 0.0,
        wallclock_median_s=float(statistics.median(walls)) if walls else 0.0,
        wallclock_max_s=float(max(walls)) if walls else 0.0,
        tolerance=opts.tolerance,
        trials=tuple(trials),
    )


__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "DEFAULT_RECOVERY_TOL",
    "DEFAULT_THETA_BOUNDS",
    "FitSwingFn",
    "RecoveryHarnessOptions",
    "RecoverySummary",
    "RecoveryTrial",
    "run_recovery_sweep",
    "sample_random_theta",
]
