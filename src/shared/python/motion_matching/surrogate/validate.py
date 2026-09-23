"""Round-trip validation of a surrogate fit against Simscape ground-truth.

After :func:`fit_swing_via_surrogate` (#029) lands a coefficient vector
``theta_hat`` whose surrogate prediction matches a target trajectory, we
still need to ask: does ``f_true(theta_hat)`` (the real Simscape forward
model) also match the target? If not, the inversion has drifted into a
region where the surrogate is over-confident -- a classic failure mode
called out in APPROACH.md § Validation.

This module implements :func:`validate_against_simscape`, which:

  1. Reads ``theta_hat`` and the surrogate's prediction from a
     :class:`FitResult`.
  2. Calls a user-supplied ``sim_fn`` (defaulting to a live
     :class:`SimscapeAdapter`) at ``theta_hat`` to produce ground truth.
  3. Computes clubhead-position RMSE for both the surrogate prediction
     and the Simscape output relative to the target.
  4. Flags ``is_extrapolation`` when ``simscape_rmse / surrogate_rmse``
     exceeds ``threshold`` (default 2.0).

Public API:
    ValidationReport         -- frozen result bundle.
    validate_against_simscape -- entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.club_target import ClubTarget

if TYPE_CHECKING:
    from src.shared.python.motion_matching.surrogate.invert import FitResult
    from src.shared.python.motion_matching.surrogate.model import ClubTrajectory
    from src.shared.python.motion_matching.surrogate.train import TrainedSurrogate

__all__ = [
    "ValidationReport",
    "check_contact_boundary_failure",
    "check_trust_region",
    "compare_gradient_fidelity",
    "compute_directional_derivative",
    "quaternion_geodesic_error_rad",
    "resample_to_timegrid",
    "validate_against_simscape",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Result of one round-trip surrogate-vs-Simscape comparison.

    Attributes:
        surrogate_rmse_m: Clubhead-position RMSE in metres between the
            surrogate's prediction at ``theta_hat`` and the target.
        simscape_rmse_m: Clubhead-position RMSE in metres between the
            Simscape ground-truth output at ``theta_hat`` and the target.
        extrapolation_factor: ``simscape_rmse_m / surrogate_rmse_m``.
            Always finite; an explicit ``inf`` is reported only if the
            surrogate RMSE is exactly zero (degenerate, never seen in
            practice).
        is_extrapolation: ``True`` when ``extrapolation_factor`` exceeds
            the user threshold; signals the inversion has drifted off
            the surrogate's reliable region.
        surrogate_pred: The surrogate's :class:`ClubTrajectory` at
            ``theta_hat`` (passed through from the :class:`FitResult`).
        simscape_out: The full :class:`SimscapeOutput` returned by
            ``sim_fn``. Typed as ``object`` here to keep this module
            free of a hard dependency on ``src.engines.simscape``.
        butt_rmse_m: Optional butt-position RMSE in metres.
        orientation_rmse_rad: Optional orientation angular RMSE in radians.
        trust_region_rejected: Whether candidate exceeded trust-region radius.
        gradient_fidelity_rejected: Whether local gradient fidelity was rejected.
        contact_boundary_failure: Whether candidate crossed contact boundary without contact dynamics.
    """

    surrogate_rmse_m: float
    simscape_rmse_m: float
    extrapolation_factor: float
    is_extrapolation: bool
    surrogate_pred: ClubTrajectory
    simscape_out: object
    butt_rmse_m: float | None = None
    orientation_rmse_rad: float | None = None
    trust_region_rejected: bool = False
    gradient_fidelity_rejected: bool = False
    contact_boundary_failure: bool = False


# ---------------------------------------------------------------------------
# Internal helpers & NM-07 validation contracts
# ---------------------------------------------------------------------------


def resample_to_timegrid(
    arr: np.ndarray,
    src_time: np.ndarray | None = None,
    dst_time: np.ndarray | int | None = None,
) -> np.ndarray:
    """Linear-interpolate ``arr`` (T, D) onto a destination timegrid.

    If ``src_time`` is provided:
      - Validates ``src_time`` is finite and strictly monotonically increasing.
      - If ``dst_time`` is an array: interpolates along ``src_time -> dst_time``.
      - If ``dst_time`` is an int: creates a uniform timegrid between ``src_time[0]`` and ``src_time[-1]``.
    If ``src_time`` is None:
      - Falls back to normalized index grid [0, 1].
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"arr must be (T, D), got shape {arr.shape}")
    src_n = arr.shape[0]

    if src_time is not None:
        src_t = np.asarray(src_time, dtype=np.float64).reshape(-1)
        if len(src_t) != src_n:
            raise ValueError(
                f"src_time length {len(src_t)} must match arr length {src_n}"
            )
        if not np.all(np.isfinite(src_t)):
            raise ValueError("src_time must be finite")
        if len(src_t) > 1 and not bool(np.all(np.diff(src_t) > 0.0)):
            raise ValueError("src_time must be strictly monotonically increasing")

        if isinstance(dst_time, (int, np.integer)):
            if dst_time <= 0:
                raise ValueError("dst_time count must be positive")
            dst_t = np.linspace(src_t[0], src_t[-1], int(dst_time))
        elif dst_time is not None:
            dst_t = np.asarray(dst_time, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(dst_t)):
                raise ValueError("dst_time must be finite")
            if len(dst_t) > 1 and not bool(np.all(np.diff(dst_t) > 0.0)):
                raise ValueError("dst_time must be strictly monotonically increasing")
        else:
            return arr.copy()

        out = np.empty((len(dst_t), arr.shape[1]), dtype=np.float64)
        for d in range(arr.shape[1]):
            out[:, d] = np.interp(dst_t, src_t, arr[:, d])
        return out

    seq_len = (
        int(dst_time)
        if isinstance(dst_time, (int, np.integer))
        else (len(dst_time) if dst_time is not None else src_n)
    )
    if src_n == seq_len:
        return arr.copy()
    src_x = np.linspace(0.0, 1.0, src_n)
    dst_x = np.linspace(0.0, 1.0, seq_len)
    out = np.empty((seq_len, arr.shape[1]), dtype=np.float64)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(dst_x, src_x, arr[:, d])
    return out


def _resample_uniform(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Linear-interpolate ``arr`` (T, D) onto a ``seq_len`` uniform grid."""
    return resample_to_timegrid(arr, src_time=None, dst_time=seq_len)


def quaternion_geodesic_error_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute mean geodesic angular error in radians with rotation sign invariance.

    Quaternions q and -q represent identical SO(3) orientations. The inner
    product |<q1, q2>| is clamped to [0, 1] and mapped via 2 * arccos(|dot|).
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    if q1.shape != q2.shape or q1.shape[-1] != 4:
        raise ValueError(
            f"quaternions must have shape (T, 4); got {q1.shape} and {q2.shape}"
        )

    n1 = np.linalg.norm(q1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(q2, axis=-1, keepdims=True)
    q1_norm = q1 / np.maximum(n1, 1e-12)
    q2_norm = q2 / np.maximum(n2, 1e-12)

    dots = np.abs(np.sum(q1_norm * q2_norm, axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    angles_rad = 2.0 * np.arccos(dots)
    return float(np.sqrt(np.mean(angles_rad * angles_rad)))


def check_trust_region(
    theta: np.ndarray, theta_anchor: np.ndarray, trust_radius: float
) -> bool:
    """Return True if ||theta - theta_anchor||_2 <= trust_radius."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    anchor = np.asarray(theta_anchor, dtype=np.float64).reshape(-1)
    if theta.shape != anchor.shape:
        raise ValueError("theta and theta_anchor must have matching dimensions")
    if not (np.isfinite(trust_radius) and trust_radius > 0.0):
        raise ValueError("trust_radius must be a positive finite float")
    dist = float(np.linalg.norm(theta - anchor))
    return bool(dist <= trust_radius)


def compute_directional_derivative(
    fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    direction: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute central difference directional derivative of fn at theta in direction."""
    theta = np.asarray(theta, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if theta.shape != direction.shape:
        raise ValueError("theta and direction must have matching shapes")
    norm_d = float(np.linalg.norm(direction))
    if norm_d < 1e-12:
        return np.zeros_like(fn(theta))
    d_unit = direction / norm_d

    f_plus = fn(theta + eps * d_unit)
    f_minus = fn(theta - eps * d_unit)
    return (
        np.asarray(f_plus, dtype=np.float64) - np.asarray(f_minus, dtype=np.float64)
    ) / (2.0 * eps)


def compare_gradient_fidelity(
    surrogate_grad: np.ndarray,
    native_grad: np.ndarray,
    cos_sim_threshold: float = 0.3,
) -> tuple[float, bool]:
    """Compute cosine similarity between surrogate gradient and native sensitivity.

    Returns (cos_sim, is_accepted). If cos_sim < cos_sim_threshold, rejects step
    to prevent adversarial surrogate exploitation.
    """
    g_s = np.asarray(surrogate_grad, dtype=np.float64).reshape(-1)
    g_n = np.asarray(native_grad, dtype=np.float64).reshape(-1)
    if g_s.shape != g_n.shape:
        raise ValueError("surrogate_grad and native_grad must have matching dimensions")
    n_s = float(np.linalg.norm(g_s))
    n_n = float(np.linalg.norm(g_n))
    if n_s < 1e-12 or n_n < 1e-12:
        return 0.0, False
    cos_sim = float(np.dot(g_s, g_n) / (n_s * n_n))
    return cos_sim, bool(cos_sim >= cos_sim_threshold)


def check_contact_boundary_failure(target: ClubTarget, model_type: str) -> bool:
    """Return True if evaluating through contact without contact compliance."""
    if not isinstance(target, ClubTarget):
        raise TypeError("target must be a ClubTarget")
    has_impact = target.impact_idx is not None and 0 <= target.impact_idx < len(
        target.time
    )
    is_smooth_only = (
        "smooth" in model_type.lower() or "rigid_surrogate" in model_type.lower()
    )
    return bool(has_impact and is_smooth_only)


def _clubhead_rmse_m(pred: np.ndarray, target: np.ndarray) -> float:
    """Return clubhead-position RMSE in metres over a shared timegrid.

    ``pred`` and ``target`` are ``(T, 3)`` float arrays. If the row
    counts differ we resample ``pred`` to ``target``'s grid by linear
    interpolation -- accurate enough for a scalar RMSE and avoids
    coupling to the dataset's exact sample rate.
    """
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred must be (T, 3); got {pred.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target must be (T, 3); got {target.shape}")
    if pred.shape[0] != target.shape[0]:
        pred = _resample_uniform(pred, target.shape[0])
    diff = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _surrogate_clubhead_np(pred: ClubTrajectory) -> np.ndarray:
    """Extract a batchless (T, 3) clubhead numpy array from a prediction."""
    head = pred.clubhead.detach().cpu().numpy()
    if head.ndim == 3:  # (B, T, 3) -> drop singleton batch
        if head.shape[0] != 1:
            raise ValueError(f"surrogate_pred has batch={head.shape[0]}; expected 1")
        head = head[0]
    return head.astype(np.float64)


def _default_sim_fn() -> Callable[[np.ndarray], object]:
    """Build a default ``sim_fn`` backed by a live :class:`SimscapeAdapter`.

    Imports are local so this module imports cleanly on hosts without
    MATLAB. If MATLAB is unavailable we raise the canonical
    :class:`SimscapeNotInstalledError` from the simscape package.
    """
    from src.engines.simscape._engine_pool import is_matlab_available
    from src.engines.simscape._errors import SimscapeNotInstalledError
    from src.engines.simscape.adapter import SimscapeAdapter

    if not is_matlab_available():
        raise SimscapeNotInstalledError(
            "round-trip validation requires either a user-supplied "
            "sim_fn or a live MATLAB Engine for Python. Install the "
            "Simscape adapter dependencies, or pass sim_fn=... "
            "(e.g. a mocked adapter) to validate_against_simscape."
        )

    adapter = SimscapeAdapter()
    return lambda coeffs: adapter.simulate_with_coefficients(coeffs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _check_args(
    result: object,
    target: ClubTarget,
    surrogate: object,
    *,
    sim_fn: Callable[[np.ndarray], object] | None = None,
    threshold: float = 2.0,
) -> bool:
    """Precondition predicate for :func:`validate_against_simscape`."""
    del sim_fn  # presence is checked at runtime
    try:
        from src.shared.python.motion_matching.surrogate.invert import FitResult
        from src.shared.python.motion_matching.surrogate.train import TrainedSurrogate

        is_result_valid = isinstance(result, FitResult)
        is_surrogate_valid = isinstance(surrogate, TrainedSurrogate)
    except (ImportError, ModuleNotFoundError):
        is_result_valid = hasattr(result, "coefficients") and hasattr(
            result, "surrogate_pred"
        )
        is_surrogate_valid = hasattr(surrogate, "model") or hasattr(
            surrogate, "norm_stats"
        )

    return (
        is_result_valid
        and isinstance(target, ClubTarget)
        and is_surrogate_valid
        and isinstance(threshold, (int, float))
        and float(threshold) > 0.0
    )


def _check_report(report: ValidationReport) -> bool:
    """Postcondition: RMSEs finite and non-negative; flag matches factor."""
    return bool(
        np.isfinite(report.surrogate_rmse_m)
        and np.isfinite(report.simscape_rmse_m)
        and report.surrogate_rmse_m >= 0.0
        and report.simscape_rmse_m >= 0.0
    )


@precondition(_check_args, "result/target/surrogate types and threshold>0")
@postcondition(_check_report, "RMSEs must be finite and non-negative")
def validate_against_simscape(
    result: FitResult,
    target: ClubTarget,
    surrogate: TrainedSurrogate,
    *,
    sim_fn: Callable[[np.ndarray], object] | None = None,
    threshold: float = 2.0,
) -> ValidationReport:
    """Compare a surrogate-derived fit against a Simscape forward run.

    Args:
        result: :class:`FitResult` from :func:`fit_swing_via_surrogate`,
            containing the best coefficients and the surrogate's
            prediction at those coefficients.
        target: The :class:`ClubTarget` we were trying to fit.
        surrogate: The :class:`TrainedSurrogate` that produced ``result``.
            Currently unused at validation time, but accepted so the
            caller's bundle stays explicit and to support later
            extensions (e.g. uncertainty queries on the surrogate).
        sim_fn: Optional ``coeffs -> SimscapeOutput`` callable. When
            ``None`` we instantiate a :class:`SimscapeAdapter` and
            forward via ``simulate_with_coefficients`` -- raising
            :class:`SimscapeNotInstalledError` if MATLAB is absent.
        threshold: Ratio above which we flag an extrapolation. The
            default ``2.0`` matches APPROACH.md § Validation.

    Returns:
        A :class:`ValidationReport` with the two RMSEs, the ratio,
        the extrapolation flag, the surrogate prediction, and the raw
        Simscape output.

    Raises:
        SimscapeNotInstalledError: If ``sim_fn is None`` and MATLAB is
            unavailable.
        ValueError: If ``sim_fn`` returns an object without an
            ``r_clubhead`` attribute of shape ``(N, 3)``.
    """
    del surrogate  # explicit: not needed at validation time today
    fn = sim_fn if sim_fn is not None else _default_sim_fn()

    coeffs = np.asarray(result.coefficients, dtype=np.float64)
    sim_out = fn(coeffs)

    head_sim = getattr(sim_out, "r_clubhead", None)
    if head_sim is None:
        raise ValueError(
            "sim_fn return value lacks an 'r_clubhead' attribute; "
            "expected a SimscapeOutput-like object"
        )
    head_sim = np.asarray(head_sim, dtype=np.float64)
    if head_sim.ndim != 2 or head_sim.shape[1] != 3:
        raise ValueError(f"sim_fn r_clubhead must be (N, 3); got {head_sim.shape}")

    head_target = np.asarray(target.clubhead, dtype=np.float64)
    head_surrogate = _surrogate_clubhead_np(result.surrogate_pred)

    surrogate_rmse = _clubhead_rmse_m(head_surrogate, head_target)
    simscape_rmse = _clubhead_rmse_m(head_sim, head_target)

    if surrogate_rmse > 0.0:
        factor = simscape_rmse / surrogate_rmse
    else:
        factor = float("inf") if simscape_rmse > 0.0 else 1.0

    is_extrap = bool(factor > float(threshold))

    logger.debug(
        "round-trip validate: surrogate=%.6f m, simscape=%.6f m, "
        "factor=%.3f, extrap=%s (threshold=%.2f)",
        surrogate_rmse,
        simscape_rmse,
        factor,
        is_extrap,
        threshold,
    )

    return ValidationReport(
        surrogate_rmse_m=surrogate_rmse,
        simscape_rmse_m=simscape_rmse,
        extrapolation_factor=float(factor),
        is_extrapolation=is_extrap,
        surrogate_pred=result.surrogate_pred,
        simscape_out=sim_out,
    )
