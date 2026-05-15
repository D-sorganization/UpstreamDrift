"""Adam-on-coefficients inversion via the trained ``SwingSurrogate`` (#029).

Implements ``fit_swing_via_surrogate`` per APPROACH.md § Inversion: K random
restarts of Adam on the input coefficient vector, with hard ``clamp_`` bound
projection after every step. The forward pass is fully differentiable (#028)
so autograd flows from a weighted-MSE trajectory loss back to the coefficients.

Public API:
    InvertOptions             -- frozen dataclass of optimizer hyperparameters.
    FitResult                 -- per-fit bundle (coefficients, loss, history).
    fit_swing_via_surrogate   -- main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Callable

import numpy as np
import torch

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.club_target import ClubTarget

from ._bounds import clamp_, default_bounds, validate_bounds
from ._normalize import NormalizationStats, zscore_coeffs
from .model import ClubTrajectory, SwingSurrogate

__all__ = ["FitResult", "InvertOptions", "fit_swing_via_surrogate"]

logger = get_logger(__name__)

BoundStrategy = Literal["clamp", "penalty"]
ScheduleName = Literal["constant", "cosine"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvertOptions:
    """Hyperparameters for :func:`fit_swing_via_surrogate`.

    Attributes:
        n_starts: Number of random restarts (parallelised across the batch).
        n_iters_per_start: Adam iterations per restart.
        lr: Initial Adam learning rate on the coefficient vector.
        schedule: ``"constant"`` or ``"cosine"`` learning-rate schedule.
        seed: RNG seed for reproducibility of the random restarts.
        bound_strategy: ``"clamp"`` for hard projection (default, per
            APPROACH.md), or ``"penalty"`` for a quadratic out-of-bounds
            penalty (kept for ablation).
        w_butt: Weight on butt-position MSE.
        w_clubhead: Weight on clubhead-position MSE.
        w_quat: Weight on the sign-invariant quaternion loss.
        penalty_lambda: Strength of the soft bound penalty
            (only used when ``bound_strategy == "penalty"``).
    """

    n_starts: int = 8
    n_iters_per_start: int = 200
    lr: float = 5.0e-2
    schedule: ScheduleName = "cosine"
    seed: int = 0
    bound_strategy: BoundStrategy = "clamp"
    w_butt: float = 1.0
    w_clubhead: float = 1.0
    w_quat: float = 0.1
    penalty_lambda: float = 1.0e2

    def __post_init__(self) -> None:
        """Validate options at construction time (DbC)."""
        if self.n_starts < 1:
            raise ValueError(f"n_starts must be >= 1, got {self.n_starts}")
        if self.n_iters_per_start < 1:
            raise ValueError(
                f"n_iters_per_start must be >= 1, got {self.n_iters_per_start}"
            )
        if not (self.lr > 0 and np.isfinite(self.lr)):
            raise ValueError(f"lr must be a positive finite float, got {self.lr}")
        if self.schedule not in ("constant", "cosine"):
            raise ValueError(
                f"schedule must be 'constant' or 'cosine', got {self.schedule!r}"
            )
        if self.bound_strategy not in ("clamp", "penalty"):
            raise ValueError(
                "bound_strategy must be 'clamp' or 'penalty', "
                f"got {self.bound_strategy!r}"
            )


@dataclass
class FitResult:
    """Return bundle for :func:`fit_swing_via_surrogate`.

    Attributes:
        coefficients: ``(coeff_dim,)`` best coefficient vector across all
            restarts (``np.float32``).
        final_loss: Loss at the best restart's last iteration.
        history: ``{"loss": (n_starts, n_iters_per_start)}`` per-iteration
            loss trajectory; rows are restarts.
        all_starts: List of length ``n_starts`` with the initial coefficient
            vector used by each restart.
        surrogate_pred: The surrogate's :class:`ClubTrajectory` prediction
            at the best coefficients (batchless, batch-1 along dim 0).
    """

    coefficients: np.ndarray
    final_loss: float
    history: dict[str, np.ndarray]
    all_starts: list[np.ndarray]
    surrogate_pred: ClubTrajectory = field(repr=False)

    @property
    def theta_optimal(self) -> np.ndarray:
        """Alias for ``coefficients`` matching the canonical FitResult schema."""
        return self.coefficients


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _target_to_tensors(
    target: ClubTarget,
    surrogate: SwingSurrogate,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ``ClubTarget`` arrays into batchless torch tensors on device.

    Resamples by simple slicing/padding to the surrogate's ``seq_len``: in
    practice the loaders already align to the surrogate's grid (1 kHz, 0.3 s)
    so this is a no-op except in tests.
    """
    seq_len = surrogate.cfg.seq_len
    butt_np = np.asarray(target.butt, dtype=np.float32)
    head_np = np.asarray(target.clubhead, dtype=np.float32)
    quat_np = np.asarray(target.club_quat, dtype=np.float32)
    if butt_np.shape[0] != seq_len:
        butt_np = _resample_uniform(butt_np, seq_len)
        head_np = _resample_uniform(head_np, seq_len)
        quat_np = _resample_uniform(quat_np, seq_len)
        norms = np.sqrt(np.einsum("...i,...i->...", quat_np, quat_np))[..., np.newaxis]
        quat_np = quat_np / np.maximum(norms, 1.0e-8)
    device = next(surrogate.parameters()).device
    butt_t = torch.from_numpy(butt_np).to(device).unsqueeze(0)
    head_t = torch.from_numpy(head_np).to(device).unsqueeze(0)
    quat_t = torch.from_numpy(quat_np).to(device).unsqueeze(0)
    return butt_t, head_t, quat_t


def _resample_uniform(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Linear-interpolate ``arr`` (T, D) onto a ``seq_len`` uniform grid."""
    src_n = arr.shape[0]
    if src_n == seq_len:
        return arr.astype(np.float32)
    src_x = np.linspace(0.0, 1.0, src_n)
    dst_x = np.linspace(0.0, 1.0, seq_len)
    out = np.empty((seq_len, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(dst_x, src_x, arr[:, d])
    return out


def _sample_initial_coeffs(
    n_starts: int,
    coeff_dim: int,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Uniformly sample ``n_starts`` initial coefficient vectors in-bounds."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(
        low=bounds_low,
        high=bounds_high,
        size=(n_starts, coeff_dim),
    )
    return raw.astype(np.float32)


def _trajectory_loss(
    pred: ClubTrajectory,
    butt: torch.Tensor,
    head: torch.Tensor,
    quat: torch.Tensor,
    opts: InvertOptions,
) -> torch.Tensor:
    """Per-restart weighted MSE + sign-invariant quat loss.

    Returns a ``(B,)`` vector of losses, one per restart in the batch, so
    that we can backprop independently for each restart in a single call.
    """
    db = pred.butt - butt
    dh = pred.clubhead - head
    pos_butt = (db * db).mean(dim=(1, 2))
    pos_head = (dh * dh).mean(dim=(1, 2))
    # Sign-invariant quaternion loss: 1 - <q_pred, q_target>^2, mean over T.
    dot = (pred.club_quat * quat).sum(dim=-1)
    quat_term = (1.0 - dot * dot).mean(dim=1)
    return opts.w_butt * pos_butt + opts.w_clubhead * pos_head + opts.w_quat * quat_term


def _bound_penalty(
    coeffs: torch.Tensor,
    bounds_low: torch.Tensor,
    bounds_high: torch.Tensor,
) -> torch.Tensor:
    """``λ · sum(max(0, x - hi)^2 + max(0, lo - x)^2)`` per restart."""
    over = torch.clamp(coeffs - bounds_high, min=0.0)
    under = torch.clamp(bounds_low - coeffs, min=0.0)
    return (over * over).sum(dim=-1) + (under * under).sum(dim=-1)


def _make_lr_schedule(opts: InvertOptions) -> Callable[[int], float]:
    """Return ``it -> lr`` for the configured schedule."""
    base = opts.lr
    n = opts.n_iters_per_start
    if opts.schedule == "constant":
        return lambda _it: base
    # Cosine annealing from base to 0 over n iterations.
    return lambda it: (
        0.5 * base * (1.0 + np.cos(np.pi * min(it, n - 1) / max(n - 1, 1)))
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _check_args(
    target: ClubTarget,
    surrogate: SwingSurrogate,
    opts: InvertOptions,
    norm_stats: NormalizationStats | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Precondition predicate for :func:`fit_swing_via_surrogate`."""
    return (
        isinstance(target, ClubTarget)
        and isinstance(surrogate, SwingSurrogate)
        and isinstance(opts, InvertOptions)
    )


def _check_result(result: FitResult) -> bool:
    """Postcondition: best coefficients finite, loss finite & non-negative."""
    coeffs = np.asarray(result.coefficients)
    return bool(
        coeffs.ndim == 1
        and np.all(np.isfinite(coeffs))
        and np.isfinite(result.final_loss)
        and result.final_loss >= 0.0
    )


@precondition(_check_args, "target/surrogate/opts must be the right types")
@postcondition(_check_result, "best coefficients must be finite, loss >= 0")
def fit_swing_via_surrogate(
    target: ClubTarget,
    surrogate: SwingSurrogate,
    opts: InvertOptions = InvertOptions(),
    *,
    norm_stats: NormalizationStats | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> FitResult:
    """Fit ``target`` by Adam-on-coefficients through the trained surrogate.

    Runs ``opts.n_starts`` random restarts in parallel along the batch
    dimension. After every step coefficients are projected back onto the
    bounds (when ``bound_strategy == "clamp"``) per APPROACH.md.

    Args:
        target: Validated :class:`ClubTarget` to fit.
        surrogate: Trained :class:`SwingSurrogate` (#028).
        opts: Inversion hyperparameters.
        norm_stats: Optional :class:`NormalizationStats` used to z-score
            coefficients before passing them through the surrogate. If the
            surrogate was trained with normalization (it is, by default),
            you must pass the stats from the :class:`TrainedSurrogate`
            bundle. The Adam optimization runs in *raw* coefficient space
            so the bounds remain physically interpretable.
        bounds: Optional ``(bounds_low, bounds_high)`` per-dimension bound
            vectors of shape ``(coeff_dim,)``. Defaults to ``[-3, 3]``.

    Returns:
        A :class:`FitResult` with the best restart's coefficients and the
        per-restart loss history.

    Raises:
        ValueError: If bounds shapes are wrong or any input is non-finite.
    """
    coeff_dim = surrogate.cfg.coeff_dim
    low_np, high_np = bounds if bounds is not None else default_bounds(coeff_dim)
    validate_bounds(low_np, high_np, coeff_dim)

    surrogate.eval()
    device = next(surrogate.parameters()).device
    bounds_low_t = torch.as_tensor(low_np, dtype=torch.float32, device=device)
    bounds_high_t = torch.as_tensor(high_np, dtype=torch.float32, device=device)

    butt_t, head_t, quat_t = _target_to_tensors(target, surrogate)
    butt_b = butt_t.expand(opts.n_starts, -1, -1).contiguous()
    head_b = head_t.expand(opts.n_starts, -1, -1).contiguous()
    quat_b = quat_t.expand(opts.n_starts, -1, -1).contiguous()

    init = _sample_initial_coeffs(opts.n_starts, coeff_dim, low_np, high_np, opts.seed)
    coeffs = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([coeffs], lr=opts.lr)
    schedule = _make_lr_schedule(opts)

    history = np.zeros((opts.n_starts, opts.n_iters_per_start), dtype=np.float32)

    for it in range(opts.n_iters_per_start):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = schedule(it)
        fed = zscore_coeffs(coeffs, norm_stats) if norm_stats is not None else coeffs
        pred = surrogate(fed)
        per_restart_loss = _trajectory_loss(pred, butt_b, head_b, quat_b, opts)
        if opts.bound_strategy == "penalty":
            per_restart_loss = per_restart_loss + opts.penalty_lambda * _bound_penalty(
                coeffs, bounds_low_t, bounds_high_t
            )
        # Sum so each restart's gradient flows independently (Adam is per-element).
        per_restart_loss.sum().backward()
        optimizer.step()
        if opts.bound_strategy == "clamp":
            clamp_(coeffs, bounds_low_t, bounds_high_t)
        history[:, it] = per_restart_loss.detach().cpu().numpy()

    final_per_restart = history[:, -1]
    best = int(np.argmin(final_per_restart))
    best_coeffs_t = coeffs.detach()[best : best + 1]
    with torch.no_grad():
        fed_best = (
            zscore_coeffs(best_coeffs_t, norm_stats)
            if norm_stats is not None
            else best_coeffs_t
        )
        surrogate_pred = surrogate(fed_best)

    logger.debug(
        "fit_swing_via_surrogate: best restart %d, loss %.6e (over %d starts)",
        best,
        float(final_per_restart[best]),
        opts.n_starts,
    )

    return FitResult(
        coefficients=best_coeffs_t.cpu().numpy().reshape(-1).astype(np.float32),
        final_loss=float(final_per_restart[best]),
        history={"loss": history},
        all_starts=[init[k].copy() for k in range(opts.n_starts)],
        surrogate_pred=surrogate_pred,
    )
