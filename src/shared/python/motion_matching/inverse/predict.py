"""CVAE inference with rejection sampling (issue #4003 / #034).

Public surface
--------------
* :func:`predict_coefficients` - sample N candidates, optionally validate via
  a round-trip forward model, return the best one (lowest round-trip RMSE).
* :class:`InverseFitResult` - frozen result bundle.
* :class:`TrainedInverseCVAE` - lightweight wrapper bundling the
  :class:`SwingInverseCVAE` model with the kinematics tensor it conditions on.
  Once #033 lands its own ``TrainedInverseCVAE`` (with norm-stats and
  metadata) we will swap this class for that one; the duck-typed
  ``model`` / ``kinematics`` attributes preserve the call-site contract.

Design notes (per APPROACH.md / INTERFACES.md §Inference)
---------------------------------------------------------
* Sampling and the kinematic encode happen on the model's device; the
  resulting samples are pulled to CPU/numpy before the round-trip forward
  model runs because ``SimscapeAdapter`` is numpy-only and we want a uniform
  surface across forward backends.
* ``forward_fn`` defaults to a wrapped :class:`SwingSurrogate`; callers can
  override with a Simscape-backed adapter when ``requires_matlab=True`` or
  inject a stub for testing.
* When *no* sample passes ``rmse_threshold_m`` we still return the best
  (least-bad) one. This keeps the API total: callers that want hard
  rejection should inspect ``len(result.accepted_samples) == 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import ClubTarget

from ._validate import (
    ForwardFn,
    RoundTripOutput,
    ValidationReport,
    round_trip_validate,
)
from .cvae import EncoderOutput, SwingInverseCVAE

__all__ = [
    "ForwardFn",
    "InverseFitResult",
    "RoundTripOutput",
    "TrainedInverseCVAE",
    "ValidationReport",
    "predict_coefficients",
]

DEFAULT_N_SAMPLES = 32
DEFAULT_RMSE_THRESHOLD_M = 0.005


class _ForwardSurrogate(Protocol):
    """Duck type covering :class:`SwingSurrogate`-like callables."""

    def __call__(self, coeffs: torch.Tensor) -> object: ...


@dataclass(frozen=True)
class TrainedInverseCVAE:
    """Bundle of model + the kinematics tensor it conditions on.

    Once issue #033 (``train_inverse_cvae``) lands and emits its own
    ``TrainedInverseCVAE`` dataclass with norm-stats / training metadata,
    we'll re-export that one. Until then the inference surface only needs
    the two attributes below; both #033 and the test stubs satisfy this
    minimal contract.

    Attributes:
        model: The trained :class:`SwingInverseCVAE`.
        kinematics: ``(1, T, n_kinematic_channels)`` conditioning tensor.
    """

    model: SwingInverseCVAE
    kinematics: torch.Tensor


@dataclass(frozen=True)
class InverseFitResult:
    """Result of a rejection-sampling inference call.

    Attributes:
        best_coefficients: Best sample under the round-trip RMSE metric
            (or the only sample drawn when ``validate=False``).
        accepted_samples: List of samples that passed
            ``rmse_threshold_m``. Empty if no sample passed.
        accepted_costs: Round-trip RMSEs for the accepted samples.
        rejected_count: Number of samples drawn but rejected.
        sampling_budget_used: Total number of samples drawn from the CVAE.
        encoder_output: The posterior parameters from the encoder pass.
    """

    best_coefficients: NDArray[np.float64]
    accepted_samples: list[NDArray[np.float64]]
    accepted_costs: list[float]
    rejected_count: int
    sampling_budget_used: int
    encoder_output: EncoderOutput


def _validate_args(
    target: ClubTarget,
    model: TrainedInverseCVAE,
    n_samples: int,
    rmse_threshold_m: float,
) -> None:
    """Eager precondition checks (DbC)."""
    if not isinstance(target, ClubTarget):
        raise TypeError(f"target must be a ClubTarget; got {type(target).__name__}")
    if not isinstance(model, TrainedInverseCVAE):
        raise TypeError(
            f"model must be a TrainedInverseCVAE bundle; got {type(model).__name__}"
        )
    if not isinstance(model.model, SwingInverseCVAE):
        raise TypeError(
            "model.model must be a SwingInverseCVAE instance; "
            f"got {type(model.model).__name__}"
        )
    if model.kinematics.dim() != 3 or model.kinematics.shape[0] != 1:
        raise ValueError(
            "model.kinematics must have shape (1, T, n_kinematic_channels); "
            f"got {tuple(model.kinematics.shape)}"
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    if not (rmse_threshold_m > 0.0):
        raise ValueError(f"rmse_threshold_m must be positive; got {rmse_threshold_m!r}")


def _draw_samples(
    bundle: TrainedInverseCVAE,
    n_samples: int,
) -> tuple[list[NDArray[np.float64]], EncoderOutput]:
    """Draw ``n_samples`` candidate coefficient vectors and the encoder output."""
    cvae = bundle.model
    cvae.eval()
    kinematics = bundle.kinematics
    with torch.no_grad():
        encoder_output = cvae.encode(kinematics, sample=False)
        sampled = cvae.sample_coefficients(kinematics, n_samples=n_samples)
    # sampled: (1, n_samples, D) on model's device.
    flat = sampled.squeeze(0).detach().cpu().numpy().astype(np.float64)
    samples: list[NDArray[np.float64]] = [np.asarray(flat[i]) for i in range(n_samples)]
    return samples, encoder_output


def _to_numpy_traj(
    out: object,
) -> RoundTripOutput:
    """Coerce a forward-model output to ``(butt, clubhead, quat)`` ndarrays.

    Accepts either a tensor-bearing dataclass (like ``ClubTrajectory``) or a
    ndarray-bearing dataclass (like ``SimscapeOutput``). Batch-first tensors
    have their leading dim squeezed.
    """
    butt = getattr(out, "butt", None)
    clubhead = getattr(out, "clubhead", None)
    quat = getattr(out, "club_quat", None)
    if butt is None or clubhead is None or quat is None:
        raise TypeError(
            "forward_fn output must expose butt / clubhead / club_quat "
            "attributes (got "
            f"{type(out).__name__})"
        )

    def _arr(x: object) -> NDArray[np.float64]:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"forward_fn field must be tensor or ndarray; got {type(x).__name__}"
            )
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        return np.asarray(x, dtype=np.float64)

    return _arr(butt), _arr(clubhead), _arr(quat)


def _surrogate_forward_fn(
    surrogate: _ForwardSurrogate,
) -> ForwardFn:
    """Wrap a ``SwingSurrogate``-like callable as a numpy ``ForwardFn``."""

    def _fn(coeffs: NDArray[np.float64]) -> RoundTripOutput:
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(coeffs, dtype=np.float32)).unsqueeze(0)
            out = surrogate(t)
        return _to_numpy_traj(out)

    return _fn


def _resolve_forward_fn(
    forward_fn: Callable[[NDArray[np.float64]], object] | None,
) -> ForwardFn | None:
    """Normalise ``forward_fn`` argument into a :data:`ForwardFn` or None.

    A user-supplied callable may return either a numpy 3-tuple (already
    matching :data:`RoundTripOutput`) or a trajectory dataclass; both are
    accepted.
    """
    if forward_fn is None:
        return None

    def _fn(coeffs: NDArray[np.float64]) -> RoundTripOutput:
        out = forward_fn(coeffs)
        if isinstance(out, tuple) and len(out) == 3:
            butt, clubhead, quat = out
            return (
                np.asarray(butt, dtype=np.float64),
                np.asarray(clubhead, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
            )
        return _to_numpy_traj(out)

    return _fn


def _select_best(
    samples: list[NDArray[np.float64]],
    report: ValidationReport,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]], list[float], int]:
    """Pick best sample and partition into accepted / rejected."""
    accepted_samples = [samples[i] for i, ok in enumerate(report.accepted) if bool(ok)]
    accepted_costs = [
        float(report.rmses_m[i]) for i, ok in enumerate(report.accepted) if bool(ok)
    ]
    rejected_count = int(len(samples) - len(accepted_samples))

    if accepted_samples:
        best_local = int(np.argmin(np.asarray(accepted_costs)))
        best = accepted_samples[best_local]
    else:
        # No sample passed the threshold -- return the least-bad one.
        best = samples[report.best_index]
    return best, accepted_samples, accepted_costs, rejected_count


def predict_coefficients(
    target: ClubTarget,
    model: TrainedInverseCVAE,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    forward_fn: Callable[[NDArray[np.float64]], object] | None = None,
    surrogate: _ForwardSurrogate | None = None,
    rmse_threshold_m: float = DEFAULT_RMSE_THRESHOLD_M,
    validate: bool = True,
) -> InverseFitResult:
    """Sample-and-validate inference for the trained inverse CVAE.

    Draws ``n_samples`` candidate coefficient vectors from the CVAE's
    posterior conditioned on ``model.kinematics`` and, when ``validate`` is
    true, scores each through ``forward_fn`` against ``target``. The
    returned :class:`InverseFitResult` carries the best (lowest round-trip
    RMSE) sample, the accepted-sample list, and the encoder posterior.

    Forward-model resolution
    ------------------------
    * Explicit ``forward_fn`` wins (used by tests with stubs).
    * Otherwise, a ``surrogate`` callable is wrapped (Option 2 fast path).
    * If neither is supplied and ``validate`` is true, the call raises
      ``ValueError`` -- the SimscapeAdapter fallback is opt-in via the
      caller because instantiating it requires a MATLAB licence.

    Args:
        target: The :class:`ClubTarget` to fit.
        model: Trained inverse CVAE bundle.
        n_samples: Number of posterior samples to draw (default 32).
        forward_fn: Optional callable mapping a coefficient ndarray to a
            trajectory tuple or dataclass. Overrides ``surrogate``.
        surrogate: Optional :class:`SwingSurrogate`-like callable used as
            the cheap forward model.
        rmse_threshold_m: Acceptance threshold for round-trip RMSE in
            metres (default 5 mm).
        validate: When false, skip the round-trip validation and return
            all samples as accepted with zero costs.

    Returns:
        An :class:`InverseFitResult` carrying the best sample, the
        accepted set, and the encoder posterior.

    Raises:
        TypeError: If ``target`` or ``model`` have the wrong type.
        ValueError: For invalid ``n_samples`` / ``rmse_threshold_m`` /
            kinematics shape, or if validation is requested without a
            forward model.
    """
    _validate_args(target, model, n_samples, rmse_threshold_m)

    samples, encoder_output = _draw_samples(model, n_samples)

    if not validate:
        # All samples accepted with zero cost; first sample is "best" by
        # convention (caller has opted out of scoring).
        return InverseFitResult(
            best_coefficients=samples[0],
            accepted_samples=list(samples),
            accepted_costs=[0.0] * len(samples),
            rejected_count=0,
            sampling_budget_used=n_samples,
            encoder_output=encoder_output,
        )

    resolved_fn: ForwardFn | None = _resolve_forward_fn(forward_fn)
    if resolved_fn is None:
        if surrogate is None:
            raise ValueError(
                "validate=True requires a forward_fn or a surrogate; got neither. "
                "Pass validate=False to skip round-trip validation."
            )
        resolved_fn = _surrogate_forward_fn(surrogate)

    report = round_trip_validate(
        samples, target, resolved_fn, rmse_threshold_m=rmse_threshold_m
    )
    best, accepted_samples, accepted_costs, rejected_count = _select_best(
        samples, report
    )

    return InverseFitResult(
        best_coefficients=best,
        accepted_samples=accepted_samples,
        accepted_costs=accepted_costs,
        rejected_count=rejected_count,
        sampling_budget_used=n_samples,
        encoder_output=encoder_output,
    )
