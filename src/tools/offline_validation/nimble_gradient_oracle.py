"""Offline Nimble gradient agreement oracle.

This module is deliberately under ``tools/`` rather than ``src/``: Nimble is an
independent differentiable-physics oracle for validation jobs, never a runtime
dependency of the library or application hot path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

NIMBLEPHYSICS_PIN = "nimblephysics==0.10.52.2"

OracleStatus = Literal["passed", "failed", "skipped"]
NimbleLoss = Callable[[Any, Any], Any]


class GradientOracleUnavailable(ImportError):
    """Raised when the optional Nimble oracle stack is not importable."""


class NimbleGradientBackend(Protocol):
    """Backend seam for Nimble-backed gradients."""

    def gradient(
        self,
        loss_fn: NimbleLoss,
        coordinates: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return ``d(loss_fn)/d(coordinates)`` from the oracle backend."""


@dataclass(frozen=True)
class GradientTolerance:
    """Tolerance envelope for candidate-vs-oracle gradient comparison."""

    rtol: float = 1.0e-4
    atol: float = 1.0e-6

    def __post_init__(self) -> None:
        _require_positive_finite(self.rtol, "rtol", allow_zero=True)
        _require_positive_finite(self.atol, "atol", allow_zero=True)


@dataclass(frozen=True)
class NimbleGradientOracleRequest:
    """Inputs for one offline gradient oracle comparison."""

    model_name: str
    coordinates: npt.NDArray[np.float64]
    candidate_gradient: npt.NDArray[np.float64]
    nimble_loss: NimbleLoss
    tolerance: GradientTolerance = field(default_factory=GradientTolerance)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        coordinates = _as_vector(self.coordinates, "coordinates")
        candidate = _as_vector(self.candidate_gradient, "candidate_gradient")
        if coordinates.shape != candidate.shape:
            raise ValueError(
                "coordinates and candidate_gradient must have matching shapes, "
                f"got {coordinates.shape} and {candidate.shape}"
            )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "candidate_gradient", candidate)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class NimbleGradientOracleResponse:
    """Result of one offline Nimble oracle comparison."""

    status: OracleStatus
    model_name: str
    oracle_gradient: npt.NDArray[np.float64] | None
    max_abs_error: float | None
    max_rel_error: float | None
    tolerance: GradientTolerance
    reason: str
    nimble_pin: str = NIMBLEPHYSICS_PIN


class TorchAutogradNimbleBackend:
    """Nimble backend using PyTorch autograd through Nimble's tensor APIs."""

    def __init__(self) -> None:
        try:
            self._nimble = import_module("nimblephysics")
            self._torch = import_module("torch")
        except ImportError as exc:
            raise GradientOracleUnavailable(
                "Install upstream-drift[nimble-oracle] to run the offline "
                f"Nimble oracle ({NIMBLEPHYSICS_PIN})."
            ) from exc

    def gradient(
        self,
        loss_fn: NimbleLoss,
        coordinates: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the PyTorch/Nimble gradient for ``loss_fn``."""

        tensor = self._torch.tensor(
            coordinates,
            dtype=self._torch.float64,
            requires_grad=True,
        )
        loss = loss_fn(self._nimble, tensor)
        if tuple(loss.shape) != ():
            raise ValueError("nimble_loss must return a scalar tensor")
        loss.backward()
        gradient = tensor.grad
        if gradient is None:
            raise ValueError("nimble_loss did not produce a coordinate gradient")
        return np.asarray(gradient.detach().cpu().numpy(), dtype=float)


def compare_nimble_gradient(
    request: NimbleGradientOracleRequest,
    *,
    backend: NimbleGradientBackend | None = None,
    require_available: bool = False,
) -> NimbleGradientOracleResponse:
    """Compare a candidate gradient against the offline Nimble oracle.

    Missing Nimble returns a structured ``skipped`` response by default so the
    normal test suite and core installs do not fail. Set ``require_available``
    for dedicated oracle jobs that must fail loudly when the optional stack is
    absent.
    """

    try:
        oracle_backend = (
            backend if backend is not None else TorchAutogradNimbleBackend()
        )
        oracle_gradient = _as_vector(
            oracle_backend.gradient(request.nimble_loss, request.coordinates),
            "oracle_gradient",
        )
    except GradientOracleUnavailable as exc:
        if require_available:
            raise
        return NimbleGradientOracleResponse(
            status="skipped",
            model_name=request.model_name,
            oracle_gradient=None,
            max_abs_error=None,
            max_rel_error=None,
            tolerance=request.tolerance,
            reason=str(exc),
        )

    if oracle_gradient.shape != request.candidate_gradient.shape:
        raise ValueError(
            "oracle_gradient and candidate_gradient must have matching shapes, "
            f"got {oracle_gradient.shape} and {request.candidate_gradient.shape}"
        )

    abs_error = np.abs(request.candidate_gradient - oracle_gradient)
    denominator = np.maximum(np.abs(oracle_gradient), request.tolerance.atol)
    rel_error = abs_error / denominator
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    max_rel_error = float(np.max(rel_error)) if rel_error.size else 0.0
    passed = bool(
        np.allclose(
            request.candidate_gradient,
            oracle_gradient,
            rtol=request.tolerance.rtol,
            atol=request.tolerance.atol,
        )
    )
    return NimbleGradientOracleResponse(
        status="passed" if passed else "failed",
        model_name=request.model_name,
        oracle_gradient=oracle_gradient,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        tolerance=request.tolerance,
        reason="gradient agreement within tolerance"
        if passed
        else "gradient disagreement exceeds tolerance",
    )


def _as_vector(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_positive_finite(
    value: float,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not valid or not np.isfinite(value):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier} and finite")
