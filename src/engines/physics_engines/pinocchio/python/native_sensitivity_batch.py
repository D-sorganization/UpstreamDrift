"""Trusted worker transport for independent native sensitivity windows.

This module is deliberately a narrow bridge between the generic shooting batch
seam and :class:`NativeWindowExecutor`.  Its pickle payload is private,
in-process transport created by the native runner; it is never a file, network,
or public API format.  Each spawned worker validates the model/candidate
identity again before constructing the native Pinocchio replay.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.native_sensitivity import (
    replay_marker_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate

Array = NDArray[np.float64]


def _readonly_finite_vector(
    value: Array | None, *, name: str, allow_none: bool
) -> Array | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeSensitivityWindowRequest:
    """Validated native sensitivity task; serialize only inside a trusted runner."""

    model_bytes: bytes
    candidate_document: dict[str, Any]
    time_s: Array
    initial_state: Array | None
    initial_sensitivity: Array | None
    first_control: int
    basis_duration_s: float

    def __post_init__(self) -> None:
        if type(self.model_bytes) is not bytes or not self.model_bytes:
            raise ValueError("model_bytes must be nonempty immutable bytes")
        try:
            spec = json.loads(self.model_bytes)
        except (TypeError, ValueError) as error:
            raise ValueError("model_bytes must contain native JSON") from error
        names = spec.get("coordinate_order")
        if not isinstance(names, list) or not names:
            raise ValueError("Native model has no coordinate order")
        if not isinstance(self.candidate_document, dict):
            raise ValueError("candidate_document must be a dictionary")
        time = _readonly_finite_vector(self.time_s, name="time_s", allow_none=False)
        assert time is not None
        if np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be strictly increasing")
        state = _readonly_finite_vector(
            self.initial_state, name="initial_state", allow_none=True
        )
        sensitivity = None
        if self.initial_sensitivity is not None:
            sensitivity = np.asarray(self.initial_sensitivity, dtype=float)
            if (
                state is None
                or sensitivity.ndim != 2
                or sensitivity.shape[0] != 2 * len(names)
                or not np.isfinite(sensitivity).all()
            ):
                raise ValueError(
                    "initial_sensitivity must match an explicit native state"
                )
            sensitivity = sensitivity.copy()
            sensitivity.setflags(write=False)
        if state is not None and state.shape != (2 * len(names),):
            raise ValueError("initial_state has the wrong native dimension")
        if (
            type(self.first_control) is not int
            or not 0 <= self.first_control <= 6
            or not np.isfinite(self.basis_duration_s)
            or self.basis_duration_s <= 0
        ):
            raise ValueError("Invalid native Bernstein window configuration")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "initial_sensitivity", sensitivity)


@dataclass(frozen=True)
class NativeSensitivityWindowResult:
    """Native worker result consumed by parent-owned cache and assembly."""

    markers_m: Array
    endpoint_state: Array
    marker_jacobian: Array
    state_jacobian: Array
    sensitivity_elapsed_s: float
    sensitivity_evaluations: int
    primal_marker_max_abs_difference_m: float


def serialize_trusted_native_sensitivity_request(
    request: NativeSensitivityWindowRequest,
) -> bytes:
    """Encode a locally validated task for private multiprocessing transport."""
    if not isinstance(request, NativeSensitivityWindowRequest):
        raise TypeError("Expected a native sensitivity window request")
    return pickle.dumps(request, protocol=5)


def evaluate_trusted_native_sensitivity_request(
    payload: bytes,
) -> NativeSensitivityWindowResult:
    """Execute a locally created request; never call this on untrusted bytes."""
    if type(payload) is not bytes:
        raise TypeError("Native sensitivity worker payload must be immutable bytes")
    try:
        request = pickle.loads(payload)  # noqa: S301 - private in-process executor transport
    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
    ) as error:
        raise ValueError("Native sensitivity worker payload type differs") from error
    if not isinstance(request, NativeSensitivityWindowRequest):
        raise ValueError("Native sensitivity worker payload type differs")
    spec = json.loads(request.model_bytes)
    candidate = NativeReplayCandidate.from_document(
        request.candidate_document,
        spec["coordinate_order"],
        hashlib.sha256(request.model_bytes).hexdigest(),
    )
    replay = replay_marker_sensitivities(
        request.model_bytes,
        candidate,
        request.time_s,
        first_control=request.first_control,
        basis_duration_s=request.basis_duration_s,
        initial_state=request.initial_state,
        initial_sensitivity=request.initial_sensitivity,
    )
    result = NativeSensitivityWindowResult(
        replay.replay.markers_m.copy(),
        replay.replay.integration.state[-1].copy(),
        replay.marker_jacobian.copy(),
        replay.state_jacobian[-1].copy(),
        float(replay.sensitivity_elapsed_s),
        int(replay.sensitivity_evaluations),
        float(replay.primal_marker_max_abs_difference_m),
    )
    for value in (
        result.markers_m,
        result.endpoint_state,
        result.marker_jacobian,
        result.state_jacobian,
    ):
        value.setflags(write=False)
    return result
