"""Replay a native candidate with one initial state and explicit weld checks."""

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.continuous_forward import (
    ContinuousForwardResult,
    integrate_forward,
)
from src.shared.python.motion_matching.marker_projection import project_markers
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile

Array = NDArray[np.float64]


class NativeReplayEngine(Protocol):
    """Public engine operations needed by the replay adapter."""

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]: ...
    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, Array]: ...
    def closure_errors(self) -> tuple[Array, Array]: ...


class NativeReplayResult(NamedTuple):
    """Requested samples and sampled closure diagnostics; no fit acceptance implied."""

    integration: ContinuousForwardResult
    markers_m: Array
    closure_pose_max_abs: float
    closure_velocity_max_abs: float
    candidate_sha256: str


def replay_candidate(
    model_bytes: bytes,
    candidate: NativeReplayCandidate,
    time_s: Array,
    *,
    model_factory: Callable[
        [Mapping[str, Any]], NativeReplayEngine
    ] = NativePinocchioModel,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.001,
    closure_tolerance: float = 1e-7,
) -> NativeReplayResult:
    """Replay full candidate coverage without intermediate state resets."""
    return _replay_native(
        model_bytes,
        candidate,
        time_s,
        initial_state=None,
        model_factory=model_factory,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        closure_tolerance=closure_tolerance,
    )


def replay_window(
    model_bytes: bytes,
    candidate: NativeReplayCandidate,
    time_s: Array,
    initial_state: Array,
    *,
    model_factory: Callable[
        [Mapping[str, Any]], NativeReplayEngine
    ] = NativePinocchioModel,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.001,
    closure_tolerance: float = 1e-7,
) -> NativeReplayResult:
    """Replay a shooting window on the unchanged absolute polynomial clock.

    The supplied state belongs exactly to time_s[0] and must already satisfy
    native position/rate closure. No assembly projection or target reset occurs.
    This is a window result, not full-candidate acceptance. Its integration
    retains the absolute clock and actual supplied initial state for provenance.
    """
    if initial_state is None:
        raise ValueError("A shooting window requires an explicit initial state")
    return _replay_native(
        model_bytes,
        candidate,
        time_s,
        initial_state=initial_state,
        model_factory=model_factory,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        closure_tolerance=closure_tolerance,
    )


def _replay_native(
    model_bytes: bytes,
    candidate: NativeReplayCandidate,
    time_s: Array,
    *,
    initial_state: Array | None,
    model_factory: Callable[
        [Mapping[str, Any]], NativeReplayEngine
    ] = NativePinocchioModel,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.001,
    closure_tolerance: float = 1e-7,
) -> NativeReplayResult:
    spec = json.loads(model_bytes)
    names = spec["coordinate_order"]
    checked = NativeReplayCandidate.from_document(
        candidate.document, names, hashlib.sha256(model_bytes).hexdigest()
    )
    data = checked.document
    clock = np.asarray(time_s, dtype=float)
    if (
        clock.ndim != 1
        or clock.size < 2
        or not np.isfinite(clock).all()
        or clock[0] < 0
        or np.any(np.diff(clock) <= 0)
        or clock[-1] > data["duration_s"]
    ):
        raise ValueError("Requested clock must be within candidate coverage")
    if initial_state is None and (clock[0] != 0 or clock[-1] != data["duration_s"]):
        raise ValueError("Requested clock must provide full candidate coverage")
    if not np.isfinite(closure_tolerance) or closure_tolerance <= 0:
        raise ValueError("Closure tolerance must be finite and positive")
    roots = [joint for joint in spec["joints"] if joint["parent"] == "world"]
    if len(roots) != 1:
        raise ValueError("Expected one native hip root")
    root = roots[0]
    if [p["primitive"] for p in root["primitives"][:3]] != ["Px", "Py", "Pz"] or [
        p["coordinate"] for p in root["primitives"][:3]
    ] != names[:3]:
        raise ValueError("Native force primitive inventory mismatch")
    rotation = np.asarray(root["parent_to_base"], dtype=float)[:3, :3].T
    profile = NativeEffortProfile(names, data["coefficients"], rotation)
    engine = model_factory(spec)
    n = len(names)

    def coordinates(values: Array) -> dict[str, float]:
        return dict(zip(names, map(float, values), strict=True))

    def derivative(t: float, state: Array) -> Array:
        acceleration = engine.accelerations(
            coordinates(state[:n]), coordinates(state[n:]), profile.evaluate(t)
        )
        if set(acceleration) != set(names):
            raise ValueError("Engine returned a different coordinate inventory")
        return np.concatenate((state[n:], [acceleration[name] for name in names]))

    def closure() -> tuple[float, float]:
        pose, rate = engine.closure_errors()
        if (
            np.shape(pose) != (6,)
            or np.shape(rate) != (6,)
            or not np.isfinite(np.concatenate((pose, rate))).all()
        ):
            raise ValueError("Invalid native closure evidence")
        errors = float(np.max(np.abs(pose))), float(np.max(np.abs(rate)))
        if max(errors) > closure_tolerance:
            raise ValueError("Native closure residual exceeds the replay bound")
        return errors

    initial = np.array(
        data["q0"] + data["qd0"] if initial_state is None else initial_state,
        dtype=float,
        copy=True,
    )
    if initial.shape != (2 * n,) or not np.isfinite(initial).all():
        raise ValueError("Initial state must be a finite native q/rate vector")
    derivative(float(clock[0]), initial)
    closure()
    project_markers(
        engine.frame_poses(coordinates(initial[:n])),
        data["marker_bodies"],
        data["marker_offsets_m"],
    )
    integration = integrate_forward(
        initial,
        clock - clock[0],
        lambda t, state: derivative(t + float(clock[0]), state),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    absolute_clock = clock.copy()
    absolute_clock.setflags(write=False)
    integration = integration._replace(time=absolute_clock)
    markers, errors = [], []
    for t, state in zip(integration.time, integration.state, strict=True):
        derivative(float(t), state)
        errors.append(closure())
        markers.append(
            project_markers(
                engine.frame_poses(coordinates(state[:n])),
                data["marker_bodies"],
                data["marker_offsets_m"],
            )
        )
    positions = np.asarray(markers)
    positions.setflags(write=False)
    maxima = np.max(errors, axis=0)
    return NativeReplayResult(
        integration, positions, float(maxima[0]), float(maxima[1]), checked.sha256
    )
