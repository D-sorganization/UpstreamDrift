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
    """Verify model identity, then integrate without feedback or state resets.

    The output clock must span the candidate's full declared duration. Build a
    new candidate identity to change that duration; coefficients remain in
    absolute seconds. Closure tolerance is a maximum component bound on native
    six-component pose/rate residuals, not a Cartesian marker-fit threshold.
    """
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
        or clock[0] != 0
        or np.any(np.diff(clock) <= 0)
        or clock[-1] != data["duration_s"]
    ):
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

    initial = np.asarray(data["q0"] + data["qd0"], dtype=float)
    derivative(0.0, initial)
    closure()
    project_markers(
        engine.frame_poses(coordinates(initial[:n])),
        data["marker_bodies"],
        data["marker_offsets_m"],
    )
    integration = integrate_forward(
        initial, clock, derivative, rtol=rtol, atol=atol, max_step=max_step
    )
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
