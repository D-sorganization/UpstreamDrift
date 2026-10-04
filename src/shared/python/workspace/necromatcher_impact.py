"""Authenticated authored-replay extraction; no reference swing substitution.

Recorded native rates use explicitly authored simulation seconds. Capture PTS is
retained as provenance only, never differentiated or identified with replay time.
Call from the existing SDK-first native context; imports alone compile nothing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.pipeline import (
    MarkerLinearization,
    MarkerLinearizationPlant,
)
from src.shared.python.physics import SwingState
from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID
from src.shared.python.simulation_backends import Trace

from .necromatcher_impact_contracts import (
    ReplayImpactGeometry,
    ReplayImpactSelection,
    finite_array,
)
from .necromatcher_native import NativeFitBinding, load_native_fit_binding

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

__all__ = [
    "ReplayImpactGeometry",
    "ReplayImpactSelection",
    "extract_replay_impact_state",
]

BASIS_PROBE_LENGTH_M = 1.0
MARKERS = ("head_point", "face_normal", "face_up", "face_tangent")


def _metadata_string(trace: Trace, key: str) -> str:
    """Require the canonical trace metadata's declared string representation."""
    value = trace.meta.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Replay metadata {key} requires a string")
    return value


def _replay_parents(library: NecromatcherLibrary, replay_id: str, trace: Trace) -> dict:
    """Recheck exact canonical asset rows and retain immutable parent receipts."""
    replay = library.load_asset(replay_id)
    if replay.kind != "authored_replay":
        raise ValueError("Impact extraction requires a registered authored replay")
    if (
        trace.meta.get("schema") != "necromatcher/authored-replay/1"
        or trace.meta.get("physical_source_time_qualified") is not False
        or trace.meta.get("scientific_qualified") is not False
        or trace.meta.get("independent_replay_executed") is not True
    ):
        raise ValueError("Only explicitly unqualified authored replay is supported")
    receipt = {"replay_id": replay_id, "replay_hash": replay.metadata["hash"]}
    for name, kind in (
        ("profile", "torque_profile"),
        ("fit", "kinematic_fit"),
        ("model", "native_model"),
        ("capture", "image_capture"),
    ):
        identity = _metadata_string(trace, name + "_id")
        digest = _metadata_string(trace, name + "_hash")
        asset = library.load_asset(identity)
        if (
            asset.kind != kind
            or asset.session_id != replay.session_id
            or asset.metadata["hash"] != digest
        ):
            raise ValueError("Replay parent kind, session or hash differs")
        receipt.update({name + "_id": identity, name + "_hash": digest})
    return receipt


def _bound_inputs(
    binding: NativeFitBinding, trace: Trace, geometry: ReplayImpactGeometry
) -> None:
    """Require compiled order/units and exact source/model parent agreement."""
    if not isinstance(binding.plant, MarkerLinearizationPlant):
        raise ValueError("Native marker linearization capability unavailable")
    for name, wanted in (
        ("fit_id", binding.fit_id),
        ("fit_hash", binding.fit_hash),
        ("model_id", binding.model_id),
        ("model_hash", binding.model_hash),
        ("capture_id", binding.fit["capture_id"]),
        ("capture_hash", binding.fit["capture_hash"]),
    ):
        if trace.meta[name] != wanted:
            raise ValueError("Compiled resource differs from replay parent")
    order = tuple(binding.plant.coordinate_order)
    if tuple(binding.fit["coordinate_order"]) != order:
        raise ValueError("Fit/native coordinate order differs")
    for key, expected in (
        ("coordinate_order", order),
        ("coordinate_units", binding.coordinate_units),
    ):
        if json.loads(_metadata_string(trace, key + "_json")) != list(expected):
            raise ValueError("Replay/native order or units differs")
    if trace.backend != binding.plant.engine_name:
        raise ValueError("Replay native engine differs")
    if geometry.body not in {
        body["name"]
        for body in binding.fit["provenance"]["native_definition"]["bodies"]
    }:
        raise ValueError("Explicit head body is absent from the exact definition")


def _marker_kinematics(
    binding: NativeFitBinding, geometry: ReplayImpactGeometry, q: np.ndarray
) -> MarkerLinearization:
    """Linearize one point and a mathematical unit-length orthonormal face triad."""
    point = np.array(geometry.local_head_point_m)
    normal, up = np.array(geometry.local_face_normal), np.array(geometry.local_face_up)
    basis = np.array([normal, up, np.cross(normal, up)])
    offsets = np.vstack((point, point + BASIS_PROBE_LENGTH_M * basis))
    attachments = {
        name: (geometry.body, tuple(offset))
        for name, offset in zip(MARKERS, offsets, strict=True)
    }
    if not isinstance(binding.plant, MarkerLinearizationPlant):
        raise ValueError("Native marker linearization capability unavailable")
    linearizer = binding.plant.create_marker_linearizer(attachments)
    row = linearizer.marker_linearization(q)
    if (
        not isinstance(row, MarkerLinearization)
        or row.marker_labels != MARKERS
        or row.coordinate_order != tuple(binding.plant.coordinate_order)
    ):
        raise ValueError("Marker derivative identity differs")
    poses = binding.plant.frame_poses(attachments, q)
    rotation, translation = poses[geometry.body]
    rotation = finite_array(rotation, (3, 3), "Native body rotation")
    translation = finite_array(translation, (3,), "Native body translation")
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12
    ) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-12):
        raise ValueError("Native body pose is not a proper rigid transform")
    expected = offsets @ rotation.T + translation
    if not np.allclose(row.positions, expected, rtol=0, atol=1e-12):
        raise ValueError("Marker points differ from public body poses")
    return row


def _world_state(
    row: MarkerLinearization, rates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use native coordinate derivatives only: Jv and rigid-triad angular rate.

    Dividing each basis displacement/derivative by the declared probe length
    yields unit vectors and 1/s derivatives. No numerical PTS differences occur.
    """
    velocities = row.jacobian @ rates
    basis = (row.positions[1:] - row.positions[0]) / BASIS_PROBE_LENGTH_M
    derivative = (velocities[1:] - velocities[0]) / BASIS_PROBE_LENGTH_M
    if not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-12):
        raise ValueError("World face triad is not orthonormal")
    if not np.allclose(
        derivative @ basis.T + basis @ derivative.T, 0, rtol=0, atol=1e-10
    ):
        raise ValueError("Marker derivatives do not preserve a rigid face triad")
    omega = 0.5 * np.sum(np.cross(basis, derivative), axis=0)
    return velocities[0], omega, basis[0]


def extract_replay_impact_state(
    library: NecromatcherLibrary,
    replay_id: str,
    geometry: ReplayImpactGeometry,
    selection: ReplayImpactSelection,
) -> SwingState:
    """Extract an explicitly selected research state through authenticated owners.

    Postconditions: all vectors use FLIGHT_FRAME_ID; metadata contains exact
    parents, recorded authored time and operator assumptions. No reference swing,
    velocity-from-footage calculation, library write or qualification occurs.
    Missing derivative capability fails explicitly, including angular velocity.
    """
    if not isinstance(geometry, ReplayImpactGeometry) or not isinstance(
        selection, ReplayImpactSelection
    ):
        raise ValueError("Impact extraction requires typed geometry and selection")
    with library.authenticated_read():
        trace = library.load_replay(replay_id)
        parents = _replay_parents(library, replay_id, trace)
        index = selection.recorded_sample_index
        if index >= len(trace.t):
            raise IndexError("Selected sample is outside the recorded replay")
        binding = load_native_fit_binding(library, parents["fit_id"])
        _bound_inputs(binding, trace, geometry)
        shape = (len(binding.plant.coordinate_order),)
        q, rates = (
            finite_array(trace.q[index], shape, "Recorded q"),
            finite_array(trace.v[index], shape, "Recorded rates"),
        )
        row = _marker_kinematics(binding, geometry, q)
        velocity, omega, normal = _world_state(row, rates)
        state = _flight_state(
            trace, selection, geometry, (velocity, omega, normal), row.positions[0]
        )
        state.metadata.update(parents)
        json.dumps(state.metadata, allow_nan=False)
        if _replay_parents(library, replay_id, trace) != parents:
            raise ValueError("Replay parents changed during extraction")
        return state


def _flight_state(
    trace: Trace,
    selection: ReplayImpactSelection,
    geometry: ReplayImpactGeometry,
    vectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    point: np.ndarray,
) -> SwingState:
    rotation = np.array(selection.world_to_flight_rotation)
    velocity, omega, normal = [rotation @ v for v in vectors]
    if not all(np.isfinite(v).all() for v in (velocity, omega, normal)):
        raise ValueError("Extracted impact vectors must be finite")
    metadata: dict[str, Any] = {
        "frame_id": FLIGHT_FRAME_ID,
        "impact_extraction_schema": "necromatcher/replay-impact/1",
        "recorded_sample_index": selection.recorded_sample_index,
        "recorded_time_s": float(trace.t[selection.recorded_sample_index]),
        "replay_clock_policy": "authored_simulation_seconds",
        "physical_source_time_qualified": False,
        "scientific_qualified": False,
        "geometry": geometry.to_record(),
        "selection": selection.to_record(),
        "head_point_flight_m": (
            rotation @ point + np.array(selection.world_to_flight_translation_m)
        ).tolist(),
        "capture_initial_frame": json.loads(
            _metadata_string(trace, "source_frame_json")
        ),
        "capture_initial_frame_index": trace.meta["source_frame_index"],
        "capture_pts_used_for_velocity": False,
        "basis_probe_length_m": BASIS_PROBE_LENGTH_M,
    }
    json.dumps(metadata, allow_nan=False)
    # Loft is an orientation descriptor; the existing solver receives the full normal.
    loft = float(np.degrees(np.arctan2(normal[2], np.hypot(normal[0], normal[1]))))
    return SwingState(
        velocity,
        omega,
        normal,
        clubhead_mass=geometry.mass_kg,
        clubhead_loft_deg=loft,
        clubhead_moi=geometry.moi_kg_m2,
        engine_name=trace.backend,
        metadata=metadata,
    )
