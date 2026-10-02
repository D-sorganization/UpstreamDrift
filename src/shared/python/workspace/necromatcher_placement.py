"""Explicit authored world placement with conserved source image projection."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.full_body_forward_dynamics import preload_feet
from src.shared.python.motion_matching.historical_fit import CameraProjection
from src.shared.python.motion_matching.pipeline.plant import ForwardSimulationPlant

from .necromatcher_native import NativeFitBinding, load_native_fit_binding

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary
    from .project_store import DatasetMetadata

PLACEMENT_SCHEMA = "necromatcher/authored-ground-placement/1"
_WORLD_TRANSLATION_ATOL_M = 1e-8
_REPROJECTION_ATOL_PIXELS = 1e-7


def author_ground_placement(
    library: NecromatcherLibrary,
    source_fit_id: str,
    new_fit_id: str,
    source_frame_index: int,
    clearance_m: float,
    description: str,
) -> DatasetMetadata:
    """Save a separate placement hypothesis; never preload a stored replay silently."""
    from .necromatcher_fit_jobs import fit_execution_stamp

    if type(source_frame_index) is not int or source_frame_index < 0:
        raise ValueError("Placement source frame must be a nonnegative integer")
    if (
        type(clearance_m) not in (float, int)
        or not np.isfinite(clearance_m)
        or clearance_m < 0
    ):
        raise ValueError("Authored clearance must be finite and nonnegative")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Placement requires an explicit authored description")
    stamp = fit_execution_stamp()
    binding = load_native_fit_binding(library, source_fit_id)
    try:
        position = binding.fit["frame_indices"].index(source_frame_index)
    except ValueError as exc:
        raise IndexError("Placement source frame is not in its stored fit") from exc
    payload = _build_revision(binding, position, clearance_m, description)
    payload["provenance"]["placement_revision"]["execution_stamp"] = stamp
    if fit_execution_stamp()["source_sha256"] != stamp["source_sha256"]:
        raise ValueError("Placement implementation changed during execution")
    session = library.load_asset(source_fit_id).session_id
    with TemporaryDirectory(prefix="necromatcher-placement-") as directory:
        source = Path(directory) / "fit.json"
        source.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        return library.add_fit(new_fit_id, session, source)


def _build_revision(
    binding: NativeFitBinding, position: int, clearance_m: float, description: str
) -> dict[str, Any]:
    if not isinstance(binding.plant, ForwardSimulationPlant):
        raise ValueError("Native plant lacks authored ground placement capability")
    simulator = binding.plant.create_forward_simulator()
    original = binding.fit
    q = np.asarray(original["q"], dtype=float)
    anchor = q[position]
    normal = np.asarray(binding.plant.ground_plane.normal, dtype=float)
    normal /= np.linalg.norm(normal)
    revised_anchor = preload_feet(simulator, anchor, preload=False)
    revised_anchor[:3] += np.linalg.solve(
        simulator.root_translation_axes(anchor), normal * clearance_m
    )
    offset = revised_anchor[:3] - anchor[:3]
    world_delta = simulator.root_translation_axes(anchor) @ offset
    revised_q = q.copy()
    revised_q[:, :3] += offset
    camera, attachments = binding.review_inputs()
    revised_camera = CameraProjection(
        camera.intrinsics,
        camera.rotation,
        camera.translation - camera.rotation @ world_delta,
    )
    diagnostics = _check_geometry(
        binding, q, revised_q, camera, revised_camera, attachments, world_delta
    )
    _, anchor_height = simulator.support(revised_q[position], np.zeros(q.shape[1]))
    if not np.isclose(
        anchor_height, clearance_m, rtol=0, atol=_WORLD_TRANSLATION_ATOL_M
    ):
        raise ValueError("Authored placement failed its anchor clearance contract")
    record = {
        "schema_version": PLACEMENT_SCHEMA,
        "kind": "authored",
        "source_fit_id": binding.fit_id,
        "source_fit_hash": binding.fit_hash,
        "source_frame_index": original["frame_indices"][position],
        "description": description,
        "requested_clearance_m": clearance_m,
        "root_coordinate_offset": offset.tolist(),
        "world_translation_m": world_delta.tolist(),
        "anchor_ground_clearance_m": anchor_height,
        **diagnostics,
    }
    geometry = {
        "camera": {
            "intrinsics": revised_camera.intrinsics.tolist(),
            "rotation": revised_camera.rotation.tolist(),
            "translation": revised_camera.translation.tolist(),
        },
        "attachments": deepcopy(attachments),
        "kind": "authored_review_geometry",
    }
    prior_geometry = original["evidence"]["original_fit"]
    if "free_coordinates" in prior_geometry:
        geometry["free_coordinates"] = deepcopy(prior_geometry["free_coordinates"])
    return {
        **deepcopy(original),
        "q": revised_q.tolist(),
        "provenance": {
            "description": description,
            "native_definition": deepcopy(original["provenance"]["native_definition"]),
            "source_provenance": deepcopy(original["provenance"]),
            "placement_revision": record,
        },
        "evidence": {
            "original_fit": geometry,
            "rejection_reasons": list(original["evidence"].get("rejection_reasons", []))
            + ["ground_placement_authored_not_observed"],
        },
    }


def _check_geometry(
    binding: NativeFitBinding,
    q: np.ndarray,
    revised_q: np.ndarray,
    camera: CameraProjection,
    revised_camera: CameraProjection,
    attachments: dict[str, Any],
    delta: np.ndarray,
) -> dict[str, float]:
    """Check every saved frame; retain remaining whole-track ground penetration."""
    max_pixels = before_penetration = after_penetration = 0.0
    rates = dict.fromkeys(binding.plant.coordinate_order, 0.0)
    for before, after in zip(q, revised_q, strict=True):
        points = binding.plant.marker_positions(before, attachments)
        revised_points = binding.plant.marker_positions(after, attachments)
        if not np.allclose(
            revised_points - points, delta, rtol=0, atol=_WORLD_TRANSLATION_ATOL_M
        ):
            raise ValueError("Root revision is not one rigid world translation")
        pixels = np.abs(camera.project(points) - revised_camera.project(revised_points))
        max_pixels = max(max_pixels, float(pixels.max()))
        for pose, label in ((before, "before"), (after, "after")):
            coordinates = dict(
                zip(binding.plant.coordinate_order, map(float, pose), strict=True)
            )
            contacts = binding.plant.contact_forces(coordinates, rates)
            penetration = max(
                float(sample.penetration_m) for sample in contacts.values()
            )
            if label == "before":
                before_penetration = max(before_penetration, penetration)
            else:
                after_penetration = max(after_penetration, penetration)
    if not np.isfinite(max_pixels) or max_pixels > _REPROJECTION_ATOL_PIXELS:
        raise ValueError("Authored world placement changed source image projection")
    return {
        "max_reprojection_difference_pixels": max_pixels,
        "max_ground_penetration_before_m": before_penetration,
        "max_ground_penetration_after_m": after_penetration,
    }


def validate_placement_lineage(
    payload: dict[str, Any], library: NecromatcherLibrary, swing_id: str
) -> None:
    """Recheck the immutable source parent and conserved sample/camera transforms."""
    record = payload["provenance"].get("placement_revision")
    if "placement_revision" not in payload["provenance"]:
        return
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != PLACEMENT_SCHEMA
        or record.get("kind") != "authored"
    ):
        raise ValueError("Invalid authored placement provenance")
    source_id = record.get("source_fit_id")
    if not isinstance(source_id, str):
        raise ValueError("Placement requires a source fit identity")
    parent = library.load_asset(source_id)
    if (
        parent.kind != "kinematic_fit"
        or parent.session_id != swing_id
        or parent.metadata["hash"] != record.get("source_fit_hash")
    ):
        raise ValueError("Placement source fit kind, session or hash mismatch")
    source = library.load_fit(source_id)
    frame_index = record.get("source_frame_index")
    clearance, achieved = (
        record.get("requested_clearance_m"),
        record.get("anchor_ground_clearance_m"),
    )
    if type(frame_index) is not int or frame_index not in source["frame_indices"]:
        raise ValueError("Placement anchor must identify an exact source fit frame")
    if (
        clearance is None
        or achieved is None
        or type(clearance) not in (float, int)
        or type(achieved) not in (float, int)
        or not np.isfinite(clearance)
        or not np.isfinite(achieved)
        or clearance < 0
        or not np.isclose(clearance, achieved, rtol=0, atol=_WORLD_TRANSLATION_ATOL_M)
    ):
        raise ValueError(
            "Placement clearance evidence differs from its authored target"
        )
    for key in (
        "model_id",
        "model_hash",
        "capture_id",
        "capture_hash",
        "coordinate_order",
        "coordinate_units",
        "frame_indices",
        "frames",
    ):
        if payload[key] != source[key]:
            raise ValueError("Placement changed its original source/model bindings")
    offset = np.asarray(record.get("root_coordinate_offset"), dtype=float)
    delta = np.asarray(record.get("world_translation_m"), dtype=float)
    if (
        offset.shape != (3,)
        or delta.shape != (3,)
        or not np.isfinite(offset).all()
        or not np.isfinite(delta).all()
    ):
        raise ValueError("Placement requires finite root and world translation vectors")
    expected = np.asarray(source["q"], dtype=float).copy()
    expected[:, :3] += offset
    if not np.array_equal(payload["q"], expected):
        raise ValueError("Placement pose samples differ from its recorded root offset")
    before, after = (
        source["evidence"]["original_fit"],
        payload["evidence"]["original_fit"],
    )
    old_camera, new_camera = (
        CameraProjection(**before["camera"]),
        CameraProjection(**after["camera"]),
    )
    if (
        after["attachments"] != before["attachments"]
        or not np.array_equal(old_camera.intrinsics, new_camera.intrinsics)
        or not np.array_equal(old_camera.rotation, new_camera.rotation)
        or not np.array_equal(
            new_camera.translation, old_camera.translation - old_camera.rotation @ delta
        )
    ):
        raise ValueError(
            "Placement camera or attachments differ from its recorded world transform"
        )
