"""Private bounded native pelvis probe with an explicit training coordinate gauge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.diagnostics.native_pelvis_geometry import (
    PELVIS_COORDINATES,
    load_isolated_capture,
    run_probe,
)
from src.engines.physics_engines.opensim.python.tour_matching.frozen_registration import (
    apply_frozen_registration,
    pelvis_coordinate_gauge,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
    parse_model,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_marker_geometry import (
    NativeMarkerGeometry,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURES,
    TourCapture,
    capture_kind,
)


def source_lateral_axis(model_path: Path) -> np.ndarray:
    """Read the pinned donor's left/right axis, not capture landmark aliases."""
    tree = parse_model(model_path)
    selected = {}
    for marker in tree.findall(".//MarkerSet/objects/Marker"):
        name = marker.get("name", "")
        if name not in ("lasi", "rasi"):
            continue
        if (
            name in selected
            or marker.findtext("socket_parent_frame") != "/bodyset/pelvis"
        ):
            raise ValueError(
                "Native reference markers must be unique and on the pelvis body"
            )
        point = np.array(
            [float(value) for value in marker.findtext("location", "").split()]
        )
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(
                "Native reference marker requires three finite local metres"
            )
        selected[name] = point
    if set(selected) != {"lasi", "rasi"}:
        raise ValueError("Native reference left/right landmark geometry is unavailable")
    axis = selected["lasi"] - selected["rasi"]
    if np.linalg.norm(axis) < 1e-6:
        raise ValueError("Native left/right reference is degenerate")
    return axis / np.linalg.norm(axis)


def run_registered_probe(
    model_path: Path,
    source_sha256: str,
    capture: TourCapture,
) -> dict[str, object]:
    """Freeze frame-zero gauge, then reuse the existing native three-pose probe."""
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("Source model hash differs from the pinned request")
    if capture.source_sha256 is None:
        raise ValueError("Pinned capture source SHA-256 identity is required")
    spec = TOUR_CAPTURES[capture_kind(capture.source_sha256)]
    if spec.vertical_axis != "y":
        raise ValueError(
            "The declared diagnostic gauge requires frozen Y-up observations"
        )
    native = NativeMarkerGeometry(model_path, PELVIS_COORDINATES)
    reference_bindings = {"native-origin": ("/bodyset/pelvis", (0.0, 0.0, 0.0))}
    reference = native.frame_poses(reference_bindings, native.initial_coordinates)[
        "/bodyset/pelvis"
    ]
    left_axis = source_lateral_axis(model_path)
    frozen = pelvis_coordinate_gauge(
        capture,
        reference,
        target_left_axis_local=left_axis,
        target_geometry_sha256=native.identity_sha256,
        training_frame=0,
    )
    transformed = apply_frozen_registration(
        capture, frozen, source_frame=frozen.transform.source_frame
    )
    receipt = run_probe(model_path, source_sha256, transformed.capture)
    receipt["gauge_driver_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    receipt["registered_points_sha256"] = hashlib.sha256(
        transformed.capture.points_m.tobytes()
    ).hexdigest()
    receipt["registration"] = {
        "identity_sha256": frozen.identity_sha256,
        "provider_sha256": frozen.provider_sha256,
        "interpretation": frozen.interpretation,
        "anatomical_admission": "missing-correspondence",
        "anatomically_qualified": frozen.anatomically_qualified,
        "source_frame": frozen.transform.source_frame,
        "target_frame": frozen.transform.target_frame,
        "rotation": frozen.transform.rotation.tolist(),
        "translation_m": frozen.transform.translation.tolist(),
        "training_frames": frozen.training_frames,
        "training_times_s": frozen.training_times_s,
        "training_points_sha256": frozen.training_points_sha256,
        "target_points_sha256": frozen.target_points_sha256,
        "constructed_training_rms_m": frozen.training_rms_m,
        "native_left_axis_local": left_axis.tolist(),
        "native_reference_coordinates": native.initial_coordinates.tolist(),
        "native_reference_body_rotation": reference[0].tolist(),
        "native_reference_body_translation_m": reference[1].tolist(),
        "local_offset_policy": "Training-frame offsets frozen after registration; no anatomical radius bound inferred",
        "ground_policy": "Training centroid is aligned to the source initial pelvis origin; capture floor is not calibrated",
        "native_direction_markers": ["lasi", "rasi"],
        "anchor_provenance": frozen.anchor_provenance,
        "limitations": [
            "waist observations are not asserted to be ASIS/PSIS landmarks",
            "centroid/reference-pose alignment is an authored coordinate gauge",
            "post-training samples are separately fitted poses, not independent predictions",
            "anatomy, ground/contact, grip and muscle dynamics remain unqualified",
        ],
    }
    return receipt


def main() -> None:
    """Keep all detailed observations, transforms and fitted states private."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        capture, transfer = load_isolated_capture(args.capture, args.capture_python)
        receipt = run_registered_probe(args.model, args.model_sha256, capture)
        receipt["capture_transfer"] = transfer
    except (ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        receipt = {
            "status": "registration-failed-not-qualified",
            "error": str(error),
            "error_type": type(error).__name__,
            "source_sha256": args.model_sha256,
        }
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if receipt["status"] == "failed-not-qualified":
        raise RuntimeError("Native geometry probe failed; see private receipt")


if __name__ == "__main__":
    main()
