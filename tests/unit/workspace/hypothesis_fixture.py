"""Real source importer/native fixtures for hypothesis admission boundary tests."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ImageSplineStart,
)
from src.shared.python.shadow_tracker.historical_capture import observation_record
from src.shared.python.shadow_tracker.ingestion import compute_frame_hash
from src.shared.python.shadow_tracker.source_records import FrameIdentity, SourceAsset
from src.shared.python.pose_estimation.interface import PoseEstimationResult


def imported_capture(
    library: Any, directory: Path, *, camera_ids: tuple[str, ...] | None = None
) -> Any:
    import cv2

    directory.mkdir()
    source = SourceAsset(
        schema_version="shadow-tracker/source/1.0.0",
        asset_id="source-" + "a" * 64,
        source_uri="https://example.invalid/fixture-source.mp4",
        content_sha256="a" * 64,
        width_px=32,
        height_px=32,
        rights_status="unknown",
        rights_note="Synthetic fixture",
    )
    rows = []
    for index in range(3):
        image = np.full((32, 32, 3), index * 30, dtype=np.uint8)
        frame = FrameIdentity(
            schema_version="shadow-tracker/frame/1.1.0",
            asset_id=source.asset_id,
            shot_id="fixture-shot",
            swing_id="practice",
            camera_id=camera_ids[index] if camera_ids else "source-camera",
            frame_id=f"frame-{index}",
            pts_ticks=index,
            timebase_numerator=1,
            timebase_denominator=10,
            physical_time_s=None,
            physical_time_reason="Unverified fixture clock",
            frame_sha256=compute_frame_hash(image.tobytes(), decoder_name="fixture"),
            timing_mode="container_pts",
            is_timing_exact=True,
            clock_evidence="Authored exact fixture PTS",
            decoder_name="fixture",
            decoder_version="1",
            pixel_format="bgr24",
        )
        ok, png = cv2.imencode(".png", image)
        assert ok
        (directory / f"{frame.frame_id}.png").write_bytes(png.tobytes())
        observation = observation_record(
            PoseEstimationResult(
                {}, 1.0, 0, raw_keypoints={"origin": np.array([0.5, 0.5])}
            )
        )
        rows.append(
            {
                "frame": frame.to_dict(),
                "image": f"{frame.frame_id}.png",
                "observation": observation,
            }
        )
    observations = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (directory / "observations.jsonl").write_bytes(observations)
    receipt = {
        "schema_version": "historical-capture/1.0.0",
        "subject_id": "hogan",
        "qualification": "image_observations_only",
        "physical_time_verified": False,
        "source": source.to_dict(),
        "window_presentation_s": [0, 0.3],
        "observations_sha256": hashlib.sha256(observations).hexdigest(),
        "frame_count": 3,
        "detected_count": 3,
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return library.add_capture("exact-capture", "practice", directory)


def saved_parent(native_fit_case: Any, tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    library, source_path, payload = native_fit_case
    capture = imported_capture(library, tmp_path / "capture-source")
    from src.shared.python.workspace.necromatcher_review import CaptureReview

    with CaptureReview(library, capture.dataset_id) as review:
        frames = [review.frame(index)["frame"] for index in range(3)]
    free = (payload["coordinate_order"][3],)
    knots = np.array([0, 0.2])
    trajectory = CubicHermiteSplineTrajectory(knots, 1)
    coefficients = trajectory.pack(np.array([[0.1], [0.2]]), np.array([[0.01], [0.02]]))
    sha = hashlib.sha256(
        json.dumps(payload["provenance"]["native_definition"], allow_nan=False).encode()
    ).hexdigest()
    start = ImageSplineStart.from_coefficients(
        knots, coefficients, tuple(payload["coordinate_order"]), free, sha
    )
    q = np.zeros((3, 44))
    q[:, 3] = trajectory.evaluate(coefficients, np.array([0, 0.1, 0.2])).q[:, 0]
    payload.update(
        capture_id=capture.dataset_id,
        capture_hash=capture.metadata["hash"],
        frame_indices=[0, 1, 2],
        frames=frames,
        q=q.tolist(),
    )
    original = payload["evidence"]["original_fit"]
    original.update(
        coordinate_order=payload["coordinate_order"],
        free_coordinates=list(free),
        knot_times=knots.tolist(),
        spline_coefficients=coefficients.tolist(),
        spline_start=start.to_record(),
        frame_indices=[0, 2],
        source_times=[0, 0.2],
        q=q[[0, 2]].tolist(),
        config=asdict(ImageFitConfig()),
    )
    payload["provenance"]["request_options"] = {
        "coordinate_scales": [1.0] * 44,
        "unknown_visibility_weight": 0.5,
        "budget_wall_s": 600.0,
    }
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("parent", "practice", source_path)
    return library, payload


def build_hypothesis_case(
    native_fit_case: Any, tmp_path: Path
) -> tuple[Any, Any, dict[str, Any]]:
    library, parent = saved_parent(native_fit_case, tmp_path)
    from src.shared.python.workspace.necromatcher_capture_identity import (
        capture_identity,
    )
    from src.shared.python.workspace.necromatcher_hypothesis_contracts import (
        NativeHypothesisRequest,
    )

    identity = capture_identity(library, parent["capture_id"])
    definition = deepcopy(parent["provenance"]["native_definition"])
    definition["bodies"][1]["solids"][0]["mass_kg"] += 0.1
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(json.dumps(definition).encode())
    path = tmp_path / "candidate.xml"
    path.write_text(xml, encoding="utf-8")
    candidate = library.add_model(
        "candidate",
        "practice",
        path,
        engine="mujoco",
        dofs=tuple(parent["coordinate_order"]),
    )
    camera = deepcopy(parent["evidence"]["original_fit"]["camera"])
    camera["translation"][0] = 0.2
    record = {
        "schema_version": "necromatcher/native-hypothesis-request/1",
        "parents": {
            "source_fit_id": "parent",
            "source_fit_hash": library.load_asset("parent").metadata["hash"],
            "capture_id": parent["capture_id"],
            "capture_hash": parent["capture_hash"],
            "source_sha256": identity.source_sha256,
            "source_clock_sha256": identity.source_clock_sha256,
        },
        "model": {
            "model_id": "candidate",
            "model_hash": candidate.metadata["hash"],
            "definition": definition,
            "attachments": parent["evidence"]["original_fit"]["attachments"],
        },
        "camera": camera,
        "mapping": {
            "coordinate_order": parent["coordinate_order"],
            "coordinate_units": parent["coordinate_units"],
            "free_coordinates": parent["evidence"]["original_fit"]["free_coordinates"],
            "reference_pose": parent["q"][0],
        },
        "gauge": {
            "stature_m": 1.71,
            "world_origin": "authored_ground",
            "world_orientation": "model_world",
            "stature_source": "Synthetic conditional fixture",
        },
    }
    return library, NativeHypothesisRequest.from_record(record), parent
