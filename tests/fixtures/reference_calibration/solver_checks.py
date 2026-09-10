"""Independent OpenCV projections qualify the consumer's transform adapter."""

from dataclasses import asdict
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from src.motion_capture.rig.capture_notes import CaptureNotes
from src.motion_capture.rig.documents import write_document
from src.tools.capture_rig.calibration_profiles import CalibrationProfile, CameraSetup
from src.tools.capture_rig.reference_calibration.frames import archive_frame
from src.tools.capture_rig.reference_calibration.session import (
    CameraSelection,
    ReferenceSession,
)
from src.tools.capture_rig.reference_calibration.solver import (
    solve_reference,
    accept_result,
)


def make_request(root: Path):
    repository = Path(__file__).resolve().parents[3]
    fixture = runpy.run_path(
        str(
            repository
            / "vendor/ud-tools/tests/shared/python/sidekick/lab/mocap/test_reference_placements.py"
        )
    )["fixture"]
    target, cameras, observations, _, camera_poses, placements = fixture()
    selected = []
    for view, camera in cameras.items():
        lens = camera.intrinsics
        path = root / f"intrinsics-{view}.json"
        write_document(
            path,
            [
                {
                    "camera_id": view,
                    "matrix": [[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1]],
                    "distortion": list(lens.distortion.coefficients),
                    "image_size_px": list(lens.resolution_px),
                    "rms_px": 0.2,
                    "frames_used": 15,
                    "frames_without_board": 0,
                    "board": {},
                }
            ],
        )
        setup = CameraSetup(
            camera_identity=f"serial-{view}",
            lens="Fixed Lens",
            zoom="Mark 1",
            focus="Mark 2",
            sensor_mode="Full Sensor",
            image_size_px=lens.resolution_px,
        )
        profile = CalibrationProfile.capture(
            name="Synthetic Lens", setup=setup, camera_id=view, intrinsics_path=path
        )
        selected.append(CameraSelection(view=view, setup=setup, profile=profile))
    session = ReferenceSession.model_validate(
        {
            "capture_id": "synthetic-reference",
            "title": "Independent Projection Test",
            "scene_id": "fixed-camera-scene",
            "cameras": selected,
            "targets": [asdict(target)],
        }
    )
    write_document(
        root / "capture_notes.json",
        CaptureNotes(capture_id=session.capture_id, title=session.title).model_dump(
            mode="json"
        ),
    )
    samples = []
    for observation in observations:
        camera = next(item for item in selected if item.view == observation.camera_key)
        width, height = camera.setup.image_size_px
        image = archive_frame(
            root,
            np.zeros((height, width, 3), dtype=np.uint8),
            capture_id=session.capture_id,
            view=camera.view,
            frame_index=observation.frame_sequence,
            timestamp_s=observation.timestamp_ns / 1_000_000_000,
            source_label="Independent Synthetic Projection",
        )
        record = asdict(observation)
        record["profile_id"] = camera.profile_key
        samples.append(
            {
                "observation": record,
                "camera_signature": camera.signature(session.scene_id),
                "source_frame": image.path,
                "source_sha256": image.sha256,
            }
        )
    session = session.revise(samples=samples)
    request = {
        "workspace": str(root),
        "session": session.model_dump(mode="json"),
        "parameters": {
            "settings_confirmed": True,
            "anchor_confirmed": True,
            "anchor_placement_id": "p0",
            "anchor_translation_m": [0.0, 0.0, 0.0],
        },
    }
    return request, camera_poses, placements


def test_solver_preserves_distortion_and_converts_world_from_camera(tmp_path):
    request, camera_poses, placements = make_request(tmp_path)
    solved = solve_reference(request)
    result = solved["result"]
    assert (tmp_path / solved["result_path"]).is_file()
    assert result["operator_reviewed"] is False
    assert (
        "reference_placements.py"
        in result["algorithm_evidence"]["provider_source_sha256"]
    )
    assert "solver.py" in result["algorithm_evidence"]["consumer_source_sha256"]
    assert len(result["residuals"]) == 12
    assert max(row["max_error_px"] for row in result["residuals"]) < 1e-5
    anchor_rotation, anchor_translation = placements["p0"]
    for record in result["cameras"]:
        cr, ct = camera_poses[record["camera_id"]]
        expected_camera_from_world = cr.as_matrix() @ anchor_rotation.as_matrix()
        expected_translation = cr.apply(anchor_translation) + ct
        assert np.asarray(
            record["extrinsics"]["rotation_world_from_camera"]
        ) == pytest.approx(expected_camera_from_world.T, abs=1e-6)
        assert np.asarray(
            record["extrinsics"]["translation_world_from_camera_m"]
        ) == pytest.approx(
            -expected_camera_from_world.T @ expected_translation, abs=1e-6
        )
        assert record["intrinsics"]["distortion"] == [0.04, -0.02, 0.001, -0.002, 0.0]
    from src.tools.capture_rig.calibration_profiles import validate_profile_set

    request["parameters"].update(
        result_id=result["layout_id"],
        reviewed=True,
        result_sha256=solved["result_sha256"],
    )
    from src.tools.capture_rig.reference_calibration.results import load_result

    restored = load_result(request)
    assert restored == json.loads(json.dumps(solved))
    request["parameters"]["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed since review"):
        accept_result(request)
    request["parameters"]["result_sha256"] = restored["result_sha256"]
    reviewed = accept_result(request)
    assert reviewed["result"]["operator_reviewed"] is True
    expected = {
        camera["view"]: (
            camera["setup"]["camera_identity"],
            tuple(camera["setup"]["image_size_px"]),
        )
        for camera in request["session"]["cameras"]
    }
    accepted_bytes = (tmp_path / reviewed["result_path"]).read_bytes()
    validate_profile_set(accepted_bytes, expected, capture_root=tmp_path)
    request["parameters"]["anchor_translation_m"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="anchor changed"):
        accept_result(request)
    revision_path = (
        tmp_path / "reference_calibration" / f"{result['reference_revision_id']}.json"
    )
    revision_path.write_bytes(b"changed observations")
    with pytest.raises(ValueError, match="observations changed"):
        validate_profile_set(accepted_bytes, expected, capture_root=tmp_path)


@pytest.mark.parametrize("confirmation", ["settings_confirmed", "anchor_confirmed"])
def test_solver_requires_fresh_operator_confirmation(tmp_path, confirmation):
    request, _, _ = make_request(tmp_path)
    request["parameters"][confirmation] = False
    with pytest.raises(ValueError, match="Confirm"):
        solve_reference(request)
    assert not (tmp_path / "reference_calibration/results").exists()


def test_reviewed_layout_projects_independent_points_through_overlay_consumer(tmp_path):
    """Persisted calibration reaches reference overlays with the same lens and frame."""
    import cv2

    from src.motion_capture.reconstruct.pipeline import start_cameras_from
    from src.motion_capture.reference.registration import project_reference_to_camera

    request, camera_poses, placements = make_request(tmp_path)
    solved = solve_reference(request)
    request["parameters"].update(
        result_id=solved["result"]["layout_id"],
        reviewed=True,
        result_sha256=solved["result_sha256"],
    )
    accepted = accept_result(request)
    path = tmp_path / accepted["result_path"]
    original = path.read_bytes()
    from reuse_checks import _write_target_bundle

    expected_views = {
        item["view"]: (
            item["setup"]["camera_identity"],
            tuple(item["setup"]["image_size_px"]),
        )
        for item in accepted["result"]["profile_selections"]
    }
    _write_target_bundle(tmp_path, expected_views)
    cameras = start_cameras_from(path, capture_root=tmp_path)
    # New non-planar points are not reference corners used by the camera solve.
    points = np.array([[0.02, 0.08, 0.12], [0.18, -0.04, 0.22], [-0.08, 0.14, 0.05]])
    anchor_rotation, anchor_translation = placements["p0"]
    world = anchor_rotation.apply(points) + anchor_translation
    for camera in cameras:
        rotation, translation = camera_poses[camera.camera_id]
        expected, _ = cv2.projectPoints(
            world,
            rotation.as_rotvec(),
            translation,
            camera.matrix,
            np.array([0.04, -0.02, 0.001, -0.002, 0.0]),
        )
        projected, visible = project_reference_to_camera(
            points, np.ones(len(points), dtype=bool), camera, clip_image=False
        )
        assert visible.all()
        np.testing.assert_allclose(
            projected, expected.reshape(-1, 2), atol=1e-5, rtol=0
        )
    assert path.read_bytes() == original
