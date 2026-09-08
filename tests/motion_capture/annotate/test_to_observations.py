"""Manual clicks as an observation set; corrections over a detector set (#9801, #9803)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.motion_capture.annotate import AnnotationSet, annotation_path
from src.motion_capture.annotate.to_observations import (
    merge,
    to_view_observations,
    write_observation_set,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.motion_capture.rig import __main__ as rig_cli
from src.tools.capture_rig import commands
from tests.tools.capture_rig.test_core import _bundle

pytestmark = pytest.mark.unit

K = len(JOINT_NAMES)


def _store() -> AnnotationSet:
    s = AnnotationSet("cam_a", 320, 200, 10.0, annotator="tester")
    s.set_point(0, "nose", 10, 20)
    s.set_point(0, "left_wrist", 30, 40)
    s.set_point(4, "nose", 12, 22)
    s.skip(4, "left_wrist")
    s.skip(6, "nose")  # only a skip: no row
    return s


def _detector(frames: int = 8) -> dict:
    rows = []
    for f in range(frames):
        rows.append(
            {
                "camera_id": "cam_a",
                "time_s": f / 10.0,
                "keypoints_px": [[100.0 + f, 50.0]] * K,
                "confidence": [0.9] * K,
            }
        )
    return {
        "schema_version": "view-observations/1.0.0",
        "view": "cam_a",
        "identity": "1",
        "camera_id": "cam_a",
        "fps": 10.0,
        "width": 320,
        "height": 200,
        "frames_total": frames,
        "frames_with_pose": frames,
        "detector_layout": {
            "name": "mediapipe_15",
            "keypoint_names": list(JOINT_NAMES),
        },
        "frames": rows,
        "provenance": {"estimator": "mediapipe"},
    }


def test_manual_view_observations_are_sparse_and_confident() -> None:
    view = to_view_observations(_store())
    assert view.detector_layout["keypoint_names"] == list(JOINT_NAMES)
    assert view.frames_with_pose == 2 and view.frames_total == 5
    first = view.frames[0]
    assert first["time_s"] == 0.0
    assert first["confidence"][JOINT_NAMES.index("nose")] == 1.0
    assert first["confidence"][JOINT_NAMES.index("left_wrist")] == 1.0
    assert sum(first["confidence"]) == 2.0
    assert first["keypoints_px"][JOINT_NAMES.index("nose")] == [10.0, 20.0]
    assert view.provenance["estimator"] == "manual" and view.provenance["points"] == 3
    with pytest.raises(Exception, match="reconstruct joints"):
        to_view_observations(AnnotationSet("v", 10, 10, 1.0, joints=("a",)))


def test_merge_replaces_rejects_and_adds() -> None:
    store = _store()
    store.set_point(20, "nose", 5, 5)  # beyond the detector's frames: added
    merged, counts = merge(store, _detector())
    assert counts == {"replaced": 3, "rejected": 2, "added": 1}
    by_frame = {round(r["time_s"] * 10): r for r in merged.frames}
    nose = JOINT_NAMES.index("nose")
    wrist = JOINT_NAMES.index("left_wrist")
    assert by_frame[0]["keypoints_px"][nose] == [10.0, 20.0]
    assert by_frame[0]["confidence"][nose] == 1.0
    assert by_frame[0]["keypoints_px"][1] == [100.0, 50.0], "untouched joints kept"
    assert by_frame[4]["confidence"][wrist] == 0.0, "skip rejects the detector point"
    assert by_frame[6]["confidence"][nose] == 0.0
    assert (
        by_frame[20]["confidence"][nose] == 1.0
        and sum(by_frame[20]["confidence"]) == 1.0
    )
    assert merged.provenance["estimator"] == "mediapipe+manual"
    assert merged.provenance["corrections"] == counts
    assert merged.frames_with_pose == 9


def test_cli_writes_manual_and_edited_sets(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    store = _store()
    store.save(annotation_path(root, "cam_a"), base=root)
    (root / "observations").mkdir()
    (root / "observations" / "cam_a.json").write_text(json.dumps(_detector()), "utf-8")
    argv = commands.annotations_command(root, views=("cam_a",))
    assert rig_cli.main(argv[3:]) == 0
    manual = json.loads(
        (root / "observations_manual" / "cam_a.json").read_text("utf-8")
    )
    assert manual["provenance"]["estimator"] == "manual"
    assert manual["provenance"]["parameters"]["annotator"] == "tester"
    assert (root / "observations_manual" / "observations.json").is_file()
    argv = commands.annotations_command(root, merge_with="observations")
    assert argv[-2:] == ["--merge-with", "observations"]
    assert rig_cli.main(argv[3:]) == 0
    edited = json.loads(
        (root / "observations_edited" / "cam_a.json").read_text("utf-8")
    )
    assert edited["provenance"]["corrections"] == {
        "replaced": 3,
        "rejected": 2,
        "added": 0,
    }
    index = json.loads(
        (root / "observations_edited" / "observations.json").read_text("utf-8")
    )
    assert (
        index["views"][0]["view"] == "cam_a"
        and index["provenance"]["merge_with"] == "observations"
    )
    with pytest.raises(SystemExit, match="no observations_x"):
        rig_cli.main(
            [
                "annotations-to-observations",
                "--session",
                str(root),
                "--merge-with",
                "observations_x",
            ]
        )


def test_write_observation_set_needs_a_view(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="at least one view"):
        write_observation_set(
            tmp_path, "observations_manual", {}, plan_name="p", inputs=[], parameters={}
        )
