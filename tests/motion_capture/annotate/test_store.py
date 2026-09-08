"""Sparse annotation store (#9798)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.motion_capture.annotate import AnnotationSet, Point, annotation_path
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit


def _set() -> AnnotationSet:
    return AnnotationSet("face_on", 320, 200, 10.0, annotator="tester")


def test_set_clear_skip_and_queries() -> None:
    s = _set()
    assert s.joints == JOINT_NAMES
    p = s.set_point(3, "left_wrist", 100.5, 40.0, note="clear view")
    assert p == Point(100.5, 40.0, "clear view")
    assert s.point(3, "left_wrist") == p and s.has_entry(3, "left_wrist")
    assert s.point(3, "right_wrist") is None
    s.skip(3, "right_wrist")
    assert s.is_skipped(3, "right_wrist") and s.point(3, "right_wrist") is None
    s.set_point(3, "right_wrist", 10, 10)  # a click clears the skip
    assert not s.is_skipped(3, "right_wrist")
    s.skip(3, "left_wrist")  # a skip clears the point
    assert s.point(3, "left_wrist") is None and s.is_skipped(3, "left_wrist")
    s.unskip(3, "left_wrist")
    assert not s.has_entry(3, "left_wrist")
    s.clear_point(3, "right_wrist")
    assert s.annotated_frames() == [] and s.count() == 0
    with pytest.raises(Exception, match="off image"):
        s.set_point(0, "nose", 320, 0)
    with pytest.raises(Exception, match="unknown joint"):
        s.set_point(0, "tail", 1, 1)
    with pytest.raises(Exception, match="frame"):
        s.set_point(-1, "nose", 1, 1)


def test_coverage_next_missing_and_interpolation() -> None:
    s = _set()
    s.set_point(0, "nose", 10, 10)
    s.set_point(10, "nose", 30, 50)
    s.skip(5, "nose")
    cov = s.coverage()["nose"]
    assert cov == {"annotated": 2, "skipped": 1, "first": 0, "last": 10}
    assert s.coverage()["left_hip"]["annotated"] == 0
    assert s.next_missing(0, "nose", stride=5) == 15
    assert s.next_missing(0, "left_hip") == 0
    mid = s.interpolate(5, "nose")
    assert mid is not None and mid.interpolated
    assert (mid.x_px, mid.y_px) == (20.0, 30.0)
    assert s.interpolate(0, "nose") == Point(10.0, 10.0)
    assert s.interpolate(12, "nose") is None  # no extrapolation
    assert s.annotated_frames() == [0, 10]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    s = _set()
    s.set_point(2, "nose", 1.5, 2.5)
    s.set_point(7, "left_wrist", 100, 120, note="n")
    s.skip(7, "right_wrist")
    path = annotation_path(tmp_path, "face_on")
    assert path == tmp_path / "annotations" / "face_on.json"
    s.save(path, base=tmp_path)
    payload = json.loads(path.read_text("utf-8"))
    assert payload["schema_version"] == "manual-annotations/1.0.0"
    assert payload["frames"]["7"]["left_wrist"] == {
        "x_px": 100.0,
        "y_px": 120.0,
        "note": "n",
    }
    assert payload["skipped"] == {"7": ["right_wrist"]}
    assert payload["provenance"]["parameters"]["annotator"] == "tester"
    created = payload["provenance"]["parameters"]["created_utc"]
    back = AnnotationSet.load(path)
    assert back == s and back.annotator == "tester" and back.created_utc == created
    back.set_point(9, "nose", 3, 3)
    back.save(path, base=tmp_path)
    again = json.loads(path.read_text("utf-8"))
    assert again["provenance"]["parameters"]["created_utc"] == created
    assert again["provenance"]["parameters"]["points"] == 3
    with pytest.raises(Exception, match="schema"):
        AnnotationSet.from_dict({"schema_version": "other/1"})
