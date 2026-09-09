"""Provenance stamping and lineage walking (#9792)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.motion_capture.provenance import (
    lineage,
    lineage_markdown,
    sha256_of,
    stamp,
    write_json,
)

pytestmark = pytest.mark.unit


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_stamp_adds_provenance_and_schema_version(tmp_path: Path) -> None:
    video = tmp_path / "face_on.avi"
    video.write_bytes(b"\x00" * 100)
    prior = _write(tmp_path / "recordings.json", {"schema_version": "rec/1.0.0"})
    payload = {"rms_px": 1.0}
    out = stamp(
        payload,
        schema_version="x/1.0.0",
        module="tests.provenance",
        inputs=[video],
        parameters={"k": 1},
        derived_from=[prior],
        base=tmp_path,
    )
    assert payload == {"rms_px": 1.0}, "stamp must not mutate its argument"
    assert out["schema_version"] == "x/1.0.0"
    prov = out["provenance"]
    assert prov["created_utc"].endswith("Z") and len(prov["created_utc"]) == 20
    assert prov["generated_by"]["package"] == "upstreamdrift"
    assert prov["generated_by"]["module"] == "tests.provenance"
    assert prov["inputs"] == [
        {
            "path": "face_on.avi",
            "sha256": sha256_of(video),
            "bytes": 100,
            "schema_version": None,
        }
    ]
    assert prov["parameters"] == {"k": 1}
    assert prov["derived_from"] == ["recordings.json"]
    # An existing schema_version is kept.
    kept = stamp({"schema_version": "y/2"}, schema_version="x/1", module="m")
    assert kept["schema_version"] == "y/2"


def test_sha256_is_a_content_hash(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"abc" * 1000)
    b.write_bytes(b"abc" * 1000)
    assert sha256_of(a) == sha256_of(b)
    b.write_bytes(b"abc" * 999 + b"abd")
    assert sha256_of(a) != sha256_of(b)
    with pytest.raises(Exception, match="must exist"):
        sha256_of(tmp_path / "missing.bin")


def test_lineage_walks_derived_from_and_inputs(tmp_path: Path) -> None:
    video = tmp_path / "cam.avi"
    video.write_bytes(b"v")
    a = write_json(
        tmp_path / "observations" / "cam.json",
        stamp(
            {"view": "cam"},
            schema_version="view-observations/1.0.0",
            module="ingest",
            inputs=[video],
            parameters={"estimator": "mediapipe"},
            base=tmp_path,
        ),
    )
    b = write_json(
        tmp_path / "reconstruct" / "session_reconstruction.json",
        stamp(
            {},
            schema_version="session-reconstruction/1.0.0",
            module="pipeline",
            inputs=[a],
            derived_from=[a],
            base=tmp_path,
        ),
    )
    c = write_json(
        tmp_path / "model" / "joint_angles.json",
        stamp(
            {},
            schema_version="model-fit/1.0.0",
            module="model",
            derived_from=[b],
            parameters={"model": "golfer"},
            base=tmp_path,
        ),
    )
    records = lineage(c, base=tmp_path)
    paths = [r.path for r in records]
    assert paths == [
        "model/joint_angles.json",
        "reconstruct/session_reconstruction.json",
        "observations/cam.json",
        "cam.avi",
    ]
    assert records[-1].is_leaf and not records[0].is_leaf
    assert records[2].parameters == {"estimator": "mediapipe"}
    text = lineage_markdown(records)
    assert text.count("\n") == 4 and "estimator=mediapipe" in text
    assert "`cam.avi` (leaf)" in text


def test_lineage_refuses_cycles(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, {"schema_version": "a/1", "provenance": {"derived_from": ["b.json"]}})
    _write(b, {"schema_version": "b/1", "provenance": {"derived_from": ["a.json"]}})
    with pytest.raises(ValueError, match="cycle"):
        lineage(a, base=tmp_path)


def test_stamp_preconditions(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="schema_version"):
        stamp({}, schema_version="", module="m")
    with pytest.raises(Exception, match="must exist"):
        stamp({}, schema_version="x/1", module="m", inputs=[tmp_path / "nope"])
