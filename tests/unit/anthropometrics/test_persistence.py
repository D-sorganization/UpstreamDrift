"""Unit tests for :mod:`anthropometrics.persistence`.

Round-trips a 16-segment :class:`SubjectAnthropometrics`, exercises
the schema-version error path, and confirms validation invariants
(triangle inequality, positive-definite inertia tensors) survive
the save/load cycle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anthropometrics import (
    SCHEMA_VERSION,
    SegmentProperties,
    Sex,
    SubjectAnthropometrics,
    default_subjects_dir,
    load_subject,
    save_subject,
)
from anthropometrics.persistence import _segment_from_dict, _segment_to_dict


# Canonical 16-segment list used by downstream estimators.
SIXTEEN_SEGMENTS: tuple[str, ...] = (
    "head",
    "torso",
    "pelvis",
    "upper_arm_left",
    "upper_arm_right",
    "forearm_left",
    "forearm_right",
    "hand_left",
    "hand_right",
    "thigh_left",
    "thigh_right",
    "shank_left",
    "shank_right",
    "foot_left",
    "foot_right",
    "club",
)


# --------------------------------------------------------------------------- #
# Builders.                                                                   #
# --------------------------------------------------------------------------- #
def _diag_inertia(ix: float, iy: float, iz: float) -> np.ndarray:
    return np.diag([ix, iy, iz]).astype(float)


def _make_segment(name: str = "upper_arm_left", **overrides: Any) -> SegmentProperties:
    defaults: dict[str, Any] = {
        "name": name,
        "body_part_id": name.split("_")[0],
        "length_m": 0.30,
        "proximal_marker": "L_SHO",
        "distal_marker": "L_ELB",
        "mass_kg": 2.0,
        "com_xyz_m": np.array([0.15, 0.0, 0.0]),
        "inertia_tensor": _diag_inertia(0.02, 0.02, 0.005),
        "source_method": "de_leva",
        "source_subject_height_m": 1.80,
        "source_subject_mass_kg": 75.0,
    }
    defaults.update(overrides)
    return SegmentProperties(**defaults)


def _make_full_subject() -> SubjectAnthropometrics:
    """Subject with all 16 canonical segments populated."""
    segments = tuple((name, _make_segment(name=name)) for name in SIXTEEN_SEGMENTS)
    return SubjectAnthropometrics(
        subject_id="SUBJ001",
        height_m=1.80,
        mass_kg=75.0,
        segments=segments,
        source_method="de_leva",
        age_years=32.5,
        sex=Sex.MALE.value,
    )


# --------------------------------------------------------------------------- #
# Round-trip.                                                                 #
# --------------------------------------------------------------------------- #
def test_round_trip_full_sixteen_segments(tmp_path: Path) -> None:
    """A 16-segment subject survives save -> load with full fidelity."""
    original = _make_full_subject()
    path = tmp_path / "subj.json"

    save_subject(original, path)
    loaded = load_subject(path)

    assert loaded.subject_id == original.subject_id
    assert loaded.height_m == pytest.approx(original.height_m)
    assert loaded.mass_kg == pytest.approx(original.mass_kg)
    assert loaded.source_method == original.source_method
    assert loaded.age_years == pytest.approx(original.age_years)
    assert loaded.sex == original.sex
    assert len(loaded.segments) == 16
    assert tuple(name for name, _ in loaded.segments) == SIXTEEN_SEGMENTS

    for (orig_name, orig_seg), (load_name, load_seg) in zip(
        original.segments, loaded.segments, strict=True
    ):
        assert orig_name == load_name
        assert load_seg.name == orig_seg.name
        assert load_seg.body_part_id == orig_seg.body_part_id
        assert load_seg.length_m == pytest.approx(orig_seg.length_m)
        assert load_seg.mass_kg == pytest.approx(orig_seg.mass_kg)
        assert load_seg.proximal_marker == orig_seg.proximal_marker
        assert load_seg.distal_marker == orig_seg.distal_marker
        assert load_seg.source_method == orig_seg.source_method
        np.testing.assert_allclose(load_seg.com_xyz_m, orig_seg.com_xyz_m)
        np.testing.assert_allclose(load_seg.inertia_tensor, orig_seg.inertia_tensor)


def test_round_trip_with_optional_fields_omitted(tmp_path: Path) -> None:
    """``age_years=None`` and unspecified sex round-trip cleanly."""
    seg = _make_segment(proximal_marker=None, distal_marker=None)
    original = SubjectAnthropometrics(
        subject_id="SUBJ002",
        height_m=1.70,
        mass_kg=65.0,
        segments=((seg.name, seg),),
        source_method="dempster",
    )
    path = tmp_path / "minimal.json"

    save_subject(original, path)
    loaded = load_subject(path)

    assert loaded.age_years is None
    assert loaded.sex == Sex.UNSPECIFIED.value
    assert loaded.segments[0][1].proximal_marker is None
    assert loaded.segments[0][1].distal_marker is None


# --------------------------------------------------------------------------- #
# On-disk file shape.                                                         #
# --------------------------------------------------------------------------- #
def test_save_writes_schema_version(tmp_path: Path) -> None:
    """The on-disk JSON includes ``schema_version`` at top level."""
    save_subject(_make_full_subject(), tmp_path / "out.json")
    raw = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION == 1


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """Missing parents are created on demand."""
    target = tmp_path / "nested" / "more" / "subj.json"
    save_subject(_make_full_subject(), target)
    assert target.exists()


def test_save_rejects_non_record() -> None:
    """``save_subject`` rejects non-:class:`SubjectAnthropometrics` inputs."""
    with pytest.raises(TypeError, match="SubjectAnthropometrics"):
        save_subject({"bogus": True}, Path("ignored.json"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Validation invariants persist across save / load.                           #
# --------------------------------------------------------------------------- #
def test_load_rejects_triangle_inequality_violation(tmp_path: Path) -> None:
    """Tampering an inertia tensor to violate triangle inequality fails on load."""
    path = tmp_path / "bad_triangle.json"
    save_subject(_make_full_subject(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    # eigenvalues (1, 1, 3) violate triangle inequality
    payload["segments"][0]["properties"]["inertia_tensor"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="triangle inequality"):
        load_subject(path)


def test_load_rejects_non_positive_definite_inertia(tmp_path: Path) -> None:
    """A negative eigenvalue in the persisted tensor fails on load."""
    path = tmp_path / "bad_pd.json"
    save_subject(_make_full_subject(), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["segments"][0]["properties"]["inertia_tensor"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -0.5],
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="positive-definite"):
        load_subject(path)


# --------------------------------------------------------------------------- #
# Schema-version handling.                                                    #
# --------------------------------------------------------------------------- #
def test_load_rejects_missing_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "no_schema.json"
    payload = {"subject_id": "X", "segments": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_subject(path)


def test_load_rejects_future_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    payload = {"schema_version": 999, "subject_id": "X"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_subject(path)


def test_load_rejects_old_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    payload = {"schema_version": 0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported schema_version"):
        load_subject(path)


# --------------------------------------------------------------------------- #
# Error paths.                                                                #
# --------------------------------------------------------------------------- #
def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_subject(tmp_path / "does_not_exist.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.json"
    path.write_text("not json {", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_subject(path)


def test_load_top_level_array_rejected(tmp_path: Path) -> None:
    path = tmp_path / "arr.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_subject(path)


def test_load_missing_required_subject_keys(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    payload = {"schema_version": SCHEMA_VERSION, "subject_id": "X"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_subject(path)


def test_load_segments_must_be_array(tmp_path: Path) -> None:
    path = tmp_path / "bad_segments.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": "X",
        "height_m": 1.8,
        "mass_kg": 75.0,
        "source_method": "de_leva",
        "segments": {"not": "an array"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_subject(path)


def test_load_segment_entry_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad_entry.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": "X",
        "height_m": 1.8,
        "mass_kg": 75.0,
        "source_method": "de_leva",
        "segments": [{"name": "head"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="'name' and 'properties'"):
        load_subject(path)


def test_segment_from_dict_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        _segment_from_dict("not a dict")


def test_segment_from_dict_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        _segment_from_dict({"name": "head"})


def test_segment_to_dict_round_trip_via_helpers() -> None:
    seg = _make_segment()
    payload = _segment_to_dict(seg)
    restored = _segment_from_dict(payload)
    assert restored.name == seg.name
    np.testing.assert_allclose(restored.inertia_tensor, seg.inertia_tensor)


# --------------------------------------------------------------------------- #
# Default-dir helper.                                                         #
# --------------------------------------------------------------------------- #
def test_default_subjects_dir_resolves_to_home() -> None:
    """The helper points at ``~/.golf_modeling_suite/subjects/``."""
    expected = Path.home() / ".golf_modeling_suite" / "subjects"
    assert default_subjects_dir() == expected


def test_default_subjects_dir_does_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper is pure — calling it does not touch the filesystem."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    # Path.home() caches nothing; both Linux and Windows respect the env.
    result = default_subjects_dir()
    # parent should not be auto-created by the helper.
    assert not result.exists()


def test_save_to_default_dir_creates_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_subject`` honours :func:`default_subjects_dir` by creating it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    target = default_subjects_dir() / "auto.json"
    save_subject(_make_full_subject(), target)
    assert target.exists()
    assert target.parent == default_subjects_dir()
