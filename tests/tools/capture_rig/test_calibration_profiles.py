"""Calibration reuse must follow the actual camera and optical configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.capture_rig.calibration_profiles import (
    CameraSetup,
    CalibrationProfile,
    ProfileHistory,
    check_profile,
    save_profile,
    ProfileAssignment,
    write_profile_set,
)

pytestmark = pytest.mark.unit


def setup(**changes: object) -> CameraSetup:
    return CameraSetup.model_validate(
        {
            "camera_identity": "serial-123",
            "lens": "ELP lens A",
            "zoom": "ring mark 2",
            "focus": "ball marker",
            "image_size_px": (1920, 1200),
            "sensor_mode": "full sensor",
        }
        | changes
    )


def calibration(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "camera_id": "face-on",
                    "matrix": [[900, 0, 960], [0, 900, 600], [0, 0, 1]],
                    "distortion": [0.0] * 5,
                    "image_size_px": [1920, 1200],
                    "rms_px": 0.2,
                    "frames_used": 15,
                    "frames_without_board": 0,
                    "board": {},
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def profile(tmp_path: Path) -> CalibrationProfile:
    return CalibrationProfile.capture(
        name="Wide view",
        setup=setup(),
        camera_id="face-on",
        intrinsics_path=calibration(tmp_path / "intrinsics.json"),
    )


def test_manual_lens_settings_must_be_confirmed_for_this_capture(
    tmp_path: Path,
) -> None:
    saved = profile(tmp_path)
    assert not check_profile(saved, setup(), settings_confirmed=False).compatible
    accepted = check_profile(saved, setup(), settings_confirmed=True)
    assert accepted.compatible and not accepted.reasons


@pytest.mark.parametrize(
    "changes",
    [
        {"camera_identity": "other-unit"},
        {"zoom": "ring mark 3"},
        {"focus": "infinity"},
        {"lens": "replacement"},
        {"image_size_px": (1280, 720)},
        {"sensor_mode": "center crop"},
    ],
)
def test_changed_camera_or_optical_configuration_requires_recalibration(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    result = check_profile(profile(tmp_path), setup(**changes), settings_confirmed=True)
    assert not result.compatible
    assert result.reasons


def test_replaced_calibration_file_does_not_rewrite_existing_profile(
    tmp_path: Path,
) -> None:
    saved = profile(tmp_path)
    Path(saved.intrinsics_path).write_text("[]", encoding="utf-8")
    result = check_profile(saved, setup(), settings_confirmed=True)
    assert not result.compatible
    assert any("changed" in reason.lower() for reason in result.reasons)


def test_missing_calibration_has_actionable_reason(tmp_path: Path) -> None:
    saved = profile(tmp_path)
    Path(saved.intrinsics_path).unlink()
    result = check_profile(saved, setup(), settings_confirmed=True)
    assert not result.compatible
    assert any("unavailable" in reason.lower() for reason in result.reasons)


def test_recalibration_history_preserves_old_revision_and_can_restore_it(
    tmp_path: Path,
) -> None:
    first = profile(tmp_path)
    second = CalibrationProfile.capture(
        name="Recalibrated",
        setup=setup(),
        camera_id="face-on",
        intrinsics_path=calibration(tmp_path / "new-intrinsics.json"),
    )
    path = tmp_path / "profiles.json"
    first = save_profile(path, first)
    second = save_profile(path, second)
    history = ProfileHistory.model_validate_json(path.read_text(encoding="utf-8"))
    assert history.profiles == (first, second)
    assert first.profile_id != second.profile_id
    assert history.active_profile_id == second.profile_id
    restored = history.select(first.profile_id)
    assert restored.active_profile_id == first.profile_id
    assert restored.profiles == history.profiles
    with pytest.raises(ValueError, match="Unknown"):
        history.select("not-a-profile")


@pytest.mark.parametrize(
    "field", ["camera_identity", "lens", "zoom", "focus", "sensor_mode"]
)
def test_unknown_or_blank_settings_cannot_form_a_qualified_profile(field: str) -> None:
    with pytest.raises(ValueError):
        setup(**{field: "  "})


@pytest.mark.parametrize("size", [(0, 1200), (1920, -1), (1.5, 1200)])
def test_invalid_image_dimensions_are_rejected(size: tuple[float, int]) -> None:
    with pytest.raises(ValueError):
        setup(image_size_px=size)


def test_capture_rejects_wrong_view_or_resolution(tmp_path: Path) -> None:
    path = calibration(tmp_path / "intrinsics.json")
    with pytest.raises(ValueError, match="camera"):
        CalibrationProfile.capture(
            name="Wrong view", setup=setup(), camera_id="other", intrinsics_path=path
        )
    with pytest.raises(ValueError, match="size"):
        CalibrationProfile.capture(
            name="Wrong size",
            setup=setup(image_size_px=(640, 480)),
            camera_id="face-on",
            intrinsics_path=path,
        )


def test_bad_quality_cannot_be_promoted_to_usable_profile(tmp_path: Path) -> None:
    path = calibration(tmp_path / "intrinsics.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    records[0]["rms_px"] = 15.0
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="quality"):
        CalibrationProfile.capture(
            name="Bad fit", setup=setup(), camera_id="face-on", intrinsics_path=path
        )


def test_existing_history_is_not_overwritten_when_it_is_malformed(
    tmp_path: Path,
) -> None:
    saved = profile(tmp_path)
    path = tmp_path / "profiles.json"
    path.write_text('{"schema_version":"future/3"}', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_profile(path, saved)
    assert path.read_bytes() == before


def test_saved_revision_survives_recalibration_overwriting_the_original(
    tmp_path: Path,
) -> None:
    original = profile(tmp_path)
    history_path = tmp_path / "profiles.json"
    saved = save_profile(history_path, original)
    Path(original.intrinsics_path).write_text("[]", encoding="utf-8")
    history = ProfileHistory.model_validate_json(history_path.read_bytes())
    assert saved == history.profiles[0]
    assert check_profile(saved, setup(), settings_confirmed=True).compatible


def test_resaving_is_idempotent_but_cannot_rewrite_a_revision(tmp_path: Path) -> None:
    original = profile(tmp_path)
    path = tmp_path / "profiles.json"
    saved = save_profile(path, original)
    assert save_profile(path, original) == saved
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        save_profile(path, original.model_copy(update={"name": "Altered history"}))
    assert path.read_bytes() == before
    assert len(ProfileHistory.model_validate_json(before).profiles) == 1


def test_corrupted_saved_artifact_is_not_silently_replaced(tmp_path: Path) -> None:
    original = profile(tmp_path)
    path = tmp_path / "profiles.json"
    saved = save_profile(path, original)
    Path(saved.intrinsics_path).write_text("corrupt", encoding="utf-8")
    repeated = CalibrationProfile.capture(
        name="Repeated",
        setup=setup(),
        camera_id="face-on",
        intrinsics_path=Path(original.intrinsics_path),
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        save_profile(path, repeated)
    assert path.read_bytes() == before
    assert not check_profile(saved, setup(), settings_confirmed=True).compatible


@pytest.mark.parametrize("matrix", [[], [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]])
def test_invalid_camera_geometry_is_rejected_even_with_low_rms(
    tmp_path: Path,
    matrix: list[list[float]],
) -> None:
    path = calibration(tmp_path / "intrinsics.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    records[0]["matrix"] = matrix
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix"):
        CalibrationProfile.capture(
            name="Malformed",
            setup=setup(),
            camera_id="face-on",
            intrinsics_path=path,
        )


def test_profile_set_requires_every_view_and_exports_only_verified_cameras(
    tmp_path: Path,
) -> None:
    saved = save_profile(tmp_path / "profiles.json", profile(tmp_path))
    assignment = ProfileAssignment("renamed-view", saved, setup(), True)
    output = tmp_path / "intrinsics-selected.json"
    with pytest.raises(ValueError, match="every view"):
        write_profile_set(
            output, [assignment], required_views=("renamed-view", "missing")
        )
    assert not output.exists()
    write_profile_set(output, [assignment], required_views=("renamed-view",))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [camera["camera_id"] for camera in payload["cameras"]] == ["renamed-view"]
    assert payload["profile_selections"][0]["profile_id"] == saved.profile_id
    assert payload["profile_selections"][0]["setup"]["camera_identity"] == "serial-123"


def test_profile_set_rejects_incompatible_or_duplicate_physical_cameras(
    tmp_path: Path,
) -> None:
    saved = profile(tmp_path)
    first = ProfileAssignment("front", saved, setup(), True)
    other = ProfileAssignment("side", saved, setup(), True)
    with pytest.raises(ValueError, match="physical camera"):
        write_profile_set(
            tmp_path / "x.json", [first, other], required_views=("front", "side")
        )
    with pytest.raises(ValueError, match="Confirm"):
        write_profile_set(
            tmp_path / "x.json",
            [ProfileAssignment("front", saved, setup(), False)],
            required_views=("front",),
        )
