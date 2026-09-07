"""Rig plan validation, persistence, and topology matching."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.motion_capture.rig.plan import (
    PLAN_SCHEMA_VERSION,
    CameraBinding,
    CameraControls,
    CaptureMode,
    RigPlan,
    check_plan,
)
from src.motion_capture.rig.topology import CameraLocation

pytestmark = pytest.mark.unit


def _cam(identity: str, port: int) -> CameraLocation:
    return CameraLocation(
        camera=f"USB\\VID_32E4&PID_5234&MI_00\\{identity}",
        composite=None,
        serial=identity if "&" not in identity else None,
        identity=identity,
        root_hub="USB\\ROOT_HUB30\\9&1d291187&0&0",
        root_port=port,
        host="PCI\\VEN_8086&DEV_15C1",
        hub_depth=0,
    )


def _plan() -> RigPlan:
    return RigPlan(
        name="three-view",
        cameras=(
            CameraBinding(view="face_on", serial="2605160001"),
            CameraBinding(
                view="down_line", serial="2601240001", mode=CaptureMode(fps=30)
            ),
            CameraBinding(view="overhead", port_path="path_D-D35A8F7-0-0000"),
        ),
    )


def test_capture_mode_defaults_and_fourcc_normalisation() -> None:
    mode = CaptureMode(fourcc="mjpg")
    assert (mode.width, mode.height, mode.fps, mode.fourcc) == (1920, 1200, 60, "MJPG")
    with pytest.raises(ValidationError):
        CaptureMode(fps=0)


def test_controls_as_overrides_skips_unset_and_casts_bool() -> None:
    assert CameraControls().as_overrides() == {}
    assert CameraControls(exposure=-6, auto_exposure=False).as_overrides() == {
        "exposure": -6.0,
        "auto_exposure": 0.0,
    }


def test_binding_requires_an_identity() -> None:
    with pytest.raises(ValidationError, match="serial or a port_path"):
        CameraBinding(view="x")
    assert CameraBinding(view="x", port_path="path_a").identity == "path_a"
    assert CameraBinding(view="x", serial="s", port_path="p").identity == "s"


def test_plan_rejects_duplicate_views_and_identities() -> None:
    with pytest.raises(ValidationError, match="duplicate views"):
        RigPlan(
            name="p",
            cameras=(
                CameraBinding(view="a", serial="1"),
                CameraBinding(view="a", serial="2"),
            ),
        )
    with pytest.raises(ValidationError, match="duplicate camera identities"):
        RigPlan(
            name="p",
            cameras=(
                CameraBinding(view="a", serial="1"),
                CameraBinding(view="b", serial="1"),
            ),
        )


def test_plan_round_trips_through_json(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plans" / "three-view.json"
    plan.save(path)
    loaded = RigPlan.load(path)
    assert loaded == plan
    assert loaded.schema_version == PLAN_SCHEMA_VERSION
    assert loaded.binding_for("2601240001") is not None
    assert loaded.binding_for("nope") is None


def test_plan_load_rejects_other_schema_versions(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        _plan().model_dump_json().replace(PLAN_SCHEMA_VERSION, "rig-plan/0.9.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        RigPlan.load(path)


def test_check_plan_matches_missing_conflicts_and_unplanned() -> None:
    cams = [
        _cam("2605160001", 4),
        _cam("2601240001", 4),  # shares root port 4 with face_on -> conflict
        _cam("path_extra", 6),  # enumerated, not planned
    ]
    check = check_plan(_plan(), cams)
    assert set(check.matched) == {"face_on", "down_line"}
    assert check.missing == ("overhead",)
    assert set(check.conflicts) == {"face_on", "down_line"}
    assert check.unplanned == ("path_extra",)
    assert not check.ok


def test_check_plan_ok_when_each_camera_has_its_own_root_port() -> None:
    cams = [
        _cam("2605160001", 4),
        _cam("2601240001", 5),
        _cam("path_D-D35A8F7-0-0000", 6),
    ]
    check = check_plan(_plan(), cams)
    assert check.ok
    assert check.conflicts == () and check.missing == () and check.unplanned == ()
