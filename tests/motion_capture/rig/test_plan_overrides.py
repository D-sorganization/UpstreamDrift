"""Operator overrides: a mode string and a view subset without editing the plan."""

from __future__ import annotations

import pytest

from src.motion_capture.rig.plan import (
    CameraBinding,
    CaptureMode,
    RigPlan,
    parse_mode,
)

pytestmark = pytest.mark.unit

PLAN = RigPlan(
    name="lab",
    cameras=(
        CameraBinding(view="a", serial="1"),
        CameraBinding(view="b", serial="2", mode=CaptureMode(fps=30)),
        CameraBinding(view="c", port_path="path_X"),
    ),
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1280x720@120", CaptureMode(width=1280, height=720, fps=120)),
        (
            "640x480@200:yuy2",
            CaptureMode(width=640, height=480, fps=200, fourcc="YUY2"),
        ),
        (" 1920X1200@60 ", CaptureMode(width=1920, height=1200, fps=60)),
    ],
)
def test_parse_mode_accepts_width_height_fps_and_optional_fourcc(
    text: str, expected: CaptureMode
) -> None:
    assert parse_mode(text) == expected


@pytest.mark.parametrize("text", ["1280x720", "1280@60", "0x480@60", "axb@c", ""])
def test_parse_mode_rejects_malformed_strings(text: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        parse_mode(text)


def test_with_overrides_applies_one_mode_to_every_view_and_renames() -> None:
    mode = parse_mode("1280x720@120")
    plan = PLAN.with_overrides(mode=mode)
    assert all(c.mode == mode for c in plan.cameras)
    assert plan.name == "lab+1280x720@120"
    assert PLAN.cameras[1].mode.fps == 30  # source plan untouched


def test_with_overrides_selects_a_view_subset_in_plan_order() -> None:
    plan = PLAN.with_overrides(views=("c", "a"))
    assert [c.view for c in plan.cameras] == ["a", "c"]
    assert plan.name == "lab+a,c"


def test_with_overrides_rejects_unknown_views_and_empty_selection() -> None:
    with pytest.raises(ValueError, match="unknown view"):
        PLAN.with_overrides(views=("zz",))
    with pytest.raises(ValueError, match="at least one view"):
        PLAN.with_overrides(views=())


def test_with_overrides_without_arguments_is_identity() -> None:
    assert PLAN.with_overrides() == PLAN
