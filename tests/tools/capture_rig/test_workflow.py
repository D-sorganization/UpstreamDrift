"""Guided workflow: readiness follows what is on disk, single vs multi camera."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.capture_rig.session import SessionMedia, ViewMedia
from src.tools.capture_rig.workflow import (
    ACTION_HELP,
    STEPS,
    Status,
    action_hints,
    current,
    enabled_actions,
    evaluate,
)

pytestmark = pytest.mark.unit


def _view(name: str, *, recorded: bool = True, ingested: bool = False) -> ViewMedia:
    return ViewMedia(
        view=name,
        identity=name,
        recording=Path(f"{name}.avi") if recorded else None,
        proxy=None,
        observations=Path(f"{name}.json") if ingested else None,
        fps=60.0,
    )


def _media(views: tuple[ViewMedia, ...], **kw) -> SessionMedia:
    fields = {
        "root": Path("s"),
        "plan_name": "p",
        "views": views,
        "swing_summary": None,
        "reconstruction": None,
        "problems": (),
    }
    fields.update(kw)
    return SessionMedia(**fields)


def test_steps_are_ordered_and_named_uniquely() -> None:
    keys = [s.key for s in STEPS]
    assert len(set(keys)) == len(keys)
    assert keys[0] == "setup" and keys[-1] == "export"
    for step in STEPS:
        assert step.requirements and step.instructions and step.actions


def test_no_session_setup_and_capture_are_ready() -> None:
    states = evaluate(None)
    by_key = {s.step.key: s for s in states}
    assert by_key["setup"].status is Status.READY
    assert by_key["capture"].status is Status.READY  # a fresh folder is the norm
    assert all(s.status is Status.BLOCKED for s in states if not s.step.starts_fresh)
    assert enabled_actions(states) == {"plan_check", "import", "record", "proxy"}


def test_action_hints_explain_every_grey_button() -> None:
    states = evaluate(None)
    hints = action_hints(states, frozenset({"stop", "load", "preview"}))
    assert set(hints) == set(ACTION_HELP)
    assert "Disabled" not in hints["record"] and hints["record"].startswith("Record")
    assert "Disabled" not in hints["preview"]
    assert "Disabled: step 'Detect" in hints["ingest"]
    assert "load or record a session first" in hints["ingest"]
    single = evaluate(_media((_view("face_on", ingested=True),)))
    hints = action_hints(single)
    assert "not for this session" in hints["reconstruct"]
    assert "Disabled" not in hints["analyze"]


def test_single_camera_route_skips_3d_steps() -> None:
    states = {s.step.key: s for s in evaluate(_media((_view("face_on"),)))}
    assert states["intrinsics"].status is Status.SKIPPED
    assert states["reconstruct"].status is Status.SKIPPED
    assert states["export"].status is Status.SKIPPED
    assert states["capture"].status is Status.DONE
    assert states["detect"].status is Status.READY
    assert states["analyze_2d"].status is Status.BLOCKED
    assert current(tuple(states.values())).step.key == "detect"
    after = {
        s.step.key: s for s in evaluate(_media((_view("face_on", ingested=True),)))
    }
    assert after["analyze_2d"].status is Status.READY
    assert after["review"].status is Status.READY


def test_multi_camera_route_gates_reconstruct_on_ingest_and_export_on_fit() -> None:
    two = (_view("a"), _view("b"))
    states = {s.step.key: s for s in evaluate(_media(two))}
    assert states["intrinsics"].status is Status.READY
    assert states["reconstruct"].status is Status.BLOCKED
    assert "two ingested views" in states["reconstruct"].reason
    assert states["analyze_2d"].status is Status.SKIPPED
    fitted = _media(
        (_view("a", ingested=True), _view("b", ingested=True)),
        intrinsics=Path("intrinsics.json"),
        reconstruction={"rms_px": 1.0},
    )
    states = {s.step.key: s for s in evaluate(fitted)}
    assert states["intrinsics"].status is Status.DONE
    assert states["reconstruct"].status is Status.DONE
    assert states["export"].status is Status.READY
    assert "export" in enabled_actions(tuple(states.values()))
    assert current(tuple(states.values())).step.key == "review"
