"""The step rail: where you are, and the one thing to press next (#9845).

Offscreen Qt tests. The rail is built standalone (it knows nothing about
the tile), fed :func:`workflow.evaluate` states, and asked the questions an
operator asks: which step am I on, what do I press, why is that grey.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from src.tools.capture_rig.session import SessionMedia, ViewMedia
from src.tools.capture_rig.step_rail import MAX_RAIL_WIDTH, StepRail
from src.tools.capture_rig.workflow import NO_SESSION, Status, evaluate

pytestmark = [pytest.mark.unit, pytest.mark.ui]

MODULE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "tools"
    / "capture_rig"
    / "step_rail.py"
)
HEX_LITERAL = re.compile(r"#[0-9a-fA-F]{6}\b")
INLINE_STYLESHEET = re.compile(r"setStyleSheet\(\s*f?[\"']")

_APP: QApplication | None = None  # keep the application alive for the module


def _app() -> QApplication:
    """The one QApplication, held at module level (a dropped one aborts)."""
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _view(name: str, *, recorded: bool = True, ingested: bool = False) -> ViewMedia:
    return ViewMedia(
        view=name,
        identity=name,
        recording=Path(f"{name}.avi") if recorded else None,
        proxy=None,
        observations=Path(f"{name}.json") if ingested else None,
        fps=60.0,
    )


def _media(views: tuple[ViewMedia, ...], **kw: object) -> SessionMedia:
    fields: dict[str, object] = {
        "root": Path("s"),
        "plan_name": "p",
        "views": views,
        "swing_summary": None,
        "reconstruction": None,
        "problems": (),
    }
    fields.update(kw)
    return SessionMedia(**fields)  # type: ignore[arg-type]


def _detected() -> SessionMedia:
    """Two ingested views with intrinsics: ready to be reviewed."""
    return _media(
        (_view("cam_a", ingested=True), _view("cam_b", ingested=True)),
        intrinsics=Path("intrinsics.json"),
    )


def _reviewed() -> SessionMedia:
    """...and reliability scored, so reconstruction is the next thing."""
    return _media(
        (_view("cam_a", ingested=True), _view("cam_b", ingested=True)),
        intrinsics=Path("intrinsics.json"),
        reliability={"cam_a": {}},
    )


def _rail(media: SessionMedia | None, **kw: object) -> StepRail:
    rail = StepRail()
    rail.set_states(evaluate(media), **kw)  # type: ignore[arg-type]
    return rail


# -- where am I ------------------------------------------------------------------
def test_rail_marks_the_current_step_at_each_stage() -> None:
    _app()
    assert _rail(None).current_key == "setup"
    two = (_view("cam_a"), _view("cam_b"))
    assert _rail(_media(two)).current_key == "intrinsics"
    calibrated = _media(two, intrinsics=Path("intrinsics.json"))
    assert _rail(calibrated).current_key == "detect"
    assert _rail(_detected()).current_key == "review"
    assert _rail(_reviewed()).current_key == "reconstruct"


def test_every_step_has_a_row_and_the_current_one_is_prominent() -> None:
    _app()
    rail = _rail(None)
    states = evaluate(None)
    assert list(rail.rows) == [s.step.key for s in states]
    assert rail.kind_of("setup") == "current"
    assert rail.kind_of("detect") == "blocked"
    assert rail.rows["setup"].styleSheet() != rail.rows["detect"].styleSheet()
    assert all(row.toolTip() for row in rail.rows.values())


def test_done_and_skipped_steps_read_as_such() -> None:
    _app()
    rail = _rail(_media((_view("face_on"),)))
    assert rail.kind_of("setup") == "done"  # views are bound
    assert rail.kind_of("intrinsics") == "skipped"  # single camera


# -- what do I press -------------------------------------------------------------
def test_primary_action_with_no_session_is_the_setup_path() -> None:
    _app()
    rail = _rail(None)
    assert rail.primary_action == "plan_check"
    assert rail.primary is not None and rail.primary.isEnabled()
    assert rail.primary.toolTip().startswith("Bind every planned view")


def test_primary_action_is_the_first_enabled_action_of_the_current_step() -> None:
    _app()
    two = (_view("cam_a"), _view("cam_b"))
    rail = _rail(_media(two))  # current step: intrinsics ("record", "calibrate")
    assert rail.primary_action == "record"
    rail.set_states(evaluate(_media(two)), enabled=frozenset({"calibrate"}))
    assert rail.primary_action == "calibrate"
    assert rail.secondary_actions == ("record",)


def test_a_disabled_primary_still_explains_itself() -> None:
    """The caller may withhold actions (a command is running); say what it does."""
    _app()
    rail = _rail(None, enabled=frozenset())
    assert rail.primary_action == "plan_check"
    assert rail.primary is not None and not rail.primary.isEnabled()
    assert rail.primary.toolTip() == rail.hint_of("plan_check")
    assert rail.primary.toolTip().strip()


def test_secondary_actions_are_the_current_steps_remainder() -> None:
    _app()
    rail = _rail(None)
    assert rail.secondary_actions == ("import",)
    assert [b.toolTip() for b in rail.secondary] == [
        rail.hint_of("import"),
    ]


def test_action_triggered_fires_with_the_action_key() -> None:
    _app()
    rail = _rail(None)
    fired: list[str] = []
    rail.action_triggered.connect(fired.append)
    assert rail.primary is not None
    rail.primary.click()
    rail.secondary[0].click()
    assert fired == ["plan_check", "import"]


# -- why is that grey ------------------------------------------------------------
def test_a_blocked_step_shows_its_reason_inline() -> None:
    _app()
    rail = _rail(None)
    assert evaluate(None)[3].status is Status.BLOCKED  # detect
    assert NO_SESSION in rail.notes["detect"].full_text
    assert "setup" not in rail.notes  # the ready step has nothing to explain


def test_the_reason_wording_comes_from_the_action_hints() -> None:
    _app()
    rail = _rail(None)
    assert rail.notes["detect"].full_text in rail.hint_of("ingest")


def test_a_skipped_step_says_so() -> None:
    _app()
    rail = _rail(_media((_view("face_on"),)))
    assert "not for this session" in rail.notes["reconstruct"].full_text


# -- it fits in a dock -----------------------------------------------------------
def test_the_rail_is_narrow_enough_for_a_side_dock() -> None:
    _app()
    rail = _rail(None)
    assert rail.minimumSizeHint().width() <= MAX_RAIL_WIDTH
    rail.set_states(evaluate(_reviewed()))
    assert rail.minimumSizeHint().width() <= MAX_RAIL_WIDTH


def test_long_row_text_elides_instead_of_widening() -> None:
    _app()
    rail = _rail(None)
    rail.resize(140, 600)
    rail.show()
    _app().processEvents()
    row = rail.rows["capture"]
    assert row.text() != row.full_text
    assert row.full_text in row.toolTip()


def test_set_states_rejects_an_empty_workflow() -> None:
    _app()
    rail = StepRail()
    with pytest.raises(ValueError):
        rail.set_states(())


# -- theme compliance ------------------------------------------------------------
def test_no_colour_literals_or_inline_stylesheets() -> None:
    lines = MODULE.read_text(encoding="utf-8").splitlines()
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(lines, 1)
        if HEX_LITERAL.search(line) or INLINE_STYLESHEET.search(line)
    ]
    assert offenders == []
