"""Regression tests for CustomTitleBar drag and event handling.

Filed under issue #5618 — the launcher's frameless window cannot be
grabbed and dragged after PR #5613 attached the Sidekick dock. These
tests pin the contract of the title bar so the regression does not
recur once fixed.

Test design follows TDD/DbC/LOD/DRY:

* TDD: each scenario was written as a failing test against current
  behaviour and only goes green after the fix is in place.
* DbC: the title bar exposes a documented invariant that its widget
  rectangle is always opaque (hit-testable) and that ``move_requested``
  emissions outside the desktop's virtual geometry are clamped to a
  visible area.
* LOD: tests only reach into ``CustomTitleBar`` and ``QApplication``
  surfaces; no walking through ``self.window()._docks._central._foo``.
* DRY: shared synthetic-event helpers live in a single ``_helpers``
  module-private factory.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QToolButton, QWidget

from src.launchers.custom_title_bar import CustomTitleBar, clamp_to_visible_screen

pytestmark = pytest.mark.ui


# ---------------------------------------------------------------------------
# Helpers (DRY)
# ---------------------------------------------------------------------------


def _press(
    widget: QWidget,
    *,
    local: QPoint,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
) -> QMouseEvent:
    """Build and post a synthetic ``mousePressEvent`` to ``widget``.

    Returns the event so callers can assert on accept/ignore state.
    """
    global_pos = widget.mapToGlobal(local)
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        local.toPointF(),
        global_pos.toPointF(),
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)
    return event


def _move(
    widget: QWidget,
    *,
    local: QPoint,
    buttons: Qt.MouseButton = Qt.MouseButton.LeftButton,
) -> QMouseEvent:
    """Build and post a synthetic ``mouseMoveEvent`` to ``widget``."""
    global_pos = widget.mapToGlobal(local)
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        local.toPointF(),
        global_pos.toPointF(),
        Qt.MouseButton.NoButton,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)
    return event


# ---------------------------------------------------------------------------
# Hit-test / opacity contract
# ---------------------------------------------------------------------------


class TestTitleBarHitTesting:
    """The title bar's widget rectangle must be opaque for click capture."""

    def test_title_bar_paints_styled_background(self, qtbot):
        """``WA_StyledBackground`` must be set so the bar is hit-testable.

        Regression: when ``WA_TranslucentBackground`` is set on the parent
        window and the title bar lacks ``WA_StyledBackground``, clicks
        within the bar's visible area fall through to the desktop instead
        of reaching ``mousePressEvent``.
        """
        bar = CustomTitleBar()
        qtbot.addWidget(bar)
        assert bar.testAttribute(Qt.WidgetAttribute.WA_StyledBackground), (
            "CustomTitleBar must set WA_StyledBackground=True so its "
            "stylesheet background-color actually paints and the bar "
            "registers mouse events when the main window uses "
            "WA_TranslucentBackground."
        )


# ---------------------------------------------------------------------------
# Drag emission
# ---------------------------------------------------------------------------


class TestTitleBarDrag:
    """Press + move on the bar emits ``move_requested`` with correct offset."""

    def test_press_and_move_emits_move_requested(self, qtbot):
        bar = CustomTitleBar()
        qtbot.addWidget(bar)
        bar.resize(400, 40)
        bar.show()
        qtbot.waitExposed(bar)

        with qtbot.waitSignal(bar.move_requested, timeout=500) as signal:
            _press(bar, local=QPoint(50, 10))
            _move(bar, local=QPoint(120, 10))

        assert isinstance(signal.args[0], QPoint)

    def test_press_only_does_not_emit_move(self, qtbot):
        """A press without a follow-up move should NOT emit ``move_requested``."""
        bar = CustomTitleBar()
        qtbot.addWidget(bar)
        bar.resize(400, 40)
        bar.show()
        qtbot.waitExposed(bar)

        with qtbot.assertNotEmitted(bar.move_requested, wait=100):
            _press(bar, local=QPoint(60, 20))


# ---------------------------------------------------------------------------
# Child propagation (decorative labels do not swallow the drag)
# ---------------------------------------------------------------------------


class TestChildLabelDoesNotSwallowDrag:
    """Clicks on decorative ``QLabel`` children must reach the title bar."""

    def test_click_on_title_label_still_initiates_drag(self, qtbot):
        bar = CustomTitleBar()
        qtbot.addWidget(bar)
        bar.resize(400, 40)
        bar.show()
        qtbot.waitExposed(bar)

        assert bar.title_label is not None
        # Press on the title_label, then move; the title bar should still
        # capture the drag start because labels are configured to forward
        # mouse events back to their parent (no event swallowing).
        with qtbot.waitSignal(bar.move_requested, timeout=500):
            _press(bar.title_label, local=QPoint(10, 5))
            _move(bar.title_label, local=QPoint(80, 5))

    def test_click_on_icon_label_still_initiates_drag(self, qtbot):
        bar = CustomTitleBar()
        qtbot.addWidget(bar)
        bar.resize(400, 40)
        bar.show()
        qtbot.waitExposed(bar)

        assert bar.icon_label is not None
        with qtbot.waitSignal(bar.move_requested, timeout=500):
            _press(bar.icon_label, local=QPoint(5, 5))
            _move(bar.icon_label, local=QPoint(40, 5))


# ---------------------------------------------------------------------------
# Buttons swallow the drag (correct behaviour)
# ---------------------------------------------------------------------------


class TestButtonsDoNotInitiateDrag:
    """Window-control buttons must consume clicks and not start a drag."""

    def test_minimize_button_click_does_not_emit_move(self, qtbot):
        bar = CustomTitleBar()
        qtbot.addWidget(bar)
        bar.resize(400, 40)
        bar.show()
        qtbot.waitExposed(bar)

        assert isinstance(bar.btn_min, QToolButton)
        with qtbot.assertNotEmitted(bar.move_requested, wait=100):
            qtbot.mouseClick(bar.btn_min, Qt.MouseButton.LeftButton)

    def test_close_button_emits_close_request_not_move(self, qtbot):
        bar = CustomTitleBar(show_close_button=True)
        qtbot.addWidget(bar)
        bar.resize(400, 40)
        bar.show()
        qtbot.waitExposed(bar)

        assert bar.btn_close is not None
        with qtbot.waitSignal(bar.close_requested, timeout=500):
            qtbot.mouseClick(bar.btn_close, Qt.MouseButton.LeftButton)


# ---------------------------------------------------------------------------
# Screen clamping (defensive against off-screen-stuck regression)
# ---------------------------------------------------------------------------


class TestClampToVisibleScreen:
    """``clamp_to_visible_screen`` is a pure function — easy to unit test."""

    def test_point_inside_screen_is_returned_unchanged(self, qtbot):
        screen = QApplication.primaryScreen()
        assert screen is not None
        center = screen.geometry().center()
        clamped = clamp_to_visible_screen(center)
        # DbC postcondition: the result lies in the virtual desktop geometry
        assert QApplication.primaryScreen().virtualGeometry().contains(clamped)
        # And inside-the-screen inputs round-trip exactly.
        assert clamped == center

    def test_far_off_screen_point_is_clamped_inside(self, qtbot):
        # DbC postcondition: any input maps to a point inside
        # virtualGeometry — never silently dropped.
        clamped = clamp_to_visible_screen(QPoint(-99999, -99999))
        vg = QApplication.primaryScreen().virtualGeometry()
        assert vg.contains(clamped), (
            f"Clamped point {clamped} must be inside virtualGeometry "
            f"{vg}; clamp_to_visible_screen must never return an "
            "off-screen coordinate."
        )

    def test_clamp_is_idempotent(self, qtbot):
        once = clamp_to_visible_screen(QPoint(50_000, 50_000))
        twice = clamp_to_visible_screen(once)
        assert once == twice


# ---------------------------------------------------------------------------
# End-to-end: title bar wired into a QMainWindow actually moves it
# ---------------------------------------------------------------------------


class TestTitleBarMovesParentWindow:
    """When the title bar's ``move_requested`` connects to ``window.move``,
    a synthetic drag moves the window and the resulting position is on screen.
    """

    def test_drag_moves_parent_main_window(self, qtbot):
        window = QMainWindow()
        bar = CustomTitleBar(window)
        bar.resize(400, 40)
        window.setCentralWidget(bar)
        bar.move_requested.connect(
            lambda pos: window.move(clamp_to_visible_screen(pos))
        )
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)

        initial = window.pos()
        _press(bar, local=QPoint(80, 10))
        _move(bar, local=QPoint(180, 10))
        qtbot.wait(50)

        # DbC postcondition: window's new position is inside virtualGeometry.
        vg = QApplication.primaryScreen().virtualGeometry()
        assert vg.contains(window.pos()), (
            f"Window moved to {window.pos()} which is outside virtualGeometry {vg}"
        )
        # And the window actually moved — we don't pin the exact delta
        # because Qt's synthetic event coordinate math depends on the
        # window's frame geometry, which is platform-dependent.
        assert window.pos() != initial, (
            "Drag synthesised but window did not move; check that "
            "move_requested is connected and that clamping did not "
            "discard a valid target."
        )
