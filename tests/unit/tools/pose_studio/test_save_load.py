"""Regression tests for #8882.

Pose Studio's Save/Load buttons were shown **enabled** but only flashed a
transient ``QToolTip`` reading "Save formats coming in #4900 ... Currently a
stub."; ``_EmbedAdapter.is_dirty`` returned a hardcoded ``False`` despite the
tool owning a 64-deep undo stack; and ``PoseStudioWindow`` had no
``closeEvent``. So an hour of joint edits was discarded on close with no
prompt -- and there had never been a way to save them.

These tests fail against unmodified ``src/``: ``save_pose_to`` /
``load_pose_from`` / ``is_dirty`` / ``confirm_close`` do not exist there, and
``_EmbedAdapter.is_dirty()`` is constant.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ui]

from src.shared.python.pose_interchange.canonical import (  # noqa: E402
    canonical_from_reference_setup,
)
from src.tools.pose_studio.core import SUPPORTED_ENGINES  # noqa: E402
from src.tools.pose_studio.gui import (  # noqa: E402
    MainWidget,
    PoseStudioWindow,
    _EmbedAdapter,
)
from src.shared.python.motion_matching.diagnostics.reference_pose import (  # noqa: E402
    REFERENCE_GOLFER_FIELDS,
)
from src.tools.pose_studio.pose_files import (  # noqa: E402
    ENGINE_FORMATS,
    engine_format,
    load_pose,
    save_pose,
)

#: A real canonical joint name, so pose edits validate.
_EDITABLE_JOINT = REFERENCE_GOLFER_FIELDS[0]


# ----------------------------------------------------------------------
# pose_files: file-format knowledge, no QApplication required
# ----------------------------------------------------------------------


def test_every_supported_engine_has_a_file_format() -> None:
    """A tile cannot offer Save for an engine with no on-disk shape."""
    for engine in SUPPORTED_ENGINES:
        fmt = engine_format(engine)
        assert fmt.suffix.startswith(".")
        assert fmt.default_name().endswith(fmt.suffix)
        assert "All Files (*)" in fmt.name_filter


def test_engine_format_rejects_an_unknown_engine() -> None:
    with pytest.raises(ValueError, match="no pose file format"):
        engine_format("not_an_engine")


@pytest.mark.parametrize("engine", sorted(ENGINE_FORMATS))
def test_save_then_load_round_trips_the_pose(engine, tmp_path) -> None:  # noqa: ANN001
    """The save path must actually be readable by the load path."""
    pose = canonical_from_reference_setup()
    target = tmp_path / engine_format(engine).default_name()
    written = save_pose(pose, engine, target)
    assert written.exists(), f"{engine}: nothing was written"

    restored = load_pose(engine, written)
    for name, value in pose.angles_full_dict_deg().items():
        assert restored.angle_deg(name) == pytest.approx(value, abs=1e-6)


# ----------------------------------------------------------------------
# MainWidget: dirty state and the real save/load handlers
# ----------------------------------------------------------------------


@pytest.fixture
def widget(qapp):  # noqa: ANN001, ANN201
    result = MainWidget()
    yield result
    result.deleteLater()


def test_is_dirty_is_false_on_a_fresh_widget(widget) -> None:  # noqa: ANN001
    assert widget.is_dirty() is False


def test_is_dirty_flips_true_after_one_pose_edit(widget) -> None:  # noqa: ANN001
    """The headline lie: is_dirty was hardcoded False (#8882)."""
    name = _EDITABLE_JOINT
    widget._on_angle_edited(name, 12.5)
    assert widget.is_dirty() is True


def test_save_clears_dirty_and_load_restores_the_pose(widget, tmp_path) -> None:  # noqa: ANN001
    name = _EDITABLE_JOINT
    widget._on_angle_edited(name, 21.0)
    assert widget.is_dirty()

    engine = widget._engine_controller.engine_name
    destination = tmp_path / engine_format(engine).default_name()
    written = widget.save_pose_to(destination)

    assert written.exists()
    assert widget.is_dirty() is False, "a successful save must clear dirty state"

    widget._on_angle_edited(name, -33.0)
    assert widget._engine_controller.pose.angle_deg(name) == pytest.approx(-33.0)
    assert widget.is_dirty()

    widget.load_pose_from(written)
    assert widget._engine_controller.pose.angle_deg(name) == pytest.approx(
        21.0, abs=1e-6
    )
    assert widget.is_dirty() is False, "a successful load must clear dirty state"


def test_save_and_load_buttons_are_enabled_and_do_real_work(widget) -> None:  # noqa: ANN001
    """No enabled control may perform nothing.

    The buttons stay enabled; the acceptance condition is that their
    handlers reach the real file path instead of a tooltip.
    """
    assert widget.btn_save.isEnabled()
    assert widget.btn_load.isEnabled()
    assert "stub" not in widget.btn_save.toolTip().lower()
    assert "stub" not in widget.btn_load.toolTip().lower()
    assert "#4900" not in widget.btn_save.toolTip()

    with patch(
        "src.tools.pose_studio.gui.QtWidgets.QFileDialog.getSaveFileName",
        return_value=("", ""),
    ) as dialog:
        widget._on_save_clicked()
    assert dialog.called, "Save must open a file dialog, not show a tooltip"


def test_save_failure_is_reported_to_the_user_not_only_logged(widget) -> None:  # noqa: ANN001
    with (
        patch(
            "src.tools.pose_studio.gui.QtWidgets.QFileDialog.getSaveFileName",
            return_value=("/nonexistent-root/\x00bad/pose.json", ""),
        ),
        patch.object(widget, "save_pose_to", side_effect=OSError("disk on fire")),
        patch("src.tools.pose_studio.gui.QtWidgets.QMessageBox.warning") as warning,
    ):
        widget._on_save_clicked()
    assert warning.called, "a failed save must produce a visible message"


def test_load_prompts_before_discarding_unsaved_edits(widget) -> None:  # noqa: ANN001
    from PyQt6 import QtWidgets

    name = _EDITABLE_JOINT
    widget._on_angle_edited(name, 7.5)
    assert widget.is_dirty()

    with (
        patch(
            "src.tools.pose_studio.gui.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
        ) as question,
        patch(
            "src.tools.pose_studio.gui.QtWidgets.QFileDialog.getOpenFileName"
        ) as dialog,
    ):
        widget._on_load_clicked()

    assert question.called
    assert not dialog.called, "Cancel must abort before the file dialog opens"
    assert widget._engine_controller.pose.angle_deg(name) == pytest.approx(7.5)


def test_confirm_close_is_true_when_clean_and_prompts_when_dirty(widget) -> None:  # noqa: ANN001
    from PyQt6 import QtWidgets

    assert widget.confirm_close() is True

    name = _EDITABLE_JOINT
    widget._on_angle_edited(name, 3.0)
    with patch(
        "src.tools.pose_studio.gui.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
    ):
        assert widget.confirm_close() is False
    with patch(
        "src.tools.pose_studio.gui.QtWidgets.QMessageBox.question",
        return_value=QtWidgets.QMessageBox.StandardButton.Discard,
    ):
        assert widget.confirm_close() is True


# ----------------------------------------------------------------------
# Standalone window + embed adapter agree on one definition of dirty
# ----------------------------------------------------------------------


def test_close_event_ignores_the_close_when_the_user_cancels(qapp) -> None:  # noqa: ANN001
    from PyQt6 import QtGui, QtWidgets

    window = PoseStudioWindow()
    try:
        name = _EDITABLE_JOINT
        window.main_widget._on_angle_edited(name, 9.0)
        event = QtGui.QCloseEvent()
        with patch(
            "src.tools.pose_studio.gui.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
        ):
            window.closeEvent(event)
        assert not event.isAccepted(), "close with unsaved edits must be refused"
    finally:
        window.deleteLater()


def test_embed_adapter_is_dirty_tracks_the_widget(qapp) -> None:  # noqa: ANN001
    adapter = _EmbedAdapter()
    assert adapter.is_dirty() is False  # no widget yet

    widget = adapter.create_main_widget(parent=None)
    try:
        assert adapter.is_dirty() is False
        name = _EDITABLE_JOINT
        widget._on_angle_edited(name, 15.0)
        assert adapter.is_dirty() is True, (
            "_EmbedAdapter.is_dirty was hardcoded False, so the launcher's "
            "dirty-close guard never fired for Pose Studio (#8882)"
        )
    finally:
        adapter.cleanup()
        widget.deleteLater()
    assert adapter.is_dirty() is False  # cleaned up
