"""Regression tests for #8884.

``_on_cancel_clicked`` / ``_on_pause_clicked`` / ``_on_resume_clicked`` each
wrapped their controller call in a ``try/except`` whose entire body was
``logger.error(...)``. A user clicking Cancel on a job the scheduler refused
to cancel saw **nothing**: no dialog, no status text, the row still reading
Running. They clicked again, or walked away believing the job had stopped
(lost GPU-hours). Cancel was also destructive with no confirmation.

These tests fail against unmodified ``src/``: ``_report_action_failure``,
``set_action_status``, ``_confirm_cancel`` and ``action_status_label`` do not
exist there, and the handlers swallow the exception.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit


def _make_widget(qapp):  # noqa: ANN001, ANN202
    """Build a MainWidget over a controller with one submitted job."""
    pytest.importorskip("PyQt6")
    from src.shared.python.training import TrainingConfig, TrainingFramework
    from src.tools.training_controller.gui import MainWidget

    from .test_gui_smoke import _make_controller

    del qapp
    controller, scheduler = _make_controller()
    job = controller.submit_job(
        TrainingConfig(
            framework=TrainingFramework.PYTORCH,
            entry_point="module:train",
            output_dir=Path("/tmp/training-controller-feedback"),
            dataset_id="dataset-1",
        )
    )
    widget = MainWidget(controller)
    widget.job_table.selectRow(0)
    return widget, controller, scheduler, job


@pytest.fixture
def dashboard(qapp):  # noqa: ANN001, ANN201
    widget, controller, scheduler, job = _make_widget(qapp)
    yield widget, controller, job
    widget.cleanup()
    widget.deleteLater()
    scheduler.shutdown()


def _raise_training_error(_self, _job_id):  # noqa: ANN001, ANN202
    from src.shared.python.training import TrainingError

    raise TrainingError("job is not in a cancellable state")


def _patch_controller(monkeypatch, method, impl) -> None:  # noqa: ANN001
    """Patch on the class: TrainingDashboardController uses __slots__."""
    from src.tools.training_controller import TrainingDashboardController

    monkeypatch.setattr(TrainingDashboardController, method, impl)


@pytest.mark.parametrize(
    ("verb", "button_attr", "controller_method"),
    [
        ("cancel", "cancel_button", "cancel_job"),
        ("pause", "pause_button", "pause_job"),
        ("resume", "resume_button", "resume_job"),
    ],
)
def test_failed_action_is_shown_to_the_user_not_only_logged(
    dashboard, monkeypatch, verb, button_attr, controller_method
) -> None:  # noqa: ANN001
    """The headline defect: a rejected action produced no visible signal."""
    from PyQt6 import QtWidgets

    widget, controller, job = dashboard
    _patch_controller(monkeypatch, controller_method, _raise_training_error)
    monkeypatch.setattr(widget, "_confirm_cancel", lambda _job_id: True)

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda _p, title, text, *a, **k: shown.append((title, text))),
    )

    getattr(widget, button_attr).click()

    assert shown, f"a failed {verb} must produce a visible message"
    title, text = shown[0]
    assert "Failed" in title or "Failed" in text
    assert verb in text
    assert job.job_id.value in text
    assert verb in widget.action_status_label.text()
    assert job.job_id.value in widget.action_status_label.text()


def test_successful_action_refreshes_the_row_immediately(
    dashboard, monkeypatch
) -> None:  # noqa: ANN001
    """The click must visibly do something, not wait for the next poll."""
    widget, controller, job = dashboard
    _patch_controller(monkeypatch, "pause_job", lambda _self, _job_id: None)

    refreshes: list[int] = []
    original = widget.update_ui
    monkeypatch.setattr(
        widget, "update_ui", lambda: (refreshes.append(1), original())[1]
    )

    widget.pause_button.click()

    assert refreshes, "a successful action must refresh the job table"
    assert "pause" in widget.action_status_label.text()
    assert job.job_id.value in widget.action_status_label.text()


def test_cancel_requires_confirmation_naming_job_and_runtime(
    dashboard, monkeypatch
) -> None:  # noqa: ANN001
    """Cancel is destructive; it must ask first, and say what is at stake."""
    from PyQt6 import QtWidgets

    widget, controller, job = dashboard
    called: list[str] = []
    _patch_controller(
        monkeypatch, "cancel_job", lambda _self, jid: called.append(jid.value)
    )

    asked: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(
            lambda _p, _t, text, *a, **k: (
                asked.append(text),
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )[1]
        ),
    )

    widget.cancel_button.click()

    assert asked, "Cancel must confirm before stopping a job"
    assert job.job_id.value in asked[0]
    assert "Progress will be lost" in asked[0]
    assert ":" in asked[0], "the prompt must include an H:MM runtime"
    assert not called, "answering Cancel must not cancel the job"


def test_confirmed_cancel_reaches_the_controller(dashboard, monkeypatch) -> None:  # noqa: ANN001
    from PyQt6 import QtWidgets

    widget, controller, job = dashboard
    called: list[str] = []
    _patch_controller(
        monkeypatch, "cancel_job", lambda _self, jid: called.append(jid.value)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes),
    )

    widget.cancel_button.click()

    assert called == [job.job_id.value]


def test_format_elapsed_hhmm() -> None:
    from src.tools.training_controller.gui import _format_elapsed_hhmm

    assert _format_elapsed_hhmm(0) == "0:00"
    assert _format_elapsed_hhmm(59) == "0:00"
    assert _format_elapsed_hhmm(60) == "0:01"
    assert _format_elapsed_hhmm(3600) == "1:00"
    assert _format_elapsed_hhmm(3600 * 2 + 60 * 7) == "2:07"
    assert _format_elapsed_hhmm(-5) == "0:00"


def test_report_action_failure_rejects_an_empty_verb(dashboard) -> None:  # noqa: ANN001
    widget, _controller, _job = dashboard
    with pytest.raises(ValueError, match="verb must be a non-empty string"):
        widget._report_action_failure("", RuntimeError("boom"))
