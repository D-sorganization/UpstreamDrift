"""Capture action feedback and navigation around the existing commands (#9913)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from . import workflow
from .capture_activity import CaptureAction, save_action

if TYPE_CHECKING:
    from .gui import CaptureRigWidget


class JourneyActions:
    """Explain every dispatched action and keep its capture/settings stable."""

    def __init__(self, host: CaptureRigWidget) -> None:
        self.host = host
        self.active_action: str | None = None
        self.cancelled = False
        self.operation: CaptureAction | None = None
        self.operation_root: Path | None = None
        self.history_warning = ""

    def _label(self, action: str) -> str:
        return dict(self.host._ACTIONS).get(action, action.replace("_", " "))

    def trigger(self, action: str) -> None:
        host = self.host
        if action == "show_log":
            host._show_log(True)
            return
        if action == "stop":
            self.cancelled = self.active_action is not None
            host.journey.notice(
                "Stopping the current action…"
                if self.cancelled
                else "No action is running."
            )
            host.runner.stop()
            return
        if host.runner.busy or self.active_action is not None:
            host.journey.notice(
                "An action is running. Wait for it to finish, or press Stop.", busy=True
            )
            return
        if action == "load":
            if host.refresh_session() is not None:
                host.journey.notice(
                    "Capture loaded. Review its status or continue with the next step."
                )
            return
        if action == "preview":
            host.journey.notice(
                "Opening cameras…" if not host.preview.active else "Stopping preview…"
            )
            host.toggle_preview()
            return
        if action == "record":
            host.journey.notice(
                "Prepare your swing. The countdown and recording status appear below the cameras."
            )
            host.record_bar.toggle()
            return
        if action == "annotate":
            dialog = host.annotate_dialog()
            if dialog is None:
                host.journey.notice(
                    "Load a capture and select a playable view before annotating."
                )
                return
            host.journey.notice(
                "Annotation editor opened. Save your points, then review this capture."
            )
            dialog.exec()
            host.refresh_session()
            return
        if action == "import" and not host.capture.pending_import:
            host.capture.choose_import_files()
            if not host.capture.pending_import:
                host.journey.notice("Import cancelled. No videos were selected.")
                return
        try:
            argv = host.command_for(action)
        except (ValueError, TypeError) as exc:
            host.journey.notice(f"Cannot {self._label(action)}: {exc}", retry=action)
            host._append_log(f"cannot run {action}: {exc}\n")
            return
        self.begin(action)
        host.runner.run(argv)
        host.library_actions.refresh()
        host.calibration_actions.refresh()

    def begin(self, action: str) -> None:
        self.active_action = action
        self.cancelled = False
        host = self.host
        context = ""
        if action == "ingest":
            context = f"{host.process.estimator()} · {host.process.ingest_out(host.capture.session_dir()) or 'observations'}"
        elif action == "compare":
            context = "mediapipe · openpose_dnn"
        elif action in {"reconstruct", "fit_model", "kinetics", "compare_models"}:
            selection = host.match.selection()
            context = (
                f"{host.process.model_name()} · {selection.name or 'default match'}"
            )
        self.operation = CaptureAction(action=action, context=context)
        self.operation_root = host.capture.session_dir()
        host.journey.active_id = self.operation.id
        self.history_warning = ""
        self._save_operation()
        host.journey.notice(
            f"Running {self._label(action)} for {host.capture.session_dir().name}"
            f"{' · ' + context if context else ''}…{self.history_warning}",
            busy=True,
        )
        for panel in (host.capture, host.process, host.match):
            panel.setEnabled(False)
        host._apply_workflow(host.media)

    def _save_operation(self) -> None:
        if self.operation is None or self.operation_root is None:
            return
        try:
            save_action(self.operation_root, self.operation)
        except (ValueError, OSError) as exc:
            self.history_warning = f" Activity history could not be saved: {exc}"
            self.host._append_log(self.history_warning + "\n")

    def complete(self, code: int) -> None:
        action = self.active_action
        self.active_action = None
        host = self.host
        if self.operation is not None:
            self.operation = self.operation.complete(code, cancelled=self.cancelled)
            self._save_operation()
            self.operation = None
        host.journey.active_id = None
        host.journey.refresh_details()
        for panel in (host.capture, host.process, host.match):
            panel.setEnabled(True)
        host._apply_workflow(host.media)
        if action is None:
            return
        label = self._label(action)
        if self.cancelled:
            host.journey.notice(
                f"{label} cancelled. Review any partial output before retrying.",
                retry=action,
            )
        elif code:
            host.journey.notice(
                f"{label} failed (exit {code}). Open Activity Log for the cause, adjust the inputs, then retry.",
                retry=action,
            )
        else:
            host.journey.notice(
                f"{label} finished. Review Capture Status and Results, or continue with the next step."
            )
        if self.history_warning:
            host.journey.message.setText(
                host.journey.message.text() + self.history_warning
            )

    def show_step(self, key: str) -> None:
        """Navigate to instructions/settings without silently starting a job."""
        host = self.host
        step = next((item for item in workflow.STEPS if item.key == key), None)
        if step is None:
            host.journey.notice(
                "That workflow step is unavailable. Open Help and Workflow."
            )
            return
        host.panes.set_visible("rail", True)
        host.rail.select_step(key)
        host.panes.docks["rail"].raise_()
        host.workflow._show(
            next(i for i, item in enumerate(workflow.STEPS) if item.key == key)
        )
        host.panes.set_visible("inputs", True)
        host.panes.docks["inputs"].raise_()
        panel = (
            host.capture
            if key in {"setup", "capture"}
            else host.match
            if key == "reconstruct"
            else host.process
        )
        tabs = panel.parentWidget()
        # Settings use Qt's stacked tab container; locate the owning tab widget.
        from PyQt6.QtWidgets import QTabWidget

        while tabs is not None and not isinstance(tabs, QTabWidget):
            tabs = tabs.parentWidget()
        if isinstance(tabs, QTabWidget):
            tabs.setCurrentWidget(panel)
        host.journey.notice(
            f"{step.title}: {step.purpose} Review Step Details and use its action button when ready."
        )

    def clear_capture(self, reason: str) -> None:
        host = self.host
        host.playback.clear_capture()
        host.match.load(None)
        host.provenance.current = None
        host.provenance.browser.setPlainText("No capture loaded.")
        for table in (
            host.swing_table,
            host.analysis_table,
            host.model_table,
            host.kinetics_table,
            host.reliability_table,
        ):
            table.fill(None)
        host.journey.set_capture(None)
        host.journey.notice(f"Cannot load this capture: {reason}")
