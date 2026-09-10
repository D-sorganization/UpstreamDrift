"""Wire the standard wizard to existing editors and the existing command runner."""

from __future__ import annotations

import sqlite3
import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QFileDialog, QPushButton

from src.motion_capture.reference.storage import ReferenceLibrary
from src.motion_capture.rig.capture_notes import read_notes

from .goal_catalog import load_catalog
from .goal_planner import CaptureGoalRequest, CaptureProgress, Readiness, restore
from .goal_wizard import CaptureWizard
from .record_bar import Phase
from .wizard_evidence import CalibrationReview, WizardEvidence, inspect_capture
from .wizard_storage import input_revision, load_progress, read_document, save_progress

if TYPE_CHECKING:
    from .gui import CaptureRigWidget


class WizardActions(QObject):
    def __init__(self, host: CaptureRigWidget) -> None:
        super().__init__(host)
        self.host = host
        self.button = QPushButton("Capture Wizard", host)
        self.button.setToolTip(
            "Choose an outcome, follow its required steps, or resume a saved capture workflow."
        )
        self.button.clicked.connect(self.show)
        self.dialog: CaptureWizard | None = None
        self.review: CalibrationReview | None = None
        self.skipped: set[str] = set()
        self.evidence: WizardEvidence | None = None
        self._future: Future[WizardEvidence] | None = None
        self._request: tuple[object, ...] | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="capture-wizard"
        )
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        host.destroyed.connect(
            lambda: self._executor.shutdown(wait=False, cancel_futures=True)
        )
        host.runner.finished.connect(lambda _code: self.refresh())
        host.runner.busy_changed.connect(self._running)
        host.record_bar.badge_changed.connect(
            lambda _badge: self._running(self._busy())
        )

    def _running(self, active: bool) -> None:
        if active and self.dialog is not None:
            self.dialog.update_evidence("Capture Operation Running", {}, busy=True)

    def _busy(self) -> bool:
        host = self.host
        return host.runner.busy or host.record_bar.phase is not Phase.IDLE

    def _start_text(self) -> str:
        host = self.host
        process = host.process
        return process.start_edit.text()

    def _notice(self, message: str) -> None:
        host = self.host
        host.journey.notice(message)
        if self.dialog is not None:
            self.dialog.set_feedback(message)

    def show(self) -> None:
        host = self.host
        try:
            if self.dialog is None:
                self.dialog = CaptureWizard(load_catalog(), self.host)
                self.dialog.action_requested.connect(self.open_step)
                self.dialog.refresh_requested.connect(self.refresh)
                self.dialog.skip_requested.connect(self.skip)
                self.dialog.save_requested.connect(self.save)
                self.dialog.resume_requested.connect(self.resume)
                self.dialog.plan_requested.connect(self.open_plan)
                self.dialog.camera_setup_requested.connect(self.open_camera_setup)
                self.dialog.helpRequested.connect(host.journey.show_help)
                self.dialog.finished.connect(self._closed)
            if host.media is not None:
                host.library_actions.library().register(host.media.root)
            self.dialog.show()
            self.dialog.raise_()
            self.dialog.activateWindow()
            self.refresh()
        except (ValueError, OSError, sqlite3.Error) as exc:
            self._notice(f"Cannot open the capture wizard: {exc}")

    def _closed(self, _result: int) -> None:
        self.review = None
        self.skipped.clear()
        if self.dialog is not None:
            self.dialog.apply_skips(())

    def _context(self) -> tuple[object, ...]:
        host = self.host
        media = host.media
        match = host.match.selection() if media is not None else None
        return (
            str(media.root) if media else None,
            self._start_text(),
            host.process.model_name(),
            repr(match),
            tuple(sorted(self.skipped)),
        )

    def _revision(self) -> str:
        host = self.host
        media = host.media
        if media is None:
            raise ValueError("Select a capture first")
        data = [input_revision(media), self._context()[1:4]]
        return hashlib.sha256(json.dumps(data).encode()).hexdigest()

    def refresh(self) -> None:
        host = self.host
        dialog = self.dialog
        if dialog is None or dialog.route is None:
            return
        if self._busy():
            dialog.update_evidence("Capture Operation Running", {}, busy=True)
            return
        if self._future is not None:
            return
        try:
            media = host.media
            if media is not None:
                notes = read_notes(media.root)
                if self.evidence and notes.capture_id != self.evidence.capture_id:
                    self.skipped.clear()
                    self.review = None
            self._request = (dialog.route, self._context())
            start = self._start_text().strip()
            self._future = self._executor.submit(
                inspect_capture,
                dialog.route,
                media,
                host.library_actions.library().root,
                review=self.review,
                start_file=Path(start) if start else None,
                skipped=frozenset(self.skipped),
                model_name=host.process.model_name(),
            )
            dialog.update_evidence("Checking Capture Status…", {})
            self._timer.start()
        except (ValueError, OSError, sqlite3.Error) as exc:
            dialog.update_evidence("Status Needs Review", {})
            self._notice(f"Cannot check this capture: {exc}")

    def _poll(self) -> None:
        host = self.host
        future, dialog = self._future, self.dialog
        if future is None or not future.done():
            return
        self._timer.stop()
        self._future = None
        if dialog is None:
            return
        try:
            if self._request != (dialog.route, self._context()):
                self.refresh()
                return
            self.evidence = future.result()
            match = host.match.selection() if host.media is not None else None
            if match and (
                match.name
                or match.observation_set != "observations"
                or match.image_space
                or match.views
            ):
                affected = [
                    "step.reconstruct",
                    "step.fit_model",
                    "step.export",
                    "compare.projected",
                ]
                if match.observation_set != "observations":
                    affected.extend(("step.detect", "step.review"))
                for key in affected:
                    if key in self.evidence.states:
                        self.evidence.states[key] = Readiness(
                            "blocked",
                            "This guided route uses the default triangulated match and all views. In Match, clear the variant name, select all views, Triangulate and observations; advanced matches remain available through their controls.",
                        )
            dialog.update_evidence(
                self.evidence.identity, self.evidence.states, busy=self._busy()
            )
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            dialog.update_evidence("Status Needs Review", {})
            self._notice(
                f"Could not inspect this capture: {exc}. Correct the input and refresh status."
            )

    def skip(self, key: str, checked: bool) -> None:
        dialog = self.dialog
        if dialog is None or dialog.route is None:
            return
        step = next((s for s in dialog.route.steps if s.id == key), None)
        if step is None or not step.optional:
            raise ValueError("Only an optional route step can be skipped")
        self.skipped.add(key) if checked else self.skipped.discard(key)
        self.refresh()

    def open_step(self, key: str) -> None:
        host = self.host
        dialog = self.dialog
        if dialog is None or dialog.route is None:
            return
        step = next((s for s in dialog.route.steps if s.id == key), None)
        if step is None:
            self._notice("That step is not in this capture route.")
            return
        if self._busy():
            self._notice("An action is running. Wait for it to finish or use Stop.")
            return
        try:
            if step.action == "workflow":
                dialog.hide()
                host.journey_actions.show_step(step.workflow_key or "")
                host.journey.append_notice(
                    " Return with Capture Wizard after reviewing the result."
                )
                return
            dialog.hide()
            self._open_editor(step.action)
            dialog.show()
            self.refresh()
        except (ValueError, OSError, sqlite3.Error) as exc:
            dialog.show()
            self._notice(f"Cannot open this step: {exc}")

    def open_camera_setup(self) -> None:
        dialog = self.dialog
        if dialog is not None:
            dialog.hide()
        setup_actions = self.host.camera_setup_actions
        setup_actions.show()
        if dialog is not None:
            dialog.show()
            self.refresh()

    def _open_editor(self, action: str) -> None:
        host = self.host
        if host.media is None and action in {
            "edit",
            "calibration",
            "draw",
            "compare_reference",
        }:
            raise ValueError("Select a capture from Library first")
        if action == "library":
            host.library_actions.show_library()
        elif action == "my_clubs":
            host.equipment_actions.show_equipment()
        elif action == "calibration":
            self.review = None
            path = host.calibration_actions.show()
            if path is not None and host.media is not None:
                self.review = CalibrationReview.confirmed(host.media.root, path)
        elif action == "edit":
            host.library_actions.edit_swing()
        else:
            from .coaching_dialog import show_coaching
            from .reference_library_dialog import ReferenceLibraryDialog
            from .reference_comparison import show_reference_comparison

            library = ReferenceLibrary(
                host.library_actions.library().root / "references"
            )
            if action == "references":
                ReferenceLibraryDialog(library, host).exec()
            elif host.media is None:
                raise ValueError("Select a capture from Library first")
            elif action == "draw":
                show_coaching(host.media.root, host)
            elif action == "compare_reference":
                show_reference_comparison(host.media.root, library, host)
            else:
                raise ValueError("Unsupported capture navigation")

    def save(self) -> None:
        host = self.host
        dialog, media = self.dialog, host.media
        if (
            dialog is None
            or dialog.route is None
            or media is None
            or dialog.current_step is None
        ):
            self._notice("Select a capture and a route step before saving progress.")
            return
        if self._busy() or self._future is not None:
            self._notice(
                "Wait for the current operation or status check before saving progress."
            )
            return
        try:
            progress = CaptureProgress(
                capture_id=read_notes(media.root).capture_id,
                goals=dialog.route.goals,
                catalog_revision=dialog.catalog.revision,
                input_revision=self._revision(),
                current_step=dialog.current_step,
                skipped=tuple(sorted(self.skipped)),
            )
            save_progress(media.root, progress)
            dialog.close()
            self._notice(
                "Workflow saved for this capture. Reopen Capture Wizard to resume."
            )
        except (ValueError, OSError) as exc:
            self._notice(f"Workflow could not be saved: {exc}")

    def resume(self) -> None:
        host = self.host
        dialog, media = self.dialog, host.media
        if dialog is None or media is None:
            self._notice("Select the capture in Library before resuming its workflow.")
            return
        try:
            progress = load_progress(media.root)
            restore(
                dialog.catalog,
                progress,
                capture_id=read_notes(media.root).capture_id,
                input_revision=self._revision(),
            )
            dialog.choose(progress.goals)
            dialog.build_route(progress.goals)
            self.skipped = set(progress.skipped)
            dialog.apply_skips(self.skipped)
            dialog.navigate(progress.current_step)
            self.refresh()
        except (ValueError, OSError) as exc:
            self._notice(
                f"Saved workflow needs review: {exc}. Choose the outcomes again; your recordings and analysis are preserved."
            )

    def open_plan(self) -> None:
        if self.dialog is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self.dialog, "Open Capture Plan", "", "Capture Plans (*.json)"
        )
        if not path:
            self._notice(
                "Plan selection cancelled. Your current workflow is unchanged."
            )
            return
        try:
            request = CaptureGoalRequest.model_validate_json(read_document(Path(path)))
            request.resolve(self.dialog.catalog)
            self.dialog.choose(request.goals)
            self._notice(
                "Plan loaded. Press Next to review the route for this capture."
            )
        except (ValueError, OSError) as exc:
            self._notice(f"Cannot open this plan: {exc}")
