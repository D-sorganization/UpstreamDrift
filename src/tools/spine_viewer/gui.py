"""PyQt6 Spine Viewer widget (SPN-15b, #12772; SPN-17c, #13260).

Shows a hash-verified study receipt as a load heatmap and provides a Run Study
panel to configure and execute studies off the GUI thread. The scientific status
and ``distribution_source`` are always visible; unqualified data is rendered in
a warning colour and never labelled qualified.

# noqa: gui-thread/ok -- background studies execute off the GUI thread in SpineStudyWorker (QThread).
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import time
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from src.engines.physics_engines.opensim.python.spine.study import SpineStudyRequest
from src.tools.spine_viewer import core

logger = logging.getLogger(__name__)

__all__ = ["SpineStudyWorker", "SpineViewerWidget"]

_NO_STUDY = "NO STUDY LOADED - open a study receipt (no data, nothing qualified)"
_QUALIFIED_STYLE = "background:#1b5e20;color:white;font-weight:bold;padding:6px;"
_WARN_STYLE = "background:#b71c1c;color:white;font-weight:bold;padding:6px;"
_NEUTRAL_STYLE = "background:#424242;color:white;font-weight:bold;padding:6px;"

_STUDY_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    RuntimeError,
    ImportError,
    ArithmeticError,
    LookupError,
    OSError,
)


class SpineStudyWorker(QtCore.QThread):
    """Background worker executing the spine study pipeline off the GUI thread."""

    finished_receipt = QtCore.pyqtSignal(Path)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        request: SpineStudyRequest,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._request = request

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            from src.engines.physics_engines.opensim.python.spine.study import (
                run_spine_study,
            )

            run_spine_study(self._request)
            output_dir = self._request.output_dir
            receipt_file = output_dir / f"{self._request.study_id}_receipt.json"
            if not receipt_file.is_file():
                candidates = sorted(output_dir.glob("*_receipt.json"))
                if candidates:
                    receipt_file = candidates[0]
            if not receipt_file.is_file():
                raise FileNotFoundError(f"Receipt file not found in {output_dir}")
            self.finished_receipt.emit(receipt_file)
        except _STUDY_ERRORS as exc:
            logger.exception("Spine study execution failed: %s", exc)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SpineViewerWidget(QtWidgets.QWidget):
    """Load-heatmap viewer and study runner for spine studies."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._view: core.SpineStudyView | None = None
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._worker: SpineStudyWorker | None = None
        self._assets_by_id: dict[str, dict[str, Any]] = {}

        # Status and distribution banner
        self._status = QtWidgets.QLabel(_NO_STUDY)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(_NEUTRAL_STYLE)
        self._source = QtWidgets.QLabel("distribution_source: -")
        self._source.setWordWrap(True)

        # Viewer controls
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(core.LOAD_QUANTITIES)
        self._combo.currentTextChanged.connect(self._redraw)
        open_button = QtWidgets.QPushButton("Open Study Receipt...")
        open_button.clicked.connect(self._choose_file)

        view_controls = QtWidgets.QHBoxLayout()
        view_controls.addWidget(open_button)
        view_controls.addWidget(QtWidgets.QLabel("Quantity:"))
        view_controls.addWidget(self._combo)
        view_controls.addStretch(1)

        # Run Study panel
        study_group = QtWidgets.QGroupBox("Run Study")
        study_grid = QtWidgets.QGridLayout(study_group)

        self._study_id_edit = QtWidgets.QLineEdit("spine_study_1")
        self._model_combo = QtWidgets.QComboBox()
        self._model_status_label = QtWidgets.QLabel("")
        self._law_combo = QtWidgets.QComboBox()
        self._motion_edit = QtWidgets.QLineEdit(
            "tests/data/motion_pipeline/golden/sample.sto"
        )
        browse_motion_btn = QtWidgets.QPushButton("Browse...")
        browse_motion_btn.clicked.connect(self._browse_motion)
        self._run_button = QtWidgets.QPushButton("Run Study")
        self._run_button.clicked.connect(lambda: self.start_study())
        self._run_status = QtWidgets.QLabel("")
        self._run_status.setWordWrap(True)

        # Populate assets
        for asset in core.list_available_assets():
            asset_id = str(asset["asset_id"])
            status = str(asset["status"])
            self._assets_by_id[asset_id] = asset
            self._model_combo.addItem(f"{asset_id} ({status})", userData=asset_id)
        fixture_idx = self._model_combo.findData("fixture")
        if fixture_idx >= 0:
            self._model_combo.setCurrentIndex(fixture_idx)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._on_model_changed()

        # Populate coupling laws
        self._law_combo.addItem("(model default)", userData=None)
        for law in core.list_available_coupling_laws():
            self._law_combo.addItem(law, userData=law)

        # Layout study grid
        study_grid.addWidget(QtWidgets.QLabel("Study ID:"), 0, 0)
        study_grid.addWidget(self._study_id_edit, 0, 1)

        study_grid.addWidget(QtWidgets.QLabel("Model:"), 1, 0)
        study_grid.addWidget(self._model_combo, 1, 1)
        study_grid.addWidget(self._model_status_label, 1, 2)

        study_grid.addWidget(QtWidgets.QLabel("Coupling Law:"), 2, 0)
        study_grid.addWidget(self._law_combo, 2, 1)

        study_grid.addWidget(QtWidgets.QLabel("Swing Source:"), 3, 0)
        motion_layout = QtWidgets.QHBoxLayout()
        motion_layout.addWidget(self._motion_edit)
        motion_layout.addWidget(browse_motion_btn)
        study_grid.addLayout(motion_layout, 3, 1)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.addWidget(self._run_button)
        action_layout.addWidget(self._run_status)
        action_layout.addStretch(1)
        study_grid.addLayout(action_layout, 4, 1)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(self._status)
        self._layout.addWidget(self._source)
        self._layout.addLayout(view_controls)
        self._layout.addWidget(study_group)

    def status_text(self) -> str:
        """Text of the always-visible status banner."""
        return self._status.text()

    def distribution_text(self) -> str:
        """Text of the always-visible distribution-source line."""
        return self._source.text()

    def current_quantity(self) -> str:
        """The load component being displayed."""
        return self._combo.currentText()

    def current_figure(self) -> Figure | None:
        """The figure on screen, or ``None`` when no valid study is loaded."""
        return self._figure

    def set_quantity(self, quantity: str) -> None:
        """Select a load component (``ValueError`` if unknown)."""
        if quantity not in core.LOAD_QUANTITIES:
            raise ValueError(f"unknown quantity {quantity!r}")
        self._combo.setCurrentText(quantity)

    def model_combo(self) -> QtWidgets.QComboBox:
        """Dropdown of available model assets."""
        return self._model_combo

    def model_status_text(self) -> str:
        """Honest admission status of the selected model asset."""
        return self._model_status_label.text()

    def coupling_law_combo(self) -> QtWidgets.QComboBox:
        """Dropdown of selectable spinal coupling laws."""
        return self._law_combo

    def motion_path_edit(self) -> QtWidgets.QLineEdit:
        """Input for the swing motion file path."""
        return self._motion_edit

    def study_id_edit(self) -> QtWidgets.QLineEdit:
        """Input for the study identifier."""
        return self._study_id_edit

    def run_study_button(self) -> QtWidgets.QPushButton:
        """Action button to trigger a study run."""
        return self._run_button

    def run_study_status_text(self) -> str:
        """Status/error text for study execution."""
        return self._run_status.text()

    def is_running_study(self) -> bool:
        """True when a study background worker is active."""
        return self._worker is not None and self._worker.isRunning()

    def load_study(self, receipt_path: str | Path) -> None:
        """Open a receipt; any failure clears the plot and shows ERROR."""
        try:
            self._view = core.open_study(receipt_path)
        except (OSError, ValueError) as exc:
            logger.warning("spine study could not be opened: %s", exc)
            self._view = None
            self._clear_plot()
            self._status.setText(f"ERROR: study not loaded ({exc})")
            self._status.setStyleSheet(_WARN_STYLE)
            self._source.setText("distribution_source: -")
            return
        self._status.setText(core.status_banner(self._view))
        qualified = self._view.scientific_status == "qualified"
        self._status.setStyleSheet(_QUALIFIED_STYLE if qualified else _WARN_STYLE)
        self._source.setText(f"distribution_source: {self._view.distribution_source}")
        self._redraw()

    def start_study(
        self,
        payload: Mapping[str, Any] | None = None,
        root: Path | None = None,
    ) -> None:
        """Launch the spine study in a background QThread worker."""
        if self.is_running_study():
            return

        if payload is None:
            study_id = self._study_id_edit.text()
            asset_id = (
                self._model_combo.currentData() or self._model_combo.currentText()
            )
            motion_path = self._motion_edit.text()
            law_data = self._law_combo.currentData()
            coupling_law = law_data if law_data is not None else None
            try:
                study_payload = core.build_study_request_payload(
                    study_id=study_id,
                    asset_id=str(asset_id),
                    motion_path=motion_path,
                    coupling_law_id=coupling_law,
                )
            except (ValueError, TypeError) as exc:
                self._run_status.setText(f"Error: {exc}")
                return
        else:
            study_payload = dict(payload)

        try:
            request = core.build_study_request(study_payload, root=root)
        except (ValueError, TypeError, OSError) as exc:
            self._run_status.setText(f"Error: {exc}")
            return

        self._run_button.setEnabled(False)
        self._run_status.setText("Running spine study...")
        worker = SpineStudyWorker(request, parent=self)
        self._worker = worker
        worker.finished_receipt.connect(self._on_study_finished)
        worker.failed.connect(self._on_study_failed)
        worker.start()

    def wait_for_study(self, timeout_ms: int = 5000) -> bool:
        """Wait for background worker to terminate, pumping Qt events (for tests)."""
        worker = self._worker
        if worker is None:
            return True
        start = time.monotonic()
        timeout_s = timeout_ms / 1000.0
        while worker.isRunning():
            QtCore.QCoreApplication.processEvents()
            if time.monotonic() - start > timeout_s:
                return False
            time.sleep(0.01)
        QtCore.QCoreApplication.processEvents()
        return True

    def cleanup(self) -> None:
        """Release plot and worker resources; safe to call repeatedly."""
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.wait(1000)
            self._worker = None
        self._view = None
        self._clear_plot()

    def _choose_file(self) -> None:  # pragma: no cover - modal dialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Study Receipt", "", "Receipt (*.json)"
        )
        if path:
            self.load_study(path)

    def _browse_motion(self) -> None:  # pragma: no cover - modal dialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Swing Motion", "", "Motion Files (*.sto *.mot)"
        )
        if path:
            self._motion_edit.setText(path)

    def _on_model_changed(self, *_args: object) -> None:
        asset_id = self._model_combo.currentData()
        if not asset_id:
            asset_id = self._model_combo.currentText()
        asset = self._assets_by_id.get(str(asset_id), {})
        status = asset.get("status", "unknown")
        if asset_id == "fixture" or asset.get("is_fixture"):
            self._model_status_label.setText(
                "Model status: fixture (synthetic, unqualified)"
            )
            self._model_status_label.setStyleSheet(_WARN_STYLE)
        elif status == "qualified":
            self._model_status_label.setText("Model status: qualified")
            self._model_status_label.setStyleSheet(_QUALIFIED_STYLE)
        else:
            self._model_status_label.setText(f"Model status: {status}")
            self._model_status_label.setStyleSheet(_WARN_STYLE)

    def _on_study_finished(self, receipt_path: Path) -> None:
        self._run_button.setEnabled(True)
        self._run_status.setText(f"Completed: {receipt_path.name}")
        self.load_study(receipt_path)

    def _on_study_failed(self, error_msg: str) -> None:
        self._run_button.setEnabled(True)
        self._run_status.setText(f"Error: {error_msg}")

    def _clear_plot(self) -> None:
        canvas, self._canvas = self._canvas, None
        self._figure = None
        if canvas is not None:
            self._layout.removeWidget(canvas)
            canvas.deleteLater()

    def _redraw(self, *_args: object) -> None:
        if self._view is None:
            return
        self._clear_plot()
        self._figure = core.heatmap_figure(self._view, self.current_quantity())
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._layout.addWidget(self._canvas, 1)
