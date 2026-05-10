#!/usr/bin/env python
"""
C3D Motion Analysis GUI

Features:
- Load C3D files (via C3DDataReader)
- Inspect metadata, markers, analog channels
- 2D plots of marker/analog time-series
- 3D marker trajectory viewer
- Basic kinematic analysis: speed, path length, extrema
- Consolidated loading path and MVC architecture
"""

import os
import sys
from pathlib import Path

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt

from .core.models import C3DDataModel
from .services.loader_thread import C3DLoaderThread
from .services.marker_export import export_markers
from .ui.tabs.analog_plot_tab import AnalogPlotTab
from .ui.tabs.analysis_tab import AnalysisTab
from .ui.tabs.force_plot_tab import ForcePlotTab
from .ui.tabs.marker_plot_tab import MarkerPlotTab
from .ui.tabs.overview_tab import OverviewTab
from .ui.tabs.segments_tab import SegmentsTab
from .ui.tabs.viewer_3d_tab import Viewer3DTab

# ---------------------------------------------------------------------------
# Embeddable Main Widget
# ---------------------------------------------------------------------------


class MainWidget(QtWidgets.QWidget):
    """Embeddable C3D Motion Analysis viewer widget.

    Hosts the tabbed C3D analysis UI without requiring a top-level
    :class:`QMainWindow`. Used by :class:`C3DViewerMainWindow` (standalone
    shell) and by the launcher embed adapter (tab/dock host).

    The widget owns the model + async loader thread; the embed adapter's
    :meth:`cleanup` is responsible for releasing matplotlib figures held
    by the various plot tabs.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the embeddable widget and create UI components."""
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.model: C3DDataModel | None = None
        self._loader_thread: C3DLoaderThread | None = None

        self._create_actions()
        self._create_central_widget()
        self._update_ui_state(False)

    # ----------------------------- UI setup --------------------------------

    def _create_actions(self) -> None:
        """Create QActions for menu wiring (host-agnostic)."""
        self.action_open = QtGui.QAction("Open &C3D…", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.setStatusTip("Open a C3D file for analysis")
        self.action_open.triggered.connect(self.open_c3d_file)

        self.action_export_markers = QtGui.QAction("&Export markers…", self)
        self.action_export_markers.setStatusTip(
            "Export selected markers, components, and frame range to CSV/JSON/NPZ"
        )
        self.action_export_markers.triggered.connect(self._export_markers_dialog)
        self.action_export_markers.setEnabled(False)
        # Backwards-compatible alias for any existing test references.
        self.action_export_csv = self.action_export_markers

        self.action_about = QtGui.QAction("&About", self)
        self.action_about.triggered.connect(self.show_about_dialog)

    def _create_central_widget(self) -> None:
        """Create the tab widget with all analysis tabs."""
        self.tabs = QtWidgets.QTabWidget(self)

        self.overview_tab = OverviewTab()
        self.marker_plot_tab = MarkerPlotTab()
        self.analog_plot_tab = AnalogPlotTab()
        self.viewer3d_tab = Viewer3DTab()
        self.segments_tab = SegmentsTab()
        self.analysis_tab = AnalysisTab()
        self.force_plot_tab = ForcePlotTab()
        # Plumb segment edits straight into the 3D viewer. Use the v2
        # signal so library / mesh / ellipsoid / capsule shapes survive
        # the trip — the legacy ``segments_changed`` signal drops them.
        self.segments_tab.viz_segments_changed.connect(
            self.viewer3d_tab.set_user_segments
        )

        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.setTabToolTip(0, "Metadata and file information")
        self.tabs.addTab(self.marker_plot_tab, "Markers (2D)")
        self.tabs.setTabToolTip(1, "2D plots of marker trajectories")
        self.tabs.addTab(self.analog_plot_tab, "Analog")
        self.tabs.setTabToolTip(2, "Analog data visualization")
        self.tabs.addTab(self.viewer3d_tab, "3D Viewer")
        self.tabs.setTabToolTip(3, "3D interactive view of markers")
        self.tabs.addTab(self.segments_tab, "Segments")
        self.tabs.setTabToolTip(4, "User-defined marker-pair segments")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.setTabToolTip(5, "Kinematic analysis and calculations")
        self.tabs.addTab(self.force_plot_tab, "Force Plates")
        self.tabs.setTabToolTip(6, "Force plate GRF and COP visualization")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    # ---------------------- UI state management ----------------------------

    def _update_ui_state(self, enabled: bool) -> None:
        """Update the enabled state of UI widgets after loading a model."""
        if not (enabled is not None):
            raise ValueError("enabled must be provided")
        widgets = [self.tabs]
        for w in widgets:
            w.setEnabled(enabled)

    def _status_message(self, text: str) -> None:
        """Forward a status message to the host window's status bar, if any.

        Walks the parent chain looking for a :class:`QMainWindow`; if found
        and it has a non-``None`` status bar, posts ``text`` to it. Embedded
        hosts without a status bar simply drop the message.
        """
        host = self.window()
        if isinstance(host, QtWidgets.QMainWindow):
            sb = host.statusBar()
            if sb is not None:
                sb.showMessage(text)

    def show_about_dialog(self) -> None:
        """Show the about dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About C3D Viewer",
            "C3D Viewer\n\nPart of the Golf Modeling Suite.\n"
            "Uses the consolidated C3DDataReader for consistent ingestion.",
        )

    # --------------------------- File I/O ----------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if not (event is not None):
            raise ValueError("event must be provided")
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                path = urls[0].toLocalFile()
                if path.lower().endswith(".c3d"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle drop event."""
        if not (event is not None):
            raise ValueError("event must be provided")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.load_c3d_file_from_path(path)

    def load_c3d_file_from_path(self, path: str) -> None:
        """Load a C3D file from the given path."""
        # Security validation (F-004)
        # shared module import must be available
        if not (path is not None):
            raise ValueError("path must be provided")
        from shared.python.security.security_utils import validate_path

        suite_root = Path(__file__).parents[6]
        allowed = [
            Path.home(),
            suite_root,
        ]
        try:
            # We use strict=False to allow checking but log/warn if outside,
            # or strict=True if we want to block. Review suggested blocking.
            path = str(validate_path(path, allowed, strict=True))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Security Warning", str(e))
            return

        self._status_message(f"Loading {os.path.basename(path)}... (Async)")

        # Ensure single cursor override
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._update_ui_state(False)  # Disable UI during load

        # Start async worker
        # Keep a reference to prevent garbage collection
        self._loader_thread = C3DLoaderThread(path)
        self._loader_thread.loaded.connect(self._on_load_success)
        self._loader_thread.failed.connect(self._on_load_failure)
        # Ensure we cleanup reference when done
        self._loader_thread.finished.connect(self._on_load_finished)
        self._loader_thread.start()

    def open_c3d_file(self) -> None:
        """Open a file dialog to load a C3D file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open C3D file",
            "",
            "C3D files (*.c3d);;All files (*.*)",
        )
        if path:
            self.load_c3d_file_from_path(path)

    def _on_load_success(self, model: C3DDataModel) -> None:
        """Handle successful model load."""
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self._populate_ui_with_model()
        self._update_ui_state(True)
        self._status_message(f"Loaded {os.path.basename(model.filepath)} successfully.")

    def _on_load_failure(self, error_msg: str) -> None:
        """Handle load failure."""
        if not (error_msg is not None):
            raise ValueError("error_msg must be provided")
        self._status_message("Error loading file.")

        QtWidgets.QMessageBox.critical(
            self,
            "Error loading C3D",
            f"Failed to load file.\n\nError:\n{error_msg}",
        )
        self._update_ui_state(True)  # Re-enable UI (at least menus)

    def _on_load_finished(self) -> None:
        """Cleanup after thread finish."""
        QtWidgets.QApplication.restoreOverrideCursor()
        self._loader_thread = None

    # --------------------- Populate UI from model --------------------------

    def _populate_ui_with_model(self) -> None:
        """Populate UI components with data from the loaded model."""
        if self.model is None:
            return

        self.overview_tab.update_from_model(self.model)
        self.marker_plot_tab.update_from_model(self.model)
        self.analog_plot_tab.update_from_model(self.model)
        self.viewer3d_tab.update_from_model(self.model)
        self.segments_tab.update_from_model(self.model)
        self.analysis_tab.update_from_model(self.model)
        self.force_plot_tab.update_from_model(self.model)
        self.action_export_markers.setEnabled(True)

    def _export_markers_dialog(self) -> None:
        """Open the selective marker-export dialog."""
        if self.model is None:
            QtWidgets.QMessageBox.information(self, "Export markers", "No file loaded.")
            return
        from .ui.dialogs.export_markers_dialog import ExportMarkersDialog

        dlg = ExportMarkersDialog(self.model, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        params = dlg.export_params()
        if params is None:
            return
        try:
            written = export_markers(
                self.model,
                params["marker_names"],
                params["components"],
                params["frame_range"],
                params["fmt"],
                params["path"],
                include_time=params["include_time"],
                include_residual=params["include_residual"],
            )
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"Could not export markers:\n{e}"
            )
            return
        self._status_message(f"Exported markers to {os.path.basename(str(written))}")


# ---------------------------------------------------------------------------
# Main Window — thin shell hosting :class:`MainWidget`.
# ---------------------------------------------------------------------------


class C3DViewerMainWindow(QtWidgets.QMainWindow):
    """Standalone main window for the C3D motion analysis viewer.

    Wraps :class:`MainWidget` with a menu bar, a status bar, and the
    standalone-app window chrome. The launcher embeds :class:`MainWidget`
    directly via the embed adapter and skips this shell.
    """

    def __init__(self) -> None:
        """Initialize the main window and create UI components."""
        super().__init__()

        self.setWindowTitle("C3D Motion Analysis Viewer")
        self.resize(1400, 900)
        # Existing test contract: ``window.acceptDrops()`` returns ``True``
        # and ``window.dropEvent`` forwards to the loader. The actual DnD
        # handling lives on :class:`MainWidget`; we mirror the flag and
        # forward the events below to keep the standalone shell usable.
        self.setAcceptDrops(True)

        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)

        self._create_menus()

        self.action_exit = QtGui.QAction("E&xit", self)
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.triggered.connect(self.close)
        # Add Exit to the File menu after construction to keep the
        # MainWidget host-agnostic (it does not own a File menu).
        if hasattr(self, "_file_menu") and self._file_menu is not None:
            self._file_menu.addSeparator()
            self._file_menu.addAction(self.action_exit)

        if (sb := self.statusBar()) is not None:
            sb.showMessage("Ready")

    # ----------------------------- UI setup --------------------------------

    def _create_menus(self) -> None:
        """Create menu bar wired to :class:`MainWidget`'s actions."""
        menubar = self.menuBar()
        if menubar is None:
            self._file_menu = None
            return

        self._file_menu = menubar.addMenu("&File")
        if self._file_menu is not None:
            self._file_menu.addAction(self._main_widget.action_open)
            self._file_menu.addAction(self._main_widget.action_export_markers)

        help_menu = menubar.addMenu("&Help")
        if help_menu is not None:
            help_menu.addAction(self._main_widget.action_about)

    # --------------- Backwards-compatible attribute proxies ----------------

    # Existing tests and external callers reference attributes that used
    # to live directly on the main window. Forward them to the embedded
    # widget so the refactor does not break the public surface.
    @property
    def model(self) -> C3DDataModel | None:
        return self._main_widget.model

    @model.setter
    def model(self, value: C3DDataModel | None) -> None:
        self._main_widget.model = value

    @property
    def tabs(self) -> QtWidgets.QTabWidget:
        return self._main_widget.tabs

    @property
    def overview_tab(self) -> OverviewTab:
        return self._main_widget.overview_tab

    @property
    def marker_plot_tab(self) -> MarkerPlotTab:
        return self._main_widget.marker_plot_tab

    @property
    def analog_plot_tab(self) -> AnalogPlotTab:
        return self._main_widget.analog_plot_tab

    @property
    def viewer3d_tab(self) -> Viewer3DTab:
        return self._main_widget.viewer3d_tab

    @property
    def segments_tab(self) -> SegmentsTab:
        return self._main_widget.segments_tab

    @property
    def analysis_tab(self) -> AnalysisTab:
        return self._main_widget.analysis_tab

    @property
    def force_plot_tab(self) -> ForcePlotTab:
        return self._main_widget.force_plot_tab

    @property
    def action_open(self) -> QtGui.QAction:
        return self._main_widget.action_open

    @property
    def action_export_markers(self) -> QtGui.QAction:
        return self._main_widget.action_export_markers

    @property
    def action_export_csv(self) -> QtGui.QAction:
        return self._main_widget.action_export_csv

    @property
    def action_about(self) -> QtGui.QAction:
        return self._main_widget.action_about

    # Forward the public file-loading entry points so existing tests that
    # call ``window.load_c3d_file_from_path(...)`` keep working.
    def load_c3d_file_from_path(self, path: str) -> None:
        self._main_widget.load_c3d_file_from_path(path)

    def open_c3d_file(self) -> None:
        self._main_widget.open_c3d_file()

    def show_about_dialog(self) -> None:
        self._main_widget.show_about_dialog()

    # ---- Drag & drop forwarding ------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Forward drag-enter to the embedded :class:`MainWidget`."""
        self._main_widget.dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Forward drop to the embedded :class:`MainWidget`."""
        self._main_widget.dropEvent(event)

    # ---- Internal-method forwarding --------------------------------------

    # Existing tests reach into ``_update_ui_state`` / ``_populate_ui_…`` /
    # ``_on_load_success`` / ``_on_load_failure`` / ``_on_load_finished`` /
    # ``_export_markers_dialog`` directly on the window. Forward them so
    # the refactor preserves the legacy public surface.

    def _update_ui_state(self, enabled: bool) -> None:
        self._main_widget._update_ui_state(enabled)

    def _populate_ui_with_model(self) -> None:
        self._main_widget._populate_ui_with_model()

    def _on_load_success(self, model: C3DDataModel) -> None:
        self._main_widget._on_load_success(model)

    def _on_load_failure(self, error_msg: str) -> None:
        self._main_widget._on_load_failure(error_msg)

    def _on_load_finished(self) -> None:
        self._main_widget._on_load_finished()

    def _export_markers_dialog(self) -> None:
        self._main_widget._export_markers_dialog()


def main() -> None:
    """Launch the C3D motion analysis viewer application."""
    app = QtWidgets.QApplication(sys.argv)
    window = C3DViewerMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
