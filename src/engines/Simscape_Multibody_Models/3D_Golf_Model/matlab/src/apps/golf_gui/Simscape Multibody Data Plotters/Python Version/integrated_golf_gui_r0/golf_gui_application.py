#!/usr/bin/env python3
"""Golf Swing Visualizer - Tabular GUI Application.

Supports multiple data sources including motion capture and future Simulink models.

Decomposed into focused modules (SRP):
- golf_playback_controller.py: SmoothPlaybackController
- golf_gui_tabs.py: MotionCaptureTab, SimulinkModelTab, ComparisonTab
- golf_visualizer_widget.py: GolfVisualizerWidget (OpenGL)
- golf_gui_styles.py: QSS stylesheet constants
"""

from __future__ import annotations

import logging
import sys

from golf_gui_styles import get_full_modern_style
from golf_gui_tabs import ComparisonTab, MotionCaptureTab, SimulinkModelTab

# Re-export extracted classes for backward compatibility
from golf_playback_controller import SmoothPlaybackController  # noqa: F401
from golf_video_export import VideoExportDialog
from golf_visualizer_widget import GolfVisualizerWidget  # noqa: F401
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN WINDOW
# ============================================================================


class GolfVisualizerMainWindow(QMainWindow):
    """Main window for the Golf Swing Visualizer with tabular interface."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Golf Swing Visualizer - Multi-Data Analysis Platform")
        self.setGeometry(100, 100, 1200, 800)

        # Apply modern white theme
        self.setStyleSheet(get_full_modern_style())

        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()

        logger.info("[*] Golf Visualizer main window created")

    def _setup_ui(self) -> None:
        """Setup the main UI with tabular structure."""
        # Create central widget with tab widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QVBoxLayout(self.central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        # Add tabs
        self.motion_capture_tab = MotionCaptureTab(self)
        self.simulink_tab = SimulinkModelTab(self)
        self.comparison_tab = ComparisonTab(self)

        self.tab_widget.addTab(self.motion_capture_tab, "Motion Capture Data")
        self.tab_widget.addTab(self.simulink_tab, "Simulink Model")
        self.tab_widget.addTab(self.comparison_tab, "Data Comparison")

        main_layout.addWidget(self.tab_widget)

        # Add global controls
        global_controls = self._create_global_controls()
        main_layout.addWidget(global_controls)

    def _create_global_controls(self) -> QGroupBox:
        """Create global control panel."""
        panel = QGroupBox("Global Controls")
        layout = QGridLayout()

        # Camera view buttons
        layout.addWidget(QLabel("Camera Views:"), 0, 0)

        self.face_on_btn = QPushButton("Face-On (1)")
        self.face_on_btn.setMaximumWidth(100)
        self.face_on_btn.clicked.connect(self._set_face_on_view)
        layout.addWidget(self.face_on_btn, 0, 1)

        self.down_line_btn = QPushButton("Down-Line (2)")
        self.down_line_btn.setMaximumWidth(100)
        self.down_line_btn.clicked.connect(self._set_down_line_view)
        layout.addWidget(self.down_line_btn, 0, 2)

        self.behind_btn = QPushButton("Behind (3)")
        self.behind_btn.setMaximumWidth(100)
        self.behind_btn.clicked.connect(self._set_behind_view)
        layout.addWidget(self.behind_btn, 1, 1)

        self.above_btn = QPushButton("Above (4)")
        self.above_btn.setMaximumWidth(100)
        self.above_btn.clicked.connect(self._set_above_view)
        layout.addWidget(self.above_btn, 1, 2)

        # Reset camera button
        self.reset_camera_btn = QPushButton("Reset Camera (R)")
        self.reset_camera_btn.setMaximumWidth(120)
        self.reset_camera_btn.clicked.connect(self._reset_camera)
        layout.addWidget(self.reset_camera_btn, 2, 1, 1, 2)

        # Visualization toggles
        layout.addWidget(QLabel("Visualization:"), 3, 0)

        self.show_face_normal_cb = QCheckBox("Face Normal")
        self.show_face_normal_cb.setChecked(True)
        self.show_face_normal_cb.stateChanged.connect(self._toggle_face_normal)
        layout.addWidget(self.show_face_normal_cb, 3, 1)

        self.show_ball_cb = QCheckBox("Ball")
        self.show_ball_cb.setChecked(True)
        self.show_ball_cb.stateChanged.connect(self._toggle_ball)
        layout.addWidget(self.show_ball_cb, 3, 2)

        panel.setLayout(layout)
        return panel

    def _setup_menu(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load Motion Capture Data", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self._load_motion_capture_data)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Export menu
        export_menu = menubar.addMenu("Export")

        export_video_action = QAction("Export Video...", self)
        export_video_action.setShortcut("Ctrl+E")
        export_video_action.triggered.connect(self._export_video)
        export_menu.addAction(export_video_action)

        # View menu
        view_menu = menubar.addMenu("View")

        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.setShortcut("R")
        reset_camera_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_camera_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Select a tab to begin analysis")

    def _load_motion_capture_data(self) -> None:
        """Load motion capture data."""
        self.tab_widget.setCurrentIndex(0)
        self.motion_capture_tab._load_motion_capture_data()

    def _export_video(self) -> None:
        """Export current animation to high-quality video."""
        tab = self.motion_capture_tab

        if not tab.frame_processor or not tab.opengl_widget.renderer:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load motion capture data before exporting video.\n\n"
                "Use File -> Load Motion Capture Data to get started.",
            )
            return

        dialog = VideoExportDialog(
            self, tab.opengl_widget.renderer, tab.frame_processor
        )
        dialog.exec()

    def _reset_camera(self) -> None:
        """Reset camera to default position."""
        if hasattr(self, "gl_widget") and self.gl_widget:
            self.gl_widget._frame_camera_to_data()
            self.gl_widget.update()

    def _set_face_on_view(self) -> None:
        """Set face-on camera view."""
        if hasattr(self, "gl_widget") and self.gl_widget:
            self.gl_widget.set_face_on_view()

    def _set_down_line_view(self) -> None:
        """Set down-the-line camera view."""
        if hasattr(self, "gl_widget") and self.gl_widget:
            self.gl_widget.set_down_the_line_view()

    def _set_behind_view(self) -> None:
        """Set behind camera view."""
        if hasattr(self, "gl_widget") and self.gl_widget:
            self.gl_widget.set_behind_view()

    def _set_above_view(self) -> None:
        """Set overhead camera view."""
        if hasattr(self, "gl_widget") and self.gl_widget:
            self.gl_widget.set_above_view()

    def _toggle_face_normal(self, state) -> None:
        """Toggle face normal visibility."""
        if (
            hasattr(self, "gl_widget")
            and self.gl_widget
            and self.gl_widget.current_render_config
        ):
            self.gl_widget.current_render_config.show_face_normal = bool(state)
            self.gl_widget.update()

    def _toggle_ball(self, state) -> None:
        """Toggle ball visibility."""
        if (
            hasattr(self, "gl_widget")
            and self.gl_widget
            and self.gl_widget.current_render_config
        ):
            self.gl_widget.current_render_config.show_ball = bool(state)
            self.gl_widget.update()

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Golf Swing Visualizer",
            "Golf Swing Visualizer - Multi-Data Analysis Platform\n\n"
            "Version: 2.0\n"
            "Features:\n"
            "- Motion capture data visualization\n"
            "- Future Simulink model integration\n"
            "- Hand midpoint tracking analysis\n"
            "- Real-time 3D rendering\n\n"
            "Built with PyQt6 and ModernGL",
        )


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Golf Swing Visualizer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Golf Analysis Lab")

    # Create and show main window
    window = GolfVisualizerMainWindow()
    window.show()

    logger.info("[*] Golf Swing Visualizer started")
    logger.info("   Tabular interface ready for multi-data analysis")
    logger.info("   Motion capture data visualization active")
    logger.info("   Simulink model integration prepared for future use")

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
