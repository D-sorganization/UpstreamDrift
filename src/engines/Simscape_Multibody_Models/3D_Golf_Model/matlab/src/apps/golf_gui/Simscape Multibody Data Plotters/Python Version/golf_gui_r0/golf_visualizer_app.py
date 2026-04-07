# mypy: disable-error-code="no-redef,union-attr"
"""Application shell for the legacy golf swing visualizer."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    from .golf_visualizer_widget import ModernGolfVisualizerWidget
except ImportError:
    from golf_visualizer_widget import ModernGolfVisualizerWidget

logger = logging.getLogger(__name__)


class ModernGolfVisualizerApp(QMainWindow):
    """Main application window with modern UI"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modern Golf Swing Visualizer")
        self.setGeometry(100, 100, 1600, 900)
        self.gl_widget = ModernGolfVisualizerWidget()
        self.setCentralWidget(self.gl_widget)
        self._create_control_panels()
        self._create_menubar()
        self._create_toolbar()
        self._create_status_bar()
        self._apply_modern_style()

    def _create_control_panels(self) -> None:
        """Create modern control panels"""
        playback_dock = QDockWidget("Playback Controls", self)
        playback_widget = self._create_playback_controls()
        playback_dock.setWidget(playback_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, playback_dock)
        vis_dock = QDockWidget("Visualization", self)
        vis_widget = self._create_visualization_controls()
        vis_dock.setWidget(vis_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, vis_dock)
        perf_dock = QDockWidget("Performance", self)
        perf_widget = self._create_performance_monitor()
        perf_dock.setWidget(perf_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, perf_dock)

    def _create_menubar(self) -> None:
        """Create a minimal menu bar for loading bundled sample data."""
        file_menu = self.menuBar().addMenu("File")
        load_sample_action = QAction("Load Sample Data", self)
        load_sample_action.triggered.connect(self._load_sample_data)
        file_menu.addAction(load_sample_action)

    def _create_toolbar(self) -> None:
        """Create a compact toolbar for playback and data loading."""
        toolbar = QToolBar("Playback", self)
        toolbar.addAction("Load Sample Data", self._load_sample_data)
        toolbar.addAction("Play/Pause", self._toggle_playback)
        self.addToolBar(toolbar)

    def _create_status_bar(self) -> None:
        """Create the status bar used by the performance panel."""
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    def _create_playback_controls(self) -> QWidget:
        """Create modern playback control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.play_button = QPushButton("Play")
        self.play_button.setMinimumHeight(40)
        self.play_button.clicked.connect(self._toggle_playback)
        layout.addWidget(self.play_button)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1000)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self.frame_slider)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)
        return widget

    def _create_visualization_controls(self) -> QWidget:
        """Create visualization control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        forces_group = QGroupBox("Forces")
        forces_layout = QVBoxLayout(forces_group)
        self.force_checkboxes = {}
        for dataset in ["BASEQ", "ZTCFQ", "DELTAQ"]:
            cb = QCheckBox(f"{dataset} Forces")
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, ds=dataset: self._toggle_forces(ds, state)
            )  # noqa: E501
            forces_layout.addWidget(cb)
            self.force_checkboxes[dataset] = cb
        layout.addWidget(forces_group)
        body_group = QGroupBox("Body Segments")
        body_layout = QVBoxLayout(body_group)
        self.body_checkboxes = {}
        segments = [
            "left_forearm",
            "left_upper_arm",
            "right_forearm",
            "right_upper_arm",
            "left_shoulder_neck",
            "right_shoulder_neck",
        ]
        for segment in segments:
            cb = QCheckBox(segment.replace("_", " ").title())
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, seg=segment: self._toggle_body_segment(seg, state)
            )
            body_layout.addWidget(cb)
            self.body_checkboxes[segment] = cb
        layout.addWidget(body_group)
        return widget

    def _create_performance_monitor(self) -> QWidget:
        """Create performance monitoring panel"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.fps_label = QLabel("FPS: 0")
        self.frame_label = QLabel("Frame: 0/0")
        self.time_label = QLabel("Time: 0.00s")
        layout.addWidget(self.fps_label)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.time_label)
        layout.addStretch()
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self._update_performance_display)
        self.perf_timer.start(100)
        return widget

    def _load_sample_data(self) -> None:
        """Load the co-located sample MAT files when present."""
        try:
            self.gl_widget.load_data("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")
            if self.gl_widget.num_frames > 0:
                self.frame_slider.setMaximum(self.gl_widget.num_frames - 1)
            self.statusBar().showMessage("Loaded bundled sample data")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Sample data load failed: %s", exc)
            self.statusBar().showMessage("Sample data not available")

    def _toggle_playback(self) -> None:
        """Toggle playback in the OpenGL widget."""
        if self.gl_widget.is_playing:
            self.gl_widget.pause_animation()
            self.play_button.setText("Play")
        else:
            self.gl_widget.play_animation()
            self.play_button.setText("Pause")

    def _on_frame_slider_changed(self, frame_idx: int) -> None:
        """Seek the visualizer to a specific frame."""
        self.gl_widget.set_frame(frame_idx)
        self.frame_label.setText(f"Frame: {frame_idx}/{self.gl_widget.num_frames}")

    def _on_speed_changed(self, speed: int) -> None:
        """Update playback speed from the slider."""
        self.gl_widget.playback_speed = max(speed, 1) / 100.0
        self.speed_label.setText(f"{self.gl_widget.playback_speed:.1f}x")
        if self.gl_widget.is_playing:
            interval = max(1, int(33 / self.gl_widget.playback_speed))
            self.gl_widget.animation_timer.start(interval)

    def _toggle_forces(self, dataset: str, state: int) -> None:
        """Toggle a force overlay dataset."""
        self.gl_widget.render_config.show_forces[dataset] = bool(state)
        self.gl_widget.update()

    def _toggle_body_segment(self, segment: str, state: int) -> None:
        """Toggle a rendered body segment."""
        self.gl_widget.render_config.show_body_segments[segment] = bool(state)
        self.gl_widget.update()

    def _update_performance_display(self) -> None:
        """Refresh the basic performance and playback labels."""
        self.fps_label.setText(f"FPS: {self.gl_widget.fps:.1f}")
        self.frame_label.setText(
            f"Frame: {self.gl_widget.current_frame}/{self.gl_widget.num_frames}"
        )
        current_time = self.gl_widget.current_frame * 0.001
        self.time_label.setText(f"Time: {current_time:.2f}s")
        if self.gl_widget.num_frames > 0:
            self.frame_slider.setMaximum(self.gl_widget.num_frames - 1)

    def _apply_modern_style(self) -> None:
        """Apply modern dark theme styling"""
        style = self._get_main_window_style()
        style += self._get_dock_widget_style()
        style += self._get_button_style()
        style += self._get_slider_style()
        style += self._get_checkbox_style()
        style += self._get_groupbox_style()
        self.setStyleSheet(style)

    @staticmethod
    def _get_main_window_style() -> str:
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        """

    @staticmethod
    def _get_dock_widget_style() -> str:
        return """
        QDockWidget {
            color: #ffffff;
            background-color: #3c3c3c;
        }
        QDockWidget::title {
            background-color: #4a4a4a;
            padding: 5px;
            border: 1px solid #5a5a5a;
        }
        """

    @staticmethod
    def _get_button_style() -> str:
        return """
        QPushButton {
            background-color: #4a4a4a;
            border: 1px solid #6a6a6a;
            color: #ffffff;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        """

    @staticmethod
    def _get_slider_style() -> str:
        return """
        QSlider::groove:horizontal {
            border: 1px solid #5a5a5a;
            height: 8px;
            background: #3a3a3a;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #005a9e;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        """

    @staticmethod
    def _get_checkbox_style() -> str:
        return """
        QCheckBox {
            color: #ffffff;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 13px;
            height: 13px;
        }
        QCheckBox::indicator:unchecked {
            background-color: #3a3a3a;
            border: 1px solid #6a6a6a;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d4;
            border: 1px solid #005a9e;
        }
        """

    @staticmethod
    def _get_groupbox_style() -> str:
        return """
        QGroupBox {
            color: #ffffff;
            border: 2px solid #5a5a5a;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        """
