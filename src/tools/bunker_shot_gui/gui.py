"""Bunker Shot 3D Simulator GUI.

Provides an experimental dashboard for sand impact simulation
using Chrono DEM (mocked for now) with particle visualization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui import HoverCopyTextBrowser

logger = logging.getLogger(__name__)

# Try to import pyqtgraph
try:
    import pyqtgraph as pg  # noqa: F401
    import pyqtgraph.opengl as gl

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class BunkerShotWidget(QWidget):
    """Central widget for the Bunker Shot simulator dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Bunker Shot 3D Simulator (Experimental)")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Impact parameters
        impact_group = QGroupBox("Impact Configuration")
        impact_form = QFormLayout(impact_group)

        self._velocity_spin = QDoubleSpinBox()
        self._velocity_spin.setRange(10.0, 60.0)
        self._velocity_spin.setValue(35.0)
        self._velocity_spin.setSuffix(" m/s")
        impact_form.addRow("Clubhead Velocity:", self._velocity_spin)

        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(10.0, 60.0)
        self._angle_spin.setValue(45.0)
        self._angle_spin.setSuffix("°")
        impact_form.addRow("Attack Angle:", self._angle_spin)

        self._depth_spin = QDoubleSpinBox()
        self._depth_spin.setRange(1.0, 10.0)
        self._depth_spin.setValue(3.0)
        self._depth_spin.setSuffix(" cm")
        impact_form.addRow("Entry Depth:", self._depth_spin)

        left_layout.addWidget(impact_group)

        # Run
        self._run_btn = QPushButton("Simulate Impact")
        self._run_btn.setStyleSheet(
            "background-color: #D2B48C; color: black; font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_simulation)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        results_group = QGroupBox("Simulation Results (Chrono DEM)")
        results_layout = QVBoxLayout(results_group)

        if PYQTGRAPH_AVAILABLE:
            self._gl_view = gl.GLViewWidget()
            self._gl_view.opts["distance"] = 0.5
            self._gl_view.opts["elevation"] = 20
            self._gl_view.opts["azimuth"] = 45

            grid = gl.GLGridItem()
            grid.setSize(x=2, y=2, z=0)
            grid.setSpacing(x=0.1, y=0.1, z=0)
            self._gl_view.addItem(grid)

            # Particles
            self._particles_item = None

            # Impact Vector
            self._vector_item = None

            results_layout.addWidget(self._gl_view, stretch=3)
        else:
            self._gl_view = None

        self._results_text = HoverCopyTextBrowser()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText(
            "Configure impact parameters and simulate.\n\n"
            "Backend: Chrono DEM (Discrete Element Method)\n"
            "Visualizing sand particle displacement and impact forces."
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        splitter.addWidget(right)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _run_simulation(self) -> None:
        """Execute the bunker shot simulation."""
        v = self._velocity_spin.value()
        angle = self._angle_spin.value()
        depth = self._depth_spin.value() / 100.0  # cm to m

        # Mock simulation of sand particles
        num_particles = int(v * depth * 50000)
        num_particles = min(max(num_particles, 500), 10000)  # limit for GUI

        # Spray pattern based on angle
        angle_rad = np.radians(angle)
        vx = v * np.cos(angle_rad)  # noqa: F841
        vy = v * np.sin(angle_rad)  # noqa: F841

        impact_force = 0.5 * 0.3 * (v**2)  # dummy kinetic energy

        self._results_text.setPlainText(
            f"Bunker Shot Impact\n"
            f"{'=' * 40}\n"
            f"Club Velocity: {v:.1f} m/s\n"
            f"Attack Angle:  {angle:.1f}°\n"
            f"Entry Depth:   {depth * 100:.1f} cm\n"
            f"Est. Force:    {impact_force:.1f} N\n\n"
            f"Displaced sand particles: ~{num_particles * 10}\n"
            f"Chrono DEM simulation mock completed."
        )

        if getattr(self, "_gl_view", None) is not None:
            if self._particles_item is not None:
                self._gl_view.removeItem(self._particles_item)
            if self._vector_item is not None:
                self._gl_view.removeItem(self._vector_item)

            # Generate random particles spraying forward and up
            pos = np.random.normal(size=(num_particles, 3)) * 0.05
            # Offset them based on impact direction
            pos[:, 0] += np.random.uniform(0, 0.5, num_particles) * (v / 60.0)
            pos[:, 2] += np.random.uniform(0, 0.3, num_particles) * np.sin(angle_rad)
            # Make sure they are mostly above ground
            pos[:, 2] = np.abs(pos[:, 2])

            color = np.ones((num_particles, 4))
            color[:, 0] = 0.8  # R
            color[:, 1] = 0.7  # G
            color[:, 2] = 0.5  # B
            color[:, 3] = 0.8  # A

            self._particles_item = gl.GLScatterPlotItem(
                pos=pos, color=color, size=3, pxMode=True
            )
            self._gl_view.addItem(self._particles_item)

            # Impact Vector (Clubhead)
            v_start = np.array([-0.2 * np.cos(angle_rad), 0, 0.2 * np.sin(angle_rad)])
            v_end = np.array([0, 0, -depth])
            self._vector_item = gl.GLLinePlotItem(
                pos=np.vstack((v_start, v_end)),
                color=(1.0, 0.2, 0.2, 1.0),
                width=5,
                antialias=True,
            )
            self._gl_view.addItem(self._vector_item)

    def cleanup(self) -> None:
        """Release resources."""


class BunkerShotWindow(QMainWindow):
    """Standalone window for the Bunker Shot simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bunker Shot 3D Simulator")
        self.setMinimumSize(1000, 700)
        self._widget = BunkerShotWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready. (Chrono DEM Experimental)")

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


class _EmbedAdapter:
    """Embed adapter for the Bunker Shot 3D Simulator."""

    tool_id = "bunker_shot_gui"

    def __init__(self) -> None:
        self._widget: BunkerShotWidget | None = None

    def embed_capabilities(self) -> Any:
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(800, 600),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        self._widget = BunkerShotWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        if self._widget is not None:
            self._widget.cleanup()
        self._widget = None

    def is_dirty(self) -> bool:
        return False


def _register() -> None:
    try:
        from src.shared.python.launcher_embed import register_embeddable_tool

        register_embeddable_tool(_EmbedAdapter())
    except Exception:  # noqa: BLE001
        logger.warning("bunker_shot_gui: EmbeddableTool registration failed")


_register()


def get_dockable_ui() -> BunkerShotWindow:
    """Return the main window instance for docking in the unified launcher."""
    return BunkerShotWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = BunkerShotWindow()
    w.show()
    sys.exit(app.exec())
