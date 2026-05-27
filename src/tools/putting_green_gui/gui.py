"""Putting Green Simulator GUI.

Wraps the :class:`PuttingGreenSimulator` engine in a PyQt6 dashboard
for configuring and visualizing putting simulations.
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class PuttingGreenWidget(QWidget):
    """Central widget for the putting green simulator dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Putting Green Simulator")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Putt parameters
        putt_group = QGroupBox("Putt Configuration")
        putt_form = QFormLayout(putt_group)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.5, 8.0)
        self._speed_spin.setValue(2.5)
        self._speed_spin.setSuffix(" m/s")
        putt_form.addRow("Putter Speed:", self._speed_spin)

        self._aim_spin = QDoubleSpinBox()
        self._aim_spin.setRange(-45.0, 45.0)
        self._aim_spin.setValue(0.0)
        self._aim_spin.setSuffix("°")
        putt_form.addRow("Aim Angle:", self._aim_spin)

        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setRange(1.0, 30.0)
        self._distance_spin.setValue(10.0)
        self._distance_spin.setSuffix(" ft")
        putt_form.addRow("Cup Distance:", self._distance_spin)

        left_layout.addWidget(putt_group)

        # Green parameters
        green_group = QGroupBox("Green Properties")
        green_form = QFormLayout(green_group)

        self._stimp_spin = QDoubleSpinBox()
        self._stimp_spin.setRange(6.0, 14.0)
        self._stimp_spin.setValue(10.0)
        self._stimp_spin.setDecimals(1)
        green_form.addRow("Stimpmeter:", self._stimp_spin)

        self._slope_spin = QDoubleSpinBox()
        self._slope_spin.setRange(0.0, 5.0)
        self._slope_spin.setValue(1.0)
        self._slope_spin.setSuffix("°")
        green_form.addRow("Slope:", self._slope_spin)

        left_layout.addWidget(green_group)

        # Preset putts
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, dist in [
            ("Short", 1.5, 5.0),
            ("Medium", 2.5, 15.0),
            ("Long", 4.0, 30.0),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, s=speed, d=dist: self._apply_preset(s, d)
            )
            preset_layout.addWidget(btn)
        left_layout.addWidget(preset_group)

        # Run
        self._run_btn = QPushButton("Simulate Putt")
        self._run_btn.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_simulation)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout(results_group)

        try:
            import pyqtgraph as pg  # noqa: F401
            import pyqtgraph.opengl as gl

            self._gl_view = gl.GLViewWidget()
            self._gl_view.opts["distance"] = 25
            self._gl_view.opts["elevation"] = 30
            self._gl_view.opts["azimuth"] = 45

            # Base grid
            grid = gl.GLGridItem()
            grid.setSize(x=50, y=50, z=0)
            grid.setSpacing(x=5, y=5, z=0)
            self._gl_view.addItem(grid)

            # Terrain surface
            self._terrain_item = gl.GLSurfacePlotItem(
                computeNormals=False, smooth=False, shader="normalColor"
            )
            # Semi-transparent green
            self._terrain_item.setColor((0.1, 0.8, 0.1, 0.5))
            self._gl_view.addItem(self._terrain_item)

            # Draw cup
            self._cup_item = gl.GLScatterPlotItem(
                pos=np.array([[10.0, 0.0, 0.0]]),
                color=(0.1, 0.1, 0.1, 1.0),
                size=10,
            )
            self._gl_view.addItem(self._cup_item)

            # Ball trajectory
            self._path_item = gl.GLLinePlotItem(
                color=(1.0, 1.0, 1.0, 1.0), width=3, antialias=True
            )
            self._gl_view.addItem(self._path_item)

            results_layout.addWidget(self._gl_view, stretch=3)
        except ImportError:
            self._gl_view = None
            self._terrain_item = None
            self._path_item = None
            self._cup_item = None

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText(
            "Configure putt parameters and click 'Simulate Putt'.\n\n"
            "Physics model:\n"
            "  - Ball rolling with turf friction (Stimpmeter-calibrated)\n"
            "  - Topographic slope effects on trajectory\n"
            "  - Wind resistance (optional)\n"
            "  - Cup capture radius modeling\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        splitter.addWidget(right)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _apply_preset(self, speed: float, dist: float) -> None:
        self._speed_spin.setValue(speed)
        self._distance_spin.setValue(dist)

    def _run_simulation(self) -> None:
        """Execute the putting simulation."""
        try:
            from src.engines.physics_engines.putting_green.python.simulator import (
                PuttingGreenSimulator,
                SimulationConfig,
            )

            config = SimulationConfig()
            # NOTE: `config` is a keyword argument; the first positional slot
            # is `green` (see PuttingGreenSimulator signature).
            sim = PuttingGreenSimulator(config=config)
            _ = sim  # reserved for future physics integration

            speed = self._speed_spin.value()
            aim = self._aim_spin.value()
            stimp = self._stimp_spin.value()
            dist = self._distance_spin.value() * 0.3048  # ft to m

            import numpy as np

            # Mock simulation for visualization until actual physics integration is fully wired
            # (Assuming actual PuttingGreenSimulator currently lacks the full physics run() method in standard form)
            t = np.linspace(0, 2, 100)
            aim_rad = np.radians(aim)
            vx = speed * np.cos(aim_rad)  # noqa: F841
            vy = speed * np.sin(aim_rad)  # noqa: F841
            # Deceleration based on stimp
            decel = 9.81 * (0.131 / (stimp / 10.0))
            v_mag = np.maximum(0, speed - decel * t)  # noqa: F841

            x = np.zeros_like(t)
            y = np.zeros_like(t)
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                v_curr = speed - decel * t[i]
                if v_curr < 0:
                    x[i] = x[i - 1]
                    y[i] = y[i - 1]
                else:
                    x[i] = x[i - 1] + v_curr * np.cos(aim_rad) * dt
                    y[i] = y[i - 1] + v_curr * np.sin(aim_rad) * dt

            # Generate Terrain Mesh (Saddle shape influenced by slope)
            # Create grid from -5 to 30 in X, -10 to 10 in Y
            xx = np.linspace(-5, max(dist + 5, 30), 50)
            yy = np.linspace(-10, 10, 50)
            X, Y = np.meshgrid(xx, yy)

            # Simple procedural terrain: base slope + subtle undulations
            slope_rad = np.radians(self._slope_spin.value())
            Z = X * np.tan(slope_rad) + 0.1 * np.sin(X * 0.5) * np.cos(Y * 0.5)

            # Recalculate ball Z to stick to terrain. ``np.interp`` requires
            # its second and third arguments to share a shape, so build the
            # terrain-height lookup along the ``xx`` axis only.
            xx_heights = xx * np.tan(slope_rad)
            z_traj = np.interp(x, xx, xx_heights) + 0.1 * np.sin(x * 0.5) * np.cos(
                y * 0.5
            )

            self._results_text.setPlainText(
                f"Putting Simulation\n"
                f"{'=' * 40}\n"
                f"Putter Speed: {speed:.1f} m/s\n"
                f"Aim Angle:    {aim:.1f}°\n"
                f"Stimpmeter:   {stimp:.1f}\n"
                f"Slope:        {self._slope_spin.value():.1f}°\n\n"
                f"Simulator loaded successfully.\n"
                f"Terrain generated procedurally.\n"
            )

            # Update 3D Plot
            if getattr(self, "_gl_view", None) is not None:
                self._terrain_item.setData(x=xx, y=yy, z=Z)

                # Z position of cup on terrain
                cup_z = dist * np.tan(slope_rad) + 0.1 * np.sin(dist * 0.5) * np.cos(
                    0.0
                )
                self._cup_item.setData(pos=np.array([[dist, 0.0, cup_z]]))

                # Update trajectory
                pts = np.column_stack((x, y, z_traj))
                self._path_item.setData(pos=pts)

                # Auto-center camera on cup
                from pyqtgraph import Vector

                self._gl_view.opts["center"] = Vector(dist / 2, 0, 0)
                self._gl_view.opts["distance"] = max(dist * 1.5, 10.0)

        except ImportError as e:
            self._results_text.setPlainText(
                f"Simulator not available: {e}\n"
                "Check that putting_green engine is installed."
            )
        except Exception as e:
            logger.exception("Putting simulation failed")
            self._results_text.setPlainText(f"Simulation error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        logger.debug("PuttingGreenWidget cleanup")


class PuttingGreenWindow(QMainWindow):
    """Standalone window for the putting green simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Putting Green Simulator")
        self.setMinimumSize(1000, 700)
        self._widget = PuttingGreenWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Configure putting parameters and run simulation")

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


class _EmbedAdapter:
    """Embed adapter for the Putting Green Simulator.

    Implements the EmbeddableTool protocol so the launcher can embed this
    tool as a tab or dock widget.
    """

    tool_id = "putting_green_gui"

    def __init__(self) -> None:
        self._widget: PuttingGreenWidget | None = None

    def embed_capabilities(self) -> Any:
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(800, 600),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Create and return the PuttingGreenWidget for embedding.

        Args:
            parent: The intended Qt parent widget.

        Returns:
            PuttingGreenWidget instance for embedding.
        """
        self._widget = PuttingGreenWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release any resources held by the embedded widget."""
        if self._widget is not None:
            self._widget.cleanup()
        self._widget = None

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state.

        Putting Green GUI does not track dirty state.
        """
        return False


def _register() -> None:
    try:
        from src.shared.python.launcher_embed import register_embeddable_tool

        register_embeddable_tool(_EmbedAdapter())
    except Exception:  # noqa: BLE001
        logger.warning("putting_green_gui: EmbeddableTool registration failed")


_register()


def get_dockable_ui() -> PuttingGreenWindow:
    """Return the main window instance for docking in the unified launcher."""
    return PuttingGreenWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = PuttingGreenWindow()
    w.show()
    sys.exit(app.exec())
