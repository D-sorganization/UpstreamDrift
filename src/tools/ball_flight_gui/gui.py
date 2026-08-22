"""Aerodynamic Ball Flight Simulator GUI.

Wraps the aerodynamics engine and ball flight simulators in a PyQt6
dashboard for configuring launch conditions and visualizing trajectories
with full aerodynamic force decomposition.
"""

from __future__ import annotations

import logging
import math
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
from src.shared.python.ui.pane_layout import install_two_pane_splitter

logger = logging.getLogger(__name__)

_MPH_TO_MS = 0.44704
_FT_TO_M = 0.3048


def build_wind_vector(speed_mph: float, direction_deg: float) -> np.ndarray:
    """Return the wind velocity vector [m/s] for the simulator frame.

    Convention (documented in the GUI): ``direction_deg`` is the direction
    the wind blows TOWARD, measured from downrange (+x) toward the left
    (+y).  0° is a pure tailwind, 180° a pure headwind.

    Args:
        speed_mph: Wind speed in mph (>= 0).
        direction_deg: Wind bearing in degrees as described above.

    Returns:
        (3,) float array; z-component is always 0.
    """
    if speed_mph < 0.0:
        raise ValueError(f"wind speed must be >= 0 mph, got {speed_mph!r}")
    speed_ms = speed_mph * _MPH_TO_MS
    theta = math.radians(direction_deg)
    return np.array([speed_ms * math.cos(theta), speed_ms * math.sin(theta), 0.0])


def combine_spins(backspin_rpm: float, sidespin_rpm: float) -> tuple[float, np.ndarray]:
    """Combine backspin and sidespin into total spin rate + unit axis.

    Backspin acts about -y; positive sidespin curves the ball to the
    right of the target line (fade/slice), i.e. about -z in the simulator
    frame [x=forward, y=left, z=up].

    Args:
        backspin_rpm: Backspin magnitude [rpm], >= 0.
        sidespin_rpm: Sidespin [rpm]; positive = rightward curve.

    Returns:
        Tuple of (total spin rate [rpm], unit spin-axis vector).
    """
    if backspin_rpm < 0.0:
        raise ValueError(f"backspin must be >= 0 rpm, got {backspin_rpm!r}")
    total = math.hypot(backspin_rpm, sidespin_rpm)
    if total < 1e-12:
        return 0.0, np.array([0.0, -1.0, 0.0])
    axis = np.array([0.0, -backspin_rpm, -sidespin_rpm]) / total
    return total, axis


class BallFlightWidget(QWidget):
    """Central widget for the aerodynamic ball flight simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        install_two_pane_splitter(
            self, self._build_controls_panel(), self._build_results_panel()
        )

    def _build_controls_panel(self) -> QWidget:
        """Left pane: title, launch conditions, environment, presets, run."""
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Aerodynamic Ball Flight Simulator")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        left_layout.addWidget(self._build_launch_group())
        left_layout.addWidget(self._build_env_group())
        left_layout.addWidget(self._build_aero_group())
        left_layout.addWidget(self._build_preset_group())

        # Run
        self._run_btn = QPushButton("Simulate Flight")
        self._run_btn.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_simulation)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        return left

    def _build_launch_group(self) -> QGroupBox:
        """Launch-condition spin boxes (speed, angle, backspin, sidespin)."""
        launch_group = QGroupBox("Launch Conditions")
        launch_form = QFormLayout(launch_group)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(50.0, 200.0)
        self._speed_spin.setValue(163.0)
        self._speed_spin.setSuffix(" mph")
        launch_form.addRow("Ball Speed:", self._speed_spin)

        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(-5.0, 45.0)
        self._angle_spin.setValue(11.0)
        self._angle_spin.setSuffix("°")
        launch_form.addRow("Launch Angle:", self._angle_spin)

        self._spin_spin = QDoubleSpinBox()
        self._spin_spin.setRange(0.0, 12000.0)
        self._spin_spin.setValue(2500.0)
        self._spin_spin.setSuffix(" rpm")
        launch_form.addRow("Backspin:", self._spin_spin)

        self._sidespin_spin = QDoubleSpinBox()
        self._sidespin_spin.setRange(-5000.0, 5000.0)
        self._sidespin_spin.setValue(0.0)
        self._sidespin_spin.setSuffix(" rpm")
        launch_form.addRow("Sidespin:", self._sidespin_spin)

        return launch_group

    def _build_env_group(self) -> QGroupBox:
        """Environment spin boxes (wind speed/direction, altitude)."""
        env_group = QGroupBox("Environment")
        env_form = QFormLayout(env_group)

        self._wind_speed = QDoubleSpinBox()
        self._wind_speed.setRange(0.0, 50.0)
        self._wind_speed.setValue(0.0)
        self._wind_speed.setSuffix(" mph")
        env_form.addRow("Wind Speed:", self._wind_speed)

        self._wind_dir = QDoubleSpinBox()
        self._wind_dir.setRange(0.0, 360.0)
        self._wind_dir.setValue(0.0)
        self._wind_dir.setSuffix("°")
        self._wind_dir.setToolTip(
            "Direction the wind blows toward, from downrange (+x) toward "
            "the left: 0° = tailwind, 90° = left-to-right carry, "
            "180° = headwind."
        )
        env_form.addRow("Wind Direction:", self._wind_dir)

        self._altitude = QDoubleSpinBox()
        self._altitude.setRange(0.0, 10000.0)
        self._altitude.setValue(0.0)
        self._altitude.setSuffix(" ft")
        env_form.addRow("Altitude:", self._altitude)

        return env_group

    def _build_aero_group(self) -> QGroupBox:
        """Aerodynamic model description. The old checkboxes (dimple geometry,
        Magnus toggle, seam orientation) were decorative — the simulator has
        no backing parameter for any of them, so they were removed rather
        than left silently ignored (issue #8818)."""
        aero_group = QGroupBox("Aerodynamic Model (fixed)")
        aero_layout = QVBoxLayout(aero_group)
        aero_label = QLabel(
            "Modeled: Reynolds-dependent drag, Magnus lift, gravity,\n"
            "wind, altitude air density.\n"
            "Not modeled: dimple geometry, seam orientation."
        )
        aero_label.setWordWrap(True)
        aero_layout.addWidget(aero_label)
        return aero_group

    def _build_preset_group(self) -> QGroupBox:
        """Club preset buttons that stamp speed/angle/backspin triples."""
        preset_group = QGroupBox("Club Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, angle, spin in [
            ("Driver", 163.0, 11.0, 2500.0),
            ("7-Iron", 118.0, 16.0, 7000.0),
            ("PW", 94.0, 23.0, 9000.0),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, s=speed, a=angle, sp=spin: self._apply_preset(s, a, sp)
            )
            preset_layout.addWidget(btn)
        return preset_group

    def _build_results_panel(self) -> QWidget:
        """Right pane: 3D trajectory view (when available) and results text."""
        right = QWidget()
        right_layout = QVBoxLayout(right)
        results_group = QGroupBox("Flight Results")
        results_layout = QVBoxLayout(results_group)

        try:
            import pyqtgraph as pg  # noqa: F401
            import pyqtgraph.opengl as gl

            self._gl_view = gl.GLViewWidget()
            self._gl_view.opts["distance"] = 250
            self._gl_view.opts["elevation"] = 20
            self._gl_view.opts["azimuth"] = 45

            grid = gl.GLGridItem()
            grid.setSize(x=500, y=500, z=0)
            grid.setSpacing(x=50, y=50, z=0)
            self._gl_view.addItem(grid)

            axis = gl.GLAxisItem()
            axis.setSize(x=50, y=50, z=50)
            self._gl_view.addItem(axis)

            self._plot_item = None
            results_layout.addWidget(self._gl_view, stretch=3)
        except ImportError:
            self._gl_view = None
            self._plot_item = None

        self._results_text = HoverCopyTextBrowser()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText(
            "Configure launch conditions and click 'Simulate Flight'.\n\n"
            "Aerodynamic forces modeled:\n"
            "  - Drag (Reynolds-dependent Cd)\n"
            "  - Magnus lift (spin-induced)\n"
            "  - Gravity\n"
            "  - Wind (speed + direction)\n"
            "  - Altitude/air density (ISA)\n\n"
            "Not modeled: dimple geometry, seam orientation.\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        return right

    def _apply_preset(self, speed: float, angle: float, spin: float) -> None:
        self._speed_spin.setValue(speed)
        self._angle_spin.setValue(angle)
        self._spin_spin.setValue(spin)

    def _build_environment(self) -> Any:
        """Build ``EnvironmentalConditions`` from the environment widgets.

        Wind speed/direction become the wind vector; altitude derives the
        air density via the ISA model (issue #8818 — these controls were
        previously never read by the simulation).
        """
        from src.shared.python.physics.ball_launch_conditions import (
            EnvironmentalConditions,
        )

        wind = build_wind_vector(self._wind_speed.value(), self._wind_dir.value())
        altitude_m = self._altitude.value() * _FT_TO_M
        return EnvironmentalConditions.from_altitude(
            altitude_m=altitude_m,
            wind_velocity=wind,
        )

    def _build_launch(self) -> Any:
        """Build ``LaunchConditions`` from the launch widgets.

        Sidespin is combined with backspin into a total spin rate and a
        tilted spin axis (issue #8818 — sidespin was previously ignored).
        """
        from src.shared.python.physics.ball_launch_conditions import (
            LaunchConditions,
        )

        speed_ms = self._speed_spin.value() * _MPH_TO_MS
        spin_rpm, spin_axis = combine_spins(
            self._spin_spin.value(), self._sidespin_spin.value()
        )
        return LaunchConditions.from_user_units(
            velocity=speed_ms,
            launch_angle_deg=self._angle_spin.value(),
            spin_rate_rpm=spin_rpm,
            spin_axis=spin_axis,
        )

    def _run_simulation(self) -> None:
        """Execute the ball flight simulation."""
        try:
            from src.shared.python.physics.ball_simulator import BallFlightSimulator

            launch = self._build_launch()
            env = self._build_environment()

            sim = BallFlightSimulator(env=env)
            trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)

            if trajectory:
                last = trajectory[-1]
                carry = float(np.sqrt(last.position[0] ** 2 + last.position[1] ** 2))
                max_h = max(p.position[2] for p in trajectory)

                self._results_text.setPlainText(
                    f"Ball Flight Results\n"
                    f"{'=' * 40}\n"
                    f"Launch: {self._speed_spin.value():.0f} mph, "
                    f"{self._angle_spin.value():.1f}°, "
                    f"{self._spin_spin.value():.0f} rpm backspin, "
                    f"{self._sidespin_spin.value():.0f} rpm sidespin\n"
                    f"Environment: wind {self._wind_speed.value():.0f} mph @ "
                    f"{self._wind_dir.value():.0f}°, altitude "
                    f"{self._altitude.value():.0f} ft "
                    f"(air density {env.air_density:.3f} kg/m³)\n\n"
                    f"Carry:       {carry:.1f} m ({carry * 1.09361:.1f} yd)\n"
                    f"Max Height:  {max_h:.1f} m ({max_h * 3.28084:.1f} ft)\n"
                    f"Flight Time: {last.time:.2f} s\n"
                    f"Points:      {len(trajectory)}\n\n"
                    f"Landing Position:\n"
                    f"  X: {last.position[0]:.1f} m\n"
                    f"  Y: {last.position[1]:.1f} m\n"
                    f"  Z: {last.position[2]:.1f} m\n"
                )

                # Update 3D Visualization
                if getattr(self, "_gl_view", None) is not None:
                    import pyqtgraph as pg  # noqa: F401
                    import pyqtgraph.opengl as gl

                    pts = np.array([p.position for p in trajectory])

                    if self._plot_item is not None:
                        self._gl_view.removeItem(self._plot_item)

                    self._plot_item = gl.GLLinePlotItem(
                        pos=pts, color=(0.0, 0.7, 1.0, 1.0), width=3, antialias=True
                    )
                    self._gl_view.addItem(self._plot_item)

                    # Auto-center camera on the trajectory midpoint
                    if len(pts) > 0:
                        from pyqtgraph import Vector

                        mid_x = max(pts[:, 0]) / 2.0
                        self._gl_view.opts["center"] = Vector(mid_x, 0, 0)
            else:
                self._results_text.setPlainText("No trajectory generated.")
        except ImportError as e:
            self._results_text.setPlainText(f"Ball flight simulator not available: {e}")
        except Exception as e:
            logger.exception("Ball flight simulation failed")
            self._results_text.setPlainText(f"Simulation error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        logger.debug("BallFlightWidget cleanup")


class BallFlightWindow(QMainWindow):
    """Standalone window for the ball flight simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aerodynamic Ball Flight Simulator")
        self.setMinimumSize(1100, 700)
        self._widget = BallFlightWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "Forces: Drag + Magnus + Gravity + Wind | Configure and run simulation"
        )

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


class _EmbedAdapter:
    """Embed adapter for the Aerodynamic Ball Flight Simulator.

    Implements the EmbeddableTool protocol so the launcher can embed this
    tool as a tab or dock widget.
    """

    tool_id = "ball_flight_gui"

    def __init__(self) -> None:
        self._widget: BallFlightWidget | None = None

    def embed_capabilities(self) -> Any:
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(800, 600),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Create and return the BallFlightWidget for embedding.

        Args:
            parent: The intended Qt parent widget.

        Returns:
            BallFlightWidget instance for embedding.
        """
        self._widget = BallFlightWidget(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release any resources held by the embedded widget."""
        if self._widget is not None:
            self._widget.cleanup()
        self._widget = None

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state.

        Ball Flight GUI does not track dirty state.
        """
        return False


def _register() -> None:
    try:
        from src.shared.python.launcher_embed import register_embeddable_tool

        register_embeddable_tool(_EmbedAdapter())
    except Exception:  # noqa: BLE001
        logger.warning("ball_flight_gui: EmbeddableTool registration failed")


_register()


def get_dockable_ui() -> BallFlightWindow:
    """Return the main window instance for docking in the unified launcher."""
    return BallFlightWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = BallFlightWindow()
    w.show()
    sys.exit(app.exec())
