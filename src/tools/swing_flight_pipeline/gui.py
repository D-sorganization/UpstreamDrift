"""Swing-to-Flight Pipeline GUI — end-to-end simulation visualizer.

Wraps :class:`SwingBallFlightPipeline` in a PyQt6 dashboard that lets
the user configure swing parameters, run the full pipeline
(swing → impact → ball flight), and visualize the trajectory.

Implements the GUI tile for the swing-to-flight pipeline registered
in ``models.yaml`` under the ``simulation`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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

from src.shared.python.physics.swing_state_providers import (
    SwingStateConfig,
    SwingStateProvider,
    available_swing_state_providers,
)
from src.shared.python.ui import HoverCopyTextBrowser

logger = logging.getLogger(__name__)


def _populate_engine_combo(
    combo: QComboBox, providers: list[SwingStateProvider]
) -> None:
    """Fill ``combo`` with providers, disabling the unavailable ones.

    Unavailable entries stay visible (so users see the roadmap) but cannot
    be selected, and carry a tooltip explaining why (issue #8819 — the old
    combo silently stamped manual numbers with engine names).
    """
    combo.clear()
    model = combo.model()
    for index, provider in enumerate(providers):
        combo.addItem(provider.provider_id)
        if not provider.is_available():
            # QComboBox uses a QStandardItemModel by default; guard anyway so
            # a custom model degrades to tooltip-only rather than crashing.
            item = model.item(index) if isinstance(model, QStandardItemModel) else None
            if item is not None:
                item.setEnabled(False)
            combo.setItemData(
                index,
                provider.availability_reason(),
                Qt.ItemDataRole.ToolTipRole,
            )


class SwingFlightWidget(QWidget):
    """Central widget for the swing-to-flight pipeline dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: Any = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _build_controls_panel(self) -> QWidget:
        """Left pane: title, swing parameters, engine source, presets, run."""
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Swing → Impact → Ball Flight Pipeline")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        left_layout.addWidget(self._build_swing_group())
        left_layout.addWidget(self._build_engine_group())
        left_layout.addWidget(self._build_preset_group())

        # Run button
        self._run_btn = QPushButton("Run Full Pipeline")
        self._run_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_pipeline)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        return left

    def _build_swing_group(self) -> QGroupBox:
        """Swing-parameter spin boxes (speed, loft, clubhead mass)."""
        swing_group = QGroupBox("Swing Parameters")
        swing_form = QFormLayout(swing_group)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(20.0, 60.0)
        self._speed_spin.setValue(45.0)
        self._speed_spin.setSuffix(" m/s")
        swing_form.addRow("Clubhead Speed:", self._speed_spin)

        self._loft_spin = QDoubleSpinBox()
        self._loft_spin.setRange(5.0, 60.0)
        self._loft_spin.setValue(10.5)
        self._loft_spin.setSuffix("°")
        swing_form.addRow("Loft Angle:", self._loft_spin)

        self._mass_spin = QDoubleSpinBox()
        self._mass_spin.setRange(0.100, 0.400)
        self._mass_spin.setValue(0.200)
        self._mass_spin.setDecimals(3)
        self._mass_spin.setSuffix(" kg")
        swing_form.addRow("Clubhead Mass:", self._mass_spin)

        return swing_group

    def _build_engine_group(self) -> QGroupBox:
        """Engine selector — entries are real SwingStateProviders; the ones
        without an implemented sourcing path are disabled (issue #8819)."""
        engine_group = QGroupBox("Physics Engine Source")
        engine_layout = QVBoxLayout(engine_group)
        self._engine_combo = QComboBox()
        self._providers = {p.provider_id: p for p in available_swing_state_providers()}
        _populate_engine_combo(self._engine_combo, list(self._providers.values()))
        self._engine_combo.setCurrentText("manual")
        engine_layout.addWidget(self._engine_combo)
        return engine_group

    def _build_preset_group(self) -> QGroupBox:
        """Club preset buttons that stamp speed/loft pairs."""
        preset_group = QGroupBox("Club Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, loft in [
            ("Driver", 50.0, 10.5),
            ("7-Iron", 35.0, 34.0),
            ("PW", 28.0, 46.0),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, s=speed, loft_val=loft: self._apply_preset(s, loft_val)
            )
            preset_layout.addWidget(btn)
        return preset_group

    def _build_results_panel(self) -> QWidget:
        """Right pane: 3D trajectory view (when available) and results text."""
        right = QWidget()
        right_layout = QVBoxLayout(right)

        results_group = QGroupBox("Pipeline Results")
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
            "Configure swing parameters and click 'Run Full Pipeline' "
            "to execute the end-to-end simulation.\n\n"
            "Pipeline stages:\n"
            "  1. Swing State → Pre-Impact kinematics\n"
            "  2. Impact Model → Post-impact ball velocity + spin\n"
            "  3. Launch Conditions → Aerodynamic trajectory\n"
            "  4. Metrics extraction (carry, height, flight time)\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        return right

    def _apply_preset(self, speed: float, loft: float) -> None:
        self._speed_spin.setValue(speed)
        self._loft_spin.setValue(loft)

    def _run_pipeline(self) -> None:
        """Execute the swing-to-flight pipeline."""
        try:
            from src.shared.python.physics.swing_ball_flight_pipeline import (
                SwingBallFlightPipeline,
            )

            # Route through the selected provider so the stamped engine_name
            # reflects the actual source of the swing state (issue #8819).
            provider = self._providers[self._engine_combo.currentText()]
            swing = provider.get_swing_state(
                SwingStateConfig(
                    clubhead_speed_ms=self._speed_spin.value(),
                    loft_deg=self._loft_spin.value(),
                    clubhead_mass_kg=self._mass_spin.value(),
                )
            )

            pipeline = SwingBallFlightPipeline()
            result = pipeline.run(swing)
            self._result = result

            self._results_text.setPlainText(
                f"Pipeline Complete\n"
                f"{'=' * 40}\n"
                f"Engine: {result.swing_state.engine_name}\n\n"
                f"Impact Results:\n"
                f"  Ball speed: {np.linalg.norm(result.impact_state.ball_velocity):.1f} m/s\n"
                f"  Ball spin:  {np.linalg.norm(result.impact_state.ball_angular_velocity):.0f} rad/s\n\n"
                f"Launch Conditions:\n"
                f"  Speed:      {result.launch_conditions.velocity:.1f} m/s\n"
                f"  Angle:      {result.launch_conditions.launch_angle:.1f}°\n"
                f"  Spin Rate:  {result.launch_conditions.spin_rate:.0f} rad/s\n\n"
                f"Flight Results:\n"
                f"  Carry:      {result.carry_m:.1f} m ({result.carry_m * 1.09361:.1f} yd)\n"
                f"  Max Height: {result.max_height_m:.1f} m\n"
                f"  Flight Time:{result.flight_time_s:.2f} s\n"
                f"  Landing:    {result.landing_angle_deg:.1f}°\n\n"
                f"Trajectory: {len(result.trajectory)} points\n"
            )

            # Update 3D Visualization
            if getattr(self, "_gl_view", None) is not None and result.trajectory:
                import pyqtgraph as pg  # noqa: F401
                import pyqtgraph.opengl as gl

                pts = np.array([p.position for p in result.trajectory])

                if self._plot_item is not None:
                    self._gl_view.removeItem(self._plot_item)

                # Shade the flight line by height (apex hot, ground cool) using
                # the shared golf_viz palette (DRY across the golf viewers).
                from src.shared.python.golf_viz import speed_colors

                flight_colors = speed_colors(pts[:, 2])
                self._plot_item = gl.GLLinePlotItem(
                    pos=pts, color=flight_colors, width=3, antialias=True
                )
                self._gl_view.addItem(self._plot_item)

                # Auto-center camera
                if len(pts) > 0:
                    from pyqtgraph import Vector

                    mid_x = max(pts[:, 0]) / 2.0
                    self._gl_view.opts["center"] = Vector(mid_x, 0, 0)

        except ImportError as e:
            self._results_text.setPlainText(
                f"Pipeline not available: {e}\n\n"
                "A required dependency for the swing-to-flight pipeline "
                "failed to import. Check the log for details."
            )
        except Exception as e:
            logger.exception("Pipeline execution failed")
            self._results_text.setPlainText(f"Pipeline error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        logger.debug("SwingFlightWidget cleanup")


class SwingFlightWindow(QMainWindow):
    """Standalone window for the swing-to-flight pipeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Swing → Flight Pipeline")
        self.setMinimumSize(1100, 700)
        self._widget = SwingFlightWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Configure swing parameters and run the pipeline")

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


def get_dockable_ui() -> SwingFlightWindow:
    """Return the main window instance for docking in the unified launcher."""
    return SwingFlightWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = SwingFlightWindow()
    w.show()
    sys.exit(app.exec())
