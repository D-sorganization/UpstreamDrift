"""Putting Green Simulator GUI.

A PyQt6 dashboard that drives the **real** :class:`PuttingGreenSimulator`
physics engine (ball roll, turf friction, slope break, cup capture) and
renders the result as a professional 3D scene: a contour-shaded green,
a roll-mode-coloured ball track, an animated ball, flagstick, and an
aim/target read line, alongside a live metrics panel.

All domain logic lives in :mod:`._scene_builder` (headless, unit-tested);
this module is a thin Qt/OpenGL renderer that consumes a :class:`PuttScene`.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, QTimer
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

from src.shared.python.golf_viz import (
    disc_mesh,
    flagstick_lines,
    grid_surface_mesh,
    roll_mode_colors,
)
from src.shared.python.ui import HoverCopyTextBrowser
from src.tools.putting_green_gui._scene_builder import (
    PuttConfig,
    PuttScene,
    build_putt_scene,
)

logger = logging.getLogger(__name__)

_M_TO_FT = 3.280839895
_PANEL_STYLE = """
QWidget#puttControls { background-color: #1b2530; }
QLabel { color: #d7e2ec; }
QLabel#puttTitle { color: #ffffff; font-size: 16px; font-weight: bold; }
QGroupBox {
    color: #9fd8a8; border: 1px solid #2f4254; border-radius: 6px;
    margin-top: 10px; padding: 8px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QDoubleSpinBox {
    background-color: #0f1822; color: #eaf2f8;
    border: 1px solid #2f4254; border-radius: 4px; padding: 3px;
}
QPushButton {
    background-color: #243343; color: #eaf2f8;
    border: 1px solid #34536b; border-radius: 5px; padding: 6px;
}
QPushButton:hover { background-color: #2c3f52; }
"""


class PuttingGreenWidget(QWidget):
    """Central widget for the putting green simulator dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene: PuttScene | None = None
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(30)
        self._anim_timer.timeout.connect(self._advance_animation)
        self._anim_index = 0
        # GL scene items (populated when pyqtgraph.opengl is available).
        self._gl_view: Any = None
        self._terrain_item: Any = None
        self._cup_item: Any = None
        self._path_item: Any = None
        self._ball_item: Any = None
        self._flag_item: Any = None
        self._aim_item: Any = None
        self._start_item: Any = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setSizes([360, 720])
        layout.addWidget(splitter)

    def _build_controls(self) -> QWidget:
        left = QWidget()
        left.setObjectName("puttControls")
        left.setStyleSheet(_PANEL_STYLE)
        left_layout = QVBoxLayout(left)

        title = QLabel("⛳  Putting Green Simulator")
        title.setObjectName("puttTitle")
        left_layout.addWidget(title)

        putt_group = QGroupBox("Putt Configuration")
        putt_form = QFormLayout(putt_group)
        self._speed_spin = self._make_spin(0.5, 8.0, 2.5, " m/s")
        putt_form.addRow("Putter Speed:", self._speed_spin)
        self._aim_spin = self._make_spin(-45.0, 45.0, 0.0, "°")
        putt_form.addRow("Aim Angle:", self._aim_spin)
        self._distance_spin = self._make_spin(1.0, 30.0, 10.0, " ft")
        putt_form.addRow("Cup Distance:", self._distance_spin)
        left_layout.addWidget(putt_group)

        green_group = QGroupBox("Green Properties")
        green_form = QFormLayout(green_group)
        self._stimp_spin = self._make_spin(6.0, 14.0, 10.0, "", decimals=1)
        green_form.addRow("Stimpmeter:", self._stimp_spin)
        self._slope_spin = self._make_spin(0.0, 5.0, 1.0, "°")
        green_form.addRow("Cross Slope:", self._slope_spin)
        left_layout.addWidget(green_group)

        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, dist in (
            ("Short", 1.5, 5.0),
            ("Medium", 2.5, 15.0),
            ("Long", 4.0, 30.0),
        ):
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda _checked, s=speed, d=dist: self._apply_preset(s, d)
            )
            preset_layout.addWidget(btn)
        left_layout.addWidget(preset_group)

        self._run_btn = QPushButton("▶  Simulate Putt")
        self._run_btn.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold;"
            " padding: 12px; border-radius: 5px;"
        )
        self._run_btn.clicked.connect(self._run_simulation)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        return left

    def _make_spin(
        self,
        lo: float,
        hi: float,
        value: float,
        suffix: str,
        *,
        decimals: int = 1,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _build_results(self) -> QWidget:
        right = QWidget()
        right_layout = QVBoxLayout(right)
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout(results_group)

        self._init_gl_view(results_layout)

        self._results_text = HoverCopyTextBrowser()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(190)
        self._results_text.setPlainText(
            "Configure putt parameters and click 'Simulate Putt'.\n\n"
            "Physics model:\n"
            "  - Real ball-roll dynamics (slide -> roll -> stop)\n"
            "  - Turf friction calibrated to the Stimpmeter rating\n"
            "  - Cross-slope break and cup-capture modelling\n"
            "  - RK4 integration of the rolling ball\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        return right

    def _init_gl_view(self, results_layout: QVBoxLayout) -> None:
        try:
            import pyqtgraph as pg  # noqa: F401
            import pyqtgraph.opengl as gl
        except ImportError:
            return

        view = gl.GLViewWidget()
        view.setBackgroundColor((14, 22, 32))
        view.opts["distance"] = 14
        view.opts["elevation"] = 32
        view.opts["azimuth"] = -78

        self._terrain_item = gl.GLMeshItem(
            smooth=True, drawEdges=False, shader="shaded", glOptions="opaque"
        )
        view.addItem(self._terrain_item)

        self._path_item = gl.GLLinePlotItem(
            width=4.0, antialias=True, mode="line_strip"
        )
        view.addItem(self._path_item)

        self._aim_item = gl.GLLinePlotItem(
            color=(0.95, 0.85, 0.25, 0.65), width=1.5, antialias=True
        )
        view.addItem(self._aim_item)

        self._cup_item = gl.GLMeshItem(
            color=(0.04, 0.04, 0.05, 1.0), smooth=True, drawEdges=False
        )
        view.addItem(self._cup_item)

        self._flag_item = gl.GLLinePlotItem(
            color=(0.95, 0.25, 0.2, 1.0), width=3.0, antialias=True
        )
        view.addItem(self._flag_item)

        self._start_item = gl.GLScatterPlotItem(color=(0.8, 0.85, 0.9, 1.0), size=9.0)
        view.addItem(self._start_item)

        self._ball_item = gl.GLScatterPlotItem(color=(1.0, 1.0, 1.0, 1.0), size=12.0)
        view.addItem(self._ball_item)

        self._gl_view = view
        results_layout.addWidget(view, stretch=3)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _apply_preset(self, speed: float, dist: float) -> None:
        self._speed_spin.setValue(speed)
        self._distance_spin.setValue(dist)

    def _current_config(self) -> PuttConfig:
        return PuttConfig(
            putter_speed_ms=self._speed_spin.value(),
            aim_deg=self._aim_spin.value(),
            cup_distance_ft=self._distance_spin.value(),
            stimp=self._stimp_spin.value(),
            slope_deg=self._slope_spin.value(),
        )

    def _run_simulation(self) -> None:
        """Run the real putting simulation and render the result."""
        self._anim_timer.stop()
        try:
            scene = build_putt_scene(self._current_config())
        except (ValueError, ImportError, RuntimeError) as exc:
            logger.exception("Putting simulation failed")
            self._results_text.setPlainText(f"Simulation error: {exc}")
            return

        self._scene = scene
        self._results_text.setPlainText(_format_metrics(scene))
        self._render_scene(scene)

    def _render_scene(self, scene: PuttScene) -> None:
        if self._gl_view is None:
            return

        verts, faces, colors = grid_surface_mesh(
            scene.grid_x, scene.grid_y, scene.grid_z
        )
        self._terrain_item.setMeshData(vertexes=verts, faces=faces, vertexColors=colors)

        path_colors = roll_mode_colors(scene.roll_modes)
        self._path_item.setData(pos=scene.trajectory_xyz, color=path_colors)
        self._aim_item.setData(pos=scene.aim_line_xyz)

        cx, cy, cz = scene.cup_xyz
        cup_v, cup_f = disc_mesh(
            (cx, cy), scene.hole_radius_m, z=cz - 0.004, segments=36
        )
        self._cup_item.setMeshData(vertexes=cup_v, faces=cup_f)
        self._flag_item.setData(pos=flagstick_lines((cx, cy), z=cz, height=1.25))

        self._start_item.setData(pos=scene.start_xyz.reshape(1, 3))
        self._ball_item.setData(pos=scene.trajectory_xyz[:1])

        self._frame_camera(scene)
        self._anim_index = 0
        self._anim_timer.start()

    def _frame_camera(self, scene: PuttScene) -> None:
        from pyqtgraph import Vector

        width, height = scene.green_size
        self._gl_view.opts["center"] = Vector(width / 2.0, height / 2.0, 0.0)
        self._gl_view.opts["distance"] = max(width, height) * 1.5 + 4.0
        self._gl_view.update()

    def _advance_animation(self) -> None:
        scene = self._scene
        if scene is None or self._ball_item is None:
            self._anim_timer.stop()
            return
        traj = scene.trajectory_xyz
        n = traj.shape[0]
        if self._anim_index >= n:
            self._ball_item.setData(pos=traj[-1:])
            self._anim_timer.stop()
            return
        self._ball_item.setData(pos=traj[self._anim_index : self._anim_index + 1])
        # Advance proportionally so every putt animates in similar wall-clock time.
        self._anim_index += max(1, n // 90)

    def cleanup(self) -> None:
        """Release resources (stop the animation timer)."""
        self._anim_timer.stop()
        logger.debug("PuttingGreenWidget cleanup")


def _format_metrics(scene: PuttScene) -> str:
    """Render a human-readable metrics panel for a finished putt."""
    outcome = "HOLED" if scene.holed else "Missed"
    dist_cm = scene.final_distance_to_cup_m * 100.0
    miss_line = (
        "Result:        HOLED"
        if scene.holed
        else f"Result:        Missed by {dist_cm:.1f} cm"
    )
    return (
        f"Putting Simulation - {outcome}\n"
        f"{'=' * 44}\n"
        f"{miss_line}\n"
        f"Total roll:    {scene.total_roll_m:.2f} m "
        f"({scene.total_roll_m * _M_TO_FT:.1f} ft)\n"
        f"Roll time:     {scene.duration_s:.2f} s\n"
        f"Peak break:    {scene.peak_break_m * 100.0:.1f} cm\n"
        f"Launch speed:  {scene.launch_speed_ms:.2f} m/s\n"
        f"Track colour:  amber = skidding, green = pure roll, grey = stopped\n"
    )


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
    """Embed adapter for the Putting Green Simulator."""

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
        self._widget = PuttingGreenWidget(parent=parent)
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
    except Exception:  # noqa: BLE001 - registration is best-effort at import time
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
