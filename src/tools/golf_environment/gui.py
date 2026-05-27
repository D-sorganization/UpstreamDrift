"""Golf Environment and Driving Range Visualization.

Provides a 3D environment for rendering ball flights within context
(e.g. driving range with distance markers, or a specific golf hole).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Try to import pyqtgraph
try:
    import pyqtgraph as pg  # noqa: F401
    import pyqtgraph.opengl as gl

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


@dataclass
class CourseHole:
    """Definition of a golf hole."""

    name: str
    par: int
    yardage: int
    tee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pin_position: tuple[float, float, float] = (100.0, 0.0, 0.0)
    # Boundaries could be defined as polygons, simplified here
    fairway_width: float = 40.0
    green_radius: float = 15.0


@dataclass
class DrivingRange:
    """Definition of a driving range."""

    name: str = "Standard Driving Range"
    width: float = 100.0
    length: float = 350.0  # meters
    markers: list[int] = field(default_factory=lambda: [50, 100, 150, 200, 250, 300])


class EnvironmentRenderer(QWidget):
    """3D Renderer for golf environments using pyqtgraph.opengl."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._environment: CourseHole | DrivingRange = DrivingRange()
        self._trajectories: list[Any] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not PYQTGRAPH_AVAILABLE:
            label = QLabel("pyqtgraph is required for 3D visualization.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self._gl_view = None
            return

        self._gl_view = gl.GLViewWidget()
        self._gl_view.opts["distance"] = 150
        self._gl_view.opts["elevation"] = 15
        self._gl_view.opts["azimuth"] = 90

        # Move center so we see down the range
        from pyqtgraph import Vector

        self._gl_view.opts["center"] = Vector(100, 0, 0)

        layout.addWidget(self._gl_view)
        self.render_environment()

    def set_environment(self, env: CourseHole | DrivingRange) -> None:
        """Change the rendered environment."""
        self._environment = env
        self.render_environment()

    def render_environment(self) -> None:
        """Draw the ground, markers, and boundaries."""
        if self._gl_view is None:
            return

        self._gl_view.clear()

        # Ground plane (Green)
        # Using a mesh for better coloring
        if isinstance(self._environment, DrivingRange):
            length = self._environment.length
            width = self._environment.width

            # Simple green rectangle for the range
            vertexes = np.array(
                [
                    [-10, -width / 2, 0],
                    [length, -width / 2, 0],
                    [-10, width / 2, 0],
                    [-10, width / 2, 0],
                    [length, -width / 2, 0],
                    [length, width / 2, 0],
                ]
            )
            colors = np.ones((6, 4))
            colors[:, 0] = 0.1  # R
            colors[:, 1] = 0.6  # G
            colors[:, 2] = 0.2  # B
            colors[:, 3] = 1.0  # A

            mesh = gl.GLMeshItem(
                vertexes=vertexes, vertexColors=colors, smooth=False, drawEdges=False
            )
            self._gl_view.addItem(mesh)

            # Draw yardage markers
            for d in self._environment.markers:
                # Convert yards to meters for positioning if we assume meters
                d_m = d * 0.9144
                marker = gl.GLLinePlotItem(
                    pos=np.array([[d_m, -width / 2, 0.1], [d_m, width / 2, 0.1]]),
                    color=(1.0, 1.0, 1.0, 0.8),
                    width=2,
                    antialias=True,
                )
                self._gl_view.addItem(marker)

                # We could add text labels if pyqtgraph supported 3D text easily,
                # but lines serve as visual markers.

        elif isinstance(self._environment, CourseHole):
            hole = self._environment

            # Tee box
            tee_mesh = gl.GLMeshItem(
                vertexes=self._create_rect(
                    hole.tee_position[0] - 2, hole.tee_position[1] - 2, 4, 4
                ),
                color=(0.3, 0.8, 0.3, 1.0),
                smooth=False,
            )
            self._gl_view.addItem(tee_mesh)

            # Fairway
            dist = np.linalg.norm(
                np.array(hole.pin_position) - np.array(hole.tee_position)
            )
            fw_mesh = gl.GLMeshItem(
                vertexes=self._create_rect(
                    0, -hole.fairway_width / 2, dist, hole.fairway_width
                ),
                color=(0.2, 0.7, 0.2, 1.0),
                smooth=False,
            )
            self._gl_view.addItem(fw_mesh)

            # Green
            green_pts = self._create_circle(
                hole.pin_position[0], hole.pin_position[1], hole.green_radius
            )
            green_mesh = gl.GLMeshItem(
                vertexes=green_pts, color=(0.1, 0.9, 0.1, 1.0), smooth=False
            )
            self._gl_view.addItem(green_mesh)

            # Pin
            pin = gl.GLLinePlotItem(
                pos=np.array(
                    [
                        hole.pin_position,
                        [hole.pin_position[0], hole.pin_position[1], 2.0],
                    ]
                ),
                color=(1.0, 1.0, 1.0, 1.0),
                width=3,
            )
            self._gl_view.addItem(pin)

    def _create_rect(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        return np.array(
            [
                [x, y, 0],
                [x + w, y, 0],
                [x, y + h, 0],
                [x, y + h, 0],
                [x + w, y, 0],
                [x + w, y + h, 0],
            ]
        )

    def _create_circle(
        self, cx: float, cy: float, r: float, segments: int = 32
    ) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, segments)
        pts = []
        for i in range(segments - 1):
            x1 = cx + r * np.cos(angles[i])
            y1 = cy + r * np.sin(angles[i])
            x2 = cx + r * np.cos(angles[i + 1])
            y2 = cy + r * np.sin(angles[i + 1])
            pts.extend([[cx, cy, 0.05], [x1, y1, 0.05], [x2, y2, 0.05]])
        return np.array(pts)

    def add_trajectory(
        self,
        points: np.ndarray,
        color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0),
    ) -> None:
        """Add a ball flight trajectory.

        Args:
            points: Array of shape (N, 3) with [x, y, z] coordinates.
            color: RGBA tuple.
        """
        if self._gl_view is None or len(points) == 0:
            return

        path = gl.GLLinePlotItem(pos=points, color=color, width=3, antialias=True)
        self._gl_view.addItem(path)
        self._trajectories.append(path)

    def clear_trajectories(self) -> None:
        if self._gl_view is None:
            return
        for t in self._trajectories:
            self._gl_view.removeItem(t)
        self._trajectories.clear()


class EnvironmentWindow(QMainWindow):
    """Standalone window for the environment renderer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Golf Environment Viewer")
        self.setMinimumSize(800, 600)

        central = QWidget()
        layout = QVBoxLayout(central)

        self.env_combo = QComboBox()
        self.env_combo.addItems(["Driving Range", "Par 3 (150y)", "Par 4 (400y)"])
        self.env_combo.currentTextChanged.connect(self._on_env_changed)
        layout.addWidget(self.env_combo)

        self.renderer = EnvironmentRenderer()
        layout.addWidget(self.renderer, stretch=1)

        self.setCentralWidget(central)

        # Add some mock trajectories for demo
        t = np.linspace(0, 5, 100)
        x = 50 * t
        y = np.zeros_like(t)
        z = 25 * t - 0.5 * 9.81 * t**2
        mask = z >= 0
        self.renderer.add_trajectory(np.column_stack((x[mask], y[mask], z[mask])))

    def _on_env_changed(self, text: str) -> None:
        self.renderer.clear_trajectories()
        if text == "Driving Range":
            self.renderer.set_environment(DrivingRange())
        elif text == "Par 3 (150y)":
            self.renderer.set_environment(
                CourseHole("Hole 1", 3, 150, pin_position=(150 * 0.9144, 0, 0))
            )
        elif text == "Par 4 (400y)":
            self.renderer.set_environment(
                CourseHole("Hole 2", 4, 400, pin_position=(400 * 0.9144, 0, 0))
            )


class _EmbedAdapter:
    """Embed adapter for the Golf Environment Viewer.

    Implements the EmbeddableTool protocol so the launcher can embed this
    tool as a tab or dock widget.
    """

    tool_id = "golf_environment"

    def __init__(self) -> None:
        self._widget: EnvironmentRenderer | None = None

    def embed_capabilities(self) -> Any:
        from src.shared.python.launcher_embed import EmbedCapabilities

        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(800, 600),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        """Create and return the EnvironmentRenderer widget for embedding.

        Args:
            parent: The intended Qt parent widget.

        Returns:
            EnvironmentRenderer widget instance for embedding.
        """
        self._widget = EnvironmentRenderer(parent=parent)
        return self._widget

    def cleanup(self) -> None:
        """Release any resources held by the embedded widget."""
        self._widget = None

    def is_dirty(self) -> bool:
        """Return True if the tool has unsaved state.

        Golf Environment Viewer does not track dirty state.
        """
        return False


def _register() -> None:
    try:
        from src.shared.python.launcher_embed import register_embeddable_tool

        register_embeddable_tool(_EmbedAdapter())
    except Exception:  # noqa: BLE001
        logger.warning("golf_environment: EmbeddableTool registration failed")


_register()


def get_dockable_ui() -> EnvironmentWindow:
    return EnvironmentWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = EnvironmentWindow()
    w.show()
    sys.exit(app.exec())
