# mypy: ignore-errors
# MuJoCo types are dynamically imported and mypy cannot resolve them statically
"""MuJoCo-based 3D visualization for URDF preview (coordinator).

Implements Task 2.1: MuJoCo Visualization Embed per Phase 2 roadmap.
Provides real-time URDF preview via MJCF conversion.

Issue #755: Enhanced visualization toggles for collision, frames, joints, and contacts.

Implementation split across:
- _mujoco_viewer_backend.py: VisualizationFlags, URDFToMJCFConverter,
  MuJoCoOffscreenRenderer
- _viewer_ui.py:            ViewerUIBuilder, disable_toggles, paint_render
- _viewer_input.py:         mouse / wheel handlers, launch_external_viewer
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QWidget

from src.shared.python.engine_core.engine_availability import MUJOCO_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from typing import Any

# Re-export public names for backward compatibility
from ._mujoco_viewer_backend import (
    MuJoCoOffscreenRenderer,
    URDFToMJCFConverter,
    VisualizationFlags,
)
from ._viewer_input import (
    handle_mouse_move,
    handle_mouse_press,
    handle_mouse_release,
    handle_wheel,
    launch_external_viewer,
)
from ._viewer_ui import ViewerUIBuilder, paint_render

logger = get_logger(__name__)

# MuJoCo is optional
if MUJOCO_AVAILABLE:
    import mujoco
else:
    mujoco = None  # type: ignore[assignment]


class MuJoCoViewerWidget(QWidget):
    """Qt widget for MuJoCo-based URDF visualization.

    Features:
    - Real-time URDF preview via MJCF conversion
    - Mouse-based camera control (rotate, zoom)
    - Visualization toggles (collision, frames, joints, contacts)
    - Physics sanity checks
    - Clear headless fallback messaging

    Issue #755: Enhanced with working toggles and contacts visualization.
    """

    # Signals
    validation_error = pyqtSignal(str)
    model_loaded = pyqtSignal(bool)
    visualization_changed = pyqtSignal(dict)  # Emitted when toggles change

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the MuJoCo viewer widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._urdf_content = ""
        self._urdf_path: str | None = None
        self._renderer: MuJoCoOffscreenRenderer | None = None
        self._last_mouse_pos: QPointF | None = None
        self._current_image = None

        # Visualization flags (using dataclass)
        self._vis_flags = VisualizationFlags()

        # UI construction delegated to ViewerUIBuilder
        ViewerUIBuilder.build(self)

        self._setup_renderer()

        # Render timer
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._update_render)
        self._render_timer.start(50)  # 20 FPS

    # ------------------------------------------------------------------
    # Renderer setup
    # ------------------------------------------------------------------

    def _setup_renderer(self) -> None:
        """Initialize the offscreen renderer."""
        if MUJOCO_AVAILABLE:
            # Use larger framebuffer to avoid dimension mismatch errors
            self._renderer = MuJoCoOffscreenRenderer(800, 800)
            # Sync initial flags
            self._renderer.vis_flags = self._vis_flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_visualization(
        self, urdf_content: str, urdf_path: str | None = None
    ) -> None:
        """Update visualization with new URDF content.

        Args:
            urdf_content: URDF XML string.
            urdf_path: Optional path to URDF file for mesh resolution.
        """
        if urdf_content is None:
            raise ValueError("urdf_content must be provided")
        self._urdf_content = urdf_content
        self._urdf_path = urdf_path

        if not MUJOCO_AVAILABLE or not self._renderer:
            link_count = urdf_content.count("<link")
            joint_count = urdf_content.count("<joint")
            self._viewport.setText(
                f"URDF Preview\n\nLinks: {link_count}\nJoints: {joint_count}\n\n"
                "(Install MuJoCo for 3D preview)"
            )
            return

        # Validate URDF
        validation_errors = self._validate_urdf(urdf_content)
        if validation_errors:
            self.validation_error.emit("\n".join(validation_errors))

        # Try to load from file path first (for mesh resolution)
        success = False
        if urdf_path:
            logger.info(f"Loading URDF from path: {urdf_path}")
            success = self._renderer.load_urdf_file(urdf_path)

        # Fallback to MJCF conversion if file loading fails
        if not success:
            logger.info("Falling back to MJCF conversion")
            mjcf_content = URDFToMJCFConverter.convert(urdf_content)
            success = self._renderer.load_mjcf(mjcf_content)

        self.model_loaded.emit(success)

        link_count = urdf_content.count("<link")
        joint_count = urdf_content.count("<joint")
        if success:
            self._status_label.setText(
                f"✓ Model loaded: {link_count} links, {joint_count} joints"
            )
        else:
            self._status_label.setText("⚠️ Failed to load model")

    def clear(self) -> None:
        """Clear the visualization."""
        self._urdf_content = ""
        self._viewport.setText("No URDF content")
        self._status_label.setText("")

    def reset_view(self) -> None:
        """Reset camera to default position."""
        if self._renderer:
            self._renderer.azimuth = 90.0
            self._renderer.elevation = -20.0
            self._renderer.distance = 3.0
            self._renderer.lookat = np.array([0.0, 0.0, 0.5])

    def get_visualization_flags(self) -> VisualizationFlags:
        """Get current visualization flags.

        Returns:
            Current visualization configuration.
        """
        return self._vis_flags

    def set_visualization_flags(self, flags: VisualizationFlags) -> None:
        """Set visualization flags programmatically.

        Args:
            flags: New visualization configuration.
        """
        if flags is None:
            raise ValueError("flags must be provided")
        self._vis_flags = flags

        # Update checkboxes to match
        self._collision_checkbox.setChecked(flags.show_collision)
        self._frames_checkbox.setChecked(flags.show_frames)
        self._joints_checkbox.setChecked(flags.show_joint_limits)
        self._contacts_checkbox.setChecked(flags.show_contacts)

        self._update_renderer_flags()

    def highlight_body(self, body_name: str | None) -> None:
        """Highlight a specific body in the visualization.

        Args:
            body_name: Name of body to highlight, or None to clear.
        """
        # Future enhancement: implement body highlighting in MuJoCo
        logger.debug(f"Body highlight requested: {body_name}")

    def is_mujoco_available(self) -> bool:
        """Check if MuJoCo rendering is available.

        Returns:
            True if MuJoCo is installed and renderer is initialized.
        """
        return MUJOCO_AVAILABLE and self._renderer is not None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Dictionary with model statistics.
        """
        info: dict[str, Any] = {
            "mujoco_available": MUJOCO_AVAILABLE,
            "model_loaded": False,
            "link_count": 0,
            "joint_count": 0,
        }

        if self._urdf_content:
            info["link_count"] = self._urdf_content.count("<link")
            info["joint_count"] = self._urdf_content.count("<joint")
            info["model_loaded"] = True

        if self._renderer and self._renderer._model is not None:
            info["bodies"] = self._renderer._model.nbody
            info["joints"] = self._renderer._model.njnt
            info["geoms"] = self._renderer._model.ngeom

        return info

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _update_render(self) -> None:
        """Update the rendered image (called by render timer)."""
        if not self._renderer:
            return
        image = self._renderer.render()
        if image is not None:
            paint_render(self, image)

    # ------------------------------------------------------------------
    # Toggle handlers (called by checkboxes wired in ViewerUIBuilder)
    # ------------------------------------------------------------------

    def _on_collision_toggled(self, checked: bool) -> None:
        """Handle collision visualization toggle."""
        self._vis_flags.show_collision = checked
        self._update_renderer_flags()
        logger.info(f"Collision visualization: {checked}")

    def _on_frames_toggled(self, checked: bool) -> None:
        """Handle frames visualization toggle."""
        self._vis_flags.show_frames = checked
        self._update_renderer_flags()
        logger.info(f"Frame visualization: {checked}")

    def _on_joints_toggled(self, checked: bool) -> None:
        """Handle joint limits visualization toggle."""
        self._vis_flags.show_joint_limits = checked
        self._update_renderer_flags()
        logger.info(f"Joint limits visualization: {checked}")

    def _on_contacts_toggled(self, checked: bool) -> None:
        """Handle contacts visualization toggle."""
        self._vis_flags.show_contacts = checked
        self._update_renderer_flags()
        logger.info(f"Contacts visualization: {checked}")

    def _update_renderer_flags(self) -> None:
        """Sync visualization flags to the renderer."""
        if self._renderer:
            self._renderer.set_visualization_flags(self._vis_flags)
            self.visualization_changed.emit(self._vis_flags.to_dict())

    # ------------------------------------------------------------------
    # Input events (delegated to _viewer_input helpers)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse press for camera control."""
        handle_mouse_press(self, event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse move for camera rotation."""
        handle_mouse_move(self, event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse release."""
        handle_mouse_release(self, event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Handle mouse wheel for zoom."""
        handle_wheel(self, event)

    def _launch_external_viewer(self) -> None:
        """Launch MuJoCo's standalone viewer."""
        launch_external_viewer(self._urdf_content, URDFToMJCFConverter)

    # ------------------------------------------------------------------
    # URDF validation (physics sanity checks)
    # ------------------------------------------------------------------

    def _validate_urdf(self, urdf_content: str) -> list[str]:
        """Validate URDF for physics sanity.

        Args:
            urdf_content: URDF XML string.

        Returns:
            List of validation error messages.
        """
        if urdf_content is None:
            raise ValueError("urdf_content must be provided")
        errors = []

        try:
            root = ET.fromstring(
                urdf_content
            )  # nosec B314 - urdf is validated tool input
        except ET.ParseError as e:
            return [f"XML Parse Error: {e}"]

        # Check inertial properties
        for link in root.findall(".//link"):
            link_name = link.get("name", "unnamed")
            inertial = link.find("inertial")

            if inertial is not None:
                inertia = inertial.find("inertia")
                if inertia is not None:
                    ixx = float(inertia.get("ixx", "0"))
                    iyy = float(inertia.get("iyy", "0"))
                    izz = float(inertia.get("izz", "0"))

                    if ixx <= 0 or iyy <= 0 or izz <= 0:
                        errors.append(
                            f"Link '{link_name}': Non-positive inertia diagonal"
                        )

        # Check joint axes
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name", "unnamed")
            axis_elem = joint.find("axis")

            if axis_elem is not None:
                xyz = axis_elem.get("xyz", "0 0 1")
                axis = [float(x) for x in xyz.split()]
                norm = sum(x * x for x in axis) ** 0.5

                if abs(norm - 1.0) > 0.01:
                    errors.append(
                        f"Joint '{joint_name}': Axis not normalized (|axis|={norm:.3f})"
                    )

        return errors


__all__ = [
    "MuJoCoOffscreenRenderer",
    "MuJoCoViewerWidget",
    "URDFToMJCFConverter",
    "VisualizationFlags",
]
