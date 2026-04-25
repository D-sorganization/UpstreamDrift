# mypy: ignore-errors
"""Input-handling helpers for MuJoCoViewerWidget.

Extracted from mujoco_viewer.py as part of issue #3060 refactor.

Contains standalone functions that implement mouse / wheel event handling
and external-viewer launching.  Each function receives the owning widget
as its first argument so that it can read/write the widget's state
without coupling this module to the full widget class hierarchy.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mouse / wheel handlers
# ---------------------------------------------------------------------------


def handle_mouse_press(owner: QWidget, event: QMouseEvent | None) -> None:
    """Record the starting position for a camera-drag gesture.

    Args:
        owner: Viewer widget; must expose ``_last_mouse_pos``.
        event: Qt mouse-press event.
    """
    if event and event.button() == Qt.MouseButton.LeftButton:
        owner._last_mouse_pos = event.position()


def handle_mouse_move(owner: QWidget, event: QMouseEvent | None) -> None:
    """Rotate the camera while the user drags with the left button held.

    Args:
        owner: Viewer widget; must expose ``_last_mouse_pos`` and ``_renderer``.
        event: Qt mouse-move event.
    """
    if event and owner._last_mouse_pos and owner._renderer:
        dx = event.position().x() - owner._last_mouse_pos.x()
        dy = event.position().y() - owner._last_mouse_pos.y()
        owner._renderer.rotate_camera(dx * 0.5, dy * 0.5)
        owner._last_mouse_pos = event.position()


def handle_mouse_release(owner: QWidget, event: QMouseEvent | None) -> None:  # noqa: ARG001
    """Clear the drag-start position on button release.

    Args:
        owner: Viewer widget; must expose ``_last_mouse_pos``.
        event: Qt mouse-release event (unused, kept for Qt signature parity).
    """
    owner._last_mouse_pos = None


def handle_wheel(owner: QWidget, event: QWheelEvent | None) -> None:
    """Zoom the camera in or out on mouse-wheel scroll.

    Args:
        owner: Viewer widget; must expose ``_renderer``.
        event: Qt wheel event.
    """
    if event and owner._renderer:
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        owner._renderer.zoom_camera(factor)


# ---------------------------------------------------------------------------
# External viewer launcher
# ---------------------------------------------------------------------------


def launch_external_viewer(urdf_content: str, converter_cls: type) -> None:
    """Launch MuJoCo's standalone viewer in a subprocess.

    Converts *urdf_content* to MJCF, writes it to a temp file, and spawns
    a ``python -c`` subprocess that opens the MuJoCo interactive viewer.

    Args:
        urdf_content: URDF XML string to visualise.
        converter_cls: ``URDFToMJCFConverter`` class (avoids a circular import).
    """
    if not urdf_content:
        logger.warning("No URDF content to view")
        return

    try:
        mjcf_content = converter_cls.convert(urdf_content)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(mjcf_content)
            temp_path = f.name

        cmd = [
            sys.executable,
            "-c",
            (
                "import mujoco; import mujoco.viewer; "
                f"m=mujoco.MjModel.from_xml_path(r'{temp_path}'); "
                "mujoco.viewer.launch(m)"
            ),
        ]
        subprocess.Popen(cmd)
        logger.info("Launched external MuJoCo viewer")

    except ImportError as e:
        logger.error(f"Failed to launch viewer: {e}")
