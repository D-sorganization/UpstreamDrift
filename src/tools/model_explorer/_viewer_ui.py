# mypy: ignore-errors
"""UI construction helpers for MuJoCoViewerWidget.

Extracted from mujoco_viewer.py as part of issue #3060 refactor.

Contains:
- ViewerUIBuilder: builds the toolbar, viewport, and status bar
- Standalone render-display helper used by the render timer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.engine_core.engine_availability import MUJOCO_AVAILABLE

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget


class ViewerUIBuilder:
    """Builds the UI elements for MuJoCoViewerWidget.

    Call ``build`` once during widget initialisation; it populates the
    widget's layout and stores named widget references back on the owner.
    """

    # ---------------------------------------------------------------------------
    # Public factory
    # ---------------------------------------------------------------------------

    @staticmethod
    def build(owner: QWidget) -> None:  # noqa: C901  (intentionally long setup)
        """Construct and attach all UI widgets to *owner*.

        Postcondition: the following attributes are set on *owner*:
            _collision_checkbox, _frames_checkbox, _joints_checkbox,
            _contacts_checkbox, _launch_btn, _viewport, _status_label
        """
        layout = QVBoxLayout(owner)

        # -- Toolbar -----------------------------------------------------------
        toolbar = QHBoxLayout()

        toggle_frame = QFrame()
        toggle_frame.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border-radius: 4px;
                padding: 2px;
            }
            QCheckBox {
                color: #ddd;
                padding: 4px 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border-radius: 2px;
            }
        """)
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(4, 2, 4, 2)
        toggle_layout.setSpacing(8)

        owner._collision_checkbox = QCheckBox("Collision")
        owner._collision_checkbox.setToolTip("Show collision geometry (red wireframe)")
        owner._collision_checkbox.toggled.connect(owner._on_collision_toggled)
        toggle_layout.addWidget(owner._collision_checkbox)

        owner._frames_checkbox = QCheckBox("Frames")
        owner._frames_checkbox.setChecked(True)
        owner._frames_checkbox.setToolTip("Show coordinate frames at each body")
        owner._frames_checkbox.toggled.connect(owner._on_frames_toggled)
        toggle_layout.addWidget(owner._frames_checkbox)

        owner._joints_checkbox = QCheckBox("Joints")
        owner._joints_checkbox.setToolTip("Show joint axes and limits")
        owner._joints_checkbox.toggled.connect(owner._on_joints_toggled)
        toggle_layout.addWidget(owner._joints_checkbox)

        owner._contacts_checkbox = QCheckBox("Contacts")
        owner._contacts_checkbox.setToolTip("Show contact points and forces")
        owner._contacts_checkbox.toggled.connect(owner._on_contacts_toggled)
        toggle_layout.addWidget(owner._contacts_checkbox)

        toolbar.addWidget(toggle_frame)
        toolbar.addStretch()

        owner._launch_btn = QPushButton("Launch Full Viewer")
        owner._launch_btn.setToolTip("Open in MuJoCo's interactive viewer")
        owner._launch_btn.clicked.connect(owner._launch_external_viewer)
        toolbar.addWidget(owner._launch_btn)

        layout.addLayout(toolbar)

        # -- Viewport ----------------------------------------------------------
        owner._viewport = QLabel()
        owner._viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        owner._viewport.setMinimumSize(320, 240)
        owner._viewport.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        owner._viewport.setMouseTracking(True)
        layout.addWidget(owner._viewport, stretch=1)

        # -- Status bar --------------------------------------------------------
        owner._status_label = QLabel()
        owner._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(owner._status_label)

        # -- Headless fallback -------------------------------------------------
        if not MUJOCO_AVAILABLE:
            owner._status_label.setText(
                "⚠️ MuJoCo not installed - running in headless mode"
            )
            disable_toggles(owner)
            show_headless_placeholder(owner)


# ---------------------------------------------------------------------------
# Standalone helpers (used by the widget directly)
# ---------------------------------------------------------------------------


def disable_toggles(owner: QWidget) -> None:
    """Disable all visualization toggles and the launch button."""
    owner._collision_checkbox.setEnabled(False)
    owner._frames_checkbox.setEnabled(False)
    owner._joints_checkbox.setEnabled(False)
    owner._contacts_checkbox.setEnabled(False)
    owner._launch_btn.setEnabled(False)


def show_headless_placeholder(owner: QWidget) -> None:
    """Render a clear headless-mode placeholder in the viewport."""
    owner._viewport.setStyleSheet("""
        QLabel {
            background-color: #1a1a2e;
            border: 2px dashed #4a4a6a;
            border-radius: 8px;
            color: #8888aa;
            font-size: 14px;
        }
    """)
    owner._viewport.setText(
        "\U0001f5a5️ Headless Mode\n\n"
        "MuJoCo is not installed.\n"
        "3D preview is unavailable.\n\n"
        "To enable 3D visualization:\n"
        "  pip install mujoco\n\n"
        "Model data is still being processed\n"
        "and exported correctly."
    )


def paint_render(owner: QWidget, image: np.ndarray) -> None:
    """Convert *image* (H, W, 3 uint8 array) to a QPixmap and update the viewport.

    Args:
        owner: The viewer widget whose ``_viewport`` should be updated.
        image: RGB numpy array produced by the offscreen renderer.
    """
    h, w, c = image.shape
    bytes_per_line = c * w
    q_image = QImage(
        image.data,
        w,
        h,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    pixmap = QPixmap.fromImage(q_image)
    scaled = pixmap.scaled(
        owner._viewport.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    owner._viewport.setPixmap(scaled)
