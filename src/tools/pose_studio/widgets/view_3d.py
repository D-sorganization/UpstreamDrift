"""3D skeleton viewport using matplotlib's QtAgg canvas.

Renders the skeleton from either:
1. Live kinematics service transforms (when available), or
2. Canonical forward kinematics (fallback)

Click-to-highlight is supported for v1; drag/IK is deferred to a follow-up issue.

Why matplotlib instead of :class:`PyQtGLRenderer` from
:mod:`body_part_viz`?  The body_part_viz renderer expects fitted shape
primitives (cylinders/ellipsoids for body segments) that are out of
scope for v1 — pose_studio only needs a lines-and-dots skeleton, which
matplotlib draws cheaply.  When IK drag-handles ship in a follow-up
the renderer can be swapped for a proper rigged view without changing
this widget's public surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    forward_kinematics,
)
from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.live_kinematics import LiveKinematicsService

logger = get_logger(__name__)


# Skeleton bones, expressed as ordered pairs of landmark names returned
# by forward_kinematics().  See SkeletonPose.points for the canonical
# landmark set.
_BONES: tuple[tuple[str, str], ...] = (
    ("pelvis", "spine_top"),
    ("spine_top", "torso_top"),
    ("torso_top", "l_shoulder"),
    ("torso_top", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("l_wrist", "l_hand"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("r_wrist", "r_hand"),
    ("butt", "clubhead"),
)


class View3D(QtWidgets.QWidget):
    """3D matplotlib canvas drawing the canonical skeleton.

    Signals
    -------
    landmark_picked(str)
        Emitted with the canonical landmark name when the user clicks a
        landmark dot.  Used by the joint panel to scroll the matching
        joint into view.
    """

    landmark_picked = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(
            "3D skeleton view. Click a landmark dot to highlight the "
            "joint. Drag-and-IK ships in a follow-up issue."
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setToolTip(self.toolTip())
        # matplotlib 3D Axes carry methods (set_zlabel, set_zlim, set_box_aspect
        # with 3-tuples) that the public Axes stub does not expose.  Use Any
        # locally rather than importing mpl_toolkits.mplot3d.Axes3D, which is
        # itself not present in older type stubs.
        self._ax: Any = self._figure.add_subplot(111, projection="3d")
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        self._ax.set_box_aspect((1, 1, 1))

        layout.addWidget(self._canvas)

        # Pre-create artists so update_pose mutates rather than rebuilds.
        # ``s=40`` is the marker size keyword for Axes3D.scatter; the 2D
        # Axes stub treats the same kwarg as positional which mypy flags.
        self._scatter: Any = self._ax.scatter(
            [0.0], [0.0], [0.0], s=40, c="#5fa8ff", picker=True, pickradius=6
        )
        (self._bone_lines,) = self._ax.plot(
            [0.0], [0.0], [0.0], lw=2.0, color="#cccccc"
        )
        self._bone_lines = cast(Any, self._bone_lines)
        self._highlighted_landmark: str | None = None
        self._landmarks_order: list[str] = []
        self._service: LiveKinematicsService | None = None

        self._canvas.mpl_connect("pick_event", self._on_pick)

        # Show a sensible initial pose so the canvas is not blank.
        self.update_pose(CanonicalPose.__new__(CanonicalPose) if False else None)

    # ---- public surface ------------------------------------------------

    def set_service(self, service: LiveKinematicsService | None) -> None:
        """Set the live kinematics service for engine-specific rendering.

        When a service is set, :meth:`update_pose` will render using the
        service's link transforms instead of canonical forward kinematics,
        allowing engine-specific kinematics and constraints to be visible.
        """
        self._service = service

    def update_pose(
        self, pose: CanonicalPose | None, *, use_service: bool = True
    ) -> None:
        """Re-draw the skeleton for *pose*.

        ``None`` clears to the all-zero canonical pose, which is the
        T-pose at the world origin.

        Parameters
        ----------
        pose
            The canonical pose to render, or ``None`` for the zero pose.
        use_service
            If ``True`` and a live service is available, render using the
            service's link transforms (showing engine-specific kinematics).
            If ``False``, always use canonical forward kinematics.
        """
        angles: Mapping[str, float]
        if pose is None:
            angles = {}
        elif isinstance(pose, CanonicalPose):
            angles = pose.angles_full_dict_deg()
        else:
            raise TypeError(
                f"pose must be a CanonicalPose or None, got {type(pose).__name__}"
            )

        # Try to render from service transforms if available and requested.
        # This shows engine-specific kinematics, constraints, and convention
        # differences that canonical forward kinematics cannot show.
        if use_service and self._service is not None:
            try:
                self._service.set_pose(pose) if pose else None
                transforms = self._service.get_link_transforms()
                self._render_from_transforms(transforms)
                self._canvas.draw_idle()
                return
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                # Fall back to canonical forward kinematics if service fails.
                logger.debug("Service render failed, using FK fallback: %s", exc)

        skeleton = forward_kinematics(angles)
        names = list(skeleton.points.keys())
        coords = np.array([skeleton.points[n] for n in names], dtype=float)

        self._landmarks_order = names
        self._update_skeleton_coords(coords)

    def update_from_service_transforms(
        self, transforms: Mapping[str, np.ndarray]
    ) -> None:
        """Re-draw the skeleton from service-provided link transforms.

        Parameters
        ----------
        transforms
            Mapping from landmark name to a 4x4 SE(3) matrix.
            The position is extracted from the translation column (index 3).

        This method renders engine-specific kinematics that may differ
        from canonical forward_kinematics due to engine conventions,
        constraints, or numerical differences.
        """
        names = list(transforms.keys())
        coords = np.array([transforms[n][:3, 3] for n in names], dtype=float)
        self._landmarks_order = names
        self._update_skeleton_coords(coords)

    def _update_skeleton_coords(self, coords: np.ndarray) -> None:
        """Update scatter and bones from landmark coordinates."""
        # Update scatter
        self._scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])

        # Update bones
        bone_x: list[float] = []
        bone_y: list[float] = []
        bone_z: list[float] = []
        for a, b in _BONES:
            if a in self._landmarks_order and b in self._landmarks_order:
                idx_a = self._landmarks_order.index(a)
                idx_b = self._landmarks_order.index(b)
                pa = coords[idx_a]
                pb = coords[idx_b]
                bone_x.extend([float(pa[0]), float(pb[0]), np.nan])
                bone_y.extend([float(pa[1]), float(pb[1]), np.nan])
                bone_z.extend([float(pa[2]), float(pb[2]), np.nan])
        self._bone_lines.set_data_3d(bone_x, bone_y, bone_z)

        # Auto-fit axes around the skeleton with a small margin.
        if coords.size:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            spans = np.maximum(maxs - mins, 0.5)
            half = float(spans.max() * 0.6)
            cx, cy, cz = (mins + maxs) / 2.0
            self._ax.set_xlim(cx - half, cx + half)
            self._ax.set_ylim(cy - half, cy + half)
            self._ax.set_zlim(cz - half, cz + half)

        self._canvas.draw_idle()

    def highlighted_landmark(self) -> str | None:
        """Return the most recently clicked landmark, or ``None``."""
        return self._highlighted_landmark

    # ---- internals -----------------------------------------------------

    def _render_from_transforms(
        self, transforms: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Render skeleton from link transforms (service-based rendering).

        Parameters
        ----------
        transforms
            Mapping of link name to 4x4 SE(3) transform matrix.
        """
        # Extract landmark positions from transforms.
        # Map canonical landmark names to expected transform keys.
        landmark_keys = {
            "pelvis": "pelvis",
            "spine_top": "spine_top",
            "torso_top": "torso_top",
            "l_shoulder": "l_shoulder",
            "r_shoulder": "r_shoulder",
            "l_elbow": "l_elbow",
            "r_elbow": "r_elbow",
            "l_wrist": "l_wrist",
            "r_wrist": "r_wrist",
            "l_hand": "l_hand",
            "r_hand": "r_hand",
            "butt": "butt",
            "clubhead": "clubhead",
        }
        names: list[str] = []
        positions: list[npt.NDArray[np.float64]] = []
        for lm_name, key in landmark_keys.items():
            if key in transforms:
                mat = transforms[key]
                pos = mat[:3, 3]  # Extract translation from SE(3)
                names.append(lm_name)
                positions.append(pos)

        if not positions:
            # No transforms available, fall back to empty render.
            self._landmarks_order = []
            self._scatter._offsets3d = ([], [], [])
            self._bone_lines.set_data_3d([], [], [])
            return

        coords = np.array(positions, dtype=float)
        self._landmarks_order = names

        # Update scatter
        self._scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])

        # Update bones
        bone_x: list[float] = []
        bone_y: list[float] = []
        bone_z: list[float] = []
        for a, b in _BONES:
            if a in names and b in names:
                idx_a = names.index(a)
                idx_b = names.index(b)
                pa = positions[idx_a]
                pb = positions[idx_b]
                bone_x.extend([float(pa[0]), float(pb[0]), np.nan])
                bone_y.extend([float(pa[1]), float(pb[1]), np.nan])
                bone_z.extend([float(pa[2]), float(pb[2]), np.nan])
        self._bone_lines.set_data_3d(bone_x, bone_y, bone_z)

        # Auto-fit axes around the skeleton with a small margin.
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        spans = np.maximum(maxs - mins, 0.5)
        half = float(spans.max() * 0.6)
        cx, cy, cz = (mins + maxs) / 2.0
        self._ax.set_xlim(cx - half, cx + half)
        self._ax.set_ylim(cy - half, cy + half)
        self._ax.set_zlim(cz - half, cz + half)

    def _on_pick(self, event: object) -> None:
        artist = getattr(event, "artist", None)
        ind = getattr(event, "ind", None)
        if artist is not self._scatter or ind is None:
            return
        try:
            idx = int(ind[0])
        except (IndexError, TypeError):
            return
        if 0 <= idx < len(self._landmarks_order):
            name = self._landmarks_order[idx]
            self._highlighted_landmark = name
            logger.debug("View3D landmark picked: %s", name)
            self.landmark_picked.emit(name)


__all__ = ["View3D"]
