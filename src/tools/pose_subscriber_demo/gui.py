"""GUI for the Pose Subscriber demo.

Subscribes to ``pose/canonical`` on construction and renders the most
recent canonical pose as a coarse forward-kinematics skeleton in an
embedded matplotlib canvas. The widget is a passive consumer — it never
publishes anything itself.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    SegmentLengths,
    SkeletonPose,
    forward_kinematics,
)
from src.shared.python.realtime import Subscription, subscribe

logger = get_logger(__name__)

__all__ = ["MainWidget"]


# Skeleton segments: each tuple is the names of two ``SkeletonPose``
# landmarks to draw a line between. Names match what
# :func:`forward_kinematics` actually emits.
_SKELETON_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("pelvis", "spine"),
    ("spine", "torso"),
    ("torso", "left_shoulder"),
    ("torso", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)


class MainWidget(QtWidgets.QWidget):
    """Live mirror of Pose Studio's canonical pose.

    The widget owns a single :class:`Subscription`; :meth:`cleanup`
    tears it down. A Qt signal is used to marshal the realtime callback
    (which runs on a transport-owned daemon thread) onto the GUI
    thread before it touches matplotlib state.
    """

    # Emitted from the realtime worker thread; the connected slot runs
    # on the GUI thread thanks to Qt's queued-connection default for
    # cross-thread signals.
    _pose_received = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._subscription: Subscription | None = None

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._configure_axes()

        self._status_label = QtWidgets.QLabel("Last update: (waiting…)")
        self._status_label.setObjectName("PoseSubscriberStatus")
        self._channel_label = QtWidgets.QLabel("Subscribed channel: pose/canonical")
        self._channel_label.setStyleSheet("color: #888;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._channel_label)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(self._status_label)

        # Cross-thread plumbing.
        self._pose_received.connect(self._on_pose_received)

        # Keep a reference to the lengths so we don't reallocate per
        # render — they're a small frozen dataclass but the call is
        # hot enough to justify it.
        self._segment_lengths = SegmentLengths()

        self._subscribe()

    # ---- subscription lifecycle --------------------------------------

    def _subscribe(self) -> None:
        try:
            self._subscription = subscribe("pose/canonical", self._on_realtime_payload)
        except Exception:
            # ``subscribe`` already returns an inert subscription on
            # transport failure, but we belt-and-brace here so that a
            # broken realtime layer never prevents the demo widget
            # from constructing.
            logger.exception("pose_subscriber_demo: failed to subscribe")
            self._subscription = None

    def cleanup(self) -> None:
        """Release the realtime subscription. Idempotent."""
        sub = self._subscription
        self._subscription = None
        if sub is None:
            return
        try:
            sub.unsubscribe()
        except Exception:  # pragma: no cover - defensive
            logger.exception("pose_subscriber_demo: unsubscribe raised")

    # ---- payload handling --------------------------------------------

    def _on_realtime_payload(self, payload: Any) -> None:
        # Runs on the realtime daemon thread; emit a queued signal to
        # hop onto the GUI thread before touching matplotlib.
        self._pose_received.emit(payload)

    @QtCore.pyqtSlot(object)
    def _on_pose_received(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            logger.debug(
                "pose_subscriber_demo: dropping non-dict payload of type %s",
                type(payload).__name__,
            )
            return
        angles = payload.get("joint_angles_deg")
        if not isinstance(angles, dict):
            logger.debug("pose_subscriber_demo: payload missing joint_angles_deg")
            return
        try:
            skeleton = forward_kinematics(angles, self._segment_lengths)
        except (TypeError, ValueError):
            logger.exception(
                "pose_subscriber_demo: forward_kinematics rejected payload"
            )
            return
        self._render_skeleton(skeleton)
        timestamp = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._status_label.setText(f"Last update: {timestamp}")

    # ---- rendering ---------------------------------------------------

    def _configure_axes(self) -> None:
        self._axes.set_xlabel("X (m)")
        self._axes.set_ylabel("Y (m)")
        # mypy: 3D axes have set_zlabel but matplotlib's stub is partial
        zlabel = getattr(self._axes, "set_zlabel", None)
        if zlabel is not None:
            zlabel("Z (m)")
        self._axes.set_xlim(-1.0, 1.0)
        self._axes.set_ylim(-1.0, 1.0)
        zlim = getattr(self._axes, "set_zlim", None)
        if zlim is not None:
            zlim(0.0, 2.0)
        self._axes.set_title("Pose Subscriber (live)")

    def _render_skeleton(self, skeleton: SkeletonPose) -> None:
        self._axes.cla()
        self._configure_axes()
        for a, b in _SKELETON_SEGMENTS:
            pa = skeleton.points.get(a)
            pb = skeleton.points.get(b)
            if pa is None or pb is None:
                continue
            xs = np.asarray([pa[0], pb[0]], dtype=float)
            ys = np.asarray([pa[1], pb[1]], dtype=float)
            zs = np.asarray([pa[2], pb[2]], dtype=float)
            self._axes.plot(xs, ys, zs, color="#1f77b4", linewidth=2.0)
            self._axes.scatter(xs, ys, zs, color="#d62728", s=12)
        self._canvas.draw_idle()
