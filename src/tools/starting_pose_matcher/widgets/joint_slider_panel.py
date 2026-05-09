"""Joint-slider + 6-DOF rigid-transform widget (issue #4706).

The starting-pose matcher previously offered no way to *adjust* a loaded
skeleton: the user could pick a snapshot but not author the seed pose
that ``fit_swing`` consumes as ``initial_pose``. This widget closes that
gap. It exposes:

* one :class:`QSlider` per joint coordinate, driven by the names returned
  from the active provider's ``coord_names()`` method when available, or
  the canonical fallback :data:`DEFAULT_JOINT_COORDS` otherwise.
* six rigid-handle sliders (``tx``, ``ty``, ``tz``, ``rx``, ``ry``,
  ``rz``) for the root frame.
* a "Reset" button that restores the dataclass defaults.

The widget emits a single :pyattr:`pose_changed` signal with a frozen
:class:`PoseState` snapshot whenever any slider moves. The owning window
forwards that snapshot to ``LiveViewController`` and to the session
serialiser; nothing about the rendering pipeline lives in this module.

Sliders carry integer positions and the widget converts to/from radians
(joints) and metres / radians (rigid transform) using the ``_SCALE``
constant. This keeps Qt's integer-only ``QSlider`` happy without a
second floating-point widget per row, which on a 23-coord skeleton would
double the layout height.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# Canonical 23-coord humanoid set (Drake convention) used as the fallback
# when the active provider does not expose ``coord_names()``. Order is
# load-bearing: the matcher's session JSON uses positional vectors keyed
# on this tuple, so changing the order is a schema break.
DEFAULT_JOINT_COORDS: tuple[str, ...] = (
    # spine / torso (3)
    "spine_yaw",
    "spine_pitch",
    "spine_roll",
    # neck (1)
    "neck_pitch",
    # left arm (4)
    "l_shoulder_yaw",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_elbow",
    # right arm (4)
    "r_shoulder_yaw",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_elbow",
    # left leg (5)
    "l_hip_yaw",
    "l_hip_pitch",
    "l_hip_roll",
    "l_knee",
    "l_ankle",
    # right leg (5)
    "r_hip_yaw",
    "r_hip_pitch",
    "r_hip_roll",
    "r_knee",
    "r_ankle",
    # club (1)
    "club_grip",
)

# 6-DOF root frame coordinates, in fixed order.
RIGID_COORDS: tuple[str, ...] = ("tx", "ty", "tz", "rx", "ry", "rz")

# Joint slider range, in radians. ±π covers every revolute joint Drake
# can exercise without re-mapping per joint.
_JOINT_MIN_RAD: float = -math.pi
_JOINT_MAX_RAD: float = math.pi

# Translation slider range, in metres. ±2 m comfortably covers any
# whole-skeleton displacement the matcher needs without drifting off the
# axes.
_TRANS_MIN_M: float = -2.0
_TRANS_MAX_M: float = 2.0

# Sliders carry int positions in [0, _SCALE]; the float value is recovered
# by linear interpolation between the row's min and max. 1000 gives
# millimetre / milliradian precision, which is well below the noise floor
# of any human-authored seed pose.
_SCALE: int = 1000


class _CoordProvider(Protocol):
    """Minimal duck-type for ``provider.coord_names()`` callers.

    Kept as a ``Protocol`` so we never import the concrete provider into
    this widget module — that would couple GUI code to motion-matching's
    physics-engine bindings.
    """

    def coord_names(self) -> tuple[str, ...]: ...


def resolve_coord_names(provider: Any | None) -> tuple[str, ...]:
    """Return the joint-coordinate vocabulary to drive the panel.

    Parameters
    ----------
    provider
        Optional object implementing :class:`_CoordProvider`. ``None``,
        a missing method, or any exception falls back to
        :data:`DEFAULT_JOINT_COORDS`.

    Returns
    -------
    tuple of str
        At least one coord name; the caller can rely on a non-empty
        result.
    """
    if provider is None:
        return DEFAULT_JOINT_COORDS
    fn = getattr(provider, "coord_names", None)
    if not callable(fn):
        return DEFAULT_JOINT_COORDS
    try:
        names = tuple(str(n) for n in fn())
    except Exception:  # noqa: BLE001 - provider errors shouldn't break GUI
        logger.exception("provider.coord_names() raised; using defaults")
        return DEFAULT_JOINT_COORDS
    return names if names else DEFAULT_JOINT_COORDS


@dataclass(frozen=True)
class PoseState:
    """Frozen snapshot of every slider in the panel.

    ``joint_angles`` is a mapping from coord-name to radians; rigid
    transform fields are metres (``tx``, ``ty``, ``tz``) and radians
    (``rx``, ``ry``, ``rz``). Frozen so it's safe to ship across the
    ``pose_changed`` signal without callers mutating the live state.
    """

    joint_angles: dict[str, float] = field(default_factory=dict)
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation for ``session_schema``."""
        return {
            "joint_angles": dict(self.joint_angles),
            "rigid_transform": {
                "tx": self.tx,
                "ty": self.ty,
                "tz": self.tz,
                "rx": self.rx,
                "ry": self.ry,
                "rz": self.rz,
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | None, *, coord_names: tuple[str, ...]
    ) -> PoseState:
        """Round-trip decoder; missing keys take their default 0.0.

        ``coord_names`` filters which joints land in the result, so a
        session JSON written with a different provider's vocabulary
        loses unknown joints rather than crashing.
        """
        if not data:
            return cls(joint_angles=dict.fromkeys(coord_names, 0.0))
        raw_joints = data.get("joint_angles") or {}
        joints = {n: float(raw_joints.get(n, 0.0)) for n in coord_names}
        rt = data.get("rigid_transform") or {}
        return cls(
            joint_angles=joints,
            tx=float(rt.get("tx", 0.0)),
            ty=float(rt.get("ty", 0.0)),
            tz=float(rt.get("tz", 0.0)),
            rx=float(rt.get("rx", 0.0)),
            ry=float(rt.get("ry", 0.0)),
            rz=float(rt.get("rz", 0.0)),
        )


def _slider_to_float(pos: int, lo: float, hi: float) -> float:
    """Map an int slider position in ``[0, _SCALE]`` to ``[lo, hi]``."""
    if _SCALE <= 0:  # pragma: no cover - constant
        return lo
    t = max(0, min(_SCALE, int(pos))) / _SCALE
    return lo + t * (hi - lo)


def _float_to_slider(value: float, lo: float, hi: float) -> int:
    """Inverse of :func:`_slider_to_float`, clamped to slider bounds."""
    if hi <= lo:  # pragma: no cover - defensive
        return 0
    t = (float(value) - lo) / (hi - lo)
    return max(0, min(_SCALE, int(round(t * _SCALE))))


class JointSliderPanel(QWidget):
    """Pose-authoring panel for the starting-pose matcher.

    Emits :pyattr:`pose_changed` with a fresh :class:`PoseState` whenever
    any slider moves. Construction is cheap and side-effect free — no
    QApplication-wide state is touched, so multiple panels can coexist
    in tests.
    """

    # Single, fully-typed signal: one PoseState per change. Listeners
    # decide whether to debounce.
    pose_changed = pyqtSignal(object)

    def __init__(
        self,
        coord_names: tuple[str, ...] | None = None,
        provider: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Caller may pass an explicit list (preferred for tests) or a
        # provider whose ``coord_names()`` we'll consult once.
        if coord_names is None:
            coord_names = resolve_coord_names(provider)
        if not coord_names:
            raise ValueError("coord_names must be non-empty")
        self._coord_names: tuple[str, ...] = tuple(coord_names)

        self._joint_sliders: dict[str, QSlider] = {}
        self._joint_labels: dict[str, QLabel] = {}
        self._rigid_sliders: dict[str, QSlider] = {}
        self._rigid_labels: dict[str, QLabel] = {}
        self._suspend_emit: bool = False

        self._build_ui()
        # Seed labels with the zero defaults so the first paint is clean.
        self._refresh_all_labels()

    # -- public accessors -------------------------------------------------- #

    @property
    def coord_names(self) -> tuple[str, ...]:
        """Return the joint coord vocabulary this panel is driving."""
        return self._coord_names

    def pose_state(self) -> PoseState:
        """Return a fresh :class:`PoseState` snapshot of every slider."""
        joints = {
            name: _slider_to_float(
                self._joint_sliders[name].value(), _JOINT_MIN_RAD, _JOINT_MAX_RAD
            )
            for name in self._coord_names
        }
        rigid: dict[str, float] = {}
        for name in RIGID_COORDS:
            lo, hi = self._rigid_bounds(name)
            rigid[name] = _slider_to_float(self._rigid_sliders[name].value(), lo, hi)
        return PoseState(joint_angles=joints, **rigid)

    def set_pose_state(self, state: PoseState) -> None:
        """Apply a :class:`PoseState`; emits one ``pose_changed`` at the end.

        Slider updates are batched under ``_suspend_emit`` so listeners
        don't see one signal per slider during a bulk reload.
        """
        self._suspend_emit = True
        try:
            for name in self._coord_names:
                angle = float(state.joint_angles.get(name, 0.0))
                self._joint_sliders[name].setValue(
                    _float_to_slider(angle, _JOINT_MIN_RAD, _JOINT_MAX_RAD)
                )
            for name in RIGID_COORDS:
                lo, hi = self._rigid_bounds(name)
                value = float(getattr(state, name))
                self._rigid_sliders[name].setValue(_float_to_slider(value, lo, hi))
        finally:
            self._suspend_emit = False
        self._refresh_all_labels()
        self._emit_pose_changed()

    def reset(self) -> None:
        """Restore every slider to its default (zero) value."""
        zeros = PoseState(joint_angles=dict.fromkeys(self._coord_names, 0.0))
        self.set_pose_state(zeros)

    # -- UI construction --------------------------------------------------- #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        outer.addWidget(self._build_rigid_group())
        outer.addWidget(self._build_joint_group(), stretch=1)
        outer.addLayout(self._build_buttons())

    def _build_rigid_group(self) -> QGroupBox:
        group = QGroupBox("Rigid transform (root)")
        form = QFormLayout(group)
        for name in RIGID_COORDS:
            slider, label = self._make_slider_row(name, is_rigid=True)
            self._rigid_sliders[name] = slider
            self._rigid_labels[name] = label
            row = QHBoxLayout()
            row.addWidget(slider, stretch=1)
            row.addWidget(label)
            form.addRow(name, row)
        return group

    def _build_joint_group(self) -> QGroupBox:
        group = QGroupBox("Joint angles")
        layout = QVBoxLayout(group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        for name in self._coord_names:
            slider, label = self._make_slider_row(name, is_rigid=False)
            self._joint_sliders[name] = slider
            self._joint_labels[name] = label
            row = QHBoxLayout()
            row.addWidget(slider, stretch=1)
            row.addWidget(label)
            form.addRow(name, row)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        return group

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset every slider to zero")
        self.btn_reset.clicked.connect(self.reset)
        row.addStretch(1)
        row.addWidget(self.btn_reset)
        return row

    def _make_slider_row(self, name: str, *, is_rigid: bool) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(_SCALE)
        slider.setValue(_SCALE // 2)  # midpoint == 0.0 in mapped units
        slider.setObjectName(f"slider::{'rigid' if is_rigid else 'joint'}::{name}")
        label = QLabel("0.000")
        label.setMinimumWidth(60)
        slider.valueChanged.connect(lambda _v, n=name: self._on_slider_changed(n))
        return slider, label

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _rigid_bounds(name: str) -> tuple[float, float]:
        # Translations in metres, rotations in radians.
        return (
            (_TRANS_MIN_M, _TRANS_MAX_M)
            if name in ("tx", "ty", "tz")
            else (_JOINT_MIN_RAD, _JOINT_MAX_RAD)
        )

    def _on_slider_changed(self, name: str) -> None:
        self._refresh_label(name)
        self._emit_pose_changed()

    def _refresh_label(self, name: str) -> None:
        if name in self._joint_sliders:
            value = _slider_to_float(
                self._joint_sliders[name].value(), _JOINT_MIN_RAD, _JOINT_MAX_RAD
            )
            self._joint_labels[name].setText(f"{value:+.3f} rad")
        elif name in self._rigid_sliders:
            lo, hi = self._rigid_bounds(name)
            value = _slider_to_float(self._rigid_sliders[name].value(), lo, hi)
            unit = "m" if name in ("tx", "ty", "tz") else "rad"
            self._rigid_labels[name].setText(f"{value:+.3f} {unit}")

    def _refresh_all_labels(self) -> None:
        for n in self._coord_names:
            self._refresh_label(n)
        for n in RIGID_COORDS:
            self._refresh_label(n)

    def _emit_pose_changed(self) -> None:
        if self._suspend_emit:
            return
        # ``replace`` on the snapshot guarantees listeners can't mutate
        # our internal dict by holding the reference.
        snapshot = self.pose_state()
        self.pose_changed.emit(replace(snapshot))


__all__ = [
    "DEFAULT_JOINT_COORDS",
    "RIGID_COORDS",
    "JointSliderPanel",
    "PoseState",
    "resolve_coord_names",
]
