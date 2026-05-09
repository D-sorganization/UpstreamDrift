"""Three-dimensional scene viewer tab for the 3-D Golf Model GUI.

Provides interactive playback (play/pause/speed/loop), keyboard
shortcuts, an optional skeleton overlay built from the canonical
anatomical-subset segment table, marker-selection helpers, view-angle
presets, event-frame quick-jump, marker-label toggling, CSV export,
and an incremental-update render path that preallocates artists once
per selection and only mutates them per frame for smooth scrubbing.

User-defined segments are rendered through the canonical
:class:`body_part_viz.renderers.MatplotlibRenderer`. Each segment's
shape is built from its v2 :class:`SegmentVizSpec`, fitted via the
appropriate :class:`ShapeFitter`, and added to the renderer once. The
per-frame path issues a single ``update_frame`` call per shape — no
artist rebuilds, no scene clears.
"""

from __future__ import annotations

import csv
import logging
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer

from src.shared.python.body_part_viz import (
    SegmentVizSpec,
    ShapeTheme,
)
from src.shared.python.body_part_viz.asset_library import ShapeLibrary
from src.shared.python.body_part_viz.fitters import (
    BetweenTwoMarkersFitter,
    ClusterKabschFitter,
    ProcrustesAnisotropicFitter,
)
from src.shared.python.body_part_viz.renderers import MatplotlibRenderer
from src.shared.python.body_part_viz.shapes import (
    CapsuleShape,
    CylinderShape,
    EllipsoidShape,
    LineShape,
    MeshShape,
)
from src.shared.python.plot_style import (
    ColormapId,
    DataChannel,
    DataDrivenColor,
    MarkerStyle,
    MatplotlibMarkerRenderer,
    StaticColor,
)
from src.shared.python.qt_utils.wheel_event_filter import suppress_wheel_on_widgets

from ...core.models import C3DDataModel
from ...services.segment_set_io import SegmentSpec, spec_v1_to_v2
from ..widgets.mpl_canvas import MplCanvas
from ._plot_style_helpers import StylePersistence, default_style_for
from .marker_plot_tab import _open_style_dialog

_LOGGER = logging.getLogger(__name__)

# Anatomical-region colour map (generic, source-agnostic).
_GROUP_COLORS: dict[str, tuple[float, float, float, float]] = {
    "pelvis": (0.20, 0.55, 0.85, 1.0),
    "torso": (0.45, 0.30, 0.75, 1.0),
    "head": (0.85, 0.60, 0.20, 1.0),
    "left_arm": (0.30, 0.75, 0.40, 1.0),
    "right_arm": (0.20, 0.50, 0.30, 1.0),
    "left_leg": (0.85, 0.30, 0.30, 1.0),
    "right_leg": (0.65, 0.20, 0.20, 1.0),
    "auto": (0.30, 0.30, 0.30, 1.0),
    "default": (0.30, 0.30, 0.30, 1.0),
}


def _color_for_group(group: str) -> tuple[float, float, float, float]:
    return _GROUP_COLORS.get(group, _GROUP_COLORS["auto"])


def _rgba_to_hex(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _a = rgba
    rh = int(round(r * 255))
    gh = int(round(g * 255))
    bh = int(round(b * 255))
    return f"#{rh:02x}{gh:02x}{bh:02x}"


try:
    from src.shared.python.motion_matching.body_skeleton import (
        default_body_segments,
    )
except ImportError:  # pragma: no cover - exercised via fallback path only
    # Fallback: when running inside the engine's pivoted sys.modules layout
    # the canonical wrapper exposes the symbol via the bare-rooted path.
    from shared.python.motion_matching.body_skeleton import (  # type: ignore
        default_body_segments,
    )


# View-angle presets: (elev, azim) tuned to match issue spec.
_VIEW_PRESETS: dict[str, tuple[float, float]] = {
    "Front": (0.0, 90.0),
    "Side": (0.0, 0.0),
    "Top": (90.0, -90.0),
    "Iso": (30.0, -60.0),
}

_PLAYBACK_SPEEDS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
_DEFAULT_SPEED_INDEX = 3  # 1.0×

# Cluster/club marker name patterns — generic, not vendor-specific.
_CLUB_MARKER_RE = re.compile(r"^Marker_\d+:\d+:", re.IGNORECASE)


def _is_club_marker(name: str) -> bool:
    """Return True when ``name`` looks like a club / cluster marker."""
    return bool(_CLUB_MARKER_RE.match(name))


def _validate_speed(speed: float) -> float:
    """Validate a playback speed multiplier."""
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise TypeError(f"speed must be a real number, got {type(speed).__name__}")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed must be positive and finite, got {speed!r}")
    return float(speed)


def _validate_frame(frame: int, n_frames: int) -> int:
    """Validate a frame index against ``n_frames``."""
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise TypeError(f"frame must be int, got {type(frame).__name__}")
    if n_frames <= 0:
        raise ValueError("no frames loaded")
    if not 0 <= frame < n_frames:
        raise ValueError(f"frame {frame} out of range [0, {n_frames})")
    return frame


def _build_shape_from_spec(
    spec: SegmentVizSpec,
    *,
    library: ShapeLibrary | None,
) -> Any:
    """Construct a :class:`BodyPartShape` from a v2 spec.

    Returns ``None`` when the shape cannot be constructed (e.g. mesh file
    missing, unknown library entry). The viewer then skips that segment
    rather than crashing the whole frame.
    """
    kind = spec.shape_kind
    params = spec.shape_params
    if kind == "line":
        length = float(params.get("length", 1.0))
        return LineShape(length=length)
    if kind == "cylinder":
        return CylinderShape(
            length=float(params.get("length", 1.0)),
            radius=float(params.get("radius", 0.015)),
            n_facets=int(params.get("n_facets", 16)),
        )
    if kind == "ellipsoid":
        return EllipsoidShape(
            a=float(params["a"]),
            b=float(params["b"]),
            c=float(params["c"]),
            n_lon=int(params.get("n_lon", 16)),
            n_lat=int(params.get("n_lat", 8)),
        )
    if kind == "capsule":
        return CapsuleShape(
            length=float(params.get("length", 1.0)),
            radius=float(params.get("radius", 0.015)),
            n_facets=int(params.get("n_facets", 16)),
            n_lat=int(params.get("n_lat", 8)),
        )
    if kind == "mesh_file":
        try:
            return MeshShape.load(
                str(params["path"]),
                max_vertices=int(params.get("max_vertices", 5000)),
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            _LOGGER.warning("could not load mesh %s: %s", params.get("path"), exc)
            return None
    if kind == "library_shape":
        if library is None:
            return None
        try:
            return library.get(str(params["shape_id"]))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            _LOGGER.warning(
                "library shape %s not available: %s", params.get("shape_id"), exc
            )
            return None
    # Composite is out-of-scope for the C3D viewer integration.
    return None


def _fitter_for_kind(kind: str) -> Any:
    if kind == "between_two":
        return BetweenTwoMarkersFitter()
    if kind == "cluster_kabsch":
        return ClusterKabschFitter()
    if kind == "procrustes_anisotropic":
        return ProcrustesAnisotropicFitter()
    raise ValueError(f"unknown fitter_kind {kind!r}")


class Viewer3DTab(QtWidgets.QWidget):
    """3D marker trajectory viewer tab with playback and skeleton overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._n_frames: int = 0
        self._selected_names: list[str] = []
        # Cached arrays for fast frame updates.
        self._selected_positions: np.ndarray = np.empty((0, 0, 3))
        self._marker_colors: list[tuple[float, float, float, float]] = []

        # Matplotlib artists pre-allocated per selection.
        self._ax: Any = None
        self._trail_lines: list[Any] = []
        self._point_artist: Any = None
        self._label_texts: list[Any] = []
        self._skeleton_collection: Line3DCollection | None = None
        self._skeleton_segments: tuple[tuple[str, str], ...] = ()
        self._event_buttons: list[QtWidgets.QToolButton] = []

        # User-defined segments (Segments tab -> here). v2 store.
        self._user_viz_segments: tuple[SegmentVizSpec, ...] = ()
        # MatplotlibRenderer + per-segment handles. Each entry is
        # (handle_or_None, shape_kind). A None handle means the segment
        # could not be built (e.g. missing markers, missing mesh file)
        # and should be skipped on frame updates.
        self._renderer: MatplotlibRenderer | None = None
        self._render_entries: list[tuple[str | None, str]] = []
        self._shape_library: ShapeLibrary | None = None

        # Playback timer.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._is_playing = False

        # plot_style integration: the current-frame marker scatter is
        # owned by a MatplotlibMarkerRenderer so it can be re-styled
        # without rebuilding the rest of the scene.
        self._marker_style_renderer: MatplotlibMarkerRenderer | None = None
        self._marker_style_handle: str | None = None
        self._persistence = StylePersistence(target_prefix="group:")
        self._persistence.load()
        # Cached color-by-channel state. When set, the renderer's style
        # uses a DataDrivenColor whose channel is recomputed per frame.
        self._color_channel: DataChannel | None = None
        self._color_range: tuple[float, float] | None = None
        self._color_colormap: ColormapId = ColormapId.VIRIDIS

        self._init_ui()
        self._install_shortcuts()

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QVBoxLayout()

        # Selection helpers.
        sel_row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("Select all")
        self.btn_select_body = QtWidgets.QPushButton("Body")
        self.btn_select_club = QtWidgets.QPushButton("Club")
        self.btn_clear_selection = QtWidgets.QPushButton("Clear")
        self.btn_select_all.clicked.connect(self.select_all_markers)
        self.btn_select_body.clicked.connect(self.select_body_markers)
        self.btn_select_club.clicked.connect(self.select_club_markers)
        self.btn_clear_selection.clicked.connect(self.clear_marker_selection)
        for btn in (
            self.btn_select_all,
            self.btn_select_body,
            self.btn_select_club,
            self.btn_clear_selection,
        ):
            sel_row.addWidget(btn)
        left_panel.addLayout(sel_row)

        left_panel.addWidget(QtWidgets.QLabel("Markers to display in 3D:"))
        self.list_markers_3d = QtWidgets.QListWidget()
        self.list_markers_3d.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.list_markers_3d.itemSelectionChanged.connect(self._on_selection_changed)
        left_panel.addWidget(self.list_markers_3d)

        # Playback controls.
        playback_row = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QToolButton()
        self.btn_play.setToolTip("Play / Pause (Space)")
        self._update_play_icon()
        self.btn_play.clicked.connect(self.toggle_play)
        playback_row.addWidget(self.btn_play)

        playback_row.addWidget(QtWidgets.QLabel("Speed:"))
        self.combo_speed = QtWidgets.QComboBox()
        for s in _PLAYBACK_SPEEDS:
            self.combo_speed.addItem(f"{s:g}×", s)
        self.combo_speed.setCurrentIndex(_DEFAULT_SPEED_INDEX)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        playback_row.addWidget(self.combo_speed)

        self.check_loop = QtWidgets.QCheckBox("Loop")
        self.check_loop.setChecked(True)
        playback_row.addWidget(self.check_loop)

        playback_row.addStretch()
        left_panel.addLayout(playback_row)

        # View / overlay toggles.
        toggles_row = QtWidgets.QHBoxLayout()
        self.check_skeleton = QtWidgets.QCheckBox("Show skeleton")
        self.check_skeleton.setChecked(True)
        self.check_skeleton.toggled.connect(self._on_skeleton_toggled)
        toggles_row.addWidget(self.check_skeleton)

        self.check_labels = QtWidgets.QCheckBox("Show labels")
        self.check_labels.setChecked(False)
        self.check_labels.toggled.connect(self._on_labels_toggled)
        toggles_row.addWidget(self.check_labels)
        toggles_row.addStretch()
        left_panel.addLayout(toggles_row)

        # plot_style controls: marker-group selector + per-group "Style…"
        # button. Groups are coarse buckets matched against the built-in
        # ``default`` preset (``body``, ``club``, ``ball``, …).
        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(QtWidgets.QLabel("Marker group:"))
        self.combo_marker_group = QtWidgets.QComboBox()
        self.combo_marker_group.setObjectName("marker_group_combo")
        for label in ("default", "body", "club", "ball", "skeleton"):
            self.combo_marker_group.addItem(label)
        suppress_wheel_on_widgets(self.combo_marker_group)
        style_row.addWidget(self.combo_marker_group)
        self.btn_marker_style = QtWidgets.QPushButton("Style…")
        self.btn_marker_style.setObjectName("marker_group_style_button")
        self.btn_marker_style.clicked.connect(self._on_marker_group_style_clicked)
        style_row.addWidget(self.btn_marker_style)
        style_row.addStretch()
        left_panel.addLayout(style_row)

        # Optional color-by-channel editor (collapsed by default).
        self._color_channel_editor: QtWidgets.QWidget | None = None
        self._color_channel_holder = QtWidgets.QWidget()
        self._color_channel_layout = QtWidgets.QVBoxLayout(self._color_channel_holder)
        self._color_channel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.addWidget(self._color_channel_holder)

        # View-angle presets.
        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("View:"))
        self._view_buttons: dict[str, QtWidgets.QToolButton] = {}
        for name in _VIEW_PRESETS:
            b = QtWidgets.QToolButton()
            b.setText(name)
            b.clicked.connect(lambda _checked=False, n=name: self.set_view_preset(n))
            self._view_buttons[name] = b
            view_row.addWidget(b)
        view_row.addStretch()
        left_panel.addLayout(view_row)

        # Frame slider with event-quick-jump strip.
        slider_row = QtWidgets.QHBoxLayout()
        self.slider_frame = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setMaximum(0)
        self.slider_frame.setValue(0)
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        suppress_wheel_on_widgets(self.slider_frame, self.combo_speed)
        slider_row.addWidget(self.slider_frame)
        left_panel.addLayout(slider_row)

        # Event-jump button row (populated when events present).
        self._event_row = QtWidgets.QHBoxLayout()
        self._event_container = QtWidgets.QWidget()
        self._event_container.setLayout(self._event_row)
        left_panel.addWidget(self._event_container)
        self._event_container.setVisible(False)

        self.label_frame_info = QtWidgets.QLabel("Frame: - / Time: -")
        left_panel.addWidget(self.label_frame_info)

        layout.addLayout(left_panel, 1)

        right_panel = QtWidgets.QVBoxLayout()
        self.canvas_3d = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas_3d)

        # Optional anthropometrics panel — hidden until a segment with a
        # SegmentProperties record is selected. Lazy import so the viewer
        # remains usable even when the anthropometrics deps are absent.
        self._segment_props_panel: QtWidgets.QWidget | None = None
        try:
            from src.shared.python.anthropometrics.ui.segment_properties_panel import (  # noqa: E501
                SegmentPropertiesPanel,
            )

            panel = SegmentPropertiesPanel(self)
            panel.setVisible(False)
            self._segment_props_panel = panel
            right_panel.addWidget(panel)
        except Exception:  # pragma: no cover - optional dep
            pass

        layout.addLayout(right_panel, 3)

    # ------------------------------------------------- Anthropometrics
    def set_selected_segment_properties(self, props: Any | None) -> None:
        """Wire-in slot for surfacing :class:`SegmentProperties`.

        Hidden when *props* is ``None`` (no SegmentProperties available
        for the current selection); shown and populated otherwise. Safe
        to call when the optional anthropometrics dependency is not
        installed (the call is a no-op).
        """
        panel = self._segment_props_panel
        if panel is None:
            return
        if props is None:
            panel.setVisible(False)
            panel.set_segment(None)  # type: ignore[attr-defined]
        else:
            panel.set_segment(props)  # type: ignore[attr-defined]
            panel.setVisible(True)

    # ---------------------------------------------------------- Shortcuts

    def _install_shortcuts(self) -> None:
        """Register keyboard shortcuts on this tab."""

        def _mk(seq: str, slot: Any) -> QtGui.QShortcut:
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            return sc

        self._shortcut_play = _mk("Space", self.toggle_play)
        self._shortcut_left = _mk("Left", lambda: self.step_frame(-1))
        self._shortcut_right = _mk("Right", lambda: self.step_frame(1))
        self._shortcut_shift_left = _mk("Shift+Left", lambda: self.step_frame(-10))
        self._shortcut_shift_right = _mk("Shift+Right", lambda: self.step_frame(10))
        self._shortcut_home = _mk("Home", lambda: self.set_frame(0))
        self._shortcut_end = _mk(
            "End", lambda: self.set_frame(max(0, self._n_frames - 1))
        )

    # ------------------------------------------------------------- Model

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Update UI with data from the model."""
        self.pause()
        self.model = model
        self.list_markers_3d.clear()
        self._clear_event_buttons()

        if model is None:
            self._n_frames = 0
            self.slider_frame.setMaximum(0)
            self._teardown_artists()
            self.canvas_3d.clear_axes()
            return

        for name in model.marker_names():
            self.list_markers_3d.addItem(name)

        if model.point_time is not None:
            self._n_frames = len(model.point_time)
            self.slider_frame.setMinimum(0)
            self.slider_frame.setMaximum(max(0, self._n_frames - 1))
            self.slider_frame.setValue(0)
        else:
            self._n_frames = 0
            self.slider_frame.setMaximum(0)
            self.slider_frame.setValue(0)

        self._populate_event_buttons()

        # Default selection: pick the body markers if any are present, else
        # fall back to the first marker (preserves prior behaviour when the
        # file has no anatomical markers).
        marker_names = model.marker_names()
        body_segments_present = default_body_segments(marker_names)
        if body_segments_present:
            self.select_body_markers()
        elif marker_names:
            self.list_markers_3d.setCurrentRow(0)

    # ------------------------------------------------------- Selection

    def select_all_markers(self) -> None:
        """Select every marker in the list."""
        self.list_markers_3d.selectAll()

    def select_body_markers(self) -> None:
        """Select the canonical anatomical-subset body markers."""
        if self.model is None:
            return
        body_names = {
            n
            for s in default_body_segments(self.model.marker_names())
            for n in (s.a, s.b)
        }
        self._set_selection(lambda n: n in body_names)

    def select_club_markers(self) -> None:
        """Select markers that match the generic club/cluster pattern."""
        self._set_selection(_is_club_marker)

    def clear_marker_selection(self) -> None:
        """Clear all marker selections."""
        self.list_markers_3d.clearSelection()

    def _set_selection(self, predicate: Any) -> None:
        self.list_markers_3d.blockSignals(True)
        try:
            for i in range(self.list_markers_3d.count()):
                item = self.list_markers_3d.item(i)
                if item is None:
                    continue
                item.setSelected(bool(predicate(item.text())))
        finally:
            self.list_markers_3d.blockSignals(False)
        self._on_selection_changed()

    # --------------------------------------------------------- Playback

    @property
    def is_playing(self) -> bool:
        """Whether playback is currently running."""
        return self._is_playing

    def toggle_play(self) -> None:
        """Toggle play/pause state."""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        """Start playback at the current speed."""
        if self._n_frames <= 1:
            return
        speed = float(self.combo_speed.currentData() or 1.0)
        rate = self.model.point_rate if self.model is not None else 0.0
        if rate <= 0.0:
            rate = 60.0  # fallback display rate
        interval_ms = max(1, int(round(1000.0 / (rate * speed))))
        self._timer.start(interval_ms)
        self._is_playing = True
        self._update_play_icon()

    def pause(self) -> None:
        """Pause playback."""
        if self._timer.isActive():
            self._timer.stop()
        self._is_playing = False
        self._update_play_icon()

    def set_speed(self, speed: float) -> None:
        """Programmatically set the playback speed multiplier."""
        speed = _validate_speed(speed)
        # Snap to the closest preset for the combo box display.
        idx = int(np.argmin([abs(s - speed) for s in _PLAYBACK_SPEEDS]))
        self.combo_speed.setCurrentIndex(idx)
        if self._is_playing:
            self.pause()
            self.play()

    def _on_speed_changed(self, _idx: int) -> None:
        if self._is_playing:
            self.pause()
            self.play()

    def _on_timer_tick(self) -> None:
        if self._n_frames <= 0:
            self.pause()
            return
        nxt = self.slider_frame.value() + 1
        if nxt >= self._n_frames:
            if self.check_loop.isChecked():
                nxt = 0
            else:
                self.pause()
                return
        self.slider_frame.setValue(nxt)

    def _update_play_icon(self) -> None:
        style = self.style()
        if style is None:
            self.btn_play.setText("Pause" if self._is_playing else "Play")
            return
        icon = (
            QtWidgets.QStyle.StandardPixmap.SP_MediaPause
            if self._is_playing
            else QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
        )
        self.btn_play.setIcon(style.standardIcon(icon))
        self.btn_play.setText("Pause" if self._is_playing else "Play")

    # ------------------------------------------------- Frame navigation

    def set_frame(self, frame: int) -> None:
        """Jump to ``frame``. Validates inputs (DbC)."""
        frame = _validate_frame(frame, self._n_frames)
        self.slider_frame.setValue(frame)

    def step_frame(self, delta: int) -> None:
        """Advance the slider by ``delta`` frames, clamped to bounds."""
        if self._n_frames <= 0:
            return
        nxt = max(0, min(self._n_frames - 1, self.slider_frame.value() + int(delta)))
        self.slider_frame.setValue(nxt)

    def _on_frame_changed(self, _value: int) -> None:
        self._render_current_frame()

    # --------------------------------------------------- Event quick-jump

    def _populate_event_buttons(self) -> None:
        self._clear_event_buttons()
        if self.model is None or not self.model.events:
            self._event_container.setVisible(False)
            return
        self._event_container.setVisible(True)
        for ev in self.model.events:
            btn = QtWidgets.QToolButton()
            btn.setText(ev.label)
            btn.setToolTip(f"Jump to event {ev.label} @ {ev.time:.3f}s")
            btn.clicked.connect(lambda _c=False, t=ev.time: self.jump_to_time(t))
            self._event_row.addWidget(btn)
            self._event_buttons.append(btn)

    def _clear_event_buttons(self) -> None:
        for btn in self._event_buttons:
            self._event_row.removeWidget(btn)
            btn.setParent(None)
            btn.deleteLater()
        self._event_buttons = []

    def jump_to_time(self, time_s: float) -> None:
        """Jump to the frame nearest ``time_s`` seconds."""
        if self.model is None or self.model.point_time is None:
            return
        if not np.isfinite(time_s):
            raise ValueError(f"time_s must be finite, got {time_s!r}")
        frame = int(np.argmin(np.abs(self.model.point_time - float(time_s))))
        self.set_frame(frame)

    # -------------------------------------------------------- View presets

    def set_view_preset(self, name: str) -> None:
        """Snap the 3D axes to a named view preset."""
        if name not in _VIEW_PRESETS:
            raise ValueError(
                f"unknown view preset {name!r}, expected one of {sorted(_VIEW_PRESETS)}"
            )
        if self._ax is None:
            return
        elev, azim = _VIEW_PRESETS[name]
        self._ax.view_init(elev=elev, azim=azim)
        self.canvas_3d.draw_idle()

    # ---------------------------------------------------- Skeleton state

    def _on_skeleton_toggled(self, _on: bool) -> None:
        if self._skeleton_collection is not None:
            self._skeleton_collection.set_visible(self.check_skeleton.isChecked())
            self._update_skeleton_segments()
            self.canvas_3d.draw_idle()

    def _on_labels_toggled(self, _on: bool) -> None:
        for txt in self._label_texts:
            txt.set_visible(self.check_labels.isChecked())
        self.canvas_3d.draw_idle()

    @property
    def skeleton_segment_count(self) -> int:
        """Number of skeleton segments currently rendered."""
        return len(self._skeleton_segments)

    # --------------------------------------------------- Selection plumbing

    def _on_selection_changed(self) -> None:
        self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        """Tear down and recreate artists when the selection changes."""
        if self.model is None:
            return
        selected = [item.text() for item in self.list_markers_3d.selectedItems()]
        self._selected_names = selected

        self._teardown_artists()
        if not selected:
            self.canvas_3d.fig.clear()
            self.canvas_3d.draw_idle()
            self.label_frame_info.setText("Frame: - / Time: -")
            return

        # Cache positions: shape (M, N, 3); pad missing with NaN.
        if self._n_frames <= 0:
            return
        positions = np.full((len(selected), self._n_frames, 3), np.nan, dtype=float)
        for i, name in enumerate(selected):
            m = self.model.markers.get(name)
            if m is None or m.position.size == 0:
                continue
            n = min(m.position.shape[0], self._n_frames)
            positions[i, :n, :] = m.position[:n, :]
        self._selected_positions = positions

        # Build colors from a perceptually-distinct map.
        cmap = plt.get_cmap("tab20" if len(selected) <= 20 else "Spectral")
        denom = max(1, len(selected) - 1)
        self._marker_colors = [
            mcolors.to_rgba(cmap(i / denom)) for i in range(len(selected))
        ]

        self.canvas_3d.fig.clear()
        ax = self.canvas_3d.add_subplot(111, projection="3d")
        self._ax = ax

        self._trail_lines = []
        for i, name in enumerate(selected):
            pos = positions[i]
            (ln,) = ax.plot(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                alpha=0.35,
                color=self._marker_colors[i],
                linewidth=1.0,
                label=name,
            )
            self._trail_lines.append(ln)

        # Current-frame point — owned by MatplotlibMarkerRenderer so the
        # user can re-style it at runtime via the Style… button.
        self._point_artist = ax.scatter(
            [],
            [],
            [],
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
        )
        self._build_marker_style_renderer(ax, positions)

        # Labels (hidden by default).
        self._label_texts = []
        for name in selected:
            txt = ax.text(0.0, 0.0, 0.0, name, fontsize=8, color="black")
            txt.set_visible(self.check_labels.isChecked())
            self._label_texts.append(txt)

        # Skeleton overlay.
        segs = default_body_segments(selected)
        self._skeleton_segments = tuple((s.a, s.b) for s in segs)
        # Seed with a degenerate origin segment because ``add_collection3d``
        # requires non-empty data to compute initial autoscale; we clear it
        # immediately afterwards.
        self._skeleton_collection = Line3DCollection(
            [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]],
            colors="black",
            linewidths=2.0,
            alpha=0.85,
        )
        ax.add_collection3d(self._skeleton_collection)
        self._skeleton_collection.set_segments([])
        self._skeleton_collection.set_visible(
            self.check_skeleton.isChecked() and bool(self._skeleton_segments)
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Marker Trajectories")

        # Equal aspect.
        finite = positions[np.isfinite(positions).all(axis=2)]
        if finite.size > 0:
            mn = finite.min(axis=0)
            mx = finite.max(axis=0)
            max_range = float(np.max(mx - mn))
            if max_range > 0.0:
                mid = 0.5 * (mx + mn)
                half = max_range / 2.0
                ax.set_xlim(mid[0] - half, mid[0] + half)
                ax.set_ylim(mid[1] - half, mid[1] + half)
                ax.set_zlim(mid[2] - half, mid[2] + half)

        if len(selected) <= 12:
            ax.legend(loc="upper right", fontsize=8)

        self._rebuild_user_segment_artists()

        self.canvas_3d.fig.tight_layout()
        self._render_current_frame()

    def _teardown_artists(self) -> None:
        self._trail_lines = []
        self._point_artist = None
        self._label_texts = []
        self._skeleton_collection = None
        self._skeleton_segments = ()
        if self._renderer is not None:
            self._renderer.clear()
        self._renderer = None
        self._render_entries = []
        # plot_style renderer is recreated per scene rebuild — drop the
        # handle so update_style/update_frame don't mis-reference the
        # previous axes.
        self._marker_style_renderer = None
        self._marker_style_handle = None
        self._ax = None

    # ---------------------------------------------------- plot_style hooks

    def _build_marker_style_renderer(self, ax: Any, positions: np.ndarray) -> None:
        """Register the current-frame scatter with a MarkerRenderer."""
        if positions.size == 0:
            return
        # Use frame 0 as a starter; subsequent frames update the artist
        # in place via update_frame.
        frame0 = positions[:, 0, :]
        finite = frame0[np.isfinite(frame0).all(axis=1)]
        if finite.size == 0:
            return
        renderer = MatplotlibMarkerRenderer(ax)
        group = (
            self.combo_marker_group.currentText()
            if hasattr(self, "combo_marker_group")
            else "default"
        )
        style = self._persistence.get(group) or default_style_for(group)
        try:
            handle = renderer.add_markers(
                np.asarray(finite, dtype=float), style, "current_frame"
            )
        except (TypeError, ValueError) as exc:
            _LOGGER.warning("could not register current-frame scatter: %s", exc)
            return
        self._marker_style_renderer = renderer
        self._marker_style_handle = handle

    def apply_marker_group_style(self, group: str, style: MarkerStyle) -> None:
        """Apply ``style`` to ``group`` and persist (debounced)."""
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        self._persistence.set(group, style)
        active_group = self.combo_marker_group.currentText()
        if (
            active_group == group
            and self._marker_style_renderer is not None
            and self._marker_style_handle is not None
        ):
            try:
                self._marker_style_renderer.update_style(
                    self._marker_style_handle, style
                )
            except (KeyError, TypeError, ValueError) as exc:
                _LOGGER.warning("update_style failed: %s", exc)
            self.canvas_3d.draw_idle()
        self._persistence.request_save()

    def _on_marker_group_style_clicked(self) -> None:
        group = self.combo_marker_group.currentText()
        current = self._persistence.get(group) or default_style_for(group)
        new_style = _open_style_dialog(self, current, f"Style — {group}")
        if new_style is None:
            return
        self.apply_marker_group_style(group, new_style)

    # ---------------------------------------------- color-by-channel mode

    def install_data_channel_editor(self, channels: tuple[DataChannel, ...]) -> None:
        """Install a :class:`DataChannelEditor` driving DataDrivenColor.

        Replaces any existing editor. Pass an empty tuple to remove.
        """
        # Lazy import — widgets package depends on PyQt6.
        from src.shared.python.plot_style.widgets.data_channel_editor import (
            DataChannelEditor,
        )

        # Drop any previous editor.
        if self._color_channel_editor is not None:
            self._color_channel_layout.removeWidget(self._color_channel_editor)
            self._color_channel_editor.setParent(None)
            self._color_channel_editor.deleteLater()
            self._color_channel_editor = None

        if not channels:
            self._color_channel = None
            self._color_range = None
            return

        editor = DataChannelEditor(channels=list(channels))
        editor.channelChanged.connect(self._on_color_channel_changed)
        editor.rangeChanged.connect(self._on_color_range_changed)
        self._color_channel_layout.addWidget(editor)
        self._color_channel_editor = editor
        # Seed cached state with the editor's initial picks.
        self._color_channel = editor.value()
        self._color_range = editor.range_value()
        self._apply_color_channel_to_active_group()

    def _on_color_channel_changed(self, channel: DataChannel) -> None:
        self._color_channel = channel
        self._apply_color_channel_to_active_group()

    def _on_color_range_changed(self, vmin: float, vmax: float) -> None:
        self._color_range = (float(vmin), float(vmax))
        self._apply_color_channel_to_active_group()

    def _apply_color_channel_to_active_group(self) -> None:
        """Re-resolve the active group's style with current data-driven color."""
        if self._color_channel is None:
            return
        group = self.combo_marker_group.currentText()
        base = self._persistence.get(group) or default_style_for(group)
        vmin, vmax = (None, None)
        if self._color_range is not None:
            vmin, vmax = self._color_range
        fill = DataDrivenColor(
            channel=self._color_channel,
            colormap=self._color_colormap,
            vmin=vmin,
            vmax=vmax,
        )
        try:
            new_style = MarkerStyle(
                shape=base.shape,
                size_px=base.size_px,
                edge_color=base.edge_color,
                edge_width=base.edge_width,
                fill_color=fill,
                opacity=base.opacity,
            )
        except (TypeError, ValueError) as exc:
            _LOGGER.warning("could not apply data-driven color: %s", exc)
            return
        # Don't persist DataDrivenColor — it's session-scoped.
        if (
            self._marker_style_renderer is not None
            and self._marker_style_handle is not None
        ):
            try:
                self._marker_style_renderer.update_style(
                    self._marker_style_handle, new_style
                )
            except (KeyError, TypeError, ValueError) as exc:
                _LOGGER.warning("update_style failed: %s", exc)
            self.canvas_3d.draw_idle()
        # Track that the active style now uses a data-driven color so
        # tests can introspect.
        # (We could also push a recompute through DataDrivenColor.resolve
        # at each frame, but update_frame on a built-in shape only moves
        # offsets — colors stay attached to the artist.)

    @property
    def has_data_channel_editor(self) -> bool:
        """Whether an active editor is wired into the panel."""
        return self._color_channel_editor is not None

    # Convenience accessor for tests.
    @property
    def active_color_uses_data_driven(self) -> bool:
        """Whether the active scatter currently uses a DataDrivenColor."""
        if self._marker_style_renderer is None or self._marker_style_handle is None:
            return False
        # Reach into the renderer's internal handle table — this is a
        # pragmatic test-hook, not a public API. The renderer's record
        # type is stable.
        record = self._marker_style_renderer._handles.get(  # type: ignore[attr-defined]
            self._marker_style_handle
        )
        if record is None:
            return False
        return isinstance(record.style.fill_color, DataDrivenColor)

    @staticmethod
    def _safe_static_color() -> StaticColor:
        return StaticColor("#1f77b4")

    # ---------------------------------------------------- User segments API

    def set_user_segments(
        self, segments: tuple[SegmentSpec | SegmentVizSpec, ...]
    ) -> None:
        """Receive a new user-defined segment set from the Segments tab.

        Accepts both the legacy v1 :class:`SegmentSpec` tuple and the v2
        :class:`SegmentVizSpec` tuple. v1 entries are converted to v2 in
        place; the underlying renderer always operates on v2.
        """
        if segments is None:
            raise ValueError("segments must be provided (use () for an empty set)")
        viz_specs: list[SegmentVizSpec] = []
        for spec in segments:
            if isinstance(spec, SegmentVizSpec):
                viz_specs.append(spec)
            elif isinstance(spec, SegmentSpec):
                viz_specs.append(spec_v1_to_v2(spec))
            else:
                raise TypeError(
                    "segments entries must be SegmentSpec or SegmentVizSpec; "
                    f"got {type(spec).__name__}"
                )
        self._user_viz_segments = tuple(viz_specs)
        if self._ax is not None:
            self._rebuild_user_segment_artists()
            self._render_current_frame()

    @property
    def user_cylinder_count(self) -> int:
        """Number of mesh-bearing user-segment artists currently allocated.

        Includes cylinders, ellipsoids, capsules, meshes, and library
        shapes. Excludes line shapes (which use ``Line3DCollection``).
        """
        return sum(
            1
            for handle, kind in self._render_entries
            if handle is not None and kind != "line"
        )

    @property
    def user_line_segment_count(self) -> int:
        """Number of line-shape user segments currently rendered."""
        return sum(
            1
            for handle, kind in self._render_entries
            if handle is not None and kind == "line"
        )

    def _resolve_library(self) -> ShapeLibrary | None:
        if self._shape_library is None:
            try:
                self._shape_library = ShapeLibrary.default()
            except (FileNotFoundError, ValueError) as exc:
                _LOGGER.warning("default shape library unavailable: %s", exc)
                return None
        return self._shape_library

    def _markers_xyz(self, names: tuple[str, ...]) -> dict[str, np.ndarray] | None:
        """Build a ``{name: (T, 3)}`` mapping for the segment's markers.

        Returns ``None`` when any required marker is missing or empty so
        the caller can skip the segment cleanly.
        """
        if self.model is None or self._n_frames <= 0:
            return None
        out: dict[str, np.ndarray] = {}
        for name in names:
            md = self.model.markers.get(name)
            if md is None or md.position.size == 0:
                return None
            arr = np.asarray(md.position, dtype=float)
            if arr.shape[0] < self._n_frames:
                padded = np.full((self._n_frames, 3), np.nan, dtype=float)
                padded[: arr.shape[0]] = arr
                arr = padded
            elif arr.shape[0] > self._n_frames:
                arr = arr[: self._n_frames]
            out[name] = arr
        return out

    def _theme_for_spec(self, spec: SegmentVizSpec) -> ShapeTheme:
        """Prefer the spec's theme; fall back to anatomical-group color."""
        rgba = _color_for_group(spec.theme.group)
        # If the theme color is the v2 default (#1f77b4) and we have a
        # group-specific color, override it for visual continuity with the
        # legacy viewer.
        if spec.theme.color == "#1f77b4" and spec.theme.group in _GROUP_COLORS:
            return ShapeTheme(
                color=_rgba_to_hex(rgba),
                opacity=spec.theme.opacity,
                edge_color=_rgba_to_hex(rgba),
                edge_width=spec.theme.edge_width,
                flat_shaded=spec.theme.flat_shaded,
                group=spec.theme.group,
            )
        return spec.theme

    def _rebuild_user_segment_artists(self) -> None:
        ax = self._ax
        if ax is None:
            return
        # Drop the previous renderer (if any) and start fresh.
        if self._renderer is not None:
            self._renderer.clear()
        self._renderer = MatplotlibRenderer(ax)
        self._render_entries = []

        if not self._user_viz_segments or self._n_frames <= 0:
            return

        library = self._resolve_library()

        for spec in self._user_viz_segments:
            shape = _build_shape_from_spec(spec, library=library)
            if shape is None:
                self._render_entries.append((None, spec.shape_kind))
                continue
            markers_xyz = self._markers_xyz(spec.binding.marker_names)
            if markers_xyz is None:
                self._render_entries.append((None, spec.shape_kind))
                continue
            try:
                fitter = _fitter_for_kind(spec.fitter_kind)
            except ValueError:
                self._render_entries.append((None, spec.shape_kind))
                continue
            try:
                # Library-shape mesh has its own shape_id; rebind the binding
                # to the shape for the fitter to find a rest-length.
                effective_binding = spec.binding
                if not effective_binding.rest_dimensions and shape.rest_dimensions:
                    # Fitter expects a rest length on the binding for
                    # between_two; sourcing it from the shape's first axis
                    # is the canonical pattern.
                    effective_binding = type(spec.binding)(
                        kind=spec.binding.kind,
                        marker_names=spec.binding.marker_names,
                        rest_dimensions=(float(shape.rest_dimensions[0]),),
                        rest_orientation_quat=spec.binding.rest_orientation_quat,
                    )
                fitted = fitter.fit(shape, effective_binding, markers_xyz)
            except (KeyError, TypeError, ValueError) as exc:
                _LOGGER.warning(
                    "fit failed for segment %s: %s",
                    spec.binding.marker_names,
                    exc,
                )
                self._render_entries.append((None, spec.shape_kind))
                continue
            theme = self._theme_for_spec(spec)
            try:
                handle = self._renderer.add_shape(shape, fitted, theme)
            except (TypeError, ValueError) as exc:
                _LOGGER.warning(
                    "renderer.add_shape failed for %s: %s", spec.shape_kind, exc
                )
                self._render_entries.append((None, spec.shape_kind))
                continue
            if not spec.visible:
                self._renderer.set_visible(handle, False)
            self._render_entries.append((handle, spec.shape_kind))

    def _update_user_segment_artists(self) -> None:
        if self._renderer is None or not self._user_viz_segments or self._n_frames <= 0:
            return
        frame = int(self.slider_frame.value())
        if not 0 <= frame < self._n_frames:
            return
        for entry, spec in zip(
            self._render_entries, self._user_viz_segments, strict=False
        ):
            handle, _kind = entry
            if handle is None:
                continue
            try:
                self._renderer.set_visible(handle, bool(spec.visible))
                if spec.visible:
                    self._renderer.update_frame(handle, frame)
            except (KeyError, IndexError, TypeError) as exc:
                _LOGGER.warning("renderer.update_frame failed: %s", exc)

    # -------------------------------------------------- Per-frame render

    def _render_current_frame(self) -> None:
        if self.model is None or self._ax is None:
            if self.model is not None:
                # Update label even when nothing is selected.
                self._update_frame_label()
            return
        frame = self.slider_frame.value()
        self._update_frame_label()
        if self._selected_positions.size == 0:
            return

        pts = self._selected_positions[:, frame, :]  # (M,3)
        finite_mask = np.isfinite(pts).all(axis=1)

        # Update current-frame scatter (legacy path retained for layout
        # bounds — the visible glyphs are owned by the plot_style
        # renderer below).
        if self._point_artist is not None:
            valid = pts[finite_mask]
            if valid.size > 0:
                self._point_artist._offsets3d = (
                    valid[:, 0],
                    valid[:, 1],
                    valid[:, 2],
                )
            else:
                self._point_artist._offsets3d = ([], [], [])

        # Push the same offsets through the plot_style renderer so its
        # scatter follows the slider too.
        if (
            self._marker_style_renderer is not None
            and self._marker_style_handle is not None
        ):
            valid = pts[finite_mask]
            if valid.size > 0:
                # The renderer's update_frame indexes into the (T, M, D)
                # array we'd have given it. Since we registered with a
                # single static frame, we instead poke the artist offsets
                # directly to keep the path simple.
                record = self._marker_style_renderer._handles.get(  # type: ignore[attr-defined]
                    self._marker_style_handle
                )
                if record is not None and record.artists:
                    art = record.artists[0]
                    if hasattr(art, "_offsets3d"):
                        art._offsets3d = (
                            valid[:, 0],
                            valid[:, 1],
                            valid[:, 2],
                        )

        # Update labels at the current point.
        for i, txt in enumerate(self._label_texts):
            if finite_mask[i]:
                p = pts[i]
                txt.set_position((float(p[0]), float(p[1])))
                # set_3d_properties for the z-coord on text3d.
                if hasattr(txt, "set_3d_properties"):
                    txt.set_3d_properties(float(p[2]))

        # Update skeleton segments.
        self._update_skeleton_segments()

        # Update user-defined segments via the body_part_viz renderer.
        self._update_user_segment_artists()

        self.canvas_3d.draw_idle()

    def _update_skeleton_segments(self) -> None:
        if (
            self._skeleton_collection is None
            or not self._skeleton_segments
            or not self._selected_names
        ):
            return
        if not self.check_skeleton.isChecked():
            self._skeleton_collection.set_segments([])
            return
        frame = self.slider_frame.value()
        idx = {n: i for i, n in enumerate(self._selected_names)}
        segs: list[list[tuple[float, float, float]]] = []
        for a, b in self._skeleton_segments:
            ia = idx.get(a)
            ib = idx.get(b)
            if ia is None or ib is None:
                continue
            pa = self._selected_positions[ia, frame, :]
            pb = self._selected_positions[ib, frame, :]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            segs.append(
                [
                    (float(pa[0]), float(pa[1]), float(pa[2])),
                    (float(pb[0]), float(pb[1]), float(pb[2])),
                ]
            )
        self._skeleton_collection.set_segments(segs)

    def _update_frame_label(self) -> None:
        frame = self.slider_frame.value()
        if (
            self.model is not None
            and self.model.point_time is not None
            and 0 <= frame < len(self.model.point_time)
        ):
            t = float(self.model.point_time[frame])
            time_str = f"{t:.4f} s"
        else:
            time_str = "-"
        self.label_frame_info.setText(
            f"Frame: {frame} / {max(0, self._n_frames - 1)}  Time: {time_str}"
        )

    # Backwards-compatible name used by existing tests / code paths.
    def update_view(self) -> None:
        """Compatibility shim: rebuild the scene from scratch."""
        self._rebuild_scene()

    # ----------------------------------------------------------- CSV

    def export_selected_markers_csv(self, path: str) -> None:
        """Write the currently selected markers' trajectories to CSV.

        Columns: frame, time_s, <marker>_x, <marker>_y, <marker>_z, …
        """
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        if self.model is None:
            raise ValueError("no model loaded")
        names = self._selected_names or self.model.marker_names()
        if not names:
            raise ValueError("no markers selected to export")

        header = ["frame", "time_s"]
        for name in names:
            header += [f"{name}_x", f"{name}_y", f"{name}_z"]

        n = self._n_frames
        time_arr = (
            self.model.point_time
            if self.model.point_time is not None
            else np.full(n, np.nan)
        )

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for fr in range(n):
                row: list[Any] = [fr, float(time_arr[fr]) if fr < len(time_arr) else ""]
                for name in names:
                    m = self.model.markers.get(name)
                    if m is None or m.position.size == 0 or fr >= m.position.shape[0]:
                        row += ["", "", ""]
                    else:
                        x, y, z = m.position[fr]
                        row += [float(x), float(y), float(z)]
                writer.writerow(row)

    # -------------------------------------------------- Show/Hide events

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802 (Qt)
        """Pause playback when the tab is hidden."""
        self.pause()
        super().hideEvent(event)
