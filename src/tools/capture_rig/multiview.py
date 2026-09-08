"""Shared multiview plumbing for the live preview and playback panes.

The live preview (#9813) and the playback pane (#9814) both show *one*
composited canvas built by :func:`layout_model.compose` from a
:class:`LayoutSpec`. Everything they need in common lives here, so there is
exactly one compositor call site per pane and no second tiling
implementation:

* :func:`theme_palette` turns the active theme into the compositor's
  :class:`~.layout_model.Palette` (the compositor is pure numpy and knows no
  Qt, so the colours are handed to it);
* :class:`CanvasLabel` is the single label a pane draws its canvas on;
* :class:`LayoutChooser` is the picker: built-in presets plus the layouts
  saved in a :class:`~.layout_presets.LayoutStore`, and an **Edit layout...**
  button that opens :class:`~.layout_editor.LayoutEditor` on the current
  spec with live thumbnails;
* :func:`frame_offsets` reads the session's strobe alignment so playback can
  put frame *k* of every view on the same canvas.

Nothing here opens a camera or a file: frames arrive as a mapping keyed by
:attr:`~.layout_model.SourceRef.key`, exactly what :func:`compose` expects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.palette import get_current_colors

from . import styling
from .layout_editor import LayoutEditor
from .layout_editor_canvas import FrameProvider
from .layout_model import (
    DEFAULT_CANVAS,
    LayoutSpec,
    Palette,
    SourceRef,
    compose,
    preset,
)
from .layout_presets import BUILTIN, LayoutStore, LayoutStoreError, PresetEntry

CANVAS_MIN = (192, 120)
EDITED = "(edited)"
EDIT_BUTTON_TEXT = "Edit…"
MIN_COMBO_CHARS = 6
#: Which compositor colour each theme palette key feeds.
PALETTE_KEYS: dict[str, str] = {
    "background": "bg",
    "placeholder": "group_bg",
    "placeholder_text": "text_secondary",
    "label_text": "text",
    "label_box": "title_bg",
}
HELP: dict[str, str] = {
    "layout": "Which multiview layout this pane is drawn through: the "
    "built-in presets and every layout saved in the user and session "
    "scopes. Views that are not in the layout are still captured.",
    "edit": "Open the layout editor on the current layout, with live "
    "thumbnails of what each tile shows. Edits apply here as you make them.",
}


def _hex(value: object) -> str | None:
    """``value`` when it is a ``#rrggbb`` string, else ``None``."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    ok = len(text) == 7 and text.startswith("#")
    return text if ok and all(c in "0123456789abcdefABCDEF" for c in text[1:]) else None


def theme_palette(colors: Mapping[str, str] | None = None) -> Palette:
    """The compositor palette derived from the active theme.

    Postcondition: never raises for an exotic palette value; a key that is
    not a ``#rrggbb`` string keeps the compositor's neutral default.
    """
    active = colors if colors is not None else get_current_colors()
    hexes = {
        field: text
        for field, key in PALETTE_KEYS.items()
        if (text := _hex(active.get(key))) is not None
    }
    return Palette.from_hex(**hexes)


def bgr_to_pixmap(frame_bgr: npt.NDArray[np.uint8]) -> QPixmap:
    """A pixmap of a BGR frame at its own size (the canvas is already sized).

    Precondition: ``frame_bgr`` is an ``HxWx3`` uint8 array.
    """
    require(frame_bgr.ndim == 3, "frame must be HxWx3", frame_bgr.shape)
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    height, width = rgb.shape[:2]
    image = QImage(rgb.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def compose_pixmap(
    frames: Mapping[str, npt.NDArray[Any]],
    spec: LayoutSpec,
    size: tuple[int, int],
    palette: Palette | None = None,
) -> tuple[npt.NDArray[np.uint8], QPixmap]:
    """``(canvas, pixmap)`` for ``frames`` through ``spec`` at ``size``.

    ``size`` of zero or less in either axis (a pane that has not been laid
    out yet) falls back to the spec's own canvas size. Postcondition: the
    pixmap is exactly the canvas, so a PNG export and what is on screen are
    the same pixels.
    """
    width, height = size
    if width <= 0 or height <= 0:
        width, height = spec.canvas or DEFAULT_CANVAS
    canvas = compose(
        frames,
        spec,
        size=(int(width), int(height)),
        palette=palette if palette is not None else theme_palette(),
    )
    return canvas, bgr_to_pixmap(canvas)


def write_png(canvas: npt.NDArray[np.uint8], path: Path) -> Path:
    """Write ``canvas`` (BGR) as a PNG, returning the path.

    Encodes in memory and writes bytes, so non-ASCII paths work on Windows
    where ``cv2.imwrite`` does not. Precondition: the canvas is a non-empty
    ``HxWx3`` uint8 array. Postcondition: ``path`` exists.
    """
    import cv2

    require(canvas.ndim == 3 and canvas.size > 0, "canvas must be HxWx3", canvas.shape)
    ok, buffer = cv2.imencode(".png", canvas)
    require(bool(ok), "could not encode the canvas as PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())
    return path


def live_sources(views: Sequence[str]) -> tuple[SourceRef, ...]:
    """One live source per view, in the given order."""
    return tuple(SourceRef(kind="live", view=view) for view in views)


def frame_offsets(
    timing: Mapping[str, Any], fps: Mapping[str, float]
) -> dict[str, int]:
    """Whole-frame shifts per view from a manifest ``timing`` block.

    A view whose arrival clock is ``offset_ns`` *later* than the reference
    view is that many nanoseconds ahead in its own file, so frame ``k`` of
    the reference lines up with frame ``k + round(offset * fps)`` of that
    view. Views without usable timing get ``0``. Postcondition: a key for
    every view in ``fps``; no key ever missing, so callers never branch.
    """
    from src.motion_capture.rig.alignment import NS_PER_S, view_timing

    out: dict[str, int] = {}
    for view, rate in fps.items():
        entry = view_timing(timing, view) if timing else None
        offset = None if entry is None else entry.get("offset_ns")
        if offset is None or rate <= 0:
            out[view] = 0
            continue
        out[view] = int(round(float(offset) / NS_PER_S * rate))
    return out


class CanvasLabel(QLabel):
    """The one label a pane draws its composited canvas on.

    Size policy ``Ignored`` in both axes: the pane decides how big the canvas
    is, never the pixmap, so a large frame cannot push the pane open.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(*CANVAS_MIN)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet(styling.tile_style())

    def restyle(self) -> None:
        self.setStyleSheet(styling.tile_style())

    def canvas_size(self) -> tuple[int, int]:
        return self.width(), self.height()


@dataclass(frozen=True)
class ChooserOptions:
    """Optional wiring for :class:`LayoutChooser` (kept off the parameter list)."""

    store: LayoutStore | None = None
    default: str = "side_by_side"
    editor_title: str = "Edit layout"


class LayoutChooser(QWidget):
    """Pick the multiview layout a pane is drawn through, or edit it.

    ``layout_changed(LayoutSpec)`` fires whenever the effective layout
    changes: a different entry chosen, the sources refilled, or an edit made
    in the editor this widget opened. :meth:`layout_name` and
    :meth:`set_layout_name` are what the tile persists.
    """

    layout_changed = pyqtSignal(object)  # LayoutSpec

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        options: ChooserOptions | None = None,
        sources: Sequence[SourceRef] = (),
    ) -> None:
        super().__init__(parent)
        self._options = options if options is not None else ChooserOptions()
        self._store = self._options.store or LayoutStore()
        self._sources: tuple[SourceRef, ...] = tuple(sources)
        self._provider: FrameProvider | None = None
        self._editor: LayoutEditor | None = None
        self._spec = preset(self._options.default, self._sources)
        self.combo = QComboBox()
        self.combo.setToolTip(HELP["layout"])
        self.combo.setMinimumContentsLength(MIN_COMBO_CHARS)
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.edit_button = QPushButton(EDIT_BUTTON_TEXT)
        self.edit_button.setToolTip(HELP["edit"])
        self.edit_button.clicked.connect(lambda: self.open_editor())
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(LayoutMetrics.SPACING_SM)
        # No caption label: the pane is narrow and both controls say what they
        # are in their tooltips (the pane must stay shrinkable, see #9813).
        row.addWidget(self.combo, 1)
        row.addWidget(self.edit_button)
        self.refresh()
        self.combo.currentIndexChanged.connect(self._on_combo)

    # -- queries -----------------------------------------------------------
    def spec(self) -> LayoutSpec:
        """The layout this pane draws through (never ``None``)."""
        return self._spec

    def layout_name(self) -> str:
        """The chosen layout's name; :data:`EDITED` for an unsaved edit.

        The bare name, not the ``name (scope)`` shown in the combo, so it
        round-trips through :meth:`set_layout_name` and the tile's settings.
        """
        data = self.combo.currentData()
        if isinstance(data, tuple):
            return str(data[0])
        return self._options.default

    def sources(self) -> tuple[SourceRef, ...]:
        return self._sources

    def entries(self) -> list[PresetEntry]:
        """Every layout offered, built-ins first (unreadable files included)."""
        return list(self._store.list())

    # -- population --------------------------------------------------------
    def refresh(self, select: str | None = None) -> None:
        """Re-read the store; keep the current choice when it still exists."""
        wanted = select if select is not None else self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        for entry in self.entries():
            label = entry.name if entry.builtin else f"{entry.name} ({entry.scope})"
            self.combo.addItem(label, (entry.name, entry.scope))
            if entry.error:
                self.combo.setItemData(
                    self.combo.count() - 1,
                    f"{entry.name}: {entry.error}",
                    Qt.ItemDataRole.ToolTipRole,
                )
        self.combo.blockSignals(False)
        if not self._select_named(wanted) and not self._select_named(
            self._options.default
        ):
            self.combo.setCurrentIndex(0 if self.combo.count() else -1)

    def _select_named(self, name: str) -> bool:
        for index in range(self.combo.count()):
            data = self.combo.itemData(index)
            if isinstance(data, tuple) and data[0] == name:
                self.combo.setCurrentIndex(index)
                return True
        return False

    def set_layout_name(self, name: str) -> bool:
        """Choose the layout called ``name``; ``False`` when there is none.

        Postcondition: on success :meth:`layout_name` starts with ``name``
        and ``layout_changed`` has fired.
        """
        if not name or name == EDITED or not self._select_named(name):
            return False
        self._load_current()
        return True

    def set_spec(self, spec: LayoutSpec) -> None:
        """Draw through ``spec`` directly, as an unsaved edit.

        Precondition: ``spec`` is a :class:`LayoutSpec`. Postcondition: the
        combo shows :data:`EDITED` and ``layout_changed`` has fired once.
        """
        require(isinstance(spec, LayoutSpec), "spec must be a LayoutSpec", type(spec))
        self._on_edited(spec)
        if self._editor is not None:
            self._editor.set_spec(spec)  # set_spec never echoes layout_changed

    def set_sources(self, sources: Sequence[SourceRef]) -> None:
        """The sources a built-in preset is filled with (live views, say).

        A built-in choice is refilled straight away so a changed camera set
        shows up; a saved layout names its own views and is left alone.
        Precondition: every item is a :class:`SourceRef`.
        """
        refs = tuple(sources)
        require(
            all(isinstance(s, SourceRef) for s in refs), "sources must be SourceRefs"
        )
        if refs == self._sources:
            return
        self._sources = refs
        if self._editor is not None:
            self._editor.set_sources(refs)
        if self._current_scope() == BUILTIN:
            self._load_current()

    def set_frame_provider(self, provider: FrameProvider | None) -> None:
        """Where the editor's thumbnails come from (the pane's latest frames)."""
        self._provider = provider
        if self._editor is not None:
            self._editor.set_frame_provider(provider)

    # -- selection ---------------------------------------------------------
    def _current_scope(self) -> str:
        data = self.combo.currentData()
        return str(data[1]) if isinstance(data, tuple) else ""

    def _on_combo(self, _index: int) -> None:
        self._load_current()

    def _load_current(self) -> None:
        data = self.combo.currentData()
        if not isinstance(data, tuple):
            return
        name, scope = str(data[0]), str(data[1])
        if scope == BUILTIN:
            self._apply(preset(name, self._sources))
            return
        try:
            self._apply(self._store.load(name, scope))
        except LayoutStoreError:
            self._apply(preset(self._options.default, self._sources))

    def _apply(self, spec: LayoutSpec) -> None:
        self._spec = spec
        if self._editor is not None:
            self._editor.set_spec(spec)
        self.layout_changed.emit(spec)

    # -- editor ------------------------------------------------------------
    @property
    def editor(self) -> LayoutEditor | None:
        """The open editor window, or ``None``."""
        return self._editor

    def open_editor(self) -> LayoutEditor:
        """Show the layout editor on the current spec (re-using an open one).

        The editor is a separate window, not a modal dialog, so the operator
        keeps watching the live canvas while rearranging it. Postcondition:
        every edit lands on this chooser through ``layout_changed``.
        """
        editor = self._editor
        if editor is None:
            editor = LayoutEditor(
                self._spec,
                sources=self._sources,
                frame_provider=self._provider,
                store=self._store,
                parent=self,
            )
            editor.setWindowFlags(Qt.WindowType.Window)
            editor.setWindowTitle(self._options.editor_title)
            editor.layout_changed.connect(self._on_edited)
            self._editor = editor
        else:
            editor.set_spec(self._spec)
        editor.show()
        return editor

    def _on_edited(self, spec: object) -> None:
        if not isinstance(spec, LayoutSpec):
            return
        self._spec = spec
        self.combo.blockSignals(True)
        if self.combo.currentText() != EDITED:
            self.combo.insertItem(0, EDITED, (EDITED, ""))
        self.combo.setCurrentIndex(self._index_of(EDITED))
        self.combo.blockSignals(False)
        self.layout_changed.emit(spec)

    def _index_of(self, text: str) -> int:
        for index in range(self.combo.count()):
            if self.combo.itemText(index) == text:
                return index
        return -1
