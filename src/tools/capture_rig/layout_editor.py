"""Interactive multiview layout editor for the Capture Rig (#9812).

:class:`LayoutEditor` lets the operator arrange the preview screen: pick a
grid (up to 4x4) or a preset, drag tiles between cells, grow or shrink a
tile's span, choose what each tile shows and how (rotate, flip, crop, fit,
label), then save the arrangement through a :class:`LayoutStore`. Every edit
goes through :meth:`_commit`, which records it for undo (20 steps) and emits
``layout_changed`` exactly once, so the widget can be tested entirely through
its programmatic API (``move_tile``, ``rotate``, ``set_crop``, ...) and the
mouse handlers merely call the same methods.

The widget owns no camera or file: frames for the thumbnails come from an
injectable ``frame_provider`` and modal dialogs are :class:`EditorDialogs`
callbacks, so tests run offscreen without blocking. Colours come from the
theme package; none are literal.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics

from .layout_editor_canvas import FrameProvider, LayoutCanvas, NormRect, check_cell
from .layout_model import (
    FITS,
    MAX_GRID,
    PRESET_NAMES,
    ROTATIONS,
    Cell,
    Crop,
    LayoutSpec,
    SourceRef,
    Tile,
    preset,
)
from .layout_presets import (
    BUILTIN,
    SESSION,
    USER,
    LayoutStore,
    LayoutStoreError,
    validate_name,
)

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 20
EMPTY_SOURCE_TEXT = "(empty)"
NO_TILE = "select a tile on the preview first"
NO_STORE = "no layout store was given to this editor"

HELP: dict[str, str] = {
    "rows": "Number of grid rows (1-4). Tiles that no longer fit are dropped.",
    "cols": "Number of grid columns (1-4). Tiles that no longer fit are dropped.",
    "preset": "Built-in presets and the layouts saved in the user and session "
    "scopes. Pick one, then Load.",
    "load": "Replace the current layout with the chosen preset, filling the "
    "cells with this editor's sources in order.",
    "scope": "Where Save as... writes: the user configuration directory or "
    "the current session folder (so the layout travels with the take).",
    "save": "Save the current layout under a new name in the chosen scope.",
    "delete": "Delete the chosen saved layout from disk.",
    "undo": "Undo the last edit (up to 20 steps).",
    "redo": "Redo the edit you just undid.",
    "source": "What the selected tile shows: a live camera view, a recorded "
    "view, or a model overlay of a view (with variants).",
    "rotate_ccw": "Rotate the selected tile 90 degrees counter-clockwise.",
    "rotate_cw": "Rotate the selected tile 90 degrees clockwise.",
    "flip_h": "Mirror the selected tile left-right.",
    "flip_v": "Mirror the selected tile top-bottom.",
    "crop": "When on, drag a rectangle over the selected tile's thumbnail "
    "to show only that part of the frame (drags crop within the current crop).",
    "reset_crop": "Show the whole frame again in the selected tile.",
    "fit": "How the frame fills the cell: fit letterboxes, fill centre-crops, "
    "stretch distorts.",
    "label": "Draw the tile's caption (its label or source name) in the corner.",
    "span_row_plus": "Make the selected tile one row taller (it takes over the "
    "cells below).",
    "span_row_minus": "Make the selected tile one row shorter (freed cells "
    "become empty tiles).",
    "span_col_plus": "Make the selected tile one column wider (it takes over "
    "the cells to its right).",
    "span_col_minus": "Make the selected tile one column narrower (freed cells "
    "become empty tiles).",
}


def hint(help_text: str, enabled: bool, why: str) -> str:
    """Tooltip: ``help_text`` plus a ``Disabled:`` line when not ``enabled``.

    Mirrors :func:`workflow.action_hints` so every grey control says why.
    """
    if enabled:
        return help_text
    return f"{help_text}\n\nDisabled: {why}."


@dataclass(frozen=True)
class EditorDialogs:
    """The three modal interactions, injectable so tests never block.

    ``ask_name(default)`` returns the chosen name or ``None`` (cancelled);
    ``confirm(text)`` returns whether the operator agreed; ``notify(text)``
    shows a message.
    """

    ask_name: Callable[[str], str | None]
    confirm: Callable[[str], bool]
    notify: Callable[[str], None]


def qt_dialogs(parent: QWidget) -> EditorDialogs:
    """The default Qt dialogs, parented to ``parent``."""

    def ask_name(default: str) -> str | None:
        text, ok = QInputDialog.getText(
            parent, "Save layout as", "Layout name:", text=default
        )
        return text if ok else None

    def confirm(text: str) -> bool:
        answer = QMessageBox.question(parent, "Delete layout", text)
        return answer == QMessageBox.StandardButton.Yes

    def notify(text: str) -> None:
        QMessageBox.information(parent, "Layout", text)

    return EditorDialogs(ask_name=ask_name, confirm=confirm, notify=notify)


# ------------------------------------------------------------ pure helpers


def fill_empty(spec: LayoutSpec) -> LayoutSpec:
    """``spec`` with an empty tile on every cell no tile covers.

    Postcondition: ``result.tile_at(r, c)`` is a tile for every cell.
    """
    covered = {rc for tile in spec.tiles for rc in tile.cell.covered()}
    extra = tuple(
        Tile(source=SourceRef.empty(), cell=Cell(r, c))
        for r in range(spec.rows)
        for c in range(spec.cols)
        if (r, c) not in covered
    )
    return spec if not extra else replace(spec, tiles=(*spec.tiles, *extra))


def resized(
    spec: LayoutSpec, rows: int, cols: int
) -> tuple[LayoutSpec, tuple[Tile, ...]]:
    """``spec`` on a ``rows`` x ``cols`` grid plus the tiles that no longer fit.

    Precondition: 1 <= rows, cols <= 4. Tiles that still fit keep their cell;
    new cells get empty tiles.
    """
    require(1 <= rows <= MAX_GRID, "rows must be 1..4", rows)
    require(1 <= cols <= MAX_GRID, "cols must be 1..4", cols)
    kept = tuple(t for t in spec.tiles if t.cell.fits(rows, cols))
    dropped = tuple(t for t in spec.tiles if not t.cell.fits(rows, cols))
    return fill_empty(replace(spec, rows=rows, cols=cols, tiles=kept)), dropped


def sub_crop(current: Crop, rect: NormRect) -> Crop:
    """The crop ``rect`` (normalised to the shown picture) inside ``current``."""
    x, y, w, h = rect
    return Crop(
        x=current.x + x * current.w,
        y=current.y + y * current.h,
        w=max(1e-6, w * current.w),
        h=max(1e-6, h * current.h),
    )


def source_text(source: SourceRef) -> str:
    return EMPTY_SOURCE_TEXT if source.is_empty else source.key


# ----------------------------------------------------------------- widget


class LayoutEditor(QWidget):
    """Interactive editor of a :class:`LayoutSpec` (see the module docstring).

    ``layout_changed(LayoutSpec)`` fires once per edit, undo or redo; never
    for :meth:`set_spec`.
    """

    layout_changed = pyqtSignal(object)

    def __init__(
        self,
        spec: LayoutSpec | None = None,
        *,
        sources: Sequence[SourceRef] = (),
        frame_provider: FrameProvider | None = None,
        store: LayoutStore | None = None,
        dialogs: EditorDialogs | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = fill_empty(spec if spec is not None else preset("two_by_two"))
        self._sources: tuple[SourceRef, ...] = tuple(sources)
        self.store = store
        self.dialogs = dialogs if dialogs is not None else qt_dialogs(self)
        self._history: list[LayoutSpec] = []
        self._future: list[LayoutSpec] = []
        self._selected: Cell | None = None
        self._syncing = False
        self._controls: list[QWidget] = []
        self.canvas = LayoutCanvas(self._spec, self)
        self.canvas.set_frame_provider(frame_provider)
        self._build()
        self._wire()
        self._refresh_presets()
        self._sync_widgets()

    # -- construction ------------------------------------------------------

    def _add(self, widget: Any, key: str, layout: QHBoxLayout) -> Any:
        widget.setObjectName(key)
        widget.setToolTip(HELP[key])
        layout.addWidget(widget)
        self._controls.append(widget)
        return widget

    def _button(self, text: str, key: str, layout: QHBoxLayout) -> QPushButton:
        button: QPushButton = self._add(QPushButton(text), key, layout)
        return button

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(LayoutMetrics.SPACING_SM)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self._build_grid_row())
        root.addWidget(self.canvas, stretch=1)
        root.addLayout(self._build_tile_row())
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    def _build_grid_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(LayoutMetrics.SPACING_SM)
        row.addWidget(QLabel("Grid"))
        self.rows_spin = self._add(QSpinBox(), "rows", row)
        self.rows_spin.setRange(1, MAX_GRID)
        row.addWidget(QLabel("x"))
        self.cols_spin = self._add(QSpinBox(), "cols", row)
        self.cols_spin.setRange(1, MAX_GRID)
        row.addSpacing(LayoutMetrics.SPACING_MD)
        self.preset_combo = self._add(QComboBox(), "preset", row)
        self.load_button = self._button("Load", "load", row)
        self.scope_combo = self._add(QComboBox(), "scope", row)
        self.scope_combo.addItem("user", USER)
        self.scope_combo.addItem("session", SESSION)
        self.save_button = self._button("Save as...", "save", row)
        self.delete_button = self._button("Delete", "delete", row)
        row.addStretch(1)
        self.undo_button = self._button("Undo", "undo", row)
        self.redo_button = self._button("Redo", "redo", row)
        return row

    def _build_tile_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(LayoutMetrics.SPACING_SM)
        row.addWidget(QLabel("Tile"))
        self.source_combo = self._add(QComboBox(), "source", row)
        self.rotate_ccw_button = self._button("⟲", "rotate_ccw", row)
        self.rotate_cw_button = self._button("⟳", "rotate_cw", row)
        self.flip_h_button = self._button("Flip H", "flip_h", row)
        self.flip_v_button = self._button("Flip V", "flip_v", row)
        self.crop_button = self._button("Crop", "crop", row)
        self.crop_button.setCheckable(True)
        self.reset_crop_button = self._button("Reset crop", "reset_crop", row)
        self.fit_combo = self._add(QComboBox(), "fit", row)
        for fit in FITS:
            self.fit_combo.addItem(fit, fit)
        self.label_check = self._add(QCheckBox("Label"), "label", row)
        row.addSpacing(LayoutMetrics.SPACING_MD)
        row.addWidget(QLabel("Span"))
        self.span_row_plus = self._button("+row", "span_row_plus", row)
        self.span_row_minus = self._button("-row", "span_row_minus", row)
        self.span_col_plus = self._button("+col", "span_col_plus", row)
        self.span_col_minus = self._button("-col", "span_col_minus", row)
        row.addStretch(1)
        return row

    def _wire(self) -> None:
        self.rows_spin.valueChanged.connect(self._on_grid_spin)
        self.cols_spin.valueChanged.connect(self._on_grid_spin)
        self.preset_combo.currentIndexChanged.connect(self._update_controls)
        self.load_button.clicked.connect(self._load_clicked)
        self.save_button.clicked.connect(self.save_as)
        self.delete_button.clicked.connect(self._delete_clicked)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.source_combo.currentIndexChanged.connect(self._on_source_combo)
        self.rotate_ccw_button.clicked.connect(lambda: self._on_tile("rotate", -1))
        self.rotate_cw_button.clicked.connect(lambda: self._on_tile("rotate", 1))
        self.flip_h_button.clicked.connect(lambda: self._on_tile("flip", "h"))
        self.flip_v_button.clicked.connect(lambda: self._on_tile("flip", "v"))
        self.crop_button.toggled.connect(self._on_crop_toggled)
        self.reset_crop_button.clicked.connect(lambda: self._on_tile("reset_crop"))
        self.fit_combo.currentIndexChanged.connect(self._on_fit_combo)
        self.label_check.toggled.connect(self._on_label_toggled)
        self.span_row_plus.clicked.connect(lambda: self._on_span(1, 0))
        self.span_row_minus.clicked.connect(lambda: self._on_span(-1, 0))
        self.span_col_plus.clicked.connect(lambda: self._on_span(0, 1))
        self.span_col_minus.clicked.connect(lambda: self._on_span(0, -1))
        self.canvas.tile_pressed.connect(lambda r, c: self.select(Cell(r, c)))
        self.canvas.tile_dropped.connect(self.move_tile)
        self.canvas.crop_dragged.connect(self._on_crop_dragged)

    # -- public state ----------------------------------------------------

    def spec(self) -> LayoutSpec:
        return self._spec

    def set_spec(self, spec: LayoutSpec) -> None:
        """Show ``spec`` (empty tiles fill uncovered cells); clears the history.

        Not an edit: ``layout_changed`` is not emitted.
        """
        require(isinstance(spec, LayoutSpec), "spec must be a LayoutSpec", type(spec))
        self._history.clear()
        self._future.clear()
        self._set(fill_empty(spec))

    @property
    def sources(self) -> tuple[SourceRef, ...]:
        return self._sources

    def set_sources(self, sources: Sequence[SourceRef]) -> None:
        """The sources presets are filled with and the source combo offers."""
        self._sources = tuple(sources)
        require(
            all(isinstance(s, SourceRef) for s in self._sources),
            "sources must be SourceRefs",
        )
        self._sync_widgets()

    def set_frame_provider(self, provider: FrameProvider | None) -> None:
        self.canvas.set_frame_provider(provider)

    def refresh_thumbnails(self) -> None:
        """Recompose the preview from fresh frames (call on a timer for live)."""
        self.canvas.redraw()

    def status_text(self) -> str:
        return self.status_label.text()

    def controls(self) -> list[QWidget]:
        """Every operator control (for tooltip audits)."""
        return list(self._controls)

    # -- selection -------------------------------------------------------

    def selected_cell(self) -> Cell | None:
        return self._selected

    def select(self, cell: Cell | None) -> None:
        """Select the tile anchored at ``cell`` (``None`` clears)."""
        if cell is not None:
            check_cell(cell)
            tile = self._spec.tile_at(cell.row, cell.col)
            cell = None if tile is None else tile.cell
        self._selected = cell
        self.canvas.set_selected(cell)
        self._sync_widgets()

    def tile_index(self, cell: Cell) -> int:
        """Index in ``spec().tiles`` of the tile covering ``cell`` (-1 if none)."""
        tile = self._spec.tile_at(cell.row, cell.col)
        tiles = self._spec.tiles
        return -1 if tile is None else tiles.index(tile)

    def selected_index(self) -> int:
        return -1 if self._selected is None else self.tile_index(self._selected)

    # -- edits (each emits layout_changed once) ---------------------------

    def _tile(self, index: int) -> Tile:
        require(
            isinstance(index, int) and 0 <= index < len(self._spec.tiles),
            "tile index out of range",
            index,
        )
        return self._spec.tiles[index]

    def _replace_tile(self, index: int, **changes: Any) -> bool:
        tile = replace(self._tile(index), **changes)
        tiles = list(self._spec.tiles)
        tiles[index] = tile
        return self._try_commit(lambda: replace(self._spec, tiles=tuple(tiles)))

    def _try_commit(self, build: Callable[[], LayoutSpec]) -> bool:
        """Commit ``build()``; a ``ValueError`` becomes a status line, not a raise."""
        try:
            spec = build()
        except ValueError as exc:
            self._status(str(exc))
            logger.info("layout edit refused: %s", exc)
            return False
        self._commit(spec)
        return True

    def _commit(self, spec: LayoutSpec) -> None:
        """Record the current spec for undo, show ``spec`` and emit once.

        Postcondition: ``can_undo()``; ``can_redo()`` is false; at most
        :data:`HISTORY_LIMIT` undo steps are kept.
        """
        self._history.append(self._spec)
        del self._history[:-HISTORY_LIMIT]
        self._future.clear()
        self._set(spec)
        self.layout_changed.emit(spec)

    def move_tile(self, src: Cell, dst: Cell) -> bool:
        """Re-anchor the tile at ``src`` on ``dst`` (swapping with an occupant).

        Returns whether the move was legal; an illegal one is reported on the
        status line. Preconditions: both arguments are :class:`Cell`.
        """
        check_cell(src)
        check_cell(dst)
        if self._spec.tile_at(src.row, src.col) is None:
            self._status(f"no tile at row {src.row}, column {src.col}")
            return False
        anchor = self._spec.tile_at(src.row, src.col)
        assert anchor is not None
        moved = self._try_commit(lambda: self._spec.move_tile(anchor.cell, dst))
        if moved:
            self.select(Cell(dst.row, dst.col))
        return moved

    def rotate(self, index: int, direction: int) -> bool:
        """Turn tile ``index`` a quarter turn; ``direction`` +1 is clockwise."""
        require(direction in (1, -1), "direction must be +1 or -1", direction)
        tile = self._tile(index)
        position = ROTATIONS.index(tile.rotation)
        return self._replace_tile(
            index, rotation=ROTATIONS[(position + direction) % len(ROTATIONS)]
        )

    def flip(self, index: int, axis: str) -> bool:
        """Toggle the ``"h"`` (left-right) or ``"v"`` (top-bottom) mirror."""
        require(axis in ("h", "v"), "axis must be 'h' or 'v'", axis)
        tile = self._tile(index)
        if axis == "h":
            return self._replace_tile(index, flip_h=not tile.flip_h)
        return self._replace_tile(index, flip_v=not tile.flip_v)

    def set_crop(self, index: int, crop: Crop) -> bool:
        require(isinstance(crop, Crop), "crop must be a Crop", crop)
        return self._replace_tile(index, crop=crop)

    def reset_crop(self, index: int) -> bool:
        return self._replace_tile(index, crop=Crop())

    def set_fit(self, index: int, fit: str) -> bool:
        require(fit in FITS, f"fit must be one of {FITS}", fit)
        return self._replace_tile(index, fit=fit)

    def set_label_visible(self, index: int, visible: bool) -> bool:
        return self._replace_tile(index, show_label=bool(visible))

    def set_label(self, index: int, label: str | None) -> bool:
        require(label is None or isinstance(label, str), "label must be text", label)
        return self._replace_tile(index, label=label)

    def set_source(self, index: int, source: SourceRef) -> bool:
        require(isinstance(source, SourceRef), "source must be a SourceRef", source)
        return self._replace_tile(index, source=source)

    def change_span(self, index: int, drow: int = 0, dcol: int = 0) -> bool:
        """Grow (+) or shrink (-) tile ``index`` by whole cells.

        Growing takes over the covered tiles; shrinking leaves empty tiles.
        Returns False (with a status line) when the span would leave the
        grid or fall below one cell.
        """
        tile = self._tile(index)
        cell = tile.cell
        new_cell = (
            Cell(cell.row, cell.col, cell.rowspan + drow, cell.colspan + dcol)
            if cell.rowspan + drow >= 1 and cell.colspan + dcol >= 1
            else None
        )
        if new_cell is None or not new_cell.fits(self._spec.rows, self._spec.cols):
            self._status("that span would leave the grid or vanish")
            return False
        grown = replace(tile, cell=new_cell)
        if drow > 0 or dcol > 0:
            return self._try_commit(lambda: fill_empty(self._spec.with_tile(grown)))
        tiles = list(self._spec.tiles)
        tiles[index] = grown
        return self._try_commit(
            lambda: fill_empty(replace(self._spec, tiles=tuple(tiles)))
        )

    def set_grid(self, rows: int, cols: int) -> tuple[Tile, ...]:
        """Resize the grid, keeping tiles that still fit; returns the dropped.

        Precondition: 1 <= rows, cols <= 4. Postcondition: the spec is
        ``rows`` x ``cols`` with every cell covered; the status line counts
        the dropped tiles.
        """
        spec, dropped = resized(self._spec, rows, cols)
        if spec == self._spec:
            return ()
        self._commit(spec)
        if dropped:
            self._status(f"{len(dropped)} tile(s) did not fit and were dropped")
        return dropped

    def apply_preset(self, name: str) -> None:
        """Load built-in preset ``name`` filled with this editor's sources."""
        require(name in PRESET_NAMES, f"preset must be one of {PRESET_NAMES}", name)
        spec = replace(preset(name, self._sources), canvas=self._spec.canvas)
        self._commit(spec)

    # -- undo / redo ------------------------------------------------------

    def can_undo(self) -> bool:
        return bool(self._history)

    def can_redo(self) -> bool:
        return bool(self._future)

    def undo(self) -> None:
        """Go back one edit (no-op without history); emits ``layout_changed``."""
        if not self._history:
            return
        self._future.append(self._spec)
        self._set(self._history.pop())
        self.layout_changed.emit(self._spec)

    def redo(self) -> None:
        """Re-apply the last undone edit (no-op when nothing was undone)."""
        if not self._future:
            return
        self._history.append(self._spec)
        self._set(self._future.pop())
        self.layout_changed.emit(self._spec)

    # -- store -------------------------------------------------------------

    def save_as(self) -> bool:
        """Ask for a name and save the layout in the chosen scope.

        Returns whether a file was written; cancellation, a bad name or a
        store error is reported through ``dialogs.notify`` / the status line.
        """
        if self.store is None:
            self._status(NO_STORE)
            return False
        name = self.dialogs.ask_name(self._spec.name)
        if name is None:
            return False
        scope = str(self.scope_combo.currentData() or USER)
        try:
            path = self.store.save(validate_name(name), self._spec, scope)
        except (ValueError, OSError) as exc:  # LayoutStoreError is a ValueError
            self.dialogs.notify(f"Could not save layout: {exc}")
            return False
        self._set(self._spec.renamed(path.stem))
        self._refresh_presets(select=(path.stem, scope))
        self._status(f"saved layout '{path.stem}' to {scope}")
        return True

    def load_preset(self, name: str, scope: str) -> bool:
        """Load ``name`` from ``scope`` (builtin, user or session) as an edit."""
        if scope == BUILTIN:
            self.apply_preset(name)
            return True
        if self.store is None:
            self._status(NO_STORE)
            return False
        try:
            spec = self.store.load(name, scope)
        except LayoutStoreError as exc:
            self.dialogs.notify(f"Could not load layout: {exc}")
            return False
        self._commit(fill_empty(replace(spec, canvas=self._spec.canvas)))
        return True

    def delete_preset(self, name: str, scope: str) -> bool:
        """Delete ``name`` from ``scope`` after confirmation; builtins refuse."""
        if self.store is None or scope == BUILTIN:
            self._status("built-in layouts cannot be deleted")
            return False
        if not self.dialogs.confirm(f"Delete layout '{name}' from {scope}?"):
            return False
        try:
            self.store.delete(name, scope)
        except (LayoutStoreError, OSError) as exc:
            self.dialogs.notify(f"Could not delete layout: {exc}")
            return False
        self._refresh_presets()
        self._status(f"deleted layout '{name}' from {scope}")
        return True

    def _find_preset(self, data: tuple[str, str]) -> int:
        """Combo index holding ``data`` (``findData`` cannot compare tuples)."""
        for index in range(self.preset_combo.count()):
            if self.preset_combo.itemData(index) == data:
                return index
        return -1

    def _chosen_preset(self) -> tuple[str, str] | None:
        data = self.preset_combo.currentData()
        return None if data is None else (str(data[0]), str(data[1]))

    def _load_clicked(self) -> None:
        chosen = self._chosen_preset()
        if chosen is not None:
            self.load_preset(*chosen)

    def _delete_clicked(self) -> None:
        chosen = self._chosen_preset()
        if chosen is not None:
            self.delete_preset(*chosen)

    def _refresh_presets(self, select: tuple[str, str] | None = None) -> None:
        current = select or self._chosen_preset()
        self._syncing = True
        self.preset_combo.clear()
        for name in PRESET_NAMES:
            self.preset_combo.addItem(name, (name, BUILTIN))
        if self.store is not None:
            for scope in (USER, SESSION):
                for entry in self.store.list(scope):
                    text = f"{scope}: {entry.name}"
                    if entry.error:
                        text += " (unreadable)"
                    self.preset_combo.addItem(text, (entry.name, scope))
        if current is not None:
            self.preset_combo.setCurrentIndex(max(self._find_preset(current), 0))
        self._syncing = False
        self._update_controls()

    # -- widget callbacks --------------------------------------------------

    def _on_grid_spin(self) -> None:
        if not self._syncing:
            self.set_grid(self.rows_spin.value(), self.cols_spin.value())

    def _on_tile(self, action: str, *args: Any) -> None:
        index = self.selected_index()
        if index < 0:
            return
        getattr(self, action)(index, *args)

    def _on_source_combo(self) -> None:
        index = self.selected_index()
        source = self.source_combo.currentData()
        if self._syncing or index < 0 or not isinstance(source, SourceRef):
            return
        if source != self._spec.tiles[index].source:
            self.set_source(index, source)

    def _on_fit_combo(self) -> None:
        index = self.selected_index()
        fit = self.fit_combo.currentData()
        if self._syncing or index < 0 or fit == self._spec.tiles[index].fit:
            return
        self.set_fit(index, str(fit))

    def _on_label_toggled(self, checked: bool) -> None:
        index = self.selected_index()
        if self._syncing or index < 0:
            return
        if checked != self._spec.tiles[index].show_label:
            self.set_label_visible(index, checked)

    def _on_crop_toggled(self, checked: bool) -> None:
        self.canvas.crop_mode = bool(checked)
        self._status(
            "drag a rectangle over the selected tile to crop it" if checked else ""
        )

    def _on_crop_dragged(self, cell: Cell, rect: NormRect) -> None:
        index = self.tile_index(cell)
        if index >= 0:
            self.set_crop(index, sub_crop(self._spec.tiles[index].crop, rect))

    def _on_span(self, drow: int, dcol: int) -> None:
        index = self.selected_index()
        if index >= 0:
            self.change_span(index, drow=drow, dcol=dcol)

    # -- syncing ------------------------------------------------------------

    def _set(self, spec: LayoutSpec) -> None:
        """Show ``spec``; the selection follows the tile at its anchor."""
        self._spec = spec
        if self._selected is not None:
            tile = spec.tile_at(self._selected.row, self._selected.col)
            self._selected = None if tile is None else tile.cell
        self.canvas.set_spec(spec, self._selected)
        self._status("")
        self._sync_widgets()

    def _status(self, text: str) -> None:
        self.status_label.setText(text)

    def _sync_widgets(self) -> None:
        """Push the spec and selection into the controls without re-editing."""
        self._syncing = True
        try:
            self.rows_spin.setValue(self._spec.rows)
            self.cols_spin.setValue(self._spec.cols)
            self._sync_source_combo()
            index = self.selected_index()
            if index >= 0:
                tile = self._spec.tiles[index]
                self.fit_combo.setCurrentIndex(
                    max(self.fit_combo.findData(tile.fit), 0)
                )
                self.label_check.setChecked(tile.show_label)
        finally:
            self._syncing = False
        self._update_controls()

    def _sync_source_combo(self) -> None:
        index = self.selected_index()
        current = self._spec.tiles[index].source if index >= 0 else None
        self.source_combo.clear()
        self.source_combo.addItem(EMPTY_SOURCE_TEXT, SourceRef.empty())
        options = list(self._sources)
        if current is not None and not current.is_empty and current not in options:
            options.append(current)
        for source in options:
            self.source_combo.addItem(source_text(source), source)
        if current is not None:
            self.source_combo.setCurrentIndex(
                max(self.source_combo.findData(current), 0)
            )

    def _update_controls(self) -> None:
        """Enable each control when it can act and say why when it cannot."""
        has_tile = self.selected_index() >= 0
        for widget in (
            self.source_combo,
            self.rotate_ccw_button,
            self.rotate_cw_button,
            self.flip_h_button,
            self.flip_v_button,
            self.crop_button,
            self.reset_crop_button,
            self.fit_combo,
            self.label_check,
        ):
            self._enable(widget, has_tile, NO_TILE)
        self._update_span_controls(has_tile)
        self._enable(self.undo_button, self.can_undo(), "nothing to undo")
        self._enable(self.redo_button, self.can_redo(), "nothing to redo")
        self._update_store_controls()

    def _update_span_controls(self, has_tile: bool) -> None:
        tile = self._spec.tiles[self.selected_index()] if has_tile else None
        cell = tile.cell if tile is not None else Cell(0, 0)
        checks = (
            (
                self.span_row_plus,
                cell.fits(self._spec.rows - 1, self._spec.cols),
                "the tile already reaches the bottom row",
            ),
            (self.span_row_minus, cell.rowspan > 1, "the tile is one row tall"),
            (
                self.span_col_plus,
                cell.fits(self._spec.rows, self._spec.cols - 1),
                "the tile already reaches the last column",
            ),
            (self.span_col_minus, cell.colspan > 1, "the tile is one column wide"),
        )
        for widget, ok, why in checks:
            self._enable(widget, has_tile and ok, NO_TILE if not has_tile else why)

    def _update_store_controls(self) -> None:
        chosen = self._chosen_preset()
        has_store = self.store is not None
        has_session = self.store is not None and self.store.session is not None
        self._enable(self.load_button, chosen is not None, "choose a preset first")
        self._enable(self.save_button, has_store, NO_STORE)
        self._enable(self.scope_combo, has_store, NO_STORE)
        session_index = self.scope_combo.findData(SESSION)
        model = self.scope_combo.model()
        item = model.item(session_index) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(bool(has_session))
            item.setToolTip(
                hint(HELP["scope"], bool(has_session), "no session folder is open")
            )
        deletable = has_store and chosen is not None and chosen[1] != BUILTIN
        why = NO_STORE if not has_store else "built-in presets are read-only"
        self._enable(self.delete_button, deletable, why)

    def _enable(self, widget: QWidget, enabled: bool, why: str) -> None:
        widget.setEnabled(enabled)
        widget.setToolTip(hint(HELP[widget.objectName()], enabled, why))
