"""QGraphicsView canvas of the multiview layout editor (#9812).

The canvas draws the whole layout through the one compositor
(:func:`layout_model.compose`) with the frames a ``frame_provider`` hands it,
then lays a transparent, clickable rectangle over every tile. Pressing a
tile selects it; dragging it onto another cell moves it (the model swaps
when the target is occupied); in crop mode a drag draws a rectangle that
becomes the tile's crop. All decisions live in :class:`LayoutEditor`: the
canvas only reports what the operator did through its signals.

Colours come from the theme (:func:`get_current_colors`); none are literal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from src.shared.python.core.contracts import require
from src.shared.python.theme.palette import get_current_colors

from .layout_model import Cell, LayoutSpec, Palette, SourceRef, cell_rect, compose

FrameProvider = Callable[[SourceRef], "npt.NDArray[Any] | None"]
NormRect = tuple[float, float, float, float]

SCENE_WIDTH = 640
DRAG_THRESHOLD_PX = 4.0
MIN_CROP_PX = 3.0
BORDER_WIDTH = 1.0
SELECTED_WIDTH = 3.0


@dataclass(frozen=True)
class CanvasColours:
    """Theme-derived colours the canvas paints with."""

    background: QColor
    border: QColor
    selected: QColor
    crop: QColor
    ghost: QColor
    palette: Palette

    @classmethod
    def from_theme(cls) -> CanvasColours:
        """Resolve every colour from the active theme palette."""
        colors = get_current_colors()
        ghost = QColor(colors["accent"])
        ghost.setAlpha(96)
        return cls(
            background=QColor(colors["bg"]),
            border=QColor(colors["border"]),
            selected=QColor(colors["accent"]),
            crop=QColor(colors["focus"]),
            ghost=ghost,
            palette=Palette.from_hex(
                background=colors["bg"],
                placeholder=colors["group_bg"],
                placeholder_text=colors["text_secondary"],
                label_text=colors["text"],
                label_box=colors["title_bg"],
            ),
        )


def scene_size_for(spec: LayoutSpec) -> tuple[int, int]:
    """Thumbnail canvas size: :data:`SCENE_WIDTH` wide at the spec's aspect."""
    width, height = spec.canvas
    return SCENE_WIDTH, max(1, round(SCENE_WIDTH * height / width))


def normalised_rect(
    rect: tuple[int, int, int, int], start: QPointF, end: QPointF
) -> NormRect | None:
    """``(x, y, w, h)`` in [0, 1] of the drag ``start``..``end`` inside ``rect``.

    The drag is clamped to the rectangle; ``None`` when it is thinner than
    :data:`MIN_CROP_PX` in either direction (a click, not a crop).
    """
    x0, y0, x1, y1 = rect
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    left = min(max(min(start.x(), end.x()), x0), x1)
    right = min(max(max(start.x(), end.x()), x0), x1)
    top = min(max(min(start.y(), end.y()), y0), y1)
    bottom = min(max(max(start.y(), end.y()), y0), y1)
    if right - left < MIN_CROP_PX or bottom - top < MIN_CROP_PX:
        return None
    return (
        (left - x0) / width,
        (top - y0) / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def bgr_to_qpixmap(frame: npt.NDArray[np.uint8]) -> QPixmap:
    """A pixmap of an ``HxWx3`` BGR uint8 frame (copied, so the array may go)."""
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    h, w = rgb.shape[:2]
    image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class LayoutCanvas(QGraphicsView):
    """Composite thumbnail of a layout with clickable, draggable tile overlays.

    Signals: ``tile_pressed(row, col)`` for the anchor of the tile under a
    click; ``tile_dropped(src_cell, dst_cell)`` after a drag between two
    different tiles; ``crop_dragged(cell, (x, y, w, h))`` for a crop-mode
    drag, normalised to the tile's rectangle.
    """

    tile_pressed = pyqtSignal(int, int)
    tile_dropped = pyqtSignal(object, object)
    crop_dragged = pyqtSignal(object, object)

    def __init__(self, spec: LayoutSpec, parent: Any = None) -> None:
        super().__init__(parent)
        self.colours = CanvasColours.from_theme()
        self._spec = spec
        self._selected: Cell | None = None
        self._frame_provider: FrameProvider | None = None
        self._composite: npt.NDArray[np.uint8] | None = None
        self._press: QPointF | None = None
        self._press_cell: Cell | None = None
        self._dragging = False
        self.crop_mode = False
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self.pixmap_item)
        self._overlays: list[QGraphicsRectItem] = []
        self._ghost = self._rubber(self.colours.ghost, fill=True)
        self._crop_band = self._rubber(self.colours.crop, fill=False)
        self.setBackgroundBrush(QBrush(self.colours.background))
        self.setMinimumHeight(160)
        self.setToolTip(
            "Layout preview. Click a tile to select it; drag a tile onto another "
            "cell to move it (the two swap); with Crop on, drag a rectangle over "
            "a tile to crop it."
        )
        self.redraw()

    def _rubber(self, colour: QColor, fill: bool) -> QGraphicsRectItem:
        item = QGraphicsRectItem()
        pen = QPen(colour, SELECTED_WIDTH)
        pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        item.setBrush(QBrush(colour) if fill else QBrush(Qt.BrushStyle.NoBrush))
        item.setZValue(10)
        item.setVisible(False)
        self._scene.addItem(item)
        return item

    # -- state -----------------------------------------------------------

    def set_spec(self, spec: LayoutSpec, selected: Cell | None) -> None:
        """Show ``spec`` with the tile anchored at ``selected`` highlighted."""
        self._spec = spec
        self._selected = selected
        self.redraw()

    def set_selected(self, selected: Cell | None) -> None:
        self._selected = selected
        self._restyle()

    def set_frame_provider(self, provider: FrameProvider | None) -> None:
        self._frame_provider = provider
        self.redraw()

    def scene_size(self) -> tuple[int, int]:
        return scene_size_for(self._spec)

    def composite(self) -> npt.NDArray[np.uint8] | None:
        """The BGR image last drawn (``None`` before the first redraw)."""
        return self._composite

    def overlay_items(self) -> list[QGraphicsRectItem]:
        """One rectangle per tile, in ``spec.tiles`` order."""
        return list(self._overlays)

    # -- drawing ---------------------------------------------------------

    def _frames(self) -> dict[str, npt.NDArray[Any]]:
        frames: dict[str, npt.NDArray[Any]] = {}
        if self._frame_provider is None:
            return frames
        for tile in self._spec.tiles:
            source = tile.source
            if source.is_empty or source.key in frames:
                continue
            frame = self._frame_provider(source)
            if frame is not None:
                frames[source.key] = frame
        return frames

    def redraw(self) -> None:
        """Recompose the thumbnail and rebuild the tile overlays.

        Postcondition: ``composite()`` is an ``(h, w, 3)`` uint8 image of
        ``scene_size()`` and ``overlay_items()`` has one item per tile.
        """
        size = self.scene_size()
        self._composite = compose(
            self._frames(), self._spec, size, palette=self.colours.palette
        )
        self.pixmap_item.setPixmap(bgr_to_qpixmap(self._composite))
        self._scene.setSceneRect(QRectF(0, 0, size[0], size[1]))
        for item in self._overlays:
            self._scene.removeItem(item)
        self._overlays = []
        for tile in self._spec.tiles:
            x0, y0, x1, y1 = cell_rect(self._spec, tile.cell, size)
            item = QGraphicsRectItem(QRectF(x0, y0, x1 - x0, y1 - y0))
            item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            item.setZValue(1)
            self._scene.addItem(item)
            self._overlays.append(item)
        self._restyle()
        self._fit_view()

    def _restyle(self) -> None:
        for tile, item in zip(self._spec.tiles, self._overlays, strict=True):
            chosen = self._selected is not None and tile.cell.anchor == (
                self._selected.row,
                self._selected.col,
            )
            colour = self.colours.selected if chosen else self.colours.border
            item.setPen(QPen(colour, SELECTED_WIDTH if chosen else BORDER_WIDTH))

    def _fit_view(self) -> None:
        viewport = self.viewport()
        if viewport is not None and viewport.width() > 0 and viewport.height() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_view()

    # -- hit testing ------------------------------------------------------

    def cell_at(self, point: QPointF) -> Cell | None:
        """Anchor cell of the tile under scene ``point`` (``None`` off-canvas)."""
        width, height = self.scene_size()
        if not (0 <= point.x() < width and 0 <= point.y() < height):
            return None
        row = min(int(point.y() * self._spec.rows / height), self._spec.rows - 1)
        col = min(int(point.x() * self._spec.cols / width), self._spec.cols - 1)
        tile = self._spec.tile_at(row, col)
        return None if tile is None else tile.cell

    def _tile_rect(self, cell: Cell) -> tuple[int, int, int, int]:
        return cell_rect(self._spec, cell, self.scene_size())

    # -- mouse (scene coordinates; the Qt events map and delegate) -------

    def begin_press(self, point: QPointF) -> None:
        """Start a press at scene ``point`` and select the tile under it."""
        self._press = point
        self._press_cell = self.cell_at(point)
        self._dragging = False
        if self._press_cell is not None:
            self.tile_pressed.emit(self._press_cell.row, self._press_cell.col)

    def drag_to(self, point: QPointF) -> None:
        """Update the drag feedback for the pointer at scene ``point``."""
        if self._press is None or self._press_cell is None:
            return
        moved = (point - self._press).manhattanLength() > DRAG_THRESHOLD_PX
        if not moved:
            return
        self._dragging = True
        x0, y0, x1, y1 = self._tile_rect(self._press_cell)
        if self.crop_mode:
            rect = QRectF(self._press, point).normalized()
            rect = rect.intersected(QRectF(x0, y0, x1 - x0, y1 - y0))
            self._crop_band.setRect(rect)
            self._crop_band.setVisible(True)
            return
        offset = point - self._press
        ghost = QRectF(x0, y0, x1 - x0, y1 - y0).translated(offset)
        self._ghost.setRect(ghost)
        self._ghost.setVisible(True)

    def release_at(self, point: QPointF) -> None:
        """Finish the press: emit a drop or a crop when the pointer moved.

        Postcondition: no feedback rectangle stays visible.
        """
        press, cell, dragging = self._press, self._press_cell, self._dragging
        self._press = self._press_cell = None
        self._dragging = False
        self._ghost.setVisible(False)
        self._crop_band.setVisible(False)
        if press is None or cell is None or not dragging:
            return
        if self.crop_mode:
            rect = normalised_rect(self._tile_rect(cell), press, point)
            if rect is not None:
                self.crop_dragged.emit(cell, rect)
            return
        target = self.cell_at(point)
        if target is not None and target.anchor != cell.anchor:
            self.tile_dropped.emit(cell, target)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin_press(self.mapToScene(event.position().toPoint()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        self.drag_to(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.release_at(self.mapToScene(event.position().toPoint()))
        super().mouseReleaseEvent(event)


def check_cell(cell: Any) -> Cell:
    """Precondition helper: ``cell`` is a :class:`Cell`."""
    require(isinstance(cell, Cell), "expected a layout Cell", cell)
    return cell
