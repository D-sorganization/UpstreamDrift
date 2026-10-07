"""Assembly canvas: drop library parts onto compatible ports (CMB-9, #11660).

The canvas is a thin view over :class:`AssemblySession`. It draws the link
tree with a circle per typed socket, colours sockets while a part is dragged
over (green = would mate, red = rejected) and forwards drops to the session.
All rules live in the session so they stay unit-testable without Qt.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

from src.tools.model_explorer.assembly_session import AssemblySession, DropDecision

PART_MIME = "application/x-upstreamdrift-part"
NODE_W, NODE_H, GAP_X, GAP_Y = 120.0, 30.0, 30.0, 95.0
PORT_R = 7.0

_FREE = QColor("#3b82f6")
_OCCUPIED = QColor("#9ca3af")
_OK = QColor("#22c55e")
_BAD = QColor("#ef4444")
_PALETTE = ("#dbeafe", "#fde68a", "#fbcfe8", "#bbf7d0", "#ddd6fe", "#fed7aa")


class AssemblyCanvas(QGraphicsView):
    """Graphics view that renders a session and accepts part drops."""

    part_dropped = pyqtSignal(str, str)  # part_id, host port name
    drop_rejected = pyqtSignal(str)  # reason
    hover_reason = pyqtSignal(str)
    instance_selected = pyqtSignal(str)

    def __init__(self, session: AssemblySession, parent: QWidget | None = None):
        if session is None:
            raise ValueError("session must be provided")
        super().__init__(parent)
        self.session = session
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMinimumSize(420, 320)
        self._port_items: dict[str, QGraphicsEllipseItem] = {}
        self._link_items: dict[str, QGraphicsRectItem] = {}
        self._drag_part: str | None = None
        self._selected: str | None = None
        self.refresh()

    # ----------------------------------------------------------------- public
    def set_session(self, session: AssemblySession) -> None:
        """Show a different session (for example after "New Assembly")."""
        if session is None:
            raise ValueError("session must be provided")
        self.session = session
        self._selected = None
        self.refresh()

    def selected_instance(self) -> str | None:
        """Instance id of the part the user last clicked, if still present."""
        return self._selected

    def port_names(self) -> tuple[str, ...]:
        """Names of the sockets currently drawn."""
        return tuple(self._port_items)

    def port_viewport_pos(self, port_name: str):
        """Viewport pixel position of a socket (used by drops and tests)."""
        item = self._port_items[port_name]
        return self.mapFromScene(item.sceneBoundingRect().center())

    def refresh(self) -> None:
        """Redraw the whole assembly from the session."""
        live = {p.instance_id for p in self.session.placed_parts}
        if self._selected not in live:
            self._selected = None
        self._scene.clear()
        self._port_items.clear()
        self._link_items.clear()
        positions = self._layout()
        owners = self._link_owners()
        self._draw_edges(positions)
        for name, (x, y) in positions.items():
            self._draw_link(name, x, y, owners.get(name, ""))
        self._draw_ports(positions)
        self._scene.setSceneRect(
            self._scene.itemsBoundingRect().adjusted(-30, -30, 30, 30)
        )
        self._fit()

    def _fit(self) -> None:
        """Shrink the view to show the whole assembly; never magnify."""
        self.resetTransform()
        rect = self._scene.sceneRect()
        viewport = self.viewport()
        if viewport is None:
            return
        if rect.width() > viewport.width() or rect.height() > viewport.height():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._fit()

    def drop_part(self, part_id: str, port_name: str) -> DropDecision:
        """Try to attach ``part_id`` at ``port_name`` and report the outcome."""
        decision = self.session.evaluate_drop(part_id, port_name)
        if decision.accepted:
            self.session.attach(part_id, port_name)
            self.refresh()
            self.part_dropped.emit(part_id, port_name)
        else:
            self.drop_rejected.emit(decision.reason)
            self._recolour_ports()
        return decision

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is None:
            return
        mime = event.mimeData()
        if mime is not None and mime.hasFormat(PART_MIME):
            self._drag_part = mime.data(PART_MIME).data().decode("utf-8")
            self._colour_ports_for(self._drag_part)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        if event is None or self._drag_part is None:
            return
        port = self._port_at(event.position())
        if port is None:
            self.hover_reason.emit("Drop on a highlighted socket")
        else:
            decision = self.session.evaluate_drop(self._drag_part, port)
            self.hover_reason.emit(
                f"{port}: {'compatible' if decision.accepted else decision.reason}"
            )
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: ANN001
        self._drag_part = None
        self._recolour_ports()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None or self._drag_part is None:
            return
        part_id, self._drag_part = self._drag_part, None
        port = self._port_at(event.position())
        if port is None:
            self.drop_rejected.emit("Drop the part on a socket")
            self._recolour_ports()
        else:
            self.drop_part(part_id, port)
        event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        item = self.itemAt(event.position().toPoint())
        while item is not None and item.data(1) is None:
            item = item.parentItem()
        if item is not None:
            self._selected = str(item.data(1))
            self.instance_selected.emit(self._selected)
        super().mousePressEvent(event)

    # --------------------------------------------------------------- internal
    def _port_at(self, pos: QPointF) -> str | None:
        scene_pos = self.mapToScene(pos.toPoint())
        for item in self._scene.items(scene_pos):
            name = item.data(0)
            if name is not None:
                return str(name)
        return None

    def _colour_ports_for(self, part_id: str) -> None:
        for name, item in self._port_items.items():
            ok = self.session.evaluate_drop(part_id, name).accepted
            item.setBrush(QBrush(_OK if ok else _BAD))

    def _recolour_ports(self) -> None:
        taken = self.session.occupied_ports()
        for name, item in self._port_items.items():
            item.setBrush(QBrush(_OCCUPIED if name in taken else _FREE))

    def _children(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {n: [] for n in self.session.link_names()}
        for parent, child in self.session.link_edges():
            children.setdefault(parent, []).append(child)
        return children

    def _layout(self) -> dict[str, tuple[float, float]]:
        children = self._children()
        kids = {c for cs in children.values() for c in cs}
        roots = [n for n in children if n not in kids]
        positions: dict[str, tuple[float, float]] = {}
        cursor = [0.0]

        def place(name: str, depth: int) -> float:
            subs = children.get(name, [])
            if not subs:
                x = cursor[0]
                cursor[0] += NODE_W + GAP_X
            else:
                xs = [place(c, depth + 1) for c in subs]
                x = sum(xs) / len(xs)
            positions[name] = (x, depth * GAP_Y)
            return x

        for root in roots:
            place(root, 0)
        return positions

    def _link_owners(self) -> dict[str, str]:
        return {
            link: placed.instance_id
            for placed in self.session.placed_parts
            for link in placed.links
        }

    def _draw_edges(self, positions: dict[str, tuple[float, float]]) -> None:
        pen = QPen(QColor("#6b7280"), 1.5)
        for name, kids in self._children().items():
            for kid in kids:
                if name in positions and kid in positions:
                    px, py = positions[name]
                    cx, cy = positions[kid]
                    self._scene.addLine(
                        px + NODE_W / 2, py + NODE_H, cx + NODE_W / 2, cy, pen
                    )

    def _draw_link(self, name: str, x: float, y: float, owner: str) -> None:
        index = (
            [p.instance_id for p in self.session.placed_parts].index(owner)
            if owner
            else 0
        )
        rect = QGraphicsRectItem(QRectF(x, y, NODE_W, NODE_H))
        rect.setBrush(QBrush(QColor(_PALETTE[index % len(_PALETTE)])))
        rect.setPen(QPen(QColor("#374151"), 1.2))
        rect.setData(1, owner)
        rect.setToolTip(f"{name}\n{owner}")
        self._scene.addItem(rect)
        label = QGraphicsSimpleTextItem(_short(name), rect)
        label.setPos(x + 6, y + 7)
        label.setBrush(QBrush(QColor("#111827")))
        self._link_items[name] = rect

    def _draw_ports(self, positions: dict[str, tuple[float, float]]) -> None:
        taken = self.session.occupied_ports()
        by_link: dict[str, list[str]] = {}
        for port in self.session.free_sockets() + tuple(
            p for p in self.session.all_ports() if p.name in taken
        ):
            by_link.setdefault(port.link_name, []).append(port.name)
        for link, names in by_link.items():
            if link not in positions:
                continue
            x, y = positions[link]
            for i, name in enumerate(names):
                cx = x + 14 + i * 24
                dot = QGraphicsEllipseItem(
                    cx - PORT_R, y + NODE_H + 4, 2 * PORT_R, 2 * PORT_R
                )
                dot.setBrush(QBrush(_OCCUPIED if name in taken else _FREE))
                dot.setPen(QPen(QColor("#1f2937"), 1.0))
                dot.setData(0, name)
                dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                dot.setToolTip(self._port_tooltip(name))
                self._scene.addItem(dot)
                self._port_items[name] = dot

    def _port_tooltip(self, name: str) -> str:
        port = self.session.host_port(name)
        if port is None or port.port_type is None:
            return name
        return f"{name}\n{port.port_type.value} socket"


def _short(name: str) -> str:
    tail = name.split("__")[-1]
    return tail if len(tail) <= 16 else tail[:15] + "…"
