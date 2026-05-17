from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QEvent, QRect, QObject

class ResizeFilter(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._resizing = False
        self._resize_edge = 0
        self._start_pos = None
        self._start_geo = None

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.HoverMove, QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            if hasattr(event, "globalPosition"):
                gpos = event.globalPosition().toPoint()
            else:
                return False

            local_pos = self.window.mapFromGlobal(gpos)
            x, y = local_pos.x(), local_pos.y()
            w, h = self.window.width(), self.window.height()
            border = 8

            if not self._resizing:
                if 0 <= x <= w and 0 <= y <= h:
                    edge = 0
                    if x < border and y < border: edge = 13
                    elif x > w - border and y < border: edge = 14
                    elif x < border and y > h - border: edge = 16
                    elif x > w - border and y > h - border: edge = 17
                    elif x < border: edge = 10
                    elif x > w - border: edge = 11
                    elif y < border: edge = 12
                    elif y > h - border: edge = 15

                    if edge != 0:
                        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                            self._resizing = True
                            self._resize_edge = edge
                            self._start_pos = gpos
                            self._start_geo = self.window.geometry()
                            return True
                        elif event.type() == QEvent.Type.HoverMove or event.type() == QEvent.Type.MouseMove:
                            if edge in (13, 17): self.window.setCursor(Qt.CursorShape.SizeFDiagCursor)
                            elif edge in (14, 16): self.window.setCursor(Qt.CursorShape.SizeBDiagCursor)
                            elif edge in (10, 11): self.window.setCursor(Qt.CursorShape.SizeHorCursor)
                            elif edge in (12, 15): self.window.setCursor(Qt.CursorShape.SizeVerCursor)
                            return True
                    else:
                        if self.window.cursor().shape() != Qt.CursorShape.ArrowCursor:
                            self.window.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                if event.type() == QEvent.Type.MouseMove:
                    delta = gpos - self._start_pos
                    rect = QRect(self._start_geo)
                    if self._resize_edge in (10, 13, 16): rect.setLeft(rect.left() + delta.x())
                    if self._resize_edge in (11, 14, 17): rect.setRight(rect.right() + delta.x())
                    if self._resize_edge in (12, 13, 14): rect.setTop(rect.top() + delta.y())
                    if self._resize_edge in (15, 16, 17): rect.setBottom(rect.bottom() + delta.y())
                    self.window.setGeometry(rect)
                    return True
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    self._resizing = False
                    self.window.setCursor(Qt.CursorShape.ArrowCursor)
                    return True
        return False

app = QApplication([])
w = QWidget()
w.setWindowFlags(Qt.WindowType.FramelessWindowHint)
w.resize(400, 300)
l = QVBoxLayout(w)
l.addWidget(QPushButton("Hello!"))
l.addWidget(QLabel("Test test"))

filter = ResizeFilter(w)
app.installEventFilter(filter)

w.show()
import sys
sys.exit(app.exec())
