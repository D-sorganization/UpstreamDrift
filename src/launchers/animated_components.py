from PyQt6.QtWidgets import QPushButton
from PyQt6.QtCore import QVariantAnimation, QEasingCurve
from PyQt6.QtGui import QColor


class HoverTransitionMixin:
    """Mixin to provide smooth color transitions on hover."""

    def init_animation(
        self,
        base_color: str,
        hover_color: str,
        text_color: str = "white",
        padding: str = "5px 10px",
        border_radius: str = "4px",
    ):
        self._base_color = QColor(base_color)
        self._hover_color = QColor(hover_color)
        self._text_color = text_color
        self._padding = padding
        self._border_radius = border_radius
        self._current_color = self._base_color

        self._hover_anim = QVariantAnimation(self)
        self._hover_anim.setDuration(150)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hover_anim.valueChanged.connect(self._on_color_changed)
        self._update_stylesheet()

    def _enter_event(self, event):
        if hasattr(self, "_hover_anim"):
            self._hover_anim.setStartValue(self._current_color)
            self._hover_anim.setEndValue(self._hover_color)
            self._hover_anim.start()

    def _leave_event(self, event):
        if hasattr(self, "_hover_anim"):
            self._hover_anim.setStartValue(self._current_color)
            self._hover_anim.setEndValue(self._base_color)
            self._hover_anim.start()

    def _on_color_changed(self, color):
        self._current_color = color
        self._update_stylesheet()

    def _update_stylesheet(self):
        css = f"""
        QPushButton {{
            background-color: {self._current_color.name()}; 
            color: {self._text_color}; 
            border: none; 
            padding: {self._padding}; 
            border-radius: {self._border_radius}; 
            font-weight: bold;
        }}
        """
        self.setStyleSheet(css)


class AnimatedButton(QPushButton, HoverTransitionMixin):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        # Default to primary colors
        self.init_animation("#0A84FF", "#0077E6")

    def enterEvent(self, event):
        self._enter_event(event)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._leave_event(event)
        super().leaveEvent(event)
