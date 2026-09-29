"""Qt utility to suppress wheel-driven value changes in input controls.

Policy: No user-editable value should change solely because of a mouse-wheel event.
Wheel input should scroll the surrounding surface or be ignored, but it must not
mutate numeric inputs, selects, combo boxes, sliders, or custom value controls.

Usage:
    from src.shared.python.qt_utils.wheel_event_filter import WheelEventFilter

    # Apply to spin boxes, combo boxes, etc.
    filter = WheelEventFilter()
    self.my_spin_box.installEventFilter(filter)
    self.my_combo_box.installEventFilter(filter)
"""

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QWheelEvent


class WheelEventFilter(QObject):
    """Event filter that blocks wheel events from reaching input controls.

    Install this filter on QSpinBox, QDoubleSpinBox, QComboBox, QSlider,
    or any other widget where wheel-driven value changes are undesirable.

    Example:
        filter = WheelEventFilter()
        self.spin_box.installEventFilter(filter)
        self.combo_box.installEventFilter(filter)
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filter out wheel events to prevent value changes.

        Args:
            obj: The object being filtered.
            event: The event to filter.

        Returns:
            True if the event should be filtered (blocked), False otherwise.
        """
        if event.type() == QEvent.Type.Wheel:
            wheel_event = event
            if isinstance(wheel_event, QWheelEvent):
                # Accept the event to prevent it from propagating
                wheel_event.accept()
                return True
        return False


# Attribute name for storing filter reference on widget
_WHEEL_FILTER_ATTR = "_wheel_event_filter"


def suppress_wheel_on_widget(widget) -> None:
    """Convenience function to install wheel event filter on a widget.

    The filter instance is stored as an attribute on the widget itself,
    ensuring it remains alive for the lifetime of the widget. When the
    widget is destroyed, the filter is automatically garbage collected.

    Args:
        widget: The widget to suppress wheel events on.
    """
    filter_instance = WheelEventFilter()
    setattr(widget, _WHEEL_FILTER_ATTR, filter_instance)
    widget.installEventFilter(filter_instance)


def suppress_wheel_on_widgets(*widgets) -> None:
    """Convenience function to install wheel event filter on multiple widgets.

    Each filter instance is stored as an attribute on its respective widget,
    ensuring it remains alive for the lifetime of the widget. When widgets
    are destroyed, their filters are automatically garbage collected.

    Args:
        widgets: Variable number of widgets to suppress wheel events on.
    """
    for widget in widgets:
        filter_instance = WheelEventFilter()
        setattr(widget, _WHEEL_FILTER_ATTR, filter_instance)
        widget.installEventFilter(filter_instance)
