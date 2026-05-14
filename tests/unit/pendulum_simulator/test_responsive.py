"""Tests for pendulum simulator responsive sizing helpers."""

from __future__ import annotations

from src.shared.python.pendulum_simulator.gui.responsive import (
    apply_overlay_checkbox_sizing,
)


class _Metrics:
    def horizontalAdvance(self, text: str) -> int:  # noqa: N802
        return len(text) * 8


class _Policy:
    def verticalPolicy(self) -> object:  # noqa: N802
        return object()


class _Checkbox:
    def __init__(self, text: str) -> None:
        self._text = text
        self.minimum_width = 0
        self.size_policy: object | None = None

    def text(self) -> str:
        return self._text

    def fontMetrics(self) -> _Metrics:  # noqa: N802
        return _Metrics()

    def setMinimumWidth(self, width: int) -> None:  # noqa: N802
        self.minimum_width = width

    def sizePolicy(self) -> _Policy:  # noqa: N802
        return _Policy()

    def setSizePolicy(self, horizontal: object, vertical: object) -> None:  # noqa: N802
        self.size_policy = (horizontal, vertical)


def test_overlay_checkbox_sizing_uses_text_aware_minimum_width() -> None:
    checkbox = _Checkbox("Mobility Ellipsoids")

    width = apply_overlay_checkbox_sizing(checkbox)

    assert width == len("Mobility Ellipsoids") * 8 + 8 + 30
    assert checkbox.minimum_width == width
    assert checkbox.size_policy is not None
