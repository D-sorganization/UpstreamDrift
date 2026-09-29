"""Tests for shared responsive PyQt sizing helpers."""

from __future__ import annotations

import pytest

from src.shared.python.theme.responsive import TextWidthSpec, readable_text_width


class _Metrics:
    def horizontalAdvance(self, text: str) -> int:  # noqa: N802
        return len(text) * 8


def test_readable_text_width_uses_widest_text_and_chrome_padding() -> None:
    width = readable_text_width(
        _Metrics(),
        ["Short", "Longer filter label"],
        TextWidthSpec(padding_px=12, chrome_px=20, minimum_px=80),
    )

    assert width == len("Longer filter label") * 8 + 12 + 20


def test_readable_text_width_respects_maximum_width() -> None:
    width = readable_text_width(
        _Metrics(),
        ["A very long responsive label"],
        TextWidthSpec(padding_px=16, minimum_px=60, maximum_px=120),
    )

    assert width == 120


def test_text_width_spec_rejects_invalid_contract() -> None:
    spec = TextWidthSpec(minimum_px=200, maximum_px=120)

    with pytest.raises(ValueError, match="maximum_px"):
        spec.validate()
