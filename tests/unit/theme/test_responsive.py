"""Tests for shared responsive PyQt sizing helpers."""

from __future__ import annotations

import pytest

from src.shared.python.theme.responsive import TextWidthSpec, readable_text_width


class _Metrics:
    def horizontalAdvance(self, text: str) -> int:  # noqa: N802
        return len(text) * 8


class _ObjectMetrics:
    def horizontalAdvance(self, _text: str) -> object:  # noqa: N802
        return object()


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


def test_readable_text_width_uses_minimum_for_short_text() -> None:
    width = readable_text_width(
        _Metrics(),
        ["tiny"],
        TextWidthSpec(padding_px=0, chrome_px=0, minimum_px=96),
    )

    assert width == 96


def test_readable_text_width_ignores_empty_candidates() -> None:
    width = readable_text_width(
        _Metrics(),
        ["", "visible"],
        TextWidthSpec(padding_px=4),
    )

    assert width == len("visible") * 8 + 4


def test_readable_text_width_rejects_all_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        readable_text_width(_Metrics(), ["", ""], TextWidthSpec())


def test_readable_text_width_falls_back_when_metrics_return_non_numeric() -> None:
    width = readable_text_width(
        _ObjectMetrics(),
        ["fallback"],
        TextWidthSpec(padding_px=1),
    )

    assert width == len("fallback") * 7 + 1


def test_text_width_spec_rejects_invalid_contract() -> None:
    spec = TextWidthSpec(minimum_px=200, maximum_px=120)

    with pytest.raises(ValueError, match="maximum_px"):
        spec.validate()


@pytest.mark.parametrize(
    ("field", "spec"),
    [
        ("padding_px", TextWidthSpec(padding_px=-1)),
        ("chrome_px", TextWidthSpec(chrome_px=-1)),
        ("minimum_px", TextWidthSpec(minimum_px=-1)),
    ],
)
def test_text_width_spec_rejects_negative_values(
    field: str,
    spec: TextWidthSpec,
) -> None:
    with pytest.raises(ValueError, match=field):
        spec.validate()
