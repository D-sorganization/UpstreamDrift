"""Responsive sizing helpers for pendulum simulator GUI widgets."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox

from src.shared.python.theme.responsive import TextWidthSpec, set_text_minimum_width

OVERLAY_CHECKBOX_WIDTH = TextWidthSpec(
    padding_px=8,
    chrome_px=30,
    minimum_px=130,
)


def apply_overlay_checkbox_sizing(checkbox: QCheckBox) -> int:
    """Keep overlay checkboxes readable while allowing layout expansion."""
    if checkbox is None:
        raise ValueError("checkbox must be provided")
    text = checkbox.text()
    return set_text_minimum_width(
        checkbox,
        OVERLAY_CHECKBOX_WIDTH,
        texts=[text if isinstance(text, str) else str(text)],
    )
