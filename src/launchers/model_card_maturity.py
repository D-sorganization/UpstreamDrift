"""Maturity chip for launcher tiles (epic #15150, E7).

Colours come from the Sidekick design tokens derived from the active theme,
never literals. PyQt6 is imported lazily so the module and its pure helpers
load headless.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.shared.python.config.maturity_scale import (
    chip_hidden_levels,
    resolve_model_maturity,
    validate_level,
)
from src.shared.python.theme.sidekick_tokens import sidekick_tokens_from_theme

#: Sidekick colour token that tints each maturity level.
CHIP_TOKENS: Mapping[str, str] = {
    "experimental": "sidekick.color.warning",
    "prototype": "sidekick.color.info",
    "beta": "sidekick.color.accent",
    "deprecated": "sidekick.color.error",
}
CHIP_TEXT_TOKEN = "sidekick.color.text"
CHIP_RADIUS_TOKEN = "sidekick.radius.lg"
CHIP_PAD_TOKENS = ("sidekick.space.1", "sidekick.space.2")


def maturity_chip_text(maturity: str) -> str | None:
    """Chip label for ``maturity`` or None when that level draws no chip.

    Raises:
        ValueError: ``maturity`` is not a canonical level.
    """
    if validate_level(maturity) in chip_hidden_levels():
        return None
    return maturity.capitalize()


def maturity_chip_style(maturity: str, colors: Mapping[str, str]) -> str:
    """Stylesheet for the chip, built only from Sidekick tokens."""
    tokens = sidekick_tokens_from_theme(colors)
    tint = tokens[CHIP_TOKENS[validate_level(maturity)]]
    pad_v, pad_h = (tokens[t] for t in CHIP_PAD_TOKENS)
    return (
        "QLabel#MaturityChip {"
        f" color: {tokens[CHIP_TEXT_TOKEN]};"
        f" border: 1px solid {tint};"
        f" border-radius: {tokens[CHIP_RADIUS_TOKEN]};"
        f" padding: {pad_v} {pad_h};"
        "}"
    )


def build_maturity_chip(model: Any, colors: Mapping[str, str], point_size: int) -> Any:
    """Return a ``QLabel`` chip for ``model`` or None for a stable tile."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel

    from src.shared.python.theme.typography import Weights, get_qfont

    maturity = resolve_model_maturity(model)
    text = maturity_chip_text(maturity)
    if text is None:
        return None
    chip = QLabel(text)
    chip.setObjectName("MaturityChip")
    chip.setFont(get_qfont(size=point_size, weight=Weights.BOLD))
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    chip.setToolTip(f"Maturity: {maturity}")
    chip.setStyleSheet(maturity_chip_style(maturity, colors))
    return chip


def add_maturity_chip(
    layout: Any, model: Any, colors: Mapping[str, str], point_size: int
) -> None:
    """Append the maturity chip to ``layout`` unless the tile is stable."""
    chip = build_maturity_chip(model, colors, point_size)
    if chip is not None:
        layout.addWidget(chip)
