"""Maturity chip for launcher tiles (epic #15150, G13).

Colours, radius and padding are Sidekick design-token names listed in
``model_card_maturity.json`` and resolved against the active theme. Which
levels draw no chip comes from ``maturity_scale.chip_hidden_levels``. PyQt6
is imported lazily so the module and its pure helpers load headless.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.shared.python.config.maturity_scale import (
    chip_hidden_levels,
    resolve_model_maturity,
    validate_level,
)
from src.shared.python.theme.sidekick_tokens import sidekick_tokens_from_theme

SETTINGS_PATH = Path(__file__).with_name("model_card_maturity.json")


@functools.cache
def _settings() -> Mapping[str, Any]:
    with SETTINGS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def _px(value: str) -> int:
    """Integer pixels of a ``"4px"`` token value."""
    return int(value.removesuffix("px"))


def maturity_chip_text(maturity: str) -> str | None:
    """Chip label for ``maturity`` or None when that level draws no chip.

    Raises:
        ValueError: ``maturity`` is not a canonical level.
    """
    if validate_level(maturity) in chip_hidden_levels():
        return None
    return maturity.capitalize()


def maturity_chip_style(maturity: str, colors: Mapping[str, str]) -> str:
    """Stylesheet for the chip, built only from Sidekick tokens.

    Raises:
        ValueError: ``maturity`` is not a canonical level or draws no chip.
    """
    cfg = _settings()
    tint_token = cfg["level_color_tokens"].get(validate_level(maturity))
    if tint_token is None:
        raise ValueError(f"maturity {maturity!r} draws no chip")
    tokens = sidekick_tokens_from_theme(colors)
    return (
        "QLabel#MaturityChip {"
        f" color: {tokens[cfg['text_token']]};"
        f" border: {cfg['border_width_px']}px solid {tokens[tint_token]};"
        f" border-radius: {tokens[cfg['radius_token']]};"
        f" padding: 0px {tokens[cfg['pad_h_token']]};"
        "}"
    )


def maturity_chip_height(line_height: int, colors: Mapping[str, str]) -> int:
    """Fixed chip height: font line height plus vertical padding and border."""
    if line_height <= 0:
        raise ValueError(f"line_height must be positive, got {line_height}")
    cfg = _settings()
    pad_v = _px(sidekick_tokens_from_theme(colors)[cfg["pad_v_token"]])
    return line_height + 2 * (pad_v + int(cfg["border_width_px"]))


def _chip_font(point_size: int) -> Any:
    from src.shared.python.theme.typography import Weights, get_qfont

    return get_qfont(size=point_size, weight=Weights.BOLD)


def reserved_chip_height(colors: Mapping[str, str], point_size: int) -> int:
    """Height a chip occupies at ``point_size`` (also reserved for stable tiles)."""
    from PyQt6.QtGui import QFontMetrics

    line = QFontMetrics(_chip_font(point_size)).height()
    return maturity_chip_height(line, colors)


def build_maturity_chip(model: Any, colors: Mapping[str, str], point_size: int) -> Any:
    """Return a fixed-height ``QLabel`` chip for ``model`` or None if stable."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QSizePolicy

    maturity = resolve_model_maturity(model)
    text = maturity_chip_text(maturity)
    if text is None:
        return None
    chip = QLabel(text)
    chip.setObjectName("MaturityChip")
    chip.setFont(_chip_font(point_size))
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    chip.setToolTip(f"Maturity: {maturity}")
    chip.setStyleSheet(maturity_chip_style(maturity, colors))
    chip.setWordWrap(False)
    chip.setFixedHeight(reserved_chip_height(colors, point_size))
    chip.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return chip


def add_maturity_chip(
    layout: Any, model: Any, colors: Mapping[str, str], point_size: int
) -> None:
    """Append the maturity chip to a list-row ``layout`` unless stable."""
    chip = build_maturity_chip(model, colors, point_size)
    if chip is not None:
        layout.addWidget(chip)


def add_maturity_row(
    layout: Any, model: Any, colors: Mapping[str, str], point_size: int
) -> None:
    """Add a centred chip row under a grid card's status chip.

    A stable tile gets an empty spacer of the same height so every card in a
    grid has identical row geometry, and the chip never widens the card.
    """
    from PyQt6.QtWidgets import QHBoxLayout, QWidget

    chip = build_maturity_chip(model, colors, point_size)
    if chip is None:
        # A placeholder widget (unlike a spacer item or empty layout) is not
        # "empty" to Qt, so it keeps the same inter-item spacing as a chip row.
        placeholder = QWidget()
        placeholder.setFixedHeight(reserved_chip_height(colors, point_size))
        placeholder.setStyleSheet("background: transparent;")
        layout.addWidget(placeholder)
        return
    row = QHBoxLayout()
    row.addStretch()
    row.addWidget(chip)
    row.addStretch()
    layout.addLayout(row)
