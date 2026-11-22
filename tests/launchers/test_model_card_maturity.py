"""Maturity chip on launcher tiles uses Sidekick tokens, not literals (#15150)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from src.launchers.model_card_maturity import (
    maturity_chip_style,
    maturity_chip_text,
)
from src.shared.python.theme.sidekick_tokens import DEFAULT_SIDEKICK_TOKENS

pytestmark = pytest.mark.unit


def test_stable_has_no_chip_others_do() -> None:
    assert maturity_chip_text("stable") is None
    assert maturity_chip_text("prototype") == "Prototype"
    assert maturity_chip_text("deprecated") == "Deprecated"
    with pytest.raises(ValueError):
        maturity_chip_text("bogus")


def test_style_uses_theme_tokens_only() -> None:
    colors = {"warning": "#123456"}
    style = maturity_chip_style("experimental", colors)
    assert "#123456" in style
    shipped = set(DEFAULT_SIDEKICK_TOKENS.values())
    literals = set(re.findall(r"#[0-9a-fA-F]{6}", style))
    assert literals <= shipped | {"#123456"}


def test_chip_widget_renders_text() -> None:
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    from src.launchers.model_card_maturity import build_maturity_chip

    app = QApplication.instance() or QApplication([])
    assert app is not None
    model = SimpleNamespace(launcher={"maturity": "prototype", "status": "ready"})
    chip = build_maturity_chip(model, {}, 8)
    assert chip.text() == "Prototype"
    assert (
        build_maturity_chip(SimpleNamespace(launcher={"status": "ready"}), {}, 8)
        is None
    )
