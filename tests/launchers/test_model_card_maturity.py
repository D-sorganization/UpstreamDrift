"""Maturity chip on launcher tiles uses Sidekick tokens, not literals (#15150)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from src.launchers.model_card_maturity import (
    maturity_chip_height,
    maturity_chip_style,
    maturity_chip_text,
)
from src.shared.python.theme.sidekick_tokens import DEFAULT_SIDEKICK_TOKENS

pytestmark = pytest.mark.unit

PHONE_WIDTH = 390


def _need_qt() -> None:
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")


def _model(maturity: str | None = None, status: str = "ready") -> SimpleNamespace:
    launcher = {"status": status}
    if maturity:
        launcher["maturity"] = maturity
    return SimpleNamespace(launcher=launcher)


def test_stable_has_no_chip_others_do() -> None:
    assert maturity_chip_text("stable") is None
    assert maturity_chip_text("prototype") == "Prototype"
    assert maturity_chip_text("deprecated") == "Deprecated"
    with pytest.raises(ValueError):
        maturity_chip_text("bogus")


def test_style_uses_theme_tokens_only() -> None:
    style = maturity_chip_style("experimental", {"warning": "#123456"})
    assert "#123456" in style
    shipped = set(DEFAULT_SIDEKICK_TOKENS.values())
    assert set(re.findall(r"#[0-9a-fA-F]{6}", style)) <= shipped | {"#123456"}


def test_style_rejects_chipless_level() -> None:
    with pytest.raises(ValueError):
        maturity_chip_style("stable", {})


def test_height_derives_from_line_height() -> None:
    base = maturity_chip_height(12, {})
    assert maturity_chip_height(13, {}) == base + 1
    with pytest.raises(ValueError):
        maturity_chip_height(0, {})


def test_chip_widget_renders_text_and_is_fixed_height(qapp) -> None:
    _need_qt()
    from src.launchers.model_card_maturity import build_maturity_chip

    chip = build_maturity_chip(_model("prototype"), {}, 8)
    assert chip.text() == "Prototype"
    assert chip.minimumHeight() == chip.maximumHeight() > 0
    assert build_maturity_chip(_model(), {}, 8) is None


@pytest.mark.parametrize("width", [1280, 390])
def test_chip_does_not_change_card_row_height(qapp, width: int) -> None:
    _need_qt()
    from unittest.mock import MagicMock

    from src.launchers.model_card import DraggableModelCard

    parent = MagicMock()
    parent.layout_edit_mode = False

    def card_for(maturity: str | None) -> DraggableModelCard:
        model = SimpleNamespace(
            id="t",
            name="A Rather Long Tile Name For Phone",
            description="d",
            engine_type="mujoco",
            type="custom_humanoid",
            path="",
            launcher={
                "status": "ready",
                **({"maturity": maturity} if maturity else {}),
            },
        )
        card = DraggableModelCard(model, parent)
        card.resize(width // 2 if width < 800 else width // 4, 300)
        card.show()
        return card

    plain, tagged = card_for(None), card_for("deprecated")
    assert plain.findChild(type(plain.lbl_name), "MaturityChip") is None
    chip = tagged.findChild(type(tagged.lbl_name), "MaturityChip")
    assert chip is not None
    assert tagged.lbl_name.height() == plain.lbl_name.height()
    assert tagged.minimumSizeHint().width() == plain.minimumSizeHint().width()
    assert tagged.sizeHint().height() == plain.sizeHint().height()
    assert chip.height() == chip.maximumHeight()


@pytest.mark.parametrize("compact", [True, False])
def test_chip_keeps_list_row_phone_sized(qapp, compact: bool) -> None:
    _need_qt()
    from unittest.mock import MagicMock

    from src.launchers.model_card import DraggableModelCard

    parent = MagicMock()
    parent.layout_edit_mode = False

    def min_width(maturity: str | None) -> int:
        launcher = {"status": "ready", **({"maturity": maturity} if maturity else {})}
        model = SimpleNamespace(
            id="t",
            name="Experimental Spine Fit",
            description="A long description",
            engine_type="mujoco",
            type="custom_humanoid",
            path="",
            launcher=launcher,
        )
        card = DraggableModelCard(model, parent, list_mode=True, list_compact=compact)
        return card.minimumSizeHint().width()

    # A chip needs room, but the row must still fit a phone-width window.
    assert min_width("deprecated") <= PHONE_WIDTH
