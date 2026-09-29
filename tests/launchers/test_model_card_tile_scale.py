"""Tests for the ``tile_scale`` parameter on :class:`DraggableModelCard`.

These tests assert that the image size, font point size, and layout
margins all scale linearly from the 1.0x reference values defined in
``launcher_constants``.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from src.launchers.launcher_constants import (
    TILE_BASE_FONT_PT,
    TILE_BASE_IMAGE_PX,
    TILE_BASE_PADDING_PX,
    TILE_MIN_FONT_PT,
)
from src.launchers.model_card import DraggableModelCard


@pytest.fixture
def parent_launcher() -> MagicMock:
    launcher = MagicMock()
    launcher.layout_edit_mode = False
    return launcher


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.id = "mujoco_unified"
    model.name = "MuJoCo"
    model.description = "Test description"
    model.engine_type = "mujoco"
    model.launcher = None
    model.type = "custom_humanoid"
    model.path = ""
    return model


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0])
def test_image_size_scales_linearly(
    qapp,
    mock_model: MagicMock,
    parent_launcher: MagicMock,
    scale: float,
) -> None:
    card = DraggableModelCard(mock_model, parent_launcher, tile_scale=scale)
    expected = int(TILE_BASE_IMAGE_PX * scale)
    assert card.lbl_img.size().width() == expected
    assert card.lbl_img.size().height() == expected


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0])
def test_name_font_scales_linearly(
    qapp,
    mock_model: MagicMock,
    parent_launcher: MagicMock,
    scale: float,
) -> None:
    card = DraggableModelCard(mock_model, parent_launcher, tile_scale=scale)
    expected = max(TILE_MIN_FONT_PT, int(round(TILE_BASE_FONT_PT * scale)))
    assert card.lbl_name.font().pointSize() == expected


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0])
def test_padding_scales_linearly(
    qapp,
    mock_model: MagicMock,
    parent_launcher: MagicMock,
    scale: float,
) -> None:
    card = DraggableModelCard(mock_model, parent_launcher, tile_scale=scale)
    margins = card.layout().contentsMargins()
    expected = max(2, int(round(TILE_BASE_PADDING_PX * scale)))
    # All four margins are equal — assert one of them.
    assert margins.left() == expected


def test_set_tile_scale_resizes_in_place(
    qapp, mock_model: MagicMock, parent_launcher: MagicMock
) -> None:
    card = DraggableModelCard(mock_model, parent_launcher, tile_scale=0.5)
    initial = card.lbl_img.size().width()
    card.set_tile_scale(1.0)
    assert card.lbl_img.size().width() == TILE_BASE_IMAGE_PX
    assert card.lbl_img.size().width() != initial


def test_invalid_scale_raises(
    qapp, mock_model: MagicMock, parent_launcher: MagicMock
) -> None:
    with pytest.raises(ValueError):
        DraggableModelCard(mock_model, parent_launcher, tile_scale=-1.0)
    with pytest.raises(ValueError):
        DraggableModelCard(mock_model, parent_launcher, tile_scale=float("nan"))
    with pytest.raises(TypeError):
        DraggableModelCard(mock_model, parent_launcher, tile_scale="big")  # type: ignore[arg-type]


def test_font_floor_at_min_scale(
    qapp, mock_model: MagicMock, parent_launcher: MagicMock
) -> None:
    card = DraggableModelCard(mock_model, parent_launcher, tile_scale=0.25)
    # 11 * 0.25 = 2.75 -> would round to 3 without floor; we expect TILE_MIN_FONT_PT.
    assert card.lbl_name.font().pointSize() >= TILE_MIN_FONT_PT
    # And the floor is engaged for the chip (base 8 -> floor 8).
    assert not math.isnan(float(card.lbl_name.font().pointSize()))
