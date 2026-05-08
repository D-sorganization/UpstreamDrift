"""Tests for the view-mode aware behaviour of :class:`LayoutManager`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.launchers.launcher_constants import ViewMode, view_mode_settings
from src.launchers.launcher_layout_manager import LayoutManager
from src.launchers.model_registry import ModelSpec


@pytest.fixture
def available_models() -> dict[str, ModelSpec]:
    return {
        f"model_{i}": ModelSpec(
            id=f"model_{i}",
            name=f"Model {i}",
            description=f"Desc {i}",
            type="managed",
            path=f"path_{i}",
        )
        for i in range(1, 5)
    }


@pytest.fixture
def get_model_func(available_models) -> Callable[[str], ModelSpec | None]:
    return lambda mid: available_models.get(mid)


@pytest.fixture
def make_layout_manager(available_models, get_model_func):
    def _make() -> LayoutManager:
        # Cards are MagicMocks that record set_tile_scale + carry _list_mode.
        def create_card(model, **kwargs):
            card = MagicMock()
            card.model_id = model.id
            card._list_mode = bool(kwargs.get("list_mode", False))
            card.tile_scale = float(kwargs.get("tile_scale", 0.5))
            return card

        return LayoutManager(
            config_file=Path("/fake/cfg.json"),
            available_models=available_models,
            get_model_func=get_model_func,
            create_card_func=create_card,
            create_header_func=lambda name: MagicMock(),
        )

    return _make


@pytest.mark.parametrize(
    "mode,expected_scale,expected_columns,expected_show_desc,expected_list",
    [
        (ViewMode.COMFORTABLE, 1.0, 4, True, False),
        (ViewMode.COMPACT, 0.5, 6, True, False),
        (ViewMode.DENSE, 0.35, 8, False, False),
        (ViewMode.LIST, 0.30, 1, True, True),
    ],
)
def test_view_mode_table_matches_spec(
    mode: ViewMode,
    expected_scale: float,
    expected_columns: int,
    expected_show_desc: bool,
    expected_list: bool,
) -> None:
    scale, cols, show_desc, is_list = view_mode_settings(mode)
    assert scale == expected_scale
    assert cols == expected_columns
    assert show_desc is expected_show_desc
    assert is_list is expected_list


@pytest.mark.parametrize(
    "mode,expected_scale,expected_columns",
    [
        (ViewMode.COMFORTABLE, 1.0, 4),
        (ViewMode.COMPACT, 0.5, 6),
        (ViewMode.DENSE, 0.35, 8),
        (ViewMode.LIST, 0.30, 1),
    ],
)
def test_set_view_mode_updates_state(
    make_layout_manager, mode, expected_scale, expected_columns
) -> None:
    lm = make_layout_manager()
    lm.set_view_mode(mode)
    assert lm.current_view_mode == mode
    assert lm.tile_scale == expected_scale
    # Columns are read from the table at rebuild time.
    _scale, cols, _sd, _il = view_mode_settings(lm.current_view_mode)
    assert cols == expected_columns


def test_set_view_mode_rejects_non_enum(make_layout_manager) -> None:
    lm = make_layout_manager()
    with pytest.raises(TypeError):
        lm.set_view_mode("compact")  # type: ignore[arg-type]


def test_list_mode_yields_one_card_per_row(make_layout_manager) -> None:
    lm = make_layout_manager()
    lm.model_order = ["model_1", "model_2", "model_3", "model_4"]
    lm.set_view_mode(ViewMode.LIST)

    grid_layout = MagicMock()
    grid_layout.count.return_value = 0

    lm.rebuild_grid(grid_layout)

    # In LIST mode each card is added as addWidget(widget, row, 0, 1, 1).
    # The header is added as addWidget(header, row, 0, 1, 1) as well (LIST has
    # column count 1). Distinguish by widget identity: cards are MagicMocks
    # registered in lm.model_cards.
    cards = {id(c) for c in lm.model_cards.values()}
    rows_seen = set()
    cols_seen = set()
    for call in grid_layout.addWidget.call_args_list:
        if len(call[0]) >= 5 and id(call[0][0]) in cards:
            _w, row, col, _rs, _cs = call[0][:5]
            rows_seen.add(row)
            cols_seen.add(col)
    # Four cards => four distinct rows, all at column 0
    assert len(rows_seen) == 4
    assert cols_seen == {0}


def test_grid_mode_columns_drive_wrap(make_layout_manager) -> None:
    lm = make_layout_manager()
    # Use 8 cards to ensure wrap in COMFORTABLE (4 cols) -> 2 rows of cards.
    for i in range(5, 9):
        lm.available_models[f"model_{i}"] = ModelSpec(
            id=f"model_{i}",
            name=f"Model {i}",
            description=f"Desc {i}",
            type="managed",
            path=f"path_{i}",
        )
    lm.model_order = [f"model_{i}" for i in range(1, 9)]
    lm.set_view_mode(ViewMode.COMFORTABLE)

    grid_layout = MagicMock()
    grid_layout.count.return_value = 0
    lm.rebuild_grid(grid_layout)

    # COMFORTABLE => 4 columns; the last card (8th) should land at col == 3.
    # Grid-mode cards are added as addWidget(widget, row, col) (3 args).
    cards = {id(c) for c in lm.model_cards.values()}
    cols_seen = set()
    for call in grid_layout.addWidget.call_args_list:
        if len(call[0]) == 3 and id(call[0][0]) in cards:
            cols_seen.add(call[0][2])
    assert max(cols_seen) == 3
