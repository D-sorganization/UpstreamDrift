"""Tests for launcher_layout_manager."""

import json  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, mock_open, patch  # noqa: E402

import pytest  # noqa: E402

from src.launchers.launcher_layout_manager import (  # noqa: E402
    LayoutConfig,
    LayoutManager,
    compute_centered_geometry,
)
from src.launchers.model_registry import ModelSpec  # noqa: E402


@pytest.fixture
def available_models():
    return {
        "model_1": ModelSpec(
            id="model_1",
            name="Model 1",
            description="A model",
            type="managed",
            path="path1",
        ),
        "model_2": ModelSpec(
            id="model_2",
            name="Model 2",
            description="B model",
            type="managed",
            path="path2",
        ),
    }


@pytest.fixture
def get_model_func(available_models):
    def get_model(model_id):
        return available_models.get(model_id)

    return get_model


@pytest.fixture
def create_card_func():
    def create_card(model):
        card = MagicMock()
        card.model_id = model.id
        return card

    return create_card


@pytest.fixture
def layout_manager(available_models, get_model_func, create_card_func):
    return LayoutManager(
        config_file=Path("/fake/config.json"),
        available_models=available_models,
        get_model_func=get_model_func,
        create_card_func=create_card_func,
    )


def test_compute_centered_geometry():
    # Example 1: Standard layout
    x, y, w, h = compute_centered_geometry(1920, 1080, 1280, 800)
    assert x == (1920 - 1280) // 2
    assert y == (1080 - 800) // 2
    assert w == 1280
    assert h == 800

    # Example 2: Ensure y is capped
    x, y, w, h = compute_centered_geometry(1920, 100, 1280, 800)
    assert y == LayoutConfig.MIN_WINDOW_Y


def test_initialize_model_order(layout_manager):
    # Tests that only available models are initialized
    layout_manager.initialize_model_order(["model_2", "non_existent_model"])
    assert layout_manager.model_order == ["model_2"]


def test_initialize_model_order_empty(layout_manager, available_models):
    # Using defaults logic
    layout_manager.initialize_model_order(None)
    # The default lists a bunch, maybe none are available, so it's empty
    assert len(layout_manager.model_order) == 0


def test_initialize_model_order_all_available(layout_manager, available_models):
    # Test path where there are no missing defaults
    layout_manager.initialize_model_order(["model_1", "model_2"])
    assert layout_manager.model_order == ["model_1", "model_2"]


def test_save_layout_success(layout_manager):
    layout_manager.model_order = ["model_1"]
    window_state = {"selected_model": "model_1", "geometry": {"x": 0}}

    mock_file = mock_open()
    with patch("builtins.open", mock_file), patch.object(Path, "mkdir"):
        layout_manager.save_layout(window_state)

    mock_file.assert_called_once_with(Path("/fake/config.json"), "w", encoding="utf-8")

    # Check that dump was called with correct data
    written_data = "".join(call.args[0] for call in mock_file().write.call_args_list)
    assert "model_1" in written_data


def test_save_layout_oserror(layout_manager):
    window_state = {}
    with patch.object(Path, "mkdir", side_effect=OSError("Boom")):
        # Should catch OSError and just log error
        layout_manager.save_layout(window_state)


def test_load_layout_no_file(layout_manager):
    with patch.object(Path, "exists", return_value=False):
        result = layout_manager.load_layout()
        assert result is None


def test_load_layout_success(layout_manager):
    layout_data = {
        "model_order": ["model_1", "model_2", "missing_model"],
        "selected_model": "model_1",
    }

    with (
        patch.object(Path, "exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(layout_data))),
    ):
        result = layout_manager.load_layout()

        assert result == layout_data
        # model_order should only contain valid / available model ids
        assert layout_manager.model_order == ["model_1", "model_2"]


def test_load_layout_json_error(layout_manager):
    with (
        patch.object(Path, "exists", return_value=True),
        patch("builtins.open", mock_open(read_data="{bad_json")),
    ):
        result = layout_manager.load_layout()
        assert result is None


def test_load_layout_no_valid_order(layout_manager):
    # Test path where saved_order ends up empty
    layout_data = {
        "model_order": ["missing_model"],
        "selected_model": "model_1",
    }
    with (
        patch.object(Path, "exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(layout_data))),
    ):
        result = layout_manager.load_layout()
        assert result == layout_data
        # model_order should remain unchanged
        assert layout_manager.model_order == []


def test_sync_model_cards(layout_manager):
    layout_manager.model_order = ["model_1"]

    # Sync creates a card
    layout_manager.sync_model_cards()
    assert "model_1" in layout_manager.model_cards
    assert "model_2" not in layout_manager.model_cards

    # Add another one and sync
    layout_manager.model_order = ["model_1", "model_2"]
    layout_manager.sync_model_cards()
    assert "model_1" in layout_manager.model_cards
    assert "model_2" in layout_manager.model_cards

    # Remove one and sync
    layout_manager.model_order = ["model_2"]
    old_card = layout_manager.model_cards["model_1"]

    layout_manager.sync_model_cards()
    assert "model_1" not in layout_manager.model_cards
    assert "model_2" in layout_manager.model_cards

    old_card.setParent.assert_called_with(None)
    old_card.deleteLater.assert_called_once()


def test_sync_model_cards_model_missing(layout_manager):
    # Test path where `_get_model` returns None
    layout_manager.model_order = ["missing_model"]
    layout_manager.sync_model_cards()
    assert "missing_model" not in layout_manager.model_cards


def test_apply_model_selection(layout_manager):
    layout_manager.model_order = ["model_1", "model_2"]
    # User selects model 1 only
    layout_manager.apply_model_selection(["model_1"])
    assert layout_manager.model_order == ["model_1"]

    # User adds model 2 back
    layout_manager.apply_model_selection(["model_1", "model_2"])
    assert layout_manager.model_order == ["model_1", "model_2"]

    # User tries to add missing model
    layout_manager.apply_model_selection(["model_1", "missing"])
    assert layout_manager.model_order == ["model_1"]


def test_swap_models(layout_manager):
    layout_manager.model_order = ["model_1", "model_2"]

    # Swap fails if not edit mode
    assert layout_manager.swap_models("model_1", "model_2") is False
    assert layout_manager.model_order == ["model_1", "model_2"]

    # Edit mode on
    layout_manager.edit_mode = True
    assert layout_manager.swap_models("model_1", "model_2") is True
    assert layout_manager.model_order == ["model_2", "model_1"]

    # Swap missing target
    assert layout_manager.swap_models("model_1", "missing_model") is False


def test_get_filtered_order(layout_manager):
    layout_manager.model_order = ["model_1", "model_2"]

    # Empty filter
    assert layout_manager.get_filtered_order() == ["model_1", "model_2"]

    # Filter matches Model 1 name
    layout_manager.update_search_filter("model 1")
    assert layout_manager.get_filtered_order() == ["model_1"]

    # Filter matches both
    layout_manager.update_search_filter("model")
    assert layout_manager.get_filtered_order() == ["model_1", "model_2"]


def test_get_filtered_order_model_missing(layout_manager):
    # Test path where `_get_model` returns None
    layout_manager.model_order = ["missing_model"]
    layout_manager.update_search_filter("missing")
    assert layout_manager.get_filtered_order() == []


def test_rebuild_grid(layout_manager):
    grid_layout = MagicMock()
    # Mock it so that count() goes 1 then 0 to test clearing
    grid_layout.count.side_effect = [1, 0, 0]

    # Setup some fake grid item
    mock_item = MagicMock()
    mock_widget = MagicMock()
    mock_item.widget.return_value = mock_widget
    grid_layout.takeAt.return_value = mock_item

    layout_manager.model_order = ["model_1", "model_2"]
    layout_manager.rebuild_grid(grid_layout)

    # Check that previous widget was cleared
    mock_widget.setParent.assert_called_with(None)

    # grid_layout should have addWidget called twice
    assert grid_layout.addWidget.call_count == 2


def test_rebuild_grid_no_widget(layout_manager):
    # Test when item has no widget, and when takeAt returns None
    grid_layout = MagicMock()
    grid_layout.count.side_effect = [2, 1, 0]

    mock_item = MagicMock()
    mock_item.widget.return_value = None
    # First returns item without widget, second returns None directly
    grid_layout.takeAt.side_effect = [mock_item, None]

    layout_manager.model_order = []
    layout_manager.rebuild_grid(grid_layout)
    mock_item.widget.assert_called_once()


def test_rebuild_grid_missing_model(layout_manager):
    # Test paths where model creation fails or is skipped
    grid_layout = MagicMock()
    grid_layout.count.return_value = 0

    layout_manager.model_order = ["missing_model"]
    layout_manager.rebuild_grid(grid_layout)
    assert grid_layout.addWidget.call_count == 0


def test_rebuild_grid_existing_card(layout_manager):
    # Test path where card is already in model_cards
    grid_layout = MagicMock()
    grid_layout.count.return_value = 0

    # Pre-populate card
    mock_card = MagicMock()
    layout_manager.model_cards["model_1"] = mock_card
    layout_manager.model_order = ["model_1"]

    with patch.object(layout_manager, "_create_card") as mock_create:
        layout_manager.rebuild_grid(grid_layout)
        mock_create.assert_not_called()
        grid_layout.addWidget.assert_called_once_with(mock_card, 0, 0)


def test_rebuild_grid_multiple_columns(layout_manager, available_models):
    # Add dummy models to trigger column wrap
    for i in range(5):
        available_models[f"model_{i + 3}"] = MagicMock()

    grid_layout = MagicMock()
    grid_layout.count.return_value = 0
    layout_manager.model_order = ["model_1", "model_2", "model_3", "model_4", "model_5"]
    layout_manager.rebuild_grid(grid_layout)

    # Check that it wrapped around
    assert grid_layout.addWidget.call_count == 5
    # The last call should be row=1, col=0 because GRID_COLUMNS=4
    last_call = grid_layout.addWidget.call_args_list[-1]
    assert last_call[0][1] == 1  # row
    assert last_call[0][2] == 0  # col


def test_set_edit_mode(layout_manager):
    layout_manager.model_cards["model_1"] = MagicMock()
    layout_manager.set_edit_mode(True)
    assert layout_manager.edit_mode is True
    layout_manager.model_cards["model_1"].setAcceptDrops.assert_called_with(True)
