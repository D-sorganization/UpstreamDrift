"""Tests for view-mode + tile-scale persistence in :class:`LayoutManager`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.launchers.launcher_constants import ViewMode
from src.launchers.launcher_layout_manager import LayoutManager
from src.launchers.model_registry import ModelSpec


@pytest.fixture
def available_models() -> dict[str, ModelSpec]:
    return {
        "model_1": ModelSpec(
            id="model_1",
            name="Model 1",
            description="A",
            type="managed",
            path="p1",
        ),
    }


@pytest.fixture
def make_layout_manager(available_models, tmp_path: Path):
    def _make(config_file: Path | None = None) -> LayoutManager:
        cfg = config_file or (tmp_path / "launcher_layout.json")

        def create_card(model, **kwargs):
            card = MagicMock()
            card.tile_scale = float(kwargs.get("tile_scale", 0.5))
            card._list_mode = bool(kwargs.get("list_mode", False))
            return card

        return LayoutManager(
            config_file=cfg,
            available_models=available_models,
            get_model_func=lambda mid: available_models.get(mid),
            create_card_func=create_card,
            create_header_func=lambda name: MagicMock(),
        )

    return _make


def test_save_and_reload_view_mode_and_scale(make_layout_manager, tmp_path) -> None:
    cfg = tmp_path / "launcher_layout.json"
    lm = make_layout_manager(cfg)
    lm.model_order = ["model_1"]
    lm.set_view_mode(ViewMode.DENSE)  # sets tile_scale=0.35
    lm.save_layout({"selected_model": "model_1", "geometry": {}})

    payload = json.loads(cfg.read_text())
    assert payload["view_mode"] == "dense"
    assert payload["tile_scale"] == pytest.approx(0.35)

    lm2 = make_layout_manager(cfg)
    lm2.load_layout()
    assert lm2.current_view_mode == ViewMode.DENSE
    assert lm2.tile_scale == pytest.approx(0.35)


def test_load_layout_missing_keys_uses_defaults(make_layout_manager, tmp_path) -> None:
    cfg = tmp_path / "old_layout.json"
    cfg.write_text(json.dumps({"model_order": ["model_1"]}))

    lm = make_layout_manager(cfg)
    lm.load_layout()
    # Default view_mode is COMPACT; default tile_scale unchanged from init.
    assert lm.current_view_mode == ViewMode.COMPACT
    # tile_scale defaults to TILE_SCALE_DEFAULT (0.5) on the manager
    assert lm.tile_scale == pytest.approx(0.5)


def test_load_layout_invalid_view_mode_falls_back(
    make_layout_manager, tmp_path
) -> None:
    cfg = tmp_path / "bad_mode.json"
    cfg.write_text(
        json.dumps(
            {
                "model_order": ["model_1"],
                "view_mode": "BOGUS",
                "tile_scale": 0.75,
            }
        )
    )
    lm = make_layout_manager(cfg)
    lm.load_layout()
    assert lm.current_view_mode == ViewMode.COMPACT
    assert lm.tile_scale == pytest.approx(0.75)


def test_load_layout_invalid_tile_scale_skipped(make_layout_manager, tmp_path) -> None:
    cfg = tmp_path / "bad_scale.json"
    cfg.write_text(
        json.dumps(
            {
                "model_order": ["model_1"],
                "view_mode": "compact",
                "tile_scale": -2.0,
            }
        )
    )
    lm = make_layout_manager(cfg)
    lm.load_layout()
    # Invalid value rejected -> tile_scale remains at the constructor default.
    assert lm.tile_scale == pytest.approx(0.5)
