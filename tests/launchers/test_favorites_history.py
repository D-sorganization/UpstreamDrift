"""Tests for Favorites and History features in LayoutManager and ModelCard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.launchers.launcher_layout_manager import LayoutManager
from src.launchers.model_registry import ModelSpec


@pytest.fixture
def available_models() -> dict[str, ModelSpec]:
    return {
        "mujoco_unified": ModelSpec(
            id="mujoco_unified",
            name="MuJoCo",
            description="Physics simulator",
            type="custom_humanoid",
            path="p1",
            launcher={"category": "physics_engine"},
        ),
        "movement_optimizer": ModelSpec(
            id="movement_optimizer",
            name="Movement Optimizer",
            description="Bio tools",
            type="special_app",
            path="p2",
            launcher={"category": "biomechanics"},
        ),
        "library_tool": ModelSpec(
            id="library_tool",
            name="Library",
            description="Research library",
            type="special_app",
            path="p3",
            launcher={"category": "documentation"},
        ),
        "putting_green": ModelSpec(
            id="putting_green",
            name="Putting Green",
            description="Putting simulation",
            type="putting_green",
            path="p4",
            launcher={"category": "simulation"},
        ),
    }


@pytest.fixture
def make_layout_manager(available_models, tmp_path: Path):
    def _make(config_file: Path | None = None) -> LayoutManager:
        cfg = config_file or (tmp_path / "launcher_layout.json")

        def create_card(model, **kwargs):
            card = MagicMock()
            card.model = model
            return card

        return LayoutManager(
            config_file=cfg,
            available_models=available_models,
            get_model_func=lambda mid: available_models.get(mid),
            create_card_func=create_card,
            create_header_func=lambda name: MagicMock(),
        )

    return _make


def test_favorites_load_save(make_layout_manager, tmp_path) -> None:
    cfg = tmp_path / "launcher_layout.json"
    lm = make_layout_manager(cfg)
    lm.model_order = ["mujoco_unified", "movement_optimizer"]

    # Check initial favorites state
    assert lm.favorites == []

    # Set and save favorites
    lm.favorites = ["mujoco_unified"]
    lm.save_layout({"selected_model": "mujoco_unified", "geometry": {}})

    # Reload and verify
    lm2 = make_layout_manager(cfg)
    lm2.load_layout()
    assert lm2.favorites == ["mujoco_unified"]


def test_launch_stats_load_save(make_layout_manager, tmp_path) -> None:
    cfg = tmp_path / "launcher_layout.json"
    lm = make_layout_manager(cfg)

    lm.record_launch("mujoco_unified")
    assert lm.launch_stats["mujoco_unified"]["count"] == 1
    assert lm.launch_stats["mujoco_unified"]["last_launched"] != ""

    lm.save_layout({"selected_model": "mujoco_unified", "geometry": {}})

    lm2 = make_layout_manager(cfg)
    lm2.load_layout()
    assert lm2.launch_stats["mujoco_unified"]["count"] == 1


def test_favorites_filter(make_layout_manager) -> None:
    lm = make_layout_manager()
    lm.model_order = ["mujoco_unified", "movement_optimizer", "putting_green"]
    lm.favorites = ["movement_optimizer"]

    # Filter by Favorites
    lm.current_category_filter = "Favorites"
    filtered = lm.get_filtered_order()
    assert filtered == ["movement_optimizer"]


def test_history_sorting(make_layout_manager) -> None:
    lm = make_layout_manager()
    lm.model_order = ["mujoco_unified", "movement_optimizer", "putting_green"]

    # Record launches: putting_green (2 times), mujoco_unified (1 time)
    lm.record_launch("putting_green")
    lm.record_launch("putting_green")
    lm.record_launch("mujoco_unified")

    # Filter by History
    lm.current_category_filter = "History"
    filtered = lm.get_filtered_order()

    # Putting green should be first (2 runs), then mujoco_unified (1 run), then unlaunched movement_optimizer
    assert filtered[0] == "putting_green"
    assert filtered[1] == "mujoco_unified"
    assert filtered[2] == "movement_optimizer"


def test_model_categories_flexible_mapping(make_layout_manager) -> None:
    lm = make_layout_manager()

    mujoco = lm._get_model("mujoco_unified")
    m_opt = lm._get_model("movement_optimizer")
    library = lm._get_model("library_tool")

    # Check Engines mapping
    assert "Engines" in lm.get_model_categories(mujoco)

    # Check flexible multi-category mapping for movement_optimizer (Biomechanics & Tools)
    m_opt_cats = lm.get_model_categories(m_opt)
    assert "Biomechanics" in m_opt_cats
    assert "Tools" in m_opt_cats

    # Check Documentation mapping
    assert "Documentation" in lm.get_model_categories(library)
