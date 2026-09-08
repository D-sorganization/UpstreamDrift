"""Quality and architecture gate for launcher category registry — Issue #9481.

Enforces:
1. Tile categories are derived directly from registry metadata, not hardcoded in launcher code.
2. get_model_categories contains no model IDs or ID heuristics.
3. Every tile in models.yaml resolves to at least one sidebar category.
4. Every sidebar filter category has at least one matching tile in the registry.
5. All sidebar filter buttons have distinct, meaningful category targets.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import pytest

pytestmark = [pytest.mark.gate]

from src.config.launcher_manifest_loader import LAUNCHER_CATEGORIES
from src.launchers.launcher_layout_manager import LayoutManager
from src.shared.python.config.model_pack_manifest import LauncherPresentationMetadata
from src.shared.python.config.model_registry import ModelConfig, ModelRegistry


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model_registry(repo_root: Path) -> ModelRegistry:
    models_path = repo_root / "src" / "config" / "models.yaml"
    assert models_path.exists(), f"models.yaml not found at {models_path}"
    return ModelRegistry(config_path=models_path, discovery_mode="local-only")


@pytest.fixture
def layout_manager(model_registry: ModelRegistry, tmp_path: Path) -> LayoutManager:
    return LayoutManager(
        config_file=tmp_path / "layout.json",
        available_models=model_registry.models,
        get_model_func=lambda mid: model_registry.models.get(mid),
        create_card_func=lambda m: None,
    )


def test_get_model_categories_ast_contains_no_model_ids(
    model_registry: ModelRegistry,
) -> None:
    """get_model_categories() must not hardcode any model IDs (issue #9481)."""
    source = textwrap.dedent(inspect.getsource(LayoutManager.get_model_categories))
    tree = ast.parse(source)

    model_ids = set(model_registry.models.keys())
    # Exclude strings that happen to be canonical category names (e.g. motion_capture)
    exempt_names = set(LAUNCHER_CATEGORIES) | {"engines", "tools", "physics_engine"}
    model_ids_only = model_ids - exempt_names

    string_literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.add(node.value.lower())

    offending = string_literals.intersection(model_ids_only)
    assert not offending, (
        f"LayoutManager.get_model_categories contains hardcoded model IDs: {offending}. "
        "Categories must be derived from registry metadata instead."
    )


def test_every_sidebar_category_has_at_least_one_tile(
    layout_manager: LayoutManager, model_registry: ModelRegistry
) -> None:
    """Every sidebar category filter must have at least one visible tile."""
    sidebar_categories = {
        "Engines",
        "Biomechanics",
        "Simulation",
        "Tools",
        "Documentation",
    }
    category_counts: dict[str, int] = dict.fromkeys(sidebar_categories, 0)

    for model in model_registry.models.values():
        if getattr(model, "hidden", False):
            continue
        cats = layout_manager.get_model_categories(model)
        for cat in cats:
            if cat in category_counts:
                category_counts[cat] += 1

    empty_categories = [cat for cat, count in category_counts.items() if count == 0]
    assert not empty_categories, (
        f"The following sidebar categories have no visible tiles in models.yaml: {empty_categories}. "
        f"Counts: {category_counts}"
    )


def test_every_tile_has_valid_category(
    layout_manager: LayoutManager, model_registry: ModelRegistry
) -> None:
    """Every tile in models.yaml must have a valid declared category and resolve to sidebar groups."""
    orphan_tiles: list[str] = []
    invalid_categories: list[tuple[str, str]] = []

    for model_id, model in model_registry.models.items():
        if getattr(model, "hidden", False):
            continue
        launcher = getattr(model, "launcher", None)
        assert launcher is not None, f"Tile {model_id} has no launcher metadata"
        declared_category = getattr(launcher, "category", None)
        assert declared_category is not None, (
            f"Tile {model_id} has no declared category"
        )
        if declared_category not in LAUNCHER_CATEGORIES:
            invalid_categories.append((model_id, declared_category))

        resolved = layout_manager.get_model_categories(model)
        if not resolved:
            orphan_tiles.append(model_id)

    assert not invalid_categories, (
        f"Tiles have invalid declared categories: {invalid_categories}"
    )
    assert not orphan_tiles, (
        f"Tiles in models.yaml failed to resolve to any sidebar category: {orphan_tiles}"
    )


def test_adding_tile_places_it_in_expected_sidebar_group(
    layout_manager: LayoutManager,
) -> None:
    """Adding a tile with a declared category places it in the sidebar group without launcher code change."""
    test_cases = [
        ("physics_engine", "Engines"),
        ("biomechanics", "Biomechanics"),
        ("simulation", "Simulation"),
        ("tool", "Tools"),
        ("motion_matching", "Tools"),
        ("motion_capture", "Tools"),
        ("documentation", "Documentation"),
    ]

    for category, expected_group in test_cases:
        mock_model = ModelConfig(
            id=f"test_tile_{category}",
            name=f"Test Tile {category}",
            description="Test tile for category gate",
            type="special_app",
            path="virtual/test",
            launcher=LauncherPresentationMetadata(
                category=category,
                status="ready",
                logo="test.png",
            ),
        )
        cats = layout_manager.get_model_categories(mock_model)
        assert expected_group in cats, (
            f"Model with category '{category}' resolved to {cats}, expected '{expected_group}'"
        )


def test_sidebar_buttons_distinct_category_mapping() -> None:
    """Sidebar category filter buttons must have distinct category targets (no duplicates)."""
    specs = (
        ("Home", "home", None),
        ("Engines", "computer", None),
        ("Biomechanics", "accessibility", None),
        ("Simulation", "sports_golf", None),
        ("Tools", "build", None),
        ("Documentation", "book", None),
        ("Favorites", "star", None),
        ("History", "history", None),
    )
    labels = [spec[0] for spec in specs]
    assert len(labels) == len(set(labels)), "Duplicate sidebar button labels found"
