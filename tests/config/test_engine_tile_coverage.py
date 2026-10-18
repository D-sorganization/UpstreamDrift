"""Engine tile coverage and JaxSim launcher availability tests (#12998).

Verifies that all supported EngineType enum members and ENGINE_TIERS have
corresponding launcher tiles or explicit documented exemptions, and that
JaxSim specifically has proper engine availability probing, configuration,
and dockable launcher UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.config.capability_state import KNOWN_PHYSICS_ENGINES
from src.engines.tiers import ENGINE_TIERS
from src.shared.python.engine_core.engine_registry import EngineType


@pytest.mark.unit
def test_every_engine_type_has_launcher_tile_or_exemption() -> None:
    """Every EngineType and ENGINE_TIERS entry should map to a launcher tile or documented alias."""
    root = Path(__file__).resolve().parent.parent.parent

    # Read launcher manifest tiles
    manifest_path = root / "src" / "config" / "launcher_manifest.json"
    assert manifest_path.is_file(), f"Missing launcher manifest: {manifest_path}"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Read models.yaml tiles
    models_path = root / "src" / "config" / "models.yaml"
    assert models_path.is_file(), f"Missing models config: {models_path}"
    with models_path.open("r", encoding="utf-8") as f:
        models_data = yaml.safe_load(f)

    # Read inventory launcher_tile_map
    inventory_path = root / "src" / "config" / "engine_model_inventory.json"
    assert inventory_path.is_file(), f"Missing inventory: {inventory_path}"
    with inventory_path.open("r", encoding="utf-8") as f:
        inventory_data = json.load(f)
    inventory_tile_map = inventory_data.get("launcher_tile_map", {})

    manifest_tile_ids: set[str] = {
        tile["id"] for tile in manifest_data.get("tiles", [])
    }
    manifest_engine_types: set[str] = {
        tile.get("engine_type")
        for tile in manifest_data.get("tiles", [])
        if tile.get("engine_type")
    }

    models_tile_ids: set[str] = {
        model["id"] for model in models_data.get("models", []) if "id" in model
    }
    models_engine_types: set[str] = {
        model.get("engine_type")
        for model in models_data.get("models", [])
        if model.get("engine_type")
    }

    # Engines covered by manifest or models.yaml
    covered_engines: set[str] = manifest_engine_types | models_engine_types

    # Add id matching engine names and tile map targets
    for tile_id, eng_key in inventory_tile_map.items():
        if tile_id in manifest_tile_ids or tile_id in models_tile_ids:
            covered_engines.add(eng_key)

    for eng_val in list(covered_engines):
        for eng in EngineType:
            if eng.value.lower() == eng_val:
                covered_engines.add(eng.name.lower())

    # Documented aliases / exemptions for specialized engine enum types
    ENGINE_TYPE_ALIASES = {
        "myosim": "myosuite",
        "matlab_2d": "simscape",
        "matlab_3d": "simscape",
        "pendulum": "pendulum_simulator",
        "golf_swing_pendulum": "golf_simulation_suite",
    }

    missing_enum: set[str] = set()
    for engine in EngineType:
        eng_val = engine.value.lower()
        alias_target = ENGINE_TYPE_ALIASES.get(eng_val)
        if eng_val not in covered_engines and (
            alias_target is None
            or alias_target not in covered_engines
            and alias_target not in models_tile_ids
        ):
            missing_enum.add(eng_val)

    assert not missing_enum, (
        f"EngineType members missing launcher tiles: {sorted(missing_enum)}"
    )

    # Also verify ENGINE_TIERS engines all have tiles
    missing_tiers: set[str] = set()
    for tier_engine in ENGINE_TIERS:
        if (
            tier_engine not in covered_engines
            and tier_engine not in manifest_tile_ids
            and tier_engine not in models_tile_ids
        ):
            missing_tiers.add(tier_engine)

    assert not missing_tiers, (
        f"ENGINE_TIERS entries missing launcher tiles: {sorted(missing_tiers)}"
    )


@pytest.mark.unit
def test_jaxsim_catalog_and_manifest_registration() -> None:
    """JaxSim must be registered in launcher_manifest.json, models.yaml, and engine inventory."""
    root = Path(__file__).resolve().parent.parent.parent

    # launcher_manifest.json
    manifest_path = root / "src" / "config" / "launcher_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    tiles = {t["id"]: t for t in manifest_data.get("tiles", [])}
    assert "jaxsim_dashboard" in tiles, (
        "jaxsim_dashboard tile missing in launcher_manifest.json"
    )
    jaxsim_tile = tiles["jaxsim_dashboard"]
    assert jaxsim_tile.get("category") == "physics_engine"
    assert jaxsim_tile.get("engine_type") == "jaxsim"

    # models.yaml
    models_path = root / "src" / "config" / "models.yaml"
    with models_path.open("r", encoding="utf-8") as f:
        models_data = yaml.safe_load(f)
    models = {m["id"]: m for m in models_data.get("models", []) if "id" in m}
    assert "jaxsim_dashboard" in models, "jaxsim_dashboard model missing in models.yaml"
    jaxsim_model = models["jaxsim_dashboard"]
    assert jaxsim_model.get("engine_type") == "jaxsim"

    # engine_model_inventory.json
    inventory_path = root / "src" / "config" / "engine_model_inventory.json"
    with inventory_path.open("r", encoding="utf-8") as f:
        inventory_data = json.load(f)
    assert (
        inventory_data.get("launcher_tile_map", {}).get("jaxsim_dashboard") == "jaxsim"
    )

    # capability_state KNOWN_PHYSICS_ENGINES
    assert "jaxsim" in KNOWN_PHYSICS_ENGINES, "jaxsim missing in KNOWN_PHYSICS_ENGINES"


@pytest.mark.unit
def test_jaxsim_engine_availability_probe() -> None:
    """JaxSim must have module probing, install hints, and model aliases configured."""
    from src.launchers.engine_availability import (
        ENGINE_INSTALL_HINTS,
        ENGINE_MODULES,
        _MODEL_ENGINE_ALIASES,
        engine_for_model,
        install_hint,
        probe_engine,
    )

    assert "jaxsim" in ENGINE_MODULES, "ENGINE_MODULES missing 'jaxsim'"
    assert "jaxsim" in ENGINE_INSTALL_HINTS, "ENGINE_INSTALL_HINTS missing 'jaxsim'"
    assert _MODEL_ENGINE_ALIASES.get("jaxsim_dashboard") == "jaxsim", (
        "_MODEL_ENGINE_ALIASES missing 'jaxsim_dashboard'"
    )
    assert engine_for_model("jaxsim_dashboard") == "jaxsim"

    hint = install_hint("jaxsim", platform="win32")
    assert "WSL" in hint or "Linux" in hint

    availability = probe_engine("jaxsim")
    assert availability.engine == "jaxsim"
    assert isinstance(availability.available, bool)
    assert availability.status in ("available", "unavailable")


@pytest.mark.unit
def test_jaxsim_dashboard_exposes_dockable_ui() -> None:
    """jaxsim_dashboard must provide get_dockable_ui for launcher docking integration."""
    import importlib

    mod = importlib.import_module("src.launchers.jaxsim_dashboard")
    assert hasattr(mod, "get_dockable_ui"), "jaxsim_dashboard missing get_dockable_ui"
    assert callable(mod.get_dockable_ui)
    assert hasattr(mod, "main"), "jaxsim_dashboard missing main"
    assert callable(mod.main)
