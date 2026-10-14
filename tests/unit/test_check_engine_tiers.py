"""Unit tests for the engine tier metadata checker."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_module() -> types.ModuleType:
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "check_engine_tiers.py"
    )
    spec = importlib.util.spec_from_file_location("check_engine_tiers", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tier(root: Path, engine_name: str, tier_value: str) -> None:
    engine_dir = root / engine_name
    engine_dir.mkdir(parents=True)
    (engine_dir / "_tier.py").write_text(
        f'TIER = "{tier_value}"\n',
        encoding="utf-8",
    )


def test_check_engine_tiers_accepts_valid_metadata(tmp_path: Path) -> None:
    module = _load_module()
    physics_root = tmp_path / "physics_engines"

    for engine_name, tier_value in module.REQUIRED_ENGINE_TIERS.items():
        _write_tier(physics_root, engine_name, tier_value)

    assert module.check_engine_tiers(physics_root) == []


def test_check_engine_tiers_reports_missing_and_invalid_metadata(
    tmp_path: Path,
) -> None:
    module = _load_module()
    physics_root = tmp_path / "physics_engines"

    for engine_name, tier_value in module.REQUIRED_ENGINE_TIERS.items():
        if engine_name != "opensim":
            _write_tier(physics_root, engine_name, tier_value)

    (physics_root / "drake" / "_tier.py").write_text(
        'TIER = "prototype"\n',
        encoding="utf-8",
    )

    violations = module.check_engine_tiers(physics_root)

    assert any("opensim is missing" in violation for violation in violations)
    assert any("drake has invalid tier" in violation for violation in violations)


def _write_all_but_simscape(module: types.ModuleType, physics_root: Path) -> None:
    for engine_name, tier_value in module.REQUIRED_ENGINE_TIERS.items():
        if engine_name != "simscape":
            _write_tier(physics_root, engine_name, tier_value)


def test_check_engine_tiers_resolves_package_under_engines_root(
    tmp_path: Path,
) -> None:
    module = _load_module()
    physics_root = tmp_path / "physics_engines"
    _write_all_but_simscape(module, physics_root)
    _write_tier(tmp_path, "simscape", module.REQUIRED_ENGINE_TIERS["simscape"])

    assert module.check_engine_tiers(physics_root) == []


def test_check_engine_tiers_missing_everywhere_reports_physics_path(
    tmp_path: Path,
) -> None:
    module = _load_module()
    physics_root = tmp_path / "physics_engines"
    _write_all_but_simscape(module, physics_root)

    violations = module.check_engine_tiers(physics_root)

    expected = f"simscape is missing {physics_root / 'simscape' / '_tier.py'}"
    assert violations == [expected]


def test_check_engine_tiers_prefers_physics_engines_location(
    tmp_path: Path,
) -> None:
    module = _load_module()
    physics_root = tmp_path / "physics_engines"
    _write_all_but_simscape(module, physics_root)
    _write_tier(physics_root, "simscape", module.REQUIRED_ENGINE_TIERS["simscape"])
    # A mismatching copy under src/engines must be ignored when the
    # physics_engines copy exists.
    _write_tier(tmp_path, "simscape", "core")

    assert module.check_engine_tiers(physics_root) == []


def test_resolve_tier_path_orders_candidates(tmp_path: Path) -> None:
    module = _load_module()
    physics_root = tmp_path / "physics_engines"

    canonical = module._resolve_tier_path(physics_root, "simscape")
    assert canonical == physics_root / "simscape" / "_tier.py"

    _write_tier(tmp_path, "simscape", "experimental")
    assert module._resolve_tier_path(physics_root, "simscape") == (
        tmp_path / "simscape" / "_tier.py"
    )
