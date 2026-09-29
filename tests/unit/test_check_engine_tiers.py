"""Unit tests for the engine tier metadata checker."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path


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
