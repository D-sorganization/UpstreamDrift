"""Validate physics engine tier metadata."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_canonical_tiers(project_root: Path) -> tuple[frozenset[str], dict[str, str]]:
    tier_module_path = project_root / "src" / "engines" / "tiers.py"
    spec = importlib.util.spec_from_file_location("engine_tiers", tier_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load engine tiers from {tier_module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _extract_tier_metadata(module)


def _extract_tier_metadata(module: ModuleType) -> tuple[frozenset[str], dict[str, str]]:
    allowed_tiers = module.ALLOWED_TIERS
    engine_tiers = module.ENGINE_TIERS
    if not isinstance(allowed_tiers, frozenset):
        raise RuntimeError("ALLOWED_TIERS must be a frozenset")
    if not isinstance(engine_tiers, dict):
        raise RuntimeError("ENGINE_TIERS must be a dictionary")

    return allowed_tiers, engine_tiers


ALLOWED_TIERS, REQUIRED_ENGINE_TIERS = _load_canonical_tiers(PROJECT_ROOT)


def _read_tier_value(tier_path: Path) -> str | None:
    tree = ast.parse(tier_path.read_text(encoding="utf-8"), filename=str(tier_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "TIER"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def check_engine_tiers(physics_root: Path) -> list[str]:
    """Return metadata violations for required engine packages.

    Postcondition: an empty list means all required packages declare the
    expected tier with one of the allowed tier values.
    """
    if not isinstance(physics_root, Path):
        raise TypeError("physics_root must be a pathlib.Path")

    violations: list[str] = []
    for engine_name, expected_tier in REQUIRED_ENGINE_TIERS.items():
        tier_path = physics_root / engine_name / "_tier.py"
        if not tier_path.exists():
            violations.append(f"{engine_name} is missing {tier_path}")
            continue

        tier_value = _read_tier_value(tier_path)
        if tier_value not in ALLOWED_TIERS:
            violations.append(
                f"{engine_name} has invalid tier {tier_value!r}; "
                f"allowed values: {sorted(ALLOWED_TIERS)}"
            )
            continue

        if tier_value != expected_tier:
            violations.append(
                f"{engine_name} declares tier {tier_value!r}; "
                f"expected {expected_tier!r}"
            )

    return violations


def main() -> int:
    physics_root = PROJECT_ROOT / "src" / "engines" / "physics_engines"
    violations = check_engine_tiers(physics_root)

    if violations:
        for violation in violations:
            print(f"ERROR: {violation}")
        return 1

    print("Engine tier metadata is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
