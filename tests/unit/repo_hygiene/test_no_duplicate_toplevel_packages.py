"""Guard against two directories claiming the same top-level package name.

Regression guard for #7995. ``./tools`` and ``./src/tools`` were both regular
packages, and both ``.`` and ``src`` are on the pytest ``pythonpath``. Because a
regular package wins ``sys.modules`` for the whole process, whichever one was
imported first made every submodule of the other permanently unimportable — so
test *collection* became order-dependent.

This test walks every ``pythonpath`` root (plus the implicit roots pytest derives
for test packages) and fails if any top-level importable name is provided by more
than one root.
"""

from __future__ import annotations

import tomllib
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit

# Roots that pytest adds implicitly because they contain a package whose parent
# has no __init__.py (rootdir of a test package under importmode=prepend).
_IMPLICIT_TEST_ROOTS = ("tests/tools",)

# Directory names that are never importable packages.
_IGNORED = {"__pycache__", "node_modules", ".venv", ".git"}


def _pythonpath_roots() -> list[Path]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = config["tool"]["pytest"]["ini_options"]["pythonpath"]
    roots = [
        REPO_ROOT / entry
        for entry in entries
        if entry != "vendor/ud-tools/src/shared/python"
    ]
    roots += [REPO_ROOT / entry for entry in _IMPLICIT_TEST_ROOTS]
    return [root for root in roots if root.is_dir()]


def _toplevel_names(root: Path) -> set[str]:
    """Importable top-level names provided by ``root``."""
    names: set[str] = set()
    for child in root.iterdir():
        if child.name in _IGNORED or child.name.startswith("."):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(child.name)
        elif child.suffix == ".py" and child.stem != "__init__":
            names.add(child.stem)
    return names


# Pre-existing collisions inherited from before this guard existed, tracked by
# issue #8053. Each entry is name -> the roots that claim it. This is a RATCHET:
# new collisions fail immediately, and entries must be deleted (never added) as
# the tracked cleanup lands. Deliberately NOT a blanket exemption — every pair is
# named so removing one is a one-line diff.
_KNOWN_COLLISIONS: dict[str, tuple[str, ...]] = {
    "ball_flight_gui": ("src/tools", "tests/tools"),
    "bunker_shot_gui": ("src/tools", "tests/tools"),
    "canonical_core": ("src/shared/python", "src/tools"),
    "config": ("src", "src/shared/python"),
    "core": ("src", "src/shared/python"),
    "examples": (".", "src/engines/physics_engines/mujoco/python"),
    "freemocap_sidecar": ("src/tools", "tests/tools"),
    "golf_environment": ("src/tools", "tests/tools"),
    "golf_simulation_suite": ("src/tools", "tests/tools"),
    "matlab_utilities": ("src/tools", "tests/tools"),
    "model_explorer": ("src/tools", "tests/tools"),
    "motion_matching": (
        "src/engines/physics_engines/mujoco/python",
        "src/shared/python",
    ),
    "perturbation": ("src/engines/physics_engines/mujoco/python", "src/shared/python"),
    "pose_studio": ("src/tools", "tests/tools"),
    "putting_green_gui": ("src/tools", "tests/tools"),
    "sg_optimizer": ("src/shared/python", "src/tools"),
    "sidekick": ("src/shared/python", "src/tools"),
    "starting_pose_matcher": ("src/tools", "tests/tools"),
    "swing_flight_pipeline": ("src/tools", "tests/tools"),
    "terrain_engine": ("src/tools", "tests/tools"),
    "tests": (".", "src/engines/physics_engines/mujoco/python", "src/shared/python"),
    "tools": ("src", "src/shared/python"),
    "training_controller": ("src/tools", "tests/tools"),
}


def _collisions() -> dict[str, tuple[str, ...]]:
    providers: dict[str, list[str]] = defaultdict(list)
    for root in _pythonpath_roots():
        for name in _toplevel_names(root):
            providers[name].append(str(root.relative_to(REPO_ROOT)).replace("\\", "/"))
    return {
        name: tuple(sorted(roots))
        for name, roots in providers.items()
        if len(roots) > 1
    }


def test_no_new_toplevel_package_name_collisions() -> None:
    """No import-name collision beyond the tracked baseline (#8053)."""
    found = _collisions()
    new = {
        name: roots
        for name, roots in found.items()
        if _KNOWN_COLLISIONS.get(name) != roots
    }

    assert not new, (
        "New top-level import-name collision(s). The first root imported wins "
        "sys.modules and makes the others' submodules permanently unimportable "
        "(see #7995). Do not add these to _KNOWN_COLLISIONS -- fix the layout:\n"
        + "\n".join(f"  {name}: {list(roots)}" for name, roots in sorted(new.items()))
    )


def test_baseline_has_no_stale_entries() -> None:
    """Force the baseline to shrink: a resolved collision must be deleted."""
    found = _collisions()
    stale = sorted(set(_KNOWN_COLLISIONS) - set(found))
    assert not stale, (
        "These collisions are resolved; remove them from _KNOWN_COLLISIONS "
        f"(#8053): {stale}"
    )


def test_root_tools_package_collision_is_gone() -> None:
    """The specific #7995 defect: ./tools vs ./src/tools."""
    assert not (REPO_ROOT / "tools" / "__init__.py").exists(), (
        "./tools must not be a regular package -- it shadows ./src/tools and "
        "makes every src/tools submodule unimportable (#7995)."
    )
    assert _KNOWN_COLLISIONS.get("tools") == ("src", "src/shared/python"), (
        "The 'tools' collision changed shape; re-verify #7995 has not regressed."
    )
