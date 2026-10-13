"""Guard against two directories claiming the same top-level package name.

Regression guard for #7995. ``./tools`` and ``./src/tools`` were both regular
packages, and both ``.`` and ``src`` are on the pytest ``pythonpath``. Because a
regular package wins ``sys.modules`` for the whole process, whichever one was
imported first made every submodule of the other permanently unimportable — so
test *collection* became order-dependent.

This test walks every ``pythonpath`` root (plus the implicit roots pytest derives
for test packages) and fails if any top-level importable name is provided by more
than one root, distinguishing verified canonical aliases and seam counterparts
from genuinely competing package implementations.
"""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit

# Roots that pytest adds implicitly because they contain a package whose parent
# has no __init__.py (rootdir of a test package under importmode=prepend).
_IMPLICIT_TEST_ROOTS = ("tests/tools",)

# Directory names that are never importable packages.
_IGNORED = {"__pycache__", "node_modules", ".venv", ".git"}

_SEAM_RULINGS_PATH = REPO_ROOT / "docs" / "shared_tools" / "seam_rulings.v1.json"
_VENDOR_SHARED_REL = "vendor/ud-tools/src/shared/python"


def _load_seam_rulings(path: Path = _SEAM_RULINGS_PATH) -> dict[str, dict[str, Any]]:
    """Load the authoritative seam rulings mapping package names to status."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rulings = data.get("rulings", {})
        return rulings if isinstance(rulings, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _is_explicit_compatibility_shim(root: Path, name: str) -> bool:
    """Return True if ``name`` under ``root`` is an explicit compatibility shim.

    A provider is an explicit shim if it delegates to canonical shared code via
    ``alias_legacy_package``, ``from shared.python.`` imports, a ``seam-shim:``
    marker, or appends canonical shared directories to ``__path__``.
    """
    pkg_dir = root / name
    if pkg_dir.is_dir():
        init_file = pkg_dir / "__init__.py"
        if init_file.is_file():
            try:
                content = init_file.read_text(encoding="utf-8", errors="replace")
                if "alias_legacy_package" in content or "seam-shim:" in content:
                    return True
                if (
                    "from shared.python." in content
                    or 'import_module("shared.python.' in content
                ):
                    return True
                if "__path__.append" in content and "shared/python" in content:
                    return True
            except OSError:
                pass
    mod_file = root / f"{name}.py"
    if mod_file.is_file():
        try:
            content = mod_file.read_text(encoding="utf-8", errors="replace")
            if (
                "from shared.python." in content
                or "alias_legacy_package" in content
                or "seam-shim:" in content
            ):
                return True
        except OSError:
            pass
    return False


def _rel_path_str(path: Path, repo_root: Path = REPO_ROOT) -> str:
    """Return a normalized repo-relative POSIX path string without resolving symlinks."""
    try:
        return str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        pass
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def _is_verified_canonical_alias(
    name: str,
    root: Path,
    repo_root: Path = REPO_ROOT,
    seam_rulings: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Return True if ``root`` is a verified canonical alias or seam counterpart for ``name``.

    Verified providers are non-competing: they either point to the canonical module
    via a compatibility shim or are governed by an approved seam ruling in
    ``docs/shared_tools/seam_rulings.v1.json``.
    """
    if _is_explicit_compatibility_shim(root, name):
        return True

    rel_root = _rel_path_str(root, repo_root)
    is_vendor_shared = rel_root == _VENDOR_SHARED_REL
    if not is_vendor_shared:
        vendor_path = repo_root / _VENDOR_SHARED_REL
        if root == vendor_path:
            is_vendor_shared = True
        elif root.is_dir() and vendor_path.is_dir():
            try:
                is_vendor_shared = root.resolve() == vendor_path.resolve()
            except OSError:
                pass

    if is_vendor_shared:
        rulings = seam_rulings if seam_rulings is not None else _load_seam_rulings()
        return name in rulings or f"{name}.py" in rulings

    return False


def _pythonpath_roots() -> list[Path]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entries = config["tool"]["pytest"]["ini_options"]["pythonpath"]
    roots = [REPO_ROOT / entry for entry in entries]
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
# issue #8053 and updated for verified layout entries. Each entry is name -> the
# competing roots that claim it. This is a RATCHET: new collisions fail immediately,
# and entries must be deleted (never added) as the tracked cleanup lands.
# Deliberately NOT a blanket exemption — every pair is named so removing one is
# a one-line diff.
_KNOWN_COLLISIONS: dict[str, tuple[str, ...]] = {
    "ball_flight_gui": ("src/tools", "tests/tools"),
    "bunker_shot_gui": ("src/tools", "tests/tools"),
    "canonical_core": ("src/shared/python", "src/tools"),
    "capture_rig": ("src/tools", "tests/tools"),
    "config": ("src", "src/shared/python"),
    "core": ("src", "src/shared/python"),
    "examples": (".", "src/engines/physics_engines/mujoco/python"),
    "freemocap_sidecar": ("src/tools", "tests/tools"),
    "golf_environment": ("src/tools", "tests/tools"),
    "golf_simulation_suite": ("src/tools", "tests/tools"),
    "golf_simulator": ("src/shared/python", "src/tools"),
    "matlab_utilities": ("src/tools", "tests/tools"),
    "model_explorer": ("src/tools", "tests/tools"),
    "motion_matching": (
        "src/engines/physics_engines/mujoco/python",
        "src/shared/python",
        "src/tools",
    ),
    "perturbation": ("src/engines/physics_engines/mujoco/python", "src/shared/python"),
    "pose_studio": ("src/tools", "tests/tools"),
    "putting_green_gui": ("src/tools", "tests/tools"),
    "sg_optimizer": ("src/shared/python", "src/tools"),
    "shadow_tracker": ("src/shared/python", "src/tools"),
    "sidekick": ("src/shared/python", "src/tools"),
    "starting_pose_matcher": ("src/tools", "tests/tools"),
    "swing_flight_pipeline": ("src/tools", "tests/tools"),
    "terrain_engine": ("src/tools", "tests/tools"),
    "tests": (".", "src/engines/physics_engines/mujoco/python"),
    "tools": ("src", "src/shared/python"),
    "training_controller": ("src/tools", "tests/tools"),
}


def _competing_collisions(
    roots: list[Path] | None = None,
    repo_root: Path = REPO_ROOT,
    seam_rulings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return top-level package collisions after filtering verified canonical aliases.

    A provider is considered a verified canonical alias (and thus non-competing) if:
    1. It is an explicit compatibility shim (e.g. delegates to canonical via
       ``alias_legacy_package``, ``seam-shim:``, or re-exports canonical symbols), or
    2. It is a documented seam counterpart under ``docs/shared_tools/seam_rulings.v1.json``.

    Returns:
        Mapping of package name to the tuple of genuinely competing roots claiming it,
        only where two or more competing roots remain.
    """
    scan_roots = roots if roots is not None else _pythonpath_roots()
    rulings = seam_rulings if seam_rulings is not None else _load_seam_rulings()

    providers: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for root in scan_roots:
        if not root.is_dir():
            continue
        rel_str = _rel_path_str(root, repo_root)
        for name in _toplevel_names(root):
            providers[name].append((rel_str, root))

    competing: dict[str, tuple[str, ...]] = {}
    for name, root_pairs in providers.items():
        if len(root_pairs) <= 1:
            continue
        competing_rel_roots = [
            rel
            for rel, path in root_pairs
            if not _is_verified_canonical_alias(
                name, path, repo_root=repo_root, seam_rulings=rulings
            )
        ]
        if len(competing_rel_roots) > 1:
            competing[name] = tuple(sorted(competing_rel_roots))

    return competing


def test_no_new_toplevel_package_name_collisions() -> None:
    """No import-name collision beyond the tracked baseline (#8053)."""
    found = _competing_collisions()
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
    found = _competing_collisions()
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
    assert _KNOWN_COLLISIONS.get("tools") == (
        "src",
        "src/shared/python",
    ), "The 'tools' collision changed shape; re-verify #7995 has not regressed."


def test_canonical_alias_root_order_independence() -> None:
    """Verified canonical aliases produce identical non-competing layout regardless of root order."""
    all_roots = _pythonpath_roots()
    forward = _competing_collisions(all_roots)
    reverse = _competing_collisions(list(reversed(all_roots)))

    assert forward == reverse, (
        "Competing collision detection must be independent of pythonpath root ordering."
    )


def test_newly_competing_provider_fails(tmp_path: Path) -> None:
    """A newly competing unverified package collision fails the collision gate."""
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    pkg_name = "unapproved_competing_pkg"

    (root_a / pkg_name).mkdir(parents=True)
    (root_a / pkg_name / "__init__.py").write_text(
        '"""Implementation A."""\n', encoding="utf-8"
    )

    (root_b / pkg_name).mkdir(parents=True)
    (root_b / pkg_name / "__init__.py").write_text(
        '"""Implementation B."""\n', encoding="utf-8"
    )

    competing = _competing_collisions(
        [root_a, root_b],
        repo_root=tmp_path,
        seam_rulings={},
    )

    assert pkg_name in competing
    assert competing[pkg_name] == ("root_a", "root_b")


def test_unruled_vendor_package_fails(tmp_path: Path) -> None:
    """A package in vendor shared lacking an approved seam ruling is flagged as competing."""
    repo_root = tmp_path / "repo"
    ud_shared = repo_root / "src" / "shared" / "python"
    vendor_shared = repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python"
    pkg_name = "unruled_vendor_candidate"

    (ud_shared / pkg_name).mkdir(parents=True)
    (ud_shared / pkg_name / "__init__.py").write_text(
        '"""UD copy."""\n', encoding="utf-8"
    )

    (vendor_shared / pkg_name).mkdir(parents=True)
    (vendor_shared / pkg_name / "__init__.py").write_text(
        '"""Vendor copy."""\n', encoding="utf-8"
    )

    # Empty seam rulings: the vendor copy has not been approved as canonical/seam counterpart.
    competing = _competing_collisions(
        [vendor_shared, ud_shared],
        repo_root=repo_root,
        seam_rulings={},
    )

    assert pkg_name in competing
    assert competing[pkg_name] == (
        "src/shared/python",
        "vendor/ud-tools/src/shared/python",
    )
