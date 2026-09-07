"""Every src/tools package is registered or explicitly excluded (#8863).

``src/config/models.yaml`` (the single tile registry, #9412) carries an
``excluded_packages`` list: a package under ``src/tools/`` must either be
reachable from a launcher tile (its path referenced by the merged launcher
surface) or carry an exclusion entry with a nonempty reason. Anything else
is "unregistered tool" drift and fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import LauncherManifest
from src.config.tile_registry import ExcludedPackage, load_tile_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "src" / "tools"

pytestmark = pytest.mark.unit


def _tool_packages() -> set[str]:
    """Directory names under src/tools/ that are importable packages."""
    return {
        d.name
        for d in TOOLS_DIR.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
    }


def _tile_referenced_packages() -> set[str]:
    """src/tools packages referenced by any tile on the merged surface."""
    manifest = LauncherManifest.load()
    pkgs: set[str] = set()
    for tile in manifest.tiles:
        path = (tile.path or "").replace("\\", "/")
        match = re.match(r"src/tools/([A-Za-z0-9_]+)", path)
        if match:
            pkgs.add(match.group(1))
    return pkgs


def _exclusions() -> tuple[ExcludedPackage, ...]:
    exclusions = load_tile_registry().excluded_packages
    assert exclusions, "models.yaml must hold an 'excluded_packages' list"
    return exclusions


class TestRegistryExclusions:
    def test_every_tool_package_is_registered_or_excluded(self) -> None:
        excluded = {entry.package for entry in _exclusions()}
        unaccounted = _tool_packages() - _tile_referenced_packages() - excluded
        assert not unaccounted, (
            f"src/tools packages in no launcher and not excluded: "
            f"{sorted(unaccounted)}. Add a models.yaml tile or a justified "
            f"'excluded_packages' entry in src/config/models.yaml."
        )

    def test_exclusions_carry_nonempty_reasons(self) -> None:
        for entry in _exclusions():
            assert entry.package, f"exclusion missing package: {entry}"
            assert len(entry.reason) >= 20, (
                f"exclusion for {entry.package!r} needs a substantive "
                f"reason, got: {entry.reason!r}"
            )

    def test_exclusions_are_not_stale_or_contradictory(self) -> None:
        packages = _tool_packages()
        referenced = _tile_referenced_packages()
        excluded = [entry.package for entry in _exclusions()]
        missing = [p for p in excluded if p not in packages]
        assert not missing, f"excluded packages no longer exist: {missing}"
        contradictory = [p for p in excluded if p in referenced]
        assert not contradictory, (
            f"packages both excluded and referenced by a tile: {contradictory} "
            f"— remove the stale exclusion entry"
        )
        assert len(excluded) == len(set(excluded)), "duplicate exclusion entries"

    def test_legacy_exclusions_file_is_gone(self) -> None:
        """The separate registry_exclusions.yaml was folded into models.yaml."""
        assert not (REPO_ROOT / "src" / "config" / "registry_exclusions.yaml").exists()
