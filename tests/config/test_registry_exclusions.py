"""Every src/tools package is registered or explicitly excluded (#8863).

``src/config/registry_exclusions.yaml`` is the documented convention for
launcher-less tool packages: a package under ``src/tools/`` must either be
reachable from a launcher tile (its path referenced by the merged launcher
surface) or carry an exclusion entry with a nonempty reason. Anything else
is "unregistered tool" drift and fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from src.config.launcher_manifest_loader import LauncherManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "src" / "tools"
EXCLUSIONS_YAML = REPO_ROOT / "src" / "config" / "registry_exclusions.yaml"

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


def _exclusions() -> list[dict[str, str]]:
    data = yaml.safe_load(EXCLUSIONS_YAML.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "exclusions" in data, (
        "registry_exclusions.yaml must hold an 'exclusions' list"
    )
    return list(data["exclusions"])


class TestRegistryExclusions:
    def test_every_tool_package_is_registered_or_excluded(self) -> None:
        excluded = {entry["package"] for entry in _exclusions()}
        unaccounted = _tool_packages() - _tile_referenced_packages() - excluded
        assert not unaccounted, (
            f"src/tools packages in no launcher and not excluded: "
            f"{sorted(unaccounted)}. Add a models.yaml tile or a justified "
            f"entry in src/config/registry_exclusions.yaml."
        )

    def test_exclusions_carry_nonempty_reasons(self) -> None:
        for entry in _exclusions():
            reason = str(entry.get("reason", "")).strip()
            assert entry.get("package"), f"exclusion missing package: {entry}"
            assert len(reason) >= 20, (
                f"exclusion for {entry.get('package')!r} needs a substantive "
                f"reason, got: {reason!r}"
            )

    def test_exclusions_are_not_stale_or_contradictory(self) -> None:
        packages = _tool_packages()
        referenced = _tile_referenced_packages()
        excluded = [entry["package"] for entry in _exclusions()]
        missing = [p for p in excluded if p not in packages]
        assert not missing, f"excluded packages no longer exist: {missing}"
        contradictory = [p for p in excluded if p in referenced]
        assert not contradictory, (
            f"packages both excluded and referenced by a tile: {contradictory} "
            f"— remove the stale exclusion entry"
        )
        assert len(excluded) == len(set(excluded)), "duplicate exclusion entries"
