"""Every registered launcher tile's launch target must resolve (issue #8854).

Loads the full launcher surface via the real shared loader
(``LauncherManifest.load`` over ``launcher_manifest.json`` +
``src/config/models.yaml``) and asserts, per tile, that the declared
``path`` resolves through the same policy the launch handlers use:

- repo-local paths must exist on disk (dotted-module strings are resolved
  to their ``.py`` file, issue #8860);
- ``provider: tools`` paths resolve inside the pinned ``vendor/ud-tools``
  gitlink (skipped with a reason when the submodule is not initialized in
  this checkout — never faked as success);
- ``source_root`` / ``shared_repo`` targets live in sibling checkouts and
  are skipped with a reason when the sibling is absent locally;
- ``virtual/*`` pseudo-paths must be registered in ``VIRTUAL_TARGETS``;
- path-less tiles must declare an honest web contract instead of a dead
  native target.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import LauncherManifest, LauncherTile
from src.shared.python.config.tile_target_resolution import (
    EXTERNAL_KINDS,
    KIND_PATHLESS,
    resolve_tile_target,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def _all_tiles() -> list[LauncherTile]:
    manifest = LauncherManifest.load()
    return list(manifest.tiles)


_TILES = _all_tiles()


def test_registry_is_not_empty() -> None:
    """Guard: the loader produced a meaningful tile surface."""
    assert len(_TILES) > 30, f"suspiciously small tile surface: {len(_TILES)}"


@pytest.mark.parametrize("tile", _TILES, ids=[t.id for t in _TILES])
def test_tile_launch_target_resolves(tile: LauncherTile) -> None:
    """Each tile's declared target exists or is explicitly external/web-only."""
    resolution = resolve_tile_target(tile, REPO_ROOT)

    if resolution.kind == KIND_PATHLESS and not resolution.resolvable:
        # No native target declared: acceptable only with an honest web
        # contract (a real route, or an explicit unavailable+reason badge).
        web = tile.web
        assert web is not None and web.mode in {"route", "unavailable"}, (
            f"Tile '{tile.id}' has no launch path and no honest web contract "
            f"({resolution.reason})"
        )
        return

    if not resolution.resolvable and resolution.kind in EXTERNAL_KINDS:
        # External targets (Tools vendor gitlink, sibling checkouts) may be
        # absent on this machine; that is an environment gap, not a registry
        # bug. Never faked as success.
        pytest.skip(f"external target not present here: {resolution.reason}")

    assert resolution.resolvable, (
        f"Tile '{tile.id}' declares an unresolvable launch target "
        f"(kind={resolution.kind}): {resolution.reason}"
    )


class TestDottedModulePaths:
    """Dotted module strings in the ``path`` field launch correctly (#8860)."""

    def test_loader_normalizes_dotted_module_paths(self) -> None:
        tile = LauncherTile.from_dict(
            {
                "id": "dotted",
                "name": "Dotted",
                "description": "d",
                "category": "tool",
                "type": "special_app",
                "path": "src.tools.simulation_backends_launcher.__main__",
                "logo": "golf_logo.svg",
            }
        )
        assert tile.path == "src/tools/simulation_backends_launcher/__main__.py"

    def test_manifest_file_contains_no_dotted_module_paths(self) -> None:
        import json

        from src.config.launcher_manifest_loader import MANIFEST_PATH
        from src.shared.python.config.tile_target_resolution import (
            module_string_to_relpath,
        )

        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        offenders = [
            t["id"]
            for t in raw["tiles"]
            if t.get("path") and module_string_to_relpath(t["path"]) is not None
        ]
        assert not offenders, (
            f"Manifest 'path' fields holding dotted module strings: {offenders}"
        )

    def test_simulation_backends_tile_targets_a_real_file(self) -> None:
        manifest = LauncherManifest.load()
        tile = manifest.get_tile("simulation_backends")
        assert tile is not None
        assert tile.path.endswith(".py")
        assert (REPO_ROOT / tile.path).exists()
