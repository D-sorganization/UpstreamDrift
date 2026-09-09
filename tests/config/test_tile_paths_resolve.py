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
- ``virtual/*`` pseudo-paths must be registered in ``VIRTUAL_TARGETS`` /
  ``VIRTUAL_PREFIXES`` *and* their backing handler artifact must exist;
- path-less tiles must declare an honest web contract instead of a dead
  native target.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typing import Any
from src.config.launcher_manifest_loader import LauncherManifest, LauncherTile
from src.shared.python.config.tile_target_resolution import (
    EXTERNAL_KINDS,
    KIND_PATHLESS,
    resolve_tile_target,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def _vendor_gitlink_materialised() -> bool:
    """True when this checkout materialised the pinned vendor/ud-tools
    gitlink as an initialized, clean worktree."""
    from src.shared.python.config.tools_vendor_authority import (
        inspect_tools_vendor_authority,
    )

    return inspect_tools_vendor_authority(REPO_ROOT).available


requires_vendor_gitlink = pytest.mark.skipif(
    not _vendor_gitlink_materialised(),
    reason="vendor/ud-tools gitlink is not materialised in this checkout",
)


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


class TestVirtualTargetValidation:
    """Virtual targets are genuinely validated, not allowlist-blessed (#8854)."""

    @staticmethod
    def _tile(path: str) -> LauncherTile:
        return LauncherTile.from_dict(
            {
                "id": "virt",
                "name": "V",
                "description": "v",
                "category": "tool",
                "type": "special_app",
                "path": path,
                "logo": "golf_logo.svg",
            }
        )

    def test_registered_virtual_targets_resolve_to_their_backing(self) -> None:
        from src.shared.python.config.tile_target_resolution import VIRTUAL_TARGETS

        for target, backing in VIRTUAL_TARGETS.items():
            resolution = resolve_tile_target(self._tile(target), REPO_ROOT)
            assert resolution.resolvable, f"{target} unresolvable: {resolution.reason}"
            assert resolution.target == REPO_ROOT / backing

    def test_virtual_prefix_targets_resolve_to_their_backing(self) -> None:
        resolution = resolve_tile_target(
            self._tile("virtual/biomech_exercise/gait"), REPO_ROOT
        )
        assert resolution.resolvable
        assert resolution.target is not None and resolution.target.exists()

    def test_unknown_virtual_target_is_a_registry_error(self) -> None:
        resolution = resolve_tile_target(
            self._tile("virtual/engine_dashboard"), REPO_ROOT
        )
        assert not resolution.resolvable
        assert "unknown virtual target" in (resolution.reason or "")

    def test_virtual_target_with_missing_backing_fails(self, tmp_path: Path) -> None:
        """A registered virtual target whose handler artifact vanished fails."""
        resolution = resolve_tile_target(self._tile("virtual/matlab_suite"), tmp_path)
        assert not resolution.resolvable
        assert "lost its backing handler artifact" in (resolution.reason or "")


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


class TestReadyMaturityGate:
    """A ready/beta tile whose entry point does not resolve fails the gate (#9478).

    Issue #9478: the registry declared tiles as ``ready``/``beta`` whose
    launch targets do not resolve from a clean checkout (the four
    ``*_models_shared`` sibling folders, the Movement_Optimizer sibling,
    and Tools paths masquerading as repo-relative ones). The gate now
    distinguishes a *false registry claim* (fail) from an *external
    surface that is simply not materialised in this checkout* (skip,
    never faked as success).
    """

    def _model(self, **overrides: Any) -> Any:
        from types import SimpleNamespace

        defaults: dict[str, Any] = {
            "id": "synthetic_tile",
            "path": "src/tools/model_explorer/main_window.py",
            "type": "special_app",
            "provider": None,
            "source_root": None,
            "status": "ready",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolving_ready_tile_passes(self, tmp_path: Path) -> None:
        """A ready tile backed by a real repo file raises no gate error."""
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        tool_dir = tmp_path / "src" / "tools" / "real_tool"
        tool_dir.mkdir(parents=True)
        (tool_dir / "gui.py").write_text("# real entry point", encoding="utf-8")
        model = self._model(path="src/tools/real_tool/gui.py")
        outcome, reason = ready_maturity_gate(model, tmp_path)
        assert outcome == "ok", reason

    def test_unresolvable_ready_tile_fails(self, tmp_path: Path) -> None:
        """A ready tile pointing at a missing local file fails the gate."""
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        model = self._model(path="src/tools/does_not_exist/gui.py")
        outcome, reason = ready_maturity_gate(model, tmp_path)
        assert outcome == "fail"
        assert reason is not None and "does_not_exist" in reason

    def test_beta_tile_missing_sibling_fails(self, tmp_path: Path) -> None:
        """A beta tile whose sibling-checkout target is absent fails."""
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        model = self._model(
            status="beta",
            path="src/movement_optimizer/__main__.py",
            source_root="Movement_Optimizer",
        )
        outcome, reason = ready_maturity_gate(model, tmp_path)
        assert outcome == "fail", reason

    def test_experimental_tile_without_claim_is_not_gated(self, tmp_path: Path) -> None:
        """Only ready/beta statuses claim launchability; others pass freely."""
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        model = self._model(
            status="experimental",
            path="src/tools/does_not_exist/gui.py",
        )
        outcome, _reason = ready_maturity_gate(model, tmp_path)
        assert outcome == "ok"

    def test_unmaterialised_tools_vendor_skips(self, tmp_path: Path) -> None:
        """A ready Tools-vendor tile in a checkout without the pinned
        vendor gitlink is an environment gap, not a registry lie."""
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        model = self._model(
            path="tools://rate_of_closure/launch_pyqt6.py",
            provider="tools",
        )
        outcome, reason = ready_maturity_gate(model, tmp_path)
        assert outcome == "skip", reason

    @requires_vendor_gitlink
    def test_tools_vendor_missing_entry_fails(self, tmp_path: Path) -> None:
        """With the vendor materialised (real repo root), a ready Tools
        tile naming a path absent from the pinned tree fails."""
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        model = self._model(
            path="tools://rate_of_closure/definitely_missing.py",
            provider="tools",
        )
        outcome, reason = ready_maturity_gate(model, REPO_ROOT)
        assert outcome == "fail", reason

    def test_registry_ready_and_beta_tiles_are_backed(self) -> None:
        """Every ready/beta tile declared by the canonical registries
        resolves (or is an unmaterialised external surface: skip)."""
        import json

        import yaml

        from src.config.launcher_manifest_loader import MANIFEST_PATH
        from src.shared.python.config.tile_target_resolution import (
            ready_maturity_gate,
        )

        raw_models = yaml.safe_load(
            (REPO_ROOT / "src" / "config" / "models.yaml").read_text(encoding="utf-8")
        )["models"]
        manifest_tiles = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["tiles"]
        seen = {m["id"] for m in raw_models}
        claimed: list[dict[str, Any]] = [
            m
            for m in raw_models + manifest_tiles
            if m.get("launcher", {}).get("status", m.get("status")) in {"ready", "beta"}
        ]
        failures: list[str] = []
        for entry in claimed:
            if entry["id"] in seen and "launcher" not in entry:
                continue  # manifest duplicate of a registry tile
            model = _EntryAdapter(entry)
            outcome, reason = ready_maturity_gate(model, REPO_ROOT)
            if outcome == "fail":
                failures.append(f"{entry['id']}: {reason}")
        assert not failures, "Registry maturity claims not backed: " + "; ".join(
            failures
        )


class _EntryAdapter:
    """Adapt a raw registry/manifest dict to the resolver attribute contract."""

    def __init__(self, entry: dict[str, Any]) -> None:
        launcher = entry.get("launcher", {})
        self.id = str(entry["id"])
        self.path = str(entry.get("path", "") or "")
        self.type = str(entry.get("type", "") or "")
        self.provider = entry.get("provider")
        self.source_root = entry.get("source_root")
        self.status = str(launcher.get("status", entry.get("status", "")) or "")


class TestToolsPathProvenance:
    """Tools-provided entry points are distinguishable from repo-relative
    ones in the registry schema (#9478): ``provider: tools`` entries must
    carry the ``tools://`` scheme so a vendor path cannot masquerade as a
    local one."""

    def test_provider_tools_paths_use_tools_scheme(self) -> None:
        import json

        import yaml

        registry_tools_entries = yaml.safe_load(
            (REPO_ROOT / "src" / "config" / "models.yaml").read_text(encoding="utf-8")
        )["models"]
        manifest_tiles = json.loads(
            (REPO_ROOT / "src" / "config" / "launcher_manifest.json").read_text(
                encoding="utf-8"
            )
        )["tiles"]
        offenders = [
            entry["id"]
            for entry in registry_tools_entries + manifest_tiles
            if entry.get("provider") == "tools"
            and not str(entry.get("path", "")).startswith("tools://")
        ]
        assert not offenders, (
            "provider: tools entries without the tools:// scheme: " + str(offenders)
        )

    @requires_vendor_gitlink
    def test_tools_scheme_resolves_against_vendor_root(self) -> None:
        """A ``tools://`` path resolves inside the pinned vendor/ud-tools
        gitlink, never against the repository root."""
        from src.shared.python.config.tile_target_resolution import (
            resolve_tile_target,
        )

        model = _EntryAdapter(
            {
                "id": "rate_of_closure",
                "type": "special_app",
                "path": "tools://src/rate_of_closure/launch_pyqt6.py",
                "provider": "tools",
            }
        )
        resolution = resolve_tile_target(model, REPO_ROOT)
        assert resolution.resolvable, resolution.reason
        assert resolution.kind == "tools-vendor"
        assert resolution.target is not None
        assert resolution.target.as_posix().endswith(
            "vendor/ud-tools/src/rate_of_closure/launch_pyqt6.py"
        )

    @requires_vendor_gitlink
    def test_tools_scheme_does_not_resolve_as_local_file(self) -> None:
        """A ``tools://`` path for a file that only exists locally still
        resolves through the vendor authority, never the repo root."""
        from src.shared.python.config.tile_target_resolution import (
            resolve_tile_target,
        )

        model = _EntryAdapter(
            {
                "id": "masquerade_probe",
                "type": "special_app",
                # model_explorer exists locally but NOT in the vendor tree
                "path": "tools://src/tools/model_explorer/main_window.py",
                "provider": "tools",
            }
        )
        resolution = resolve_tile_target(model, REPO_ROOT)
        assert not resolution.resolvable
        assert resolution.kind == "tools-vendor"
        assert "vendor/ud-tools" in (resolution.reason or "")
