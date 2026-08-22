"""Shared launch-target resolvability for launcher tiles.

Single source of truth for deciding whether a launcher tile's declared
target can actually be launched from this checkout (issues #8854, #8855,
#8860). Both the registry resolution test
(``tests/config/test_tile_paths_resolve.py``) and the desktop status chip
(``src.launchers.model_card.ModelCard``) consume this module, so the chip
can never claim "Ready" for a target the test would flag as dead.

Design by Contract:
    Preconditions:
        - ``resolve_tile_target`` requires a model-like object with an ``id``
          and a repo root directory that exists.
    Postconditions:
        - The returned resolution always carries a ``kind`` from ``KINDS``
          and, when ``resolvable`` is False, a non-empty ``reason``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.shared.python.config.tools_vendor_authority import (
    ToolsVendorAuthority,
    inspect_tools_vendor_authority,
)
from src.shared.python.core.contracts import ensure, require

# Pseudo-paths that are dispatched to dedicated launcher handlers rather
# than resolved on disk. Machine-checkable registry of "virtual" targets
# (issue #8854): anything else under ``virtual/`` is a registry error.
# Each virtual target maps to the repo-relative *backing* artifact its
# launcher handler actually dispatches to; a virtual target is genuinely
# resolvable only when that backing exists, so a renamed or deleted
# handler surface fails the registry test instead of launching dead.
VIRTUAL_TARGETS: dict[str, str] = {
    # MatlabSuiteHandler → MatlabSuiteWidget (launcher_simulation.py)
    "virtual/matlab_suite": "src/launchers/matlab_suite_dialog.py",
    # Library tile → LibraryWidget (launcher_layout_manager.py)
    "virtual/library": "src/launchers/library_widget.py",
}

# Virtual namespaces synthesized by the model registry for pathless model
# types (exercise presets, PINN modes); dispatched by dedicated handlers.
# Prefix → repo-relative backing artifact the handler launches/imports.
VIRTUAL_PREFIXES: dict[str, str] = {
    # BiomechExerciseHandler launches the exercise dashboard script.
    "virtual/biomech_exercise/": "src/launchers/exercise_dashboard.py",
    # PINN modes are a library-only surface (launcher_model_handlers.py).
    "virtual/physics_informed/": "src/shared/python/physics_informed",
}

# Tile/model types that legitimately declare no filesystem path because a
# dedicated handler synthesizes the launch (exercise presets, PINN modes)
# or the tile is an API/web-only catalog entry.
PATHLESS_TYPES = frozenset({"biomech_exercise", "physics_informed", "api_backed"})

KIND_FILE = "file"
KIND_TOOLS_VENDOR = "tools-vendor"
KIND_SIBLING = "sibling"
KIND_SHARED_REPO = "shared-repo"
KIND_VIRTUAL = "virtual"
KIND_PATHLESS = "pathless"
KINDS = frozenset(
    {
        KIND_FILE,
        KIND_TOOLS_VENDOR,
        KIND_SIBLING,
        KIND_SHARED_REPO,
        KIND_VIRTUAL,
        KIND_PATHLESS,
    }
)

# Kinds whose targets live outside this repository checkout; an absent
# target of these kinds means "not available here", not "registry is wrong".
EXTERNAL_KINDS = frozenset({KIND_TOOLS_VENDOR, KIND_SIBLING, KIND_SHARED_REPO})


@dataclass(frozen=True)
class TileTargetResolution:
    """Outcome of resolving one tile's declared launch target."""

    resolvable: bool
    kind: str
    target: Path | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Enforce the postcondition contract."""
        ensure(self.kind in KINDS, "kind must be a known resolution kind", self.kind)
        ensure(
            self.resolvable or bool(self.reason),
            "unresolvable targets must carry a reason",
            self,
        )


def module_string_to_relpath(path: str) -> str | None:
    """Return the repo-relative ``.py`` path for a dotted module string.

    Returns ``None`` when ``path`` is not a dotted module string (already a
    file path, a directory name, or empty). This is the schema bridge for
    manifests that historically stored ``src.tools.x.__main__`` in the
    filesystem ``path`` field (issue #8860).
    """
    if "/" in path or "\\" in path or "." not in path:
        return None
    if path.endswith((".py", ".pyw", ".md", ".json", ".yaml", ".yml", ".xml")):
        return None
    parts = path.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    return "/".join(parts) + ".py"


def _get(model: Any, attr: str) -> str:
    value = getattr(model, attr, None)
    return value.strip() if isinstance(value, str) else ""


@lru_cache(maxsize=8)
def _cached_tools_authority(repo_root: str) -> ToolsVendorAuthority:
    """Cache the (subprocess-backed) vendor authority check per repo root."""
    return inspect_tools_vendor_authority(Path(repo_root))


def _resolve_tools_vendor(path: str, repo_root: Path) -> TileTargetResolution:
    authority = _cached_tools_authority(str(repo_root))
    if not authority.available:
        return TileTargetResolution(
            resolvable=False,
            kind=KIND_TOOLS_VENDOR,
            reason=authority.reason or "vendor/ud-tools authority unavailable",
        )
    target = authority.root / path
    if target.exists():
        return TileTargetResolution(True, KIND_TOOLS_VENDOR, target)
    return TileTargetResolution(
        resolvable=False,
        kind=KIND_TOOLS_VENDOR,
        target=target,
        reason=f"declared Tools path does not exist in vendor/ud-tools: {path}",
    )


def _resolve_sibling(
    path: str, source_root: str, repo_root: Path
) -> TileTargetResolution:
    root = Path(source_root)
    if not root.is_absolute():
        in_repo = repo_root / root
        root = in_repo if in_repo.exists() else repo_root.parent / root
    if not root.exists():
        return TileTargetResolution(
            resolvable=False,
            kind=KIND_SIBLING,
            reason=f"sibling checkout not found: {source_root}",
        )
    target = root / path
    if target.exists():
        return TileTargetResolution(True, KIND_SIBLING, target)
    return TileTargetResolution(
        resolvable=False,
        kind=KIND_SIBLING,
        target=target,
        reason=f"declared path missing inside sibling {source_root}: {path}",
    )


def _resolve_shared_repo(path: str, repo_root: Path) -> TileTargetResolution:
    for candidate in (repo_root / path, repo_root.parent / path):
        if candidate.is_dir():
            return TileTargetResolution(True, KIND_SHARED_REPO, candidate)
    return TileTargetResolution(
        resolvable=False,
        kind=KIND_SHARED_REPO,
        reason=f"shared repo folder not found beside checkout: {path}",
    )


def _resolve_local_file(path: str, repo_root: Path) -> TileTargetResolution:
    as_module = module_string_to_relpath(path)
    candidates = [repo_root / path]
    if as_module is not None:
        module_path = repo_root / as_module
        candidates = [
            module_path,
            module_path.with_suffix("") / "__init__.py",
        ]
    for candidate in candidates:
        if candidate.exists():
            return TileTargetResolution(True, KIND_FILE, candidate)
    return TileTargetResolution(
        resolvable=False,
        kind=KIND_FILE,
        target=candidates[0],
        reason=f"declared path does not exist in the repository: {path}",
    )


def _resolve_virtual(path: str, repo_root: Path) -> TileTargetResolution:
    """Genuinely validate a ``virtual/*`` pseudo-path (issue #8854).

    A virtual target is resolvable only when it is registered in
    ``VIRTUAL_TARGETS`` (or matches a ``VIRTUAL_PREFIXES`` namespace) *and*
    the backing artifact its handler dispatches to exists in this checkout.
    """
    backing_rel = VIRTUAL_TARGETS.get(path)
    if backing_rel is None:
        for prefix, prefix_backing in VIRTUAL_PREFIXES.items():
            if path.startswith(prefix):
                backing_rel = prefix_backing
                break
    if backing_rel is None:
        return TileTargetResolution(
            resolvable=False,
            kind=KIND_VIRTUAL,
            reason=f"unknown virtual target (not in VIRTUAL_TARGETS): {path}",
        )
    backing = repo_root / backing_rel
    if backing.exists():
        return TileTargetResolution(True, KIND_VIRTUAL, backing)
    return TileTargetResolution(
        resolvable=False,
        kind=KIND_VIRTUAL,
        target=backing,
        reason=(
            f"virtual target {path} lost its backing handler artifact: "
            f"{backing_rel} does not exist"
        ),
    )


def resolve_tile_target(model: Any, repo_root: Path) -> TileTargetResolution:
    """Resolve a tile/model's declared launch target against this checkout.

    Args:
        model: A ``ModelConfig`` or ``LauncherTile``-like object exposing
            ``id`` and optionally ``path``, ``type``, ``provider``, and
            ``source_root`` attributes.
        repo_root: The UpstreamDrift checkout root.

    Returns:
        A :class:`TileTargetResolution` describing whether the target can
        be launched from this machine, and why not when it cannot.
    """
    require(bool(_get(model, "id")), "model must declare a non-empty id", model)
    require(repo_root.is_dir(), "repo_root must be an existing directory", repo_root)

    path = _get(model, "path")
    tile_type = _get(model, "type")

    if not path:
        if tile_type in PATHLESS_TYPES:
            return TileTargetResolution(True, KIND_PATHLESS)
        return TileTargetResolution(
            resolvable=False,
            kind=KIND_PATHLESS,
            reason=f"tile type {tile_type!r} declares no launch path",
        )
    if path.startswith("virtual/"):
        return _resolve_virtual(path, repo_root)
    if _get(model, "provider") == "tools":
        return _resolve_tools_vendor(path, repo_root)
    source_root = _get(model, "source_root")
    if source_root:
        return _resolve_sibling(path, source_root, repo_root)
    if tile_type == "shared_repo":
        return _resolve_shared_repo(path, repo_root)
    return _resolve_local_file(path, repo_root)
