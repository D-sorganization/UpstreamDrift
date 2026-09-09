#!/usr/bin/env python3
"""Launcher logo-family gate (#9482).

A launcher tile may share a logo file only with tiles in its declared family.
The family is derived from the manifest's own fields, never from taste:

- a tile with ``engine_type`` belongs to the ``engine:<engine_type>`` family
  (one icon per physics-engine family, mirroring
  ``launcher_manifest_loader._ENGINE_LOGOS``);
- every other tile belongs to the ``category:<category>`` family.

Two rules are enforced:

1. **No cross-family reuse.** A logo used by more than one tile must be used
   exclusively by tiles of a single family.
2. **No accidental reuse.** A shared logo must be declared in
   ``DECLARED_SHARED_LOGOS`` with the family it belongs to and a rationale,
   so that reuse is a decision rather than an accident.

Structural checks (no Qt required): the manifest parses as JSON, every
referenced logo file exists under ``assets/logos``, and the family/declaration
rules above hold. Run ``python scripts/check_launcher_logo_families.py`` from
the repo root; exit code 0 means the gate passes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "src" / "config" / "launcher_manifest.json"
LOGOS_DIR = REPO_ROOT / "assets" / "logos"


@dataclass(frozen=True)
class SharedLogoDeclaration:
    """A deliberate, documented decision to share one logo within a family.

    Attributes:
        family: The single family key allowed to use the logo.
        rationale: Why the sharing is intentional for these tiles.
    """

    family: str
    rationale: str


#: Every logo used by more than one tile MUST appear here. Anything shared
#: without a declaration fails the gate: reuse must be a decision.
DECLARED_SHARED_LOGOS: dict[str, SharedLogoDeclaration] = {
    # Engine families: one icon per engine, mirroring the loader's
    # ``_ENGINE_LOGOS`` — the dashboard tiles are views onto the same engine.
    "mujoco_humanoid.svg": SharedLogoDeclaration(
        "engine:mujoco",
        "MuJoCo interface and its dashboard are one engine family.",
    ),
    "drake.svg": SharedLogoDeclaration(
        "engine:drake",
        "Drake golf interface and its dashboard are one engine family.",
    ),
    "pinocchio.svg": SharedLogoDeclaration(
        "engine:pinocchio",
        "Pinocchio golf interface and its dashboard are one engine family.",
    ),
    # Category families where the repo owns fewer distinct icons than tiles.
    "data_explorer.svg": SharedLogoDeclaration(
        "category:tool",
        "Data import/processing/generation tools from the Tools provider "
        "share the data-explorer artwork.",
    ),
    "urdf_icon.svg": SharedLogoDeclaration(
        "category:tool",
        "URDF/humanoid model building tools (explorer, builder, robotics).",
    ),
    "project_map.svg": SharedLogoDeclaration(
        "category:tool",
        "Map/diagram tools: suite map, terrain configuration, P&ID diagrams.",
    ),
    "movement_optimizer_icon.svg": SharedLogoDeclaration(
        "category:tool",
        "Live model-interaction tools: force overlays, actuator tuning, "
        "realtime stream (verbatim copy of the movement-optimizer icon).",
    ),
    "pose_studio.svg": SharedLogoDeclaration(
        "category:tool",
        "Pose manipulation tools; starting_pose_matcher is a hidden alias "
        "for the motion-target preview.",
    ),
    "capture_rig.svg": SharedLogoDeclaration(
        "category:tool",
        "3-D scene ingest/visualization tools (capture rig, Unreal/VR).",
    ),
    "video_analyzer.svg": SharedLogoDeclaration(
        "category:tool",
        "Tools-provider video analysis and processing tiles.",
    ),
    "sidekick.svg": SharedLogoDeclaration(
        "category:tool",
        "AI assistant/agent tiles (chat assistant, AIP protocol); verbatim "
        "copy of the Sidekick chat icon.",
    ),
    "biomechanics.svg": SharedLogoDeclaration(
        "category:biomechanics",
        "Human-body figure for the biomechanics workspace/analysis tiles.",
    ),
    "exercise_dashboard.svg": SharedLogoDeclaration(
        "category:analysis",
        "Analysis dashboards (API endpoints, perturbation analysis).",
    ),
    "motion_target_preview.svg": SharedLogoDeclaration(
        "category:motion_matching",
        "Motion-matching tiles share the multi-source target preview icon.",
    ),
    "golf_logo.svg": SharedLogoDeclaration(
        "category:simulation",
        "Golf simulation suite and shot tracer share the suite brand icon.",
    ),
    "cross_engine.svg": SharedLogoDeclaration(
        "category:simulation",
        "Cross-engine comparison tiles (dashboard, backend comparison).",
    ),
    "pendulum.svg": SharedLogoDeclaration(
        "category:physics_engine",
        "Double-pendulum golfer dynamics companions share the pendulum icon.",
    ),
}


def family_key(tile: dict[str, Any]) -> str:
    """Derive the declared family of a manifest tile from its own fields.

    Args:
        tile: Raw tile dictionary from ``launcher_manifest.json``.

    Returns:
        ``engine:<engine_type>`` when the tile declares an engine type,
        otherwise ``category:<category>``.
    """
    engine_type = tile.get("engine_type")
    if engine_type:
        return f"engine:{engine_type}"
    return f"category:{tile.get('category')}"


def load_manifest_tiles() -> list[dict[str, Any]]:
    """Load the raw launcher manifest tiles.

    Returns:
        The list of tile dictionaries exactly as declared in the manifest.

    Raises:
        ValueError: If the manifest does not parse as JSON with a tile list.
    """
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{MANIFEST_PATH} does not parse as JSON: {exc}") from exc
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError(f"{MANIFEST_PATH} declares no tile list")
    return tiles


def tiles_by_logo(tiles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group tiles by the logo file they reference.

    Args:
        tiles: Raw tile dictionaries from the manifest.

    Returns:
        Mapping of logo filename to the tiles using it.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for tile in tiles:
        grouped.setdefault(tile["logo"], []).append(tile)
    return grouped


def find_missing_logos(tiles: list[dict[str, Any]]) -> list[str]:
    """Return tile ids whose logo file does not exist under ``assets/logos``.

    Args:
        tiles: Raw tile dictionaries from the manifest.

    Returns:
        Tile ids with a missing logo file.
    """
    return [
        tile["id"]
        for tile in tiles
        if not (LOGOS_DIR / Path(tile["logo"]).name).is_file()
    ]


def find_logo_family_violations(
    tiles: list[dict[str, Any]],
) -> list[str]:
    """Report every logo-reuse violation against the declared family rule.

    Args:
        tiles: Raw tile dictionaries from the manifest.

    Returns:
        One human-readable message per violation: cross-family reuse,
        undeclared sharing, or a declaration whose family does not match.
    """
    violations: list[str] = []
    for logo, users in tiles_by_logo(tiles).items():
        families = {family_key(tile) for tile in users}
        if len(users) == 1:
            continue
        if len(families) > 1:
            joined = ", ".join(sorted(families))
            violations.append(
                f"{logo} is used by {len(users)} tiles across families "
                f"[{joined}]: " + ", ".join(sorted(t["id"] for t in users))
            )
            continue
        declaration = DECLARED_SHARED_LOGOS.get(logo)
        family = next(iter(families))
        if declaration is None:
            violations.append(
                f"{logo} is shared by {len(users)} {family} tiles but is not "
                "declared in DECLARED_SHARED_LOGOS: "
                + ", ".join(sorted(t["id"] for t in users))
            )
        elif declaration.family != family:
            violations.append(
                f"{logo} declares family '{declaration.family}' but is used "
                f"by '{family}' tiles: " + ", ".join(sorted(t["id"] for t in users))
            )
    return violations


def find_stale_declarations(tiles: list[dict[str, Any]]) -> list[str]:
    """Return declarations whose logo is no longer shared or no longer used.

    Args:
        tiles: Raw tile dictionaries from the manifest.

    Returns:
        One message per declaration that no longer matches manifest reality.
    """
    grouped = tiles_by_logo(tiles)
    return [
        f"{logo} is declared as a shared logo but is used by "
        f"{len(grouped.get(logo, []))} tiles"
        for logo in DECLARED_SHARED_LOGOS
        if len(grouped.get(logo, [])) < 2
    ]


def main() -> int:
    """Run the logo-family gate and exit nonzero on any violation."""
    try:
        tiles = load_manifest_tiles()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    failures = (
        find_missing_logos(tiles)
        + find_logo_family_violations(tiles)
        + find_stale_declarations(tiles)
    )
    if failures:
        print("Launcher logo-family gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        f"Launcher logo-family gate passed: {len(tiles)} tiles, "
        f"{len(tiles_by_logo(tiles))} distinct logos, "
        f"{len(DECLARED_SHARED_LOGOS)} declared families."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
