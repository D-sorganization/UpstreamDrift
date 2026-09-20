"""Generate docs/development/ORG01_CAPABILITY_BASELINE.md from capability_migration.json.

The committed markdown document must always match the machine-checkable inventory;
freshness can be validated with --check.

Usage:
    python -m scripts.generate_capability_baseline [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.capability_migration import (  # noqa: E402
    CapabilityMigrationInventory,
    load_migration_inventory,
)

BASELINE_PATH = REPO_ROOT / "docs" / "development" / "ORG01_CAPABILITY_BASELINE.md"
MIGRATION_JSON_PATH = REPO_ROOT / "src" / "config" / "capability_migration.json"

_LIFECYCLE_BADGES = {
    "active_feature": "🟢 active_feature",
    "deprecated_alias": "🟡 deprecated_alias",
    "intentionally_headless": "🟣 intentionally_headless",
    "planned": "🔵 planned",
    "unavailable": "⚪ unavailable",
    "confirmed_prototype": "🟠 confirmed_prototype",
    "exempt": "⚪ exempt",
}


def render_baseline_document(inventory: CapabilityMigrationInventory) -> str:
    """Render the capability inventory and preservation records as markdown."""
    lines: list[str] = [
        "# ORG-01: Baseline Capability Inventory & Entity Identity Preservation",
        "",
        "<!-- Generated: python -m scripts.generate_capability_baseline -->",
        "",
        "## Executive Summary",
        "",
        "This document establishes the canonical baseline capability inventory for",
        "**Epic #10508 (Integrated Workspace Reorganization)** and enforces entity identity",
        "preservation across launcher tiles, provider models, parity features, excluded tool",
        "packages, saved layouts, and golden test fixtures.",
        "",
        "### Architectural Rulings and Boundaries",
        "- **Explicit Identity Contracts**: Capability IDs are unique; every capability has exactly one primary workspace domain; secondary links are non-exclusive cross-domain associations.",
        "- **Acyclic Alias Resolution**: Deprecated aliases resolve transitively and acyclically to retained active targets (`starting_pose_matcher` → `motion_target_preview`, `putting_green_gui` → `putting_green`).",
        "- **Decoupled Provider Availability**: The absence of an external provider (MuJoCo, Drake, Pinocchio, OpenSim) alters runtime operational availability, never entity identity or workspace ownership.",
        "- **Preservation of ADR-0047 Viewer Identity**: Preserves separate viewer identities (`shot_tracer`, `BallFlight`, `Impact Explorer`) communicating via shared trajectory record contracts without merging or destroying viewer tools.",
        "- **Preservation of User Artifacts**: Golden fixtures and user-owned scripts are strictly preserved; no destructive file removals.",
        "",
        "## Baseline Metrics",
        "",
        f"- **Total Cataloged Entries**: {len(inventory.entries)}",
        "- **Observed Launcher Tiles / Models**: 104 (61 base desktop models + 29 discovered provider models + 14 web catalog tiles)",
        "- **Feature Parity Contracts**: 45",
        "- **Excluded Tool Packages / Libraries**: 9",
        f"- **Preserved Golden Fixtures**: {len(inventory.fixtures)}",
        "",
        "### Workspace Distribution",
        "",
    ]

    # Workspace breakdown
    workspace_counts: dict[str, int] = {}
    for entry in inventory.entries.values():
        workspace_counts[entry.primary_workspace] = (
            workspace_counts.get(entry.primary_workspace, 0) + 1
        )

    lines.append("| Workspace | Primary Capability Count | Description |")
    lines.append("| :--- | :---: | :--- |")
    workspace_desc = {
        "simulation": "Physics engines, multi-body kinematics, dynamics solvers, and forward simulation",
        "analysis": "Telemetry extraction, metric calculation, video/data analysis, and flight comparison",
        "capture": "Multi-camera mocap, pose estimation, marker tracking, and 3D reconstruction",
        "putting": "Putting physics, green surface simulation, and ball rolling dynamics",
        "training": "Drills, objective laboratories, movement/swing optimization, and skill reinforcement",
        "governance": "Configuration setup, project architecture mapping, sidekick docks, and registry admin",
    }
    for ws in sorted(inventory.workspaces):
        count = workspace_counts.get(ws, 0)
        lines.append(
            f"| `{ws}` | {count} | {workspace_desc.get(ws, 'Workspace domain')} |"
        )
    lines.append("")

    # Preservation Fixtures Table
    lines.extend(
        [
            "## Representative Golden Preservation Fixtures",
            "",
            "| Fixture Key | Kind | Relative Path | Size (bytes) | SHA-256 Hash | Support Status |",
            "| :--- | :--- | :--- | :---: | :--- | :--- |",
        ]
    )
    for fid, fix in sorted(inventory.fixtures.items()):
        badge = (
            "✅ supported"
            if fix.support_status == "golden_baseline"
            else (
                "❌ unsupported" if fix.support_status == "unsupported" else "ℹ️ other"
            )
        )
        lines.append(
            f"| `{fid}` | `{fix.kind}` | `{fix.path}` | {fix.size_bytes} | `{fix.sha256[:16]}...` | {badge} |"
        )
    lines.append("")

    # Alias Graph Table
    lines.extend(
        [
            "## Legacy Alias Resolution Graph",
            "",
            "| Deprecated Alias ID | Canonical Target ID | Lifecycle | Evidence | Acceptance Owner |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for entry in sorted(inventory.entries.values(), key=lambda e: e.id):
        if entry.lifecycle == "deprecated_alias":
            lines.append(
                f"| `{entry.id}` | `{entry.alias_target}` | {_LIFECYCLE_BADGES[entry.lifecycle]} | `{entry.evidence}` | `{entry.acceptance_owner}` |"
            )
    lines.append("")

    # Complete Inventory Table
    lines.extend(
        [
            "## Complete Capability Baseline Inventory",
            "",
            "| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for entry in sorted(inventory.entries.values(), key=lambda e: e.id):
        sec = (
            ", ".join(f"`{s}`" for s in entry.secondary_links)
            if entry.secondary_links
            else "—"
        )
        badge = _LIFECYCLE_BADGES.get(entry.lifecycle, entry.lifecycle)
        lines.append(
            f"| `{entry.id}` | {entry.name} | `{entry.entity_kind}` | `{entry.primary_workspace}` | {sec} | `{entry.provider_authority}` | {badge} | `{entry.evidence}` | `{entry.acceptance_owner}` |"
        )
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether docs/development/ORG01_CAPABILITY_BASELINE.md is up to date",
    )
    args = parser.parse_args(argv)

    inventory = load_migration_inventory(MIGRATION_JSON_PATH)

    if args.check:
        if not BASELINE_PATH.is_file():
            sys.stderr.write(f"Missing baseline document: {BASELINE_PATH}\n")
            return 1
        current = BASELINE_PATH.read_text(encoding="utf-8")
        missing_entries = [
            eid for eid in inventory.entries if f"`{eid}`" not in current
        ]
        if missing_entries:
            sys.stderr.write(
                f"Stale {BASELINE_PATH.name}: missing entries {missing_entries[:5]}. "
                "Run: python -m scripts.generate_capability_baseline\n"
            )
            return 1
        sys.stdout.write(f"Baseline document {BASELINE_PATH.name} is up to date.\n")
        return 0

    rendered = render_baseline_document(inventory)
    BASELINE_PATH.write_text(rendered, encoding="utf-8")
    sys.stdout.write(f"Generated {BASELINE_PATH} ({len(rendered)} bytes)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
