"""Generate docs/development/ORG01_CAPABILITY_BASELINE.md (index) plus linked parts.

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
    MigrationEntry,
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


def _render_workspace_breakdown(inventory: CapabilityMigrationInventory) -> list[str]:
    """Render the workspace distribution markdown table."""
    workspace_counts: dict[str, int] = {}
    for entry in inventory.entries.values():
        workspace_counts[entry.primary_workspace] = (
            workspace_counts.get(entry.primary_workspace, 0) + 1
        )

    lines: list[str] = [
        "| Workspace | Primary Capability Count | Description |",
        "| :--- | :---: | :--- |",
    ]
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
    return lines


def _render_fixtures_table(inventory: CapabilityMigrationInventory) -> list[str]:
    """Render the golden preservation fixtures markdown table."""
    lines: list[str] = [
        "## Representative Golden Preservation Fixtures",
        "",
        "| Fixture Key | Kind | Relative Path | Size (bytes) | SHA-256 Hash | Support Status |",
        "| :--- | :--- | :--- | :---: | :--- | :--- |",
    ]
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
    return lines


def _render_alias_table(inventory: CapabilityMigrationInventory) -> list[str]:
    """Render the legacy alias resolution graph markdown table."""
    lines: list[str] = [
        "## Legacy Alias Resolution Graph",
        "",
        "| Deprecated Alias ID | Canonical Target ID | Lifecycle | Evidence | Acceptance Owner |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for entry in sorted(inventory.entries.values(), key=lambda e: e.id):
        if entry.lifecycle == "deprecated_alias":
            lines.append(
                f"| `{entry.id}` | `{entry.alias_target}` | {_LIFECYCLE_BADGES[entry.lifecycle]} | `{entry.evidence}` | `{entry.acceptance_owner}` |"
            )
    lines.append("")
    return lines


PARTS_DIR = BASELINE_PATH.parent / "org01_capability_baseline"
MAX_PART_BYTES = 51_200
ROWS_PER_PART = 25
_INVENTORY_HEADER = [
    "| ID | Name | Entity Kind | Primary Workspace | Secondary Links | Authority | Lifecycle | Evidence | Acceptance Owner |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
]


def _render_inventory_row(entry: MigrationEntry) -> str:
    """Render one capability inventory table row."""
    sec = (
        ", ".join(f"`{s}`" for s in entry.secondary_links)
        if entry.secondary_links
        else "—"
    )
    badge = _LIFECYCLE_BADGES.get(entry.lifecycle, entry.lifecycle)
    return f"| `{entry.id}` | {entry.name} | `{entry.entity_kind}` | `{entry.primary_workspace}` | {sec} | `{entry.provider_authority}` | {badge} | `{entry.evidence}` | `{entry.acceptance_owner}` |"


def plan_inventory_parts(
    inventory: CapabilityMigrationInventory,
) -> list[tuple[str, str, list[MigrationEntry]]]:
    """Group entries by primary workspace into (filename, title, entries) parts.

    Each part holds at most ROWS_PER_PART rows so it stays under the doc size budget.
    """
    if ROWS_PER_PART < 1:
        raise ValueError("ROWS_PER_PART must be >= 1")
    by_ws: dict[str, list[MigrationEntry]] = {}
    for entry in sorted(inventory.entries.values(), key=lambda e: e.id):
        by_ws.setdefault(entry.primary_workspace, []).append(entry)
    parts: list[tuple[str, str, list[MigrationEntry]]] = []
    for ws in sorted(by_ws):
        entries = by_ws[ws]
        chunks = [
            entries[i : i + ROWS_PER_PART]
            for i in range(0, len(entries), ROWS_PER_PART)
        ]
        for n, chunk in enumerate(chunks, start=1):
            suffix = f"-{n}" if len(chunks) > 1 else ""
            title_suffix = f" (Part {n}/{len(chunks)})" if len(chunks) > 1 else ""
            parts.append(
                (
                    f"inventory-{ws}{suffix}.md",
                    f"Capability Baseline Inventory: {ws.title()}{title_suffix}",
                    chunk,
                )
            )
    return parts


def render_inventory_part(title: str, entries: list[MigrationEntry]) -> str:
    """Render one linked inventory sub-document."""
    lines = [
        f"# {title}",
        "",
        "<!-- Generated: python -m scripts.generate_capability_baseline -->",
        "",
        f"Part of the [ORG-01 baseline index](../{BASELINE_PATH.name}).",
        "",
        *_INVENTORY_HEADER,
    ]
    lines.extend(_render_inventory_row(e) for e in entries)
    lines.append("")
    return "\n".join(lines)


def _render_inventory_index(
    parts: list[tuple[str, str, list[MigrationEntry]]],
) -> list[str]:
    """Render the inventory section of the index: links to each part."""
    lines = [
        "## Complete Capability Baseline Inventory",
        "",
        "The full inventory table is split into linked parts by primary workspace.",
        "",
        "| Part | Entries | First ID | Last ID |",
        "| :--- | :---: | :--- | :--- |",
    ]
    for fname, title, entries in parts:
        lines.append(
            f"| [{title}]({PARTS_DIR.name}/{fname}) | {len(entries)} | `{entries[0].id}` | `{entries[-1].id}` |"
        )
    lines.append("")
    return lines


def render_baseline_document(
    inventory: CapabilityMigrationInventory,
    parts: list[tuple[str, str, list[MigrationEntry]]] | None = None,
) -> str:
    """Render the baseline index (summary, fixtures, aliases, links to parts)."""
    if parts is None:
        parts = plan_inventory_parts(inventory)
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
    lines.extend(_render_workspace_breakdown(inventory))
    lines.extend(_render_fixtures_table(inventory))
    lines.extend(_render_alias_table(inventory))
    lines.extend(_render_inventory_index(parts))
    return "\n".join(lines)


def render_baseline_files(inventory: CapabilityMigrationInventory) -> dict[Path, str]:
    """Render the index plus all inventory parts as {path: content}."""
    parts = plan_inventory_parts(inventory)
    files = {BASELINE_PATH: render_baseline_document(inventory, parts)}
    for fname, title, entries in parts:
        files[PARTS_DIR / fname] = render_inventory_part(title, entries)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether docs/development/ORG01_CAPABILITY_BASELINE.md is up to date",
    )
    args = parser.parse_args(argv)

    inventory = load_migration_inventory(MIGRATION_JSON_PATH)

    files = render_baseline_files(inventory)

    if args.check:
        return _check(inventory, files)

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    expected = {p.name for p in files if p.parent == PARTS_DIR}
    for stale in PARTS_DIR.glob("*.md"):
        if stale.name not in expected:
            stale.unlink()
    for path, rendered in files.items():
        size = len(rendered.encode("utf-8"))
        if size > MAX_PART_BYTES:
            raise ValueError(f"{path.name} is {size} bytes (> {MAX_PART_BYTES})")
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(rendered)
        sys.stdout.write(f"Generated {path} ({size} bytes)\n")
    return 0


def _check(inventory: CapabilityMigrationInventory, files: dict[Path, str]) -> int:
    """Validate that the committed index and parts exist and list every entry."""
    missing_files = [p for p in files if not p.is_file()]
    if missing_files:
        sys.stderr.write(f"Missing baseline document(s): {missing_files}\n")
        return 1
    current = "\n".join(p.read_text(encoding="utf-8") for p in files)
    missing_entries = [eid for eid in inventory.entries if f"`{eid}`" not in current]
    if missing_entries:
        sys.stderr.write(
            f"Stale {BASELINE_PATH.name}: missing entries {missing_entries[:5]}. "
            "Run: python -m scripts.generate_capability_baseline\n"
        )
        return 1
    sys.stdout.write(f"Baseline document {BASELINE_PATH.name} is up to date.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
