"""Generate docs/development/feature_parity_matrix.md from feature_parity.json.

The committed markdown matrix must always match the registry; the freshness
test in tests/config/feature_parity/test_matrix_freshness.py regenerates the
doc and compares it byte-for-byte (CI gate for issue #7445 / epic #7462).

Usage:
    python -m scripts.generate_feature_parity_matrix [--check]

``--check`` exits non-zero (without writing) when the committed doc is stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.feature_parity_loader import (  # noqa: E402
    REGISTRY_PATH,
    FeatureParityEntry,
    FeatureParityRegistry,
)

MATRIX_PATH = REPO_ROOT / "docs" / "development" / "feature_parity_matrix.md"

_STATUS_BADGES = {
    "parity": "✅ parity",
    "gap": "🔴 gap",
    "exempt": "⚪ exempt",
    "api_only": "🔌 api_only",
}


def _format_cell(value: str | None) -> str:
    """Render a path cell, escaping nothing (paths contain no pipes)."""
    return f"`{value}`" if value else "—"


def _format_tracking(entry: FeatureParityEntry) -> str:
    """Render the tracking column (issue link or exemption reason)."""
    parts: list[str] = []
    if entry.issue is not None:
        parts.append(f"#{entry.issue}")
    if entry.reason:
        parts.append(entry.reason)
    if entry.pending_decision:
        parts.append("**pending decision (#7460)**")
    return " — ".join(parts) if parts else "—"


def render_matrix(registry: FeatureParityRegistry) -> str:
    """Render the feature-parity registry as a markdown matrix.

    Args:
        registry: Loaded feature-parity registry

    Returns:
        Full markdown document content (deterministic for a given registry)
    """
    counts = {
        status: len(registry.by_status(status))
        for status in ("parity", "gap", "exempt", "api_only")
    }
    lines = [
        "# Feature Parity Matrix (PyQt6 ↔ Tauri/React)",
        "",
        "<!-- AUTO-GENERATED — do not edit by hand. -->",
        "<!-- Regenerate with: python -m scripts.generate_feature_parity_matrix -->",
        "",
        "Generated from [`src/config/feature_parity.json`](../../src/config/feature_parity.json)"
        f" (registry v{registry.version}).",
        "The PyQt6 desktop app is the canonical model; the web app must match",
        "(epic #7462, registry mechanism #7445).",
        "",
        f"**Summary:** {counts['parity']} parity · {counts['gap']} gap ·"
        f" {counts['exempt']} exempt · {counts['api_only']} api_only"
        f" ({sum(1 for e in registry.exemptions if e.pending_decision)}"
        " pending decision in #7460).",
        "",
        "| Feature | Status | PyQt6 | API | Web | Tracking |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in registry.entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{entry.feature_id}`<br>{entry.title}",
                    _STATUS_BADGES[entry.status],
                    _format_cell(entry.pyqt),
                    _format_cell(entry.api),
                    _format_cell(entry.web),
                    _format_tracking(entry),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Launcher tile coverage",
        "",
        "Tiles from `src/config/launcher_manifest.json` mapped to registry entries:",
        "",
        "| Tile id | Feature |",
        "| --- | --- |",
    ]
    tile_to_feature = {
        tile: entry.feature_id for entry in registry.entries for tile in entry.tiles
    }
    for tile_id, feature_id in sorted(tile_to_feature.items()):
        lines.append(f"| `{tile_id}` | `{feature_id}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success / fresh, 1 when ``--check`` finds a stale doc.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed matrix matches the registry; do not write.",
    )
    args = parser.parse_args(argv)

    registry = FeatureParityRegistry.load(REGISTRY_PATH)
    rendered = render_matrix(registry)

    if args.check:
        current = (
            MATRIX_PATH.read_text(encoding="utf-8") if MATRIX_PATH.exists() else ""
        )
        if current != rendered:
            print(  # noqa: T201 - CLI tool output
                f"STALE: {MATRIX_PATH} does not match feature_parity.json. "
                "Run: python -m scripts.generate_feature_parity_matrix"
            )
            return 1
        print(f"OK: {MATRIX_PATH} is up to date.")  # noqa: T201 - CLI tool output
        return 0

    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {MATRIX_PATH}")  # noqa: T201 - CLI tool output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
