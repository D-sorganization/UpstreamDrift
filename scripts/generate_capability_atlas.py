"""Regenerate the Project Map companion and offline website atlas; --check is read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.capability_atlas.model import build
from scripts.capability_atlas.goals import prerequisite_mermaid, routes
from scripts.capability_atlas.render import (
    catalog,
    document,
    edge_table,
    mermaid,
    svg,
    tile_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


def outputs(root: Path) -> dict[Path, str]:
    """Return deterministic expected artifacts without writing to the checkout."""
    graph = build(root)
    template = (root / "scripts/capability_atlas/template.html").read_text(
        encoding="utf-8"
    )
    for key, value in {
        "SYSTEM_SVG": svg(graph, "system"),
        "WORKFLOW_SVG": svg(graph, "workflow"),
        "SYSTEM_TABLE": edge_table(graph, "system"),
        "WORKFLOW_TABLE": edge_table(graph, "workflow"),
        "FEATURES": catalog(graph),
        "TILES": tile_catalog(graph),
        "CAPTURE_GOALS": routes(graph),
        "FEATURE_COUNT": str(len(graph["features"])),
        "TILE_COUNT": str(len(graph["tiles"])),
    }.items():
        template = template.replace("{{" + key + "}}", value)
    public = root / "ui/public/capability-atlas"
    return {
        public / "index.html": template,
        public / "graph.json": json.dumps(graph, indent=2, ensure_ascii=False) + "\n",
        public / "system.mmd": mermaid(graph, "system"),
        public / "capture-workflow.mmd": mermaid(graph, "workflow"),
        public / "capture-goals.mmd": prerequisite_mermaid(graph),
        root / "docs/architecture/CAPABILITY_ATLAS.md": document(graph),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for path, content in outputs(ROOT).items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        print("Stale capability atlas: " + ", ".join(stale))
        print("Run python3 -m scripts.generate_capability_atlas")
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
