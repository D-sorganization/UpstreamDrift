"""Project registry-owned tile fields into launcher_manifest.json (issue #8089).

``src/config/models.yaml`` is the single authored source of truth for how a
launcher tile *behaves* (name, description, category, handler type, entry
point, engine, capabilities, provider). ``src/config/launcher_manifest.json``
owns only the web presentation (order, SVG logo, readiness chip, web launch
contract).

This script rewrites the registry-owned fields of every *shared* tile ID in
``launcher_manifest.json`` from ``models.yaml``. Tiles that exist only in the
manifest (web-only surfaces such as ``realtime_ws`` or ``aip``) are left
untouched, as are the manifest-owned fields of shared tiles.

Usage:
    python -m scripts.sync_launcher_manifest [--check]

``--check`` exits non-zero (without writing) when the committed manifest is
stale. The same comparison runs in
``tests/config/launcher_manifest/test_registry_manifest_parity.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.launcher_parity import (  # noqa: E402
    REGISTRY_OWNED_FIELDS,
    is_exempt,
)
from src.shared.python.config.model_registry import ModelConfig, ModelRegistry  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "src" / "config" / "launcher_manifest.json"
REGISTRY_PATH = REPO_ROOT / "src" / "config" / "models.yaml"

#: Registry-owned fields that are dropped from the JSON entirely when the
#: registry value is empty, rather than written as ``null``/``[]``.
_OMIT_WHEN_EMPTY = frozenset(
    {"engine_type", "provider", "source_root", "python_paths", "capabilities", "hidden"}
)


def registry_value(model: ModelConfig, field: str) -> Any:
    """Return the JSON-shaped value of a registry-owned ``field``."""
    if field == "category":
        return model.launcher.category if model.launcher else None
    value = getattr(model, field, None)
    if isinstance(value, tuple):
        return list(value)
    return value


def project_tile(tile: dict[str, Any], model: ModelConfig) -> dict[str, Any]:
    """Return ``tile`` with its registry-owned fields taken from ``model``.

    Key order is preserved for fields already present so the committed JSON
    stays reviewable; newly added fields are appended.
    """
    if not isinstance(tile, dict):
        raise TypeError("tile must be a mapping")
    updated = dict(tile)
    tile_id = str(tile.get("id", ""))
    for field in REGISTRY_OWNED_FIELDS:
        if is_exempt(tile_id, field):
            continue
        value = registry_value(model, field)
        if field in _OMIT_WHEN_EMPTY and not value:
            updated.pop(field, None)
            continue
        if value is None:
            continue
        updated[field] = value
    return updated


def build_manifest(
    *,
    manifest_path: Path = MANIFEST_PATH,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Return the manifest document with shared tiles projected from the registry."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Launcher manifest not found: {manifest_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {
        m.id: m for m in ModelRegistry(config_path=registry_path).get_all_models()
    }

    raw["tiles"] = [
        project_tile(tile, models[tile["id"]]) if tile.get("id") in models else tile
        for tile in raw["tiles"]
    ]
    return raw


def render(document: dict[str, Any]) -> str:
    """Serialize the manifest as JSON (prettier re-flows it on commit)."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def is_in_sync(*, manifest_path: Path = MANIFEST_PATH) -> bool:
    """Return True when the committed manifest carries the projected values.

    The comparison is on parsed JSON, not bytes: ``prettier`` (a pre-commit
    hook) re-flows short arrays onto one line, so a byte comparison would
    report a permanently stale file.
    """
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    return committed == build_manifest(manifest_path=manifest_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed manifest is stale (do not write).",
    )
    args = parser.parse_args(argv)

    if is_in_sync():
        sys.stdout.write("launcher_manifest.json is in sync with models.yaml\n")
        return 0

    if args.check:
        sys.stderr.write(
            "launcher_manifest.json is stale.\n"
            "Run: python3 -m scripts.sync_launcher_manifest\n"
        )
        return 1

    MANIFEST_PATH.write_text(render(build_manifest()), encoding="utf-8")
    sys.stdout.write(f"Rewrote {MANIFEST_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
