#!/usr/bin/env python3
"""Enumerate the testable surface of UpstreamDrift.

Feeds ``docs/testing/functional-test-plan.md``. Everything the functional test
matrix claims about inventory size must be reproducible from this script, so a
matrix row can never be written against a feature that does not exist.

Surfaces enumerated:

* native launcher tiles      -- ``src/config/models.yaml`` via ``ModelRegistry``
* shared/web launcher tiles  -- ``src/config/launcher_manifest.json``
* React routes               -- ``ui/src/App.tsx``
* backend routes             -- ``src.api.local_server.create_local_app()``
* PyQt6 menu entries         -- the built launcher (requires a Qt platform)
* feature-parity entries     -- ``src/config/feature_parity.json``

Usage::

    QT_QPA_PLATFORM=offscreen python3 scripts/testing/enumerate_test_surface.py --json
    python3 scripts/testing/enumerate_test_surface.py --native-table
    python3 scripts/testing/enumerate_test_surface.py --summary

Menu enumeration is skipped unless ``--menus`` is given, because it constructs
the real launcher and is therefore the slowest and most fragile probe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tools_source_root(repo_root: Path, env_value: str | None) -> Path:
    """Resolve the Tools source tree used for shared Sidekick imports."""
    if env_value:
        tools_root = Path(env_value).expanduser().resolve()
        tools_src = tools_root / "src"
        if not tools_src.is_dir():
            raise RuntimeError(
                "TOOLS_REPO_PATH must point to a Tools checkout containing "
                f"a src/ directory, got: {tools_root}"
            )
        return tools_src

    vendor_src = repo_root / "vendor" / "ud-tools" / "src"
    if vendor_src.is_dir():
        return vendor_src

    sibling_src = repo_root.parent / "Tools" / "src"
    if sibling_src.is_dir():
        return sibling_src

    return vendor_src


def _bootstrap_paths(repo_root: Path, env_value: str | None) -> tuple[Path, ...]:
    """Return import roots needed before route modules are imported."""
    tools_src = _tools_source_root(repo_root, env_value)
    return (
        tools_src / "shared" / "python",
        tools_src,
        repo_root,
        repo_root / "src",
        repo_root / "src" / "shared" / "python",
    )


for _p in reversed(_bootstrap_paths(REPO_ROOT, os.environ.get("TOOLS_REPO_PATH"))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def native_tiles() -> list[dict[str, Any]]:
    """Return the native launcher inventory from ``models.yaml``."""
    import yaml

    raw = yaml.safe_load(
        (REPO_ROOT / "src" / "config" / "models.yaml").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for model in raw["models"]:
        launcher = model.get("launcher") or {}
        rows.append(
            {
                "id": model["id"],
                "name": model.get("name"),
                "category": launcher.get("category"),
                "status": launcher.get("status"),
                "hidden": bool(launcher.get("hidden", False)),
                "type": model.get("type"),
                "path": model.get("path"),
            }
        )
    return sorted(rows, key=lambda r: (r["category"] or "", r["id"]))


def shared_tiles() -> list[dict[str, Any]]:
    """Return the shared-manifest inventory as the loader resolves it."""
    from src.config.launcher_manifest_loader import LauncherManifest

    manifest = LauncherManifest.load()
    rows = []
    for tile in manifest.visible_tiles:
        web = getattr(tile, "web", None)
        rows.append(
            {
                "id": tile.id,
                "name": tile.name,
                "category": tile.category,
                "status": tile.status,
                "type": tile.type,
                "order": getattr(tile, "order", None),
                "web_route": getattr(web, "route", None) if web else None,
            }
        )
    return sorted(rows, key=lambda r: (r["category"] or "", r["id"]))


def runtime_models() -> list[str]:
    """Return every model id the native launcher would offer at runtime."""
    from src.config.launcher_manifest_loader import REGISTRY_PATH
    from src.shared.python.config.model_registry import ModelRegistry

    return sorted(m.id for m in ModelRegistry(REGISTRY_PATH).get_all_models())


def react_routes() -> list[str]:
    """Return every route path declared in the React shell."""
    app = (REPO_ROOT / "ui" / "src" / "App.tsx").read_text(encoding="utf-8")
    return re.findall(r'path="([^"]+)"', app)


def backend_routes() -> list[dict[str, str]]:
    """Return the mounted FastAPI route table."""
    from src.api.local_server import create_local_app

    app = create_local_app()
    rows = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or {"WEBSOCKET"}
        rows.append(
            {
                "path": getattr(route, "path", "?"),
                "methods": ",".join(sorted(methods - {"HEAD", "OPTIONS"})),
            }
        )
    return sorted(rows, key=lambda r: (r["path"], r["methods"]))


def parity_entries() -> dict[str, Any]:
    """Return the feature-parity registry keyed by feature id."""
    raw = json.loads(
        (REPO_ROOT / "src" / "config" / "feature_parity.json").read_text(
            encoding="utf-8"
        )
    )
    return raw["features"]


def menu_entries() -> list[dict[str, Any]]:
    """Return every menu entry of the built PyQt6 launcher."""
    from PyQt6.QtWidgets import QApplication, QMenuBar

    app = QApplication.instance() or QApplication([])
    assert app is not None
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    launcher = UpstreamDriftLauncher()
    rows: list[dict[str, Any]] = []

    def walk(menu: Any, path: str) -> None:
        for action in menu.actions():
            if action.isSeparator():
                continue
            label = action.text()
            here = f"{path} → {label}" if path else label
            submenu = action.menu()
            rows.append(
                {
                    "path": here,
                    "enabled": action.isEnabled(),
                    "shortcut": action.shortcut().toString() or None,
                    "is_submenu": submenu is not None,
                }
            )
            if submenu is not None:
                walk(submenu, here)

    bars = launcher.findChildren(QMenuBar)
    if bars:
        walk(bars[0], "")
    return rows


def collect(include_menus: bool) -> dict[str, Any]:
    """Assemble the whole surface snapshot."""
    native = native_tiles()
    shared = shared_tiles()
    runtime = runtime_models()
    native_ids = set(runtime)
    shared_ids = {r["id"] for r in shared}
    data: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "native_tiles": native,
        "shared_tiles": shared,
        "runtime_model_ids": runtime,
        "react_routes": react_routes(),
        "backend_routes": backend_routes(),
        "parity": parity_entries(),
        "divergence": {
            "shared_by_both": sorted(native_ids & shared_ids),
            "native_only": sorted(native_ids - shared_ids),
            "web_only": sorted(shared_ids - native_ids),
        },
    }
    if include_menus:
        data["menu_entries"] = menu_entries()
    return data


def summarise(data: dict[str, Any]) -> str:
    """Render the one-screen counts the matrix quotes."""
    backend = data["backend_routes"]
    non_v1 = {r["path"] for r in backend if not r["path"].startswith("/api/v1")}
    lines = [
        f"native models.yaml tiles : {len(data['native_tiles'])}",
        f"runtime native models    : {len(data['runtime_model_ids'])}",
        f"shared manifest tiles    : {len(data['shared_tiles'])}",
        f"  shared by both surfaces: {len(data['divergence']['shared_by_both'])}",
        f"  native-only            : {len(data['divergence']['native_only'])}",
        f"  web-only               : {len(data['divergence']['web_only'])}",
        f"react routes             : {len(data['react_routes'])}",
        f"backend routes (mounted) : {len(backend)}",
        f"backend paths (non-/v1)  : {len(non_v1)}",
        f"parity entries           : {len(data['parity'])}",
    ]
    if "menu_entries" in data:
        lines.append(f"menu entries             : {len(data['menu_entries'])}")
    return "\n".join(lines)


def native_table(data: dict[str, Any]) -> str:
    """Render the NAT-* matrix table body."""
    classes = {"ready": "A", "beta": "B", "experimental": "B", "deprecated": "C"}
    out = []
    for i, row in enumerate(data["native_tiles"], 1):
        cls = classes.get(row["status"] or "", "B")
        out.append(
            f"| NAT-{i:02d} | `{row['id']}` | {row['category']} | "
            f"{row['name']} | {row['status']} | {cls} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="dump the full snapshot")
    parser.add_argument("--summary", action="store_true", help="counts only (default)")
    parser.add_argument("--native-table", action="store_true", help="NAT-* table body")
    parser.add_argument("--menus", action="store_true", help="also enumerate menus")
    args = parser.parse_args(argv)

    data = collect(include_menus=args.menus)

    if args.json:
        json.dump(data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif args.native_table:
        print(native_table(data))
    else:
        print(summarise(data))
        if args.menus:
            for entry in data["menu_entries"]:
                mark = "" if entry["enabled"] else "  (disabled)"
                sc = f"  <{entry['shortcut']}>" if entry["shortcut"] else ""
                print(f"  {entry['path']}{mark}{sc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
