"""Source-backed capability model; imports no GUI or physics engine runtime."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from src.config.feature_parity_loader import FeatureParityRegistry
from src.tools.capture_rig.goal_catalog import validate_bindings
from src.tools.capture_rig.goal_planner import CaptureGoalCatalog

SOURCES = (
    "src/config/models.yaml",
    "src/config/launcher_manifest.json",
    "src/config/feature_parity.json",
    "src/config/capability_connections.json",
    "src/tools/capture_rig/workflow.py",
)
Graph = dict[str, Any]


def tile_path(root: Path, target: str) -> str:
    """Resolve Tools targets; virtual/external entries link to their registration."""
    path = target.replace("tools://", "vendor/ud-tools/")
    candidate = root / path
    if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
        return path
    return SOURCES[0]


def workflow_nodes(root: Path) -> list[dict[str, Any]]:
    """Read literal Step metadata without executing readiness or importing Qt."""
    path = "src/tools/capture_rig/workflow.py"
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "Step":
            continue
        fields = {item.arg: item.value for item in call.keywords}
        data = {
            key: ast.literal_eval(fields[key])
            for key in (
                "key",
                "title",
                "purpose",
                "requirements",
                "instructions",
                "actions",
            )
        }
        nodes.append(
            dict(
                id=f"step.{data['key']}",
                title=data["title"],
                description=data["purpose"],
                evidence=path,
                view="workflow",
                **{k: data[k] for k in ("requirements", "instructions", "actions")},
            )
        )
    if not nodes:
        raise ValueError("Capture workflow has no discoverable Step declarations")
    return nodes


def build(root: Path) -> Graph:
    """Build a complete feature/tile catalog with curated, evidenced flow edges."""
    registry = FeatureParityRegistry.load(root / SOURCES[2])
    features = [asdict(entry) | {"id": entry.feature_id} for entry in registry.entries]
    models = yaml.safe_load((root / SOURCES[0]).read_text(encoding="utf-8"))["models"]
    metadata = json.loads((root / SOURCES[1]).read_text(encoding="utf-8"))
    web = {tile["id"]: tile for tile in metadata["tiles"]}
    tiles = [
        {
            "id": m["id"],
            "title": m["name"],
            "description": m.get("description", ""),
            "path": tile_path(root, m.get("path", SOURCES[0])),
            "launch_target": m.get("path", "registered launcher action"),
            "category": m.get("launcher", {}).get("category", "tool"),
            "status": m.get("launcher", {}).get("status", "unknown"),
            "capabilities": web.get(m["id"], {}).get("capabilities", []),
        }
        for m in models
    ]
    connections = json.loads((root / SOURCES[3]).read_text(encoding="utf-8"))
    graph = {
        "version": 1,
        "features": features,
        "tiles": tiles,
        "nodes": connections["nodes"] + workflow_nodes(root),
        "edges": connections["edges"],
        "capture_goals": connections["capture_goals"],
        "inputs": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in SOURCES
        },
    }
    validate_graph(graph, root)
    return graph


def source_path(root: Path, value: str) -> None:
    """Require an existing source inside the checkout, including pinned vendors."""
    path = root / value
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"Unsafe evidence path: {value}")
    if not path.exists() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            f"Missing evidence path: {value}; initialize pinned submodules"
        )


def validate_graph(graph: Graph, root: Path) -> None:
    """Reject incomplete identities, dangling edges and unsupported integration claims."""
    ids = [node["id"] for node in graph["nodes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate graph node IDs")
    goals = CaptureGoalCatalog.model_validate(graph["capture_goals"])
    validate_bindings(
        goals,
        ids,
        (
            node["id"].removeprefix("step.")
            for node in graph["nodes"]
            if node["id"].startswith("step.") and "actions" in node
        ),
    )
    for node in graph["nodes"]:
        if not node["id"] or not node["title"]:
            raise ValueError("Nodes need an ID and title")
        source_path(root, node["evidence"])
    for edge in graph["edges"]:
        if edge["source"] not in ids or edge["target"] not in ids:
            raise ValueError("Unknown edge endpoint")
        if edge["kind"] not in {"runtime", "file", "workflow", "contract"}:
            raise ValueError("Unknown integration kind")
        if not edge["artifact"] or not edge["constraint"]:
            raise ValueError("Edges need artifact and scope/constraint evidence")
        source_path(root, edge["evidence"])
    for feature in graph["features"]:
        if feature["status"] not in {"parity", "gap", "exempt"}:
            raise ValueError("Unknown feature status")
        for surface in ("pyqt", "api", "web"):
            if feature.get(surface):
                source_path(root, feature[surface])
    tile_ids = [tile["id"] for tile in graph["tiles"]]
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("Duplicate launcher tile IDs")
    for tile in graph["tiles"]:
        source_path(root, tile["path"])
