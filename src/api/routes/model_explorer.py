"""Model explorer routes.

Provides endpoints for browsing, inspecting, and comparing
URDF/MJCF models. Includes tree view data, property inspection,
and Frankenstein mode (side-by-side comparison).

See issue #1200

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_logger
from ..models.requests import ModelCompareRequest, ModelExplorerRequest
from ..models.responses import (
    ModelCompareResponse,
    ModelExplorerResponse,
    URDFTreeNode,
)

router = APIRouter()

# Reuse model directory discovery from models.py
_MODEL_DIRS = [
    Path("src/shared/urdf"),
    Path("src/engines/physics_engines/pinocchio/models/generated"),
    Path("tests/fixtures/models"),
]


def _find_project_root() -> Path:
    """Find the project root directory by looking for known markers."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "shared" / "urdf").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _parse_urdf_tree(urdf_content: str, file_path: str) -> ModelExplorerResponse:
    """Parse URDF XML into a tree structure for the model explorer.

    Args:
        urdf_content: Raw URDF XML string.
        file_path: Path to the URDF file.

    Returns:
        Model explorer response with tree nodes.

    Raises:
        ValueError: If the URDF cannot be parsed.
    """
    try:
        root = ElementTree.fromstring(urdf_content)  # noqa: S314
    except ElementTree.ParseError as e:
        raise ValueError(f"Invalid URDF XML: {e}") from e

    model_name = root.get("name", "unknown")
    nodes: list[URDFTreeNode] = []

    # Parse links
    link_names: set[str] = set()
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "unnamed")
        link_names.add(link_name)

        # Extract link properties
        properties: dict[str, Any] = {"type": "link"}

        # Visual geometry
        visual = link_elem.find("visual")
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                for child in geom:
                    properties["geometry_type"] = child.tag
                    properties.update(child.attrib)

        # Inertial
        inertial = link_elem.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                properties["mass"] = float(mass_elem.get("value", "0"))

        nodes.append(
            URDFTreeNode(
                id=f"link_{link_name}",
                name=link_name,
                node_type="link",
                parent_id=None,  # Filled in during joint processing
                children=[],
                properties=properties,
            )
        )

    # Parse joints and build parent-child relationships
    child_links: set[str] = set()
    joint_count = 0
    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name", "unnamed")
        joint_type = joint_elem.get("type", "fixed")
        joint_count += 1

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        parent_link = parent_elem.get("link", "") if parent_elem is not None else ""
        child_link = child_elem.get("link", "") if child_elem is not None else ""
        child_links.add(child_link)

        # Joint properties
        properties: dict[str, Any] = {
            "type": "joint",
            "joint_type": joint_type,
            "parent_link": parent_link,
            "child_link": child_link,
        }

        # Origin
        origin_elem = joint_elem.find("origin")
        if origin_elem is not None:
            properties["xyz"] = origin_elem.get("xyz", "0 0 0")
            properties["rpy"] = origin_elem.get("rpy", "0 0 0")

        # Axis
        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            properties["axis"] = axis_elem.get("xyz", "0 0 1")

        # Limits
        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            properties["lower"] = float(limit_elem.get("lower", "0"))
            properties["upper"] = float(limit_elem.get("upper", "0"))
            if limit_elem.get("effort"):
                properties["effort"] = float(limit_elem.get("effort", "0"))
            if limit_elem.get("velocity"):
                properties["velocity"] = float(limit_elem.get("velocity", "0"))

        parent_node_id = f"link_{parent_link}"
        child_node_id = f"link_{child_link}"

        nodes.append(
            URDFTreeNode(
                id=f"joint_{joint_name}",
                name=joint_name,
                node_type="joint",
                parent_id=parent_node_id,
                children=[child_node_id],
                properties=properties,
            )
        )

        # Update parent link's children
        for node in nodes:
            if node.id == parent_node_id:
                node.children.append(f"joint_{joint_name}")
                break

        # Update child link's parent
        for node in nodes:
            if node.id == child_node_id:
                node.parent_id = f"joint_{joint_name}"
                break

    # Find root link (not a child of any joint)
    root_links = link_names - child_links
    root_link_name = (
        next(iter(root_links))
        if root_links
        else (next(iter(link_names)) if link_names else "base")
    )

    # Mark root node
    for node in nodes:
        if node.id == f"link_{root_link_name}":
            node.node_type = "root"
            break

    return ModelExplorerResponse(
        model_name=model_name,
        tree=nodes,
        joint_count=joint_count,
        link_count=len(link_names),
        model_format="urdf",
        file_path=file_path,
    )


def _resolve_model_path(model_path: str) -> Path:
    """Resolve a model path relative to the project root.

    Args:
        model_path: Relative or absolute model path.

    Returns:
        Resolved absolute path.

    Raises:
        HTTPException: If file not found.
    """
    root = _find_project_root()
    resolved = root / model_path
    if resolved.exists():
        return resolved

    # Try direct path
    direct = Path(model_path)
    if direct.exists():
        return direct

    # Search in model directories
    for model_dir in _MODEL_DIRS:
        candidate = root / model_dir / Path(model_path).name
        if candidate.exists():
            return candidate

    raise HTTPException(
        status_code=404,
        detail=f"Model file not found: {model_path}",
    )


@router.get(
    "/tools/model-explorer/{model_name}",
    response_model=ModelExplorerResponse,
)
async def get_model_explorer(
    model_name: str,
    logger: Any = Depends(get_logger),
) -> ModelExplorerResponse:
    """Get model explorer tree data for a model.

    Parses the URDF/MJCF file and returns a tree structure
    suitable for rendering in a collapsible tree view.

    Args:
        model_name: Name or path of the model.
        logger: Injected logger.

    Returns:
        Model explorer data with tree nodes.
    """
    # Import model discovery from models route
    from .models import _discover_models

    root = _find_project_root()
    models = _discover_models()

    # Find model by name
    model_entry = None
    for m in models:
        if m["name"] == model_name or model_name in m["name"]:
            model_entry = m
            break

    if model_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. "
            f"Available: {[m['name'] for m in models[:10]]}",
        )

    filepath = root / model_entry["path"]
    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_entry['path']}",
        )

    try:
        content = filepath.read_text(encoding="utf-8")
        return _parse_urdf_tree(content, model_entry["path"])
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse model: {str(exc)}"
        ) from exc
    except Exception as exc:
        if logger:
            logger.error("Error in model explorer for %s: %s", model_name, exc)
        raise HTTPException(
            status_code=500, detail=f"Model explorer error: {str(exc)}"
        ) from exc


@router.post(
    "/tools/model-explorer/inspect",
    response_model=ModelExplorerResponse,
)
async def inspect_model(
    request: ModelExplorerRequest,
    logger: Any = Depends(get_logger),
) -> ModelExplorerResponse:
    """Inspect a model file by path.

    Args:
        request: Model explorer request with file path.
        logger: Injected logger.

    Returns:
        Model explorer data.
    """
    try:
        filepath = _resolve_model_path(request.model_path)
        content = filepath.read_text(encoding="utf-8")
        return _parse_urdf_tree(content, request.model_path)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse model: {str(exc)}"
        ) from exc
    except Exception as exc:
        if logger:
            logger.error("Error inspecting model: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Model inspection error: {str(exc)}"
        ) from exc


@router.post(
    "/tools/model-explorer/compare",
    response_model=ModelCompareResponse,
)
async def compare_models(
    request: ModelCompareRequest,
    logger: Any = Depends(get_logger),
) -> ModelCompareResponse:
    """Compare two models side by side (Frankenstein mode).

    Parses both models and identifies shared/unique joints.

    Args:
        request: Comparison request with two model paths.
        logger: Injected logger.

    Returns:
        Comparison data with both models and diff analysis.
    """
    try:
        path_a = _resolve_model_path(request.model_a_path)
        path_b = _resolve_model_path(request.model_b_path)

        content_a = path_a.read_text(encoding="utf-8")
        content_b = path_b.read_text(encoding="utf-8")

        model_a = _parse_urdf_tree(content_a, request.model_a_path)
        model_b = _parse_urdf_tree(content_b, request.model_b_path)

        # Find shared and unique joints
        joints_a = {node.name for node in model_a.tree if node.node_type == "joint"}
        joints_b = {node.name for node in model_b.tree if node.node_type == "joint"}

        shared = sorted(joints_a & joints_b)
        only_a = sorted(joints_a - joints_b)
        only_b = sorted(joints_b - joints_a)

        return ModelCompareResponse(
            model_a=model_a,
            model_b=model_b,
            shared_joints=shared,
            unique_to_a=only_a,
            unique_to_b=only_b,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse models: {str(exc)}"
        ) from exc
    except Exception as exc:
        if logger:
            logger.error("Error comparing models: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Model comparison error: {str(exc)}"
        ) from exc
