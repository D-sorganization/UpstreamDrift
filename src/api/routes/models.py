"""URDF/MJCF model serving routes.

Provides endpoints for listing and retrieving parsed URDF/MJCF models
for 3D rendering in the frontend.

See issue #1201

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_logger
from ..models.responses import (
    ModelListResponse,
    URDFJointDescriptor,
    URDFLinkGeometry,
    URDFModelResponse,
)

router = APIRouter()

# Base directories for model discovery
_MODEL_DIRS = [
    Path("src/shared/urdf"),
    Path("src/engines/physics_engines/pinocchio/models/generated"),
    Path("tests/fixtures/models"),
]


def _find_project_root() -> Path:
    """Find the project root directory by looking for known markers."""
    # Walk up from this file's location
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "shared" / "urdf").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _discover_models() -> list[dict[str, str]]:
    """Discover available URDF/MJCF model files.

    Returns:
        List of dicts with name, format, and path keys.
    """
    root = _find_project_root()
    models: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for model_dir in _MODEL_DIRS:
        full_dir = root / model_dir
        if not full_dir.exists():
            continue

        for ext in ("*.urdf", "*.xml"):
            for filepath in full_dir.rglob(ext):
                name = filepath.stem
                if name in seen_names:
                    # Disambiguate with parent directory
                    name = f"{filepath.parent.name}/{name}"
                seen_names.add(name)

                fmt = "urdf" if filepath.suffix == ".urdf" else "mjcf"
                models.append({
                    "name": name,
                    "format": fmt,
                    "path": str(filepath.relative_to(root)),
                })

    return models


def _parse_urdf_geometry(visual_elem: Any, materials: dict[str, list[float]]) -> dict:
    """Parse a single <visual> element into geometry data.

    Args:
        visual_elem: XML element for <visual>.
        materials: Dictionary mapping material names to RGBA color lists.

    Returns:
        Dictionary with geometry_type, dimensions, origin, rotation, and color.
    """
    result: dict[str, Any] = {
        "geometry_type": "box",
        "dimensions": {},
        "origin": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "color": [0.5, 0.5, 0.5, 1.0],
        "mesh_path": None,
    }

    # Parse origin
    origin_elem = visual_elem.find("origin")
    if origin_elem is not None:
        xyz = origin_elem.get("xyz", "0 0 0")
        rpy = origin_elem.get("rpy", "0 0 0")
        result["origin"] = [float(x) for x in xyz.split()]
        result["rotation"] = [float(x) for x in rpy.split()]

    # Parse geometry
    geom_elem = visual_elem.find("geometry")
    if geom_elem is not None:
        box = geom_elem.find("box")
        cylinder = geom_elem.find("cylinder")
        sphere = geom_elem.find("sphere")
        mesh = geom_elem.find("mesh")

        if box is not None:
            result["geometry_type"] = "box"
            size = box.get("size", "0.1 0.1 0.1")
            dims = [float(x) for x in size.split()]
            result["dimensions"] = {
                "width": dims[0] if len(dims) > 0 else 0.1,
                "height": dims[1] if len(dims) > 1 else 0.1,
                "depth": dims[2] if len(dims) > 2 else 0.1,
            }
        elif cylinder is not None:
            result["geometry_type"] = "cylinder"
            result["dimensions"] = {
                "radius": float(cylinder.get("radius", "0.05")),
                "length": float(cylinder.get("length", "0.3")),
            }
        elif sphere is not None:
            result["geometry_type"] = "sphere"
            result["dimensions"] = {
                "radius": float(sphere.get("radius", "0.1")),
            }
        elif mesh is not None:
            result["geometry_type"] = "mesh"
            result["mesh_path"] = mesh.get("filename", "")
            scale = mesh.get("scale", "1 1 1")
            scale_vals = [float(x) for x in scale.split()]
            result["dimensions"] = {
                "scale_x": scale_vals[0] if len(scale_vals) > 0 else 1.0,
                "scale_y": scale_vals[1] if len(scale_vals) > 1 else 1.0,
                "scale_z": scale_vals[2] if len(scale_vals) > 2 else 1.0,
            }

    # Parse material/color
    mat_elem = visual_elem.find("material")
    if mat_elem is not None:
        mat_name = mat_elem.get("name", "")
        color_elem = mat_elem.find("color")
        if color_elem is not None:
            rgba = color_elem.get("rgba", "0.5 0.5 0.5 1.0")
            result["color"] = [float(x) for x in rgba.split()]
        elif mat_name in materials:
            result["color"] = materials[mat_name]

    return result


def _parse_urdf(urdf_content: str) -> URDFModelResponse:
    """Parse a URDF XML string into a URDFModelResponse.

    Args:
        urdf_content: Raw URDF XML string.

    Returns:
        Parsed model data.

    Raises:
        ValueError: If the URDF cannot be parsed.
    """
    try:
        root = ElementTree.fromstring(urdf_content)  # noqa: S314
    except ElementTree.ParseError as e:
        raise ValueError(f"Invalid URDF XML: {e}") from e

    model_name = root.get("name", "unknown")

    # Parse top-level materials
    materials: dict[str, list[float]] = {}
    for mat_elem in root.findall("material"):
        mat_name = mat_elem.get("name", "")
        color_elem = mat_elem.find("color")
        if color_elem is not None:
            rgba = color_elem.get("rgba", "0.5 0.5 0.5 1.0")
            materials[mat_name] = [float(x) for x in rgba.split()]

    # Parse links
    links: list[URDFLinkGeometry] = []
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "unnamed")
        visual_elem = link_elem.find("visual")
        if visual_elem is not None:
            geom_data = _parse_urdf_geometry(visual_elem, materials)
            links.append(URDFLinkGeometry(
                link_name=link_name,
                **geom_data,
            ))

    # Parse joints
    joints: list[URDFJointDescriptor] = []
    child_links: set[str] = set()
    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name", "unnamed")
        joint_type = joint_elem.get("type", "fixed")

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        parent_link = parent_elem.get("link", "") if parent_elem is not None else ""
        child_link = child_elem.get("link", "") if child_elem is not None else ""
        child_links.add(child_link)

        origin = [0.0, 0.0, 0.0]
        rotation = [0.0, 0.0, 0.0]
        origin_elem = joint_elem.find("origin")
        if origin_elem is not None:
            xyz = origin_elem.get("xyz", "0 0 0")
            rpy = origin_elem.get("rpy", "0 0 0")
            origin = [float(x) for x in xyz.split()]
            rotation = [float(x) for x in rpy.split()]

        axis = [0.0, 0.0, 1.0]
        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            axis_str = axis_elem.get("xyz", "0 0 1")
            axis = [float(x) for x in axis_str.split()]

        lower_limit = None
        upper_limit = None
        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            lower_limit = float(limit_elem.get("lower", "0"))
            upper_limit = float(limit_elem.get("upper", "0"))

        joints.append(URDFJointDescriptor(
            name=joint_name,
            joint_type=joint_type,
            parent_link=parent_link,
            child_link=child_link,
            origin=origin,
            rotation=rotation,
            axis=axis,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ))

    # Find root link (not a child of any joint)
    all_link_names = {link.link_name for link in links}
    root_candidates = all_link_names - child_links
    root_link = next(iter(root_candidates)) if root_candidates else (
        links[0].link_name if links else "base"
    )

    return URDFModelResponse(
        model_name=model_name,
        links=links,
        joints=joints,
        root_link=root_link,
        urdf_raw=urdf_content,
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    logger: Any = Depends(get_logger),
) -> ModelListResponse:
    """List available URDF/MJCF models.

    Returns:
        List of available model files.
    """
    try:
        models = _discover_models()
        return ModelListResponse(models=models)
    except Exception as exc:
        if logger:
            logger.error("Error listing models: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(exc)}"
        ) from exc


@router.get("/models/{model_name}/urdf", response_model=URDFModelResponse)
async def get_model_urdf(
    model_name: str,
    logger: Any = Depends(get_logger),
) -> URDFModelResponse:
    """Get parsed URDF model data for 3D rendering.

    Parses the URDF XML and returns structured geometry, joint,
    and kinematic chain data that can be directly consumed by
    the frontend URDFViewer component.

    Args:
        model_name: Model identifier (from /models list).
        logger: Injected logger.

    Returns:
        Parsed URDF model data.

    Raises:
        HTTPException: If model not found or parse fails.
    """
    # Find the model file
    root = _find_project_root()
    models = _discover_models()
    model_entry = None

    for m in models:
        if m["name"] == model_name:
            model_entry = m
            break

    if model_entry is None:
        # Try partial match
        for m in models:
            if model_name in m["name"] or m["name"].endswith(model_name):
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
        urdf_content = filepath.read_text(encoding="utf-8")
        result = _parse_urdf(urdf_content)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse URDF: {str(exc)}"
        ) from exc
    except Exception as exc:
        if logger:
            logger.error("Error loading model %s: %s", model_name, exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to load model: {str(exc)}"
        ) from exc
