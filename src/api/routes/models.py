"""URDF/MJCF model serving routes.

Provides endpoints for listing and retrieving parsed URDF/MJCF models
for 3D rendering in the frontend.

See issue #1201

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ElementTree
from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_logger
from ..models.responses import (
    ModelListResponse,
    URDFJointDescriptor,
    URDFLinkGeometry,
    URDFModelResponse,
)
from ..utils.path_validation import resolve_contained_path
from ._route_utils import find_project_root

__all__ = ["discover_models", "router"]

router = APIRouter()

# Base directories for model discovery
_MODEL_DIRS = [
    Path("src/shared/urdf"),
    Path("src/engines/physics_engines/pinocchio/models/generated"),
    Path("tests/fixtures/models"),
]


def _find_project_root() -> Path:
    """Find the project root directory by looking for known markers."""
    return find_project_root()


# Default attribute values for URDF vector/color attributes.
_DEFAULT_XYZ = "0 0 0"
_DEFAULT_RPY = "0 0 0"
_DEFAULT_AXIS = "0 0 1"
_DEFAULT_RGBA = "0.5 0.5 0.5 1.0"


def _parse_floats(text: str, element: str) -> list[float]:
    """Parse a whitespace-separated list of floats from a URDF attribute.

    Args:
        text: Raw attribute text (e.g. ``"0 0 0.2"``).
        element: Human-readable name of the offending attribute/element,
            used in the error message.

    Returns:
        List of parsed floats.

    Raises:
        ValueError: If any token is not a valid float.
    """
    try:
        return [float(token) for token in text.split()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value in URDF {element} attribute: {text!r}"
        ) from exc


def _parse_scalar(text: str, element: str) -> float:
    """Parse a single float from a URDF attribute.

    Args:
        text: Raw attribute text.
        element: Name of the offending attribute/element for errors.

    Returns:
        The parsed float.

    Raises:
        ValueError: If the value is not a valid float.
    """
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value in URDF {element} attribute: {text!r}"
        ) from exc


def _parse_vec3(text: str, element: str) -> list[float]:
    """Parse and validate a 3-component vector from a URDF attribute.

    Args:
        text: Raw attribute text.
        element: Name of the offending attribute/element for errors.

    Returns:
        A list of exactly three floats.

    Raises:
        ValueError: If the value is non-numeric or not exactly 3 components.
    """
    values = _parse_floats(text, element)
    if len(values) != 3:
        raise ValueError(
            f"URDF {element} must have 3 components, got {len(values)}: {text!r}"
        )
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError(f"URDF {element} must contain finite numbers, got {text!r}")
    return values


def _parse_color(text: str, element: str = "color rgba") -> list[float]:
    """Parse and validate an RGBA color from a URDF attribute.

    Args:
        text: Raw ``rgba`` attribute text.
        element: Name of the offending attribute/element for errors.

    Returns:
        A list of exactly four floats.

    Raises:
        ValueError: If the value is non-numeric or not exactly 4 components.
    """
    values = _parse_floats(text, element)
    if len(values) != 4:
        raise ValueError(
            f"URDF {element} must have 4 components, got {len(values)}: {text!r}"
        )
    return values


def discover_models() -> list[dict[str, str]]:
    """Discover available URDF/MJCF model files.

    Returns:
        List of dicts with name, format, and path keys.
    """
    root = _find_project_root()
    models: list[dict[str, str]] = []
    seen_names: set[str] = set()
    allowed_roots: list[Path] = []

    for model_dir in _MODEL_DIRS:
        full_dir = root / model_dir
        if full_dir.exists():
            allowed_roots.append(full_dir)

    for model_dir in _MODEL_DIRS:
        full_dir = root / model_dir
        if not full_dir.exists():
            continue

        for ext in ("*.urdf", "*.xml"):
            for filepath in full_dir.rglob(ext):
                try:
                    resolve_contained_path(filepath, allowed_roots)
                except HTTPException:
                    continue

                name = filepath.stem
                if name in seen_names:
                    # Disambiguate with parent directory
                    name = f"{filepath.parent.name}/{name}"
                seen_names.add(name)

                fmt = "urdf" if filepath.suffix == ".urdf" else "mjcf"
                models.append(
                    {
                        "name": name,
                        "format": fmt,
                        "path": str(filepath.relative_to(root)),
                    }
                )

    return models


def _discover_models() -> list[dict[str, str]]:
    """Backward-compatible alias for older local callers.

    Cross-module imports should use ``discover_models()`` so the public
    contract is explicit and resilient to future refactors.
    """
    return discover_models()


def _parse_urdf_geometry(  # noqa: C901
    visual_elem: Any, materials: dict[str, list[float]]
) -> dict[str, Any]:
    """Parse a single <visual> element into geometry data.

    Args:
        visual_elem: XML element for <visual>.
        materials: Dictionary mapping material names to RGBA color lists.

    Returns:
        Dictionary with geometry_type, dimensions, origin, rotation, and color.
    """
    if materials is None:
        raise ValueError("materials must be provided")
    if visual_elem is None:
        raise ValueError("visual_elem must be provided")
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
        result["origin"] = _parse_vec3(
            origin_elem.get("xyz", _DEFAULT_XYZ), "visual origin xyz"
        )
        result["rotation"] = _parse_vec3(
            origin_elem.get("rpy", _DEFAULT_RPY), "visual origin rpy"
        )

    # Parse geometry
    geom_elem = visual_elem.find("geometry")
    if geom_elem is not None:
        box = geom_elem.find("box")
        cylinder = geom_elem.find("cylinder")
        sphere = geom_elem.find("sphere")
        mesh = geom_elem.find("mesh")

        if box is not None:
            result["geometry_type"] = "box"
            dims = _parse_floats(box.get("size", "0.1 0.1 0.1"), "box size")
            result["dimensions"] = {
                "width": dims[0] if len(dims) > 0 else 0.1,
                "height": dims[1] if len(dims) > 1 else 0.1,
                "depth": dims[2] if len(dims) > 2 else 0.1,
            }
        elif cylinder is not None:
            result["geometry_type"] = "cylinder"
            result["dimensions"] = {
                "radius": _parse_scalar(
                    cylinder.get("radius", "0.05"), "cylinder radius"
                ),
                "length": _parse_scalar(
                    cylinder.get("length", "0.3"), "cylinder length"
                ),
            }
        elif sphere is not None:
            result["geometry_type"] = "sphere"
            result["dimensions"] = {
                "radius": _parse_scalar(sphere.get("radius", "0.1"), "sphere radius"),
            }
        elif mesh is not None:
            result["geometry_type"] = "mesh"
            result["mesh_path"] = mesh.get("filename", "")
            scale_vals = _parse_floats(mesh.get("scale", "1 1 1"), "mesh scale")
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
            result["color"] = _parse_color(color_elem.get("rgba", _DEFAULT_RGBA))
        elif mat_name in materials:
            result["color"] = materials[mat_name]

    return result


def _parse_urdf_materials(root: Any) -> dict[str, list[float]]:
    """Parse top-level URDF material definitions into an RGBA color map.

    Args:
        root: The parsed URDF XML root element.

    Returns:
        Dictionary mapping material names to RGBA color lists.
    """
    materials: dict[str, list[float]] = {}
    for mat_elem in root.findall("material"):
        mat_name = mat_elem.get("name", "")
        color_elem = mat_elem.find("color")
        if color_elem is not None:
            materials[mat_name] = _parse_color(color_elem.get("rgba", _DEFAULT_RGBA))
    return materials


def _parse_urdf_links(
    root: Any, materials: dict[str, list[float]]
) -> list[URDFLinkGeometry]:
    """Parse URDF <link> elements into geometry descriptors.

    Args:
        root: The parsed URDF XML root element.
        materials: Material name-to-RGBA mapping from top-level definitions.

    Returns:
        List of URDFLinkGeometry descriptors for each link with visual data.
    """
    if materials is None:
        raise ValueError("materials must be provided")
    if root is None:
        raise ValueError("root must be provided")
    links: list[URDFLinkGeometry] = []
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "unnamed")
        visual_elem = link_elem.find("visual")
        if visual_elem is not None:
            geom_data = _parse_urdf_geometry(visual_elem, materials)
            links.append(
                URDFLinkGeometry(
                    link_name=link_name,
                    **geom_data,
                )
            )
        else:
            links.append(
                URDFLinkGeometry(
                    link_name=link_name,
                    geometry_type="none",
                    dimensions={},
                    origin=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0],
                    color=[0.0, 0.0, 0.0, 0.0],
                    mesh_path=None,
                )
            )
    return links


def _parse_urdf_link_names(root: Any) -> list[str]:
    """Parse all declared URDF link names, including visual-less topology nodes."""
    if root is None:
        raise ValueError("root must be provided")
    return [link_elem.get("name", "unnamed") for link_elem in root.findall("link")]


def _parse_urdf_joint_element(joint_elem: Any) -> URDFJointDescriptor:
    """Parse a single URDF <joint> element into a joint descriptor.

    Args:
        joint_elem: XML element for a <joint> tag.

    Returns:
        URDFJointDescriptor with joint name, type, parent/child links,
        origin, rotation, axis, and optional limits.
    """
    joint_name = joint_elem.get("name", "unnamed")
    joint_type = joint_elem.get("type", "fixed")

    parent_elem = joint_elem.find("parent")
    child_elem = joint_elem.find("child")
    parent_link = parent_elem.get("link", "") if parent_elem is not None else ""
    child_link = child_elem.get("link", "") if child_elem is not None else ""

    origin = [0.0, 0.0, 0.0]
    rotation = [0.0, 0.0, 0.0]
    origin_elem = joint_elem.find("origin")
    if origin_elem is not None:
        origin = _parse_vec3(origin_elem.get("xyz", _DEFAULT_XYZ), "joint origin xyz")
        rotation = _parse_vec3(origin_elem.get("rpy", _DEFAULT_RPY), "joint origin rpy")

    axis = [0.0, 0.0, 1.0]
    axis_elem = joint_elem.find("axis")
    if axis_elem is not None:
        axis = _parse_vec3(axis_elem.get("xyz", _DEFAULT_AXIS), "joint axis xyz")
        axis_norm = math.sqrt(sum(c * c for c in axis))
        if axis_norm < 1e-9:
            raise ValueError(
                f"Joint {joint_name!r} axis cannot be zero vector: {axis!r}"
            )

    lower_limit = None
    upper_limit = None
    limit_elem = joint_elem.find("limit")
    if limit_elem is not None:
        lower_limit = _parse_scalar(limit_elem.get("lower", "0"), "joint limit lower")
        upper_limit = _parse_scalar(limit_elem.get("upper", "0"), "joint limit upper")

    return URDFJointDescriptor(
        name=joint_name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        origin=origin,
        rotation=rotation,
        axis=axis,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )


def _parse_urdf_joints(root: Any) -> tuple[list[URDFJointDescriptor], set[str]]:
    """Parse all URDF <joint> elements and collect child link names.

    Args:
        root: The parsed URDF XML root element.

    Returns:
        Tuple of (joint descriptors list, set of child link names).
    """
    joints: list[URDFJointDescriptor] = []
    child_links: set[str] = set()
    for joint_elem in root.findall("joint"):
        descriptor = _parse_urdf_joint_element(joint_elem)
        joints.append(descriptor)
        child_links.add(descriptor.child_link)
    return joints, child_links


def _validate_urdf_joint_links(
    joints: list[URDFJointDescriptor], declared_link_names: set[str]
) -> None:
    """Ensure every joint endpoint references a declared URDF link."""
    if joints is None:
        raise ValueError("joints must be provided")
    if declared_link_names is None:
        raise ValueError("declared_link_names must be provided")
    for joint in joints:
        missing = [
            link_name
            for link_name in (joint.parent_link, joint.child_link)
            if link_name not in declared_link_names
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Joint {joint.name!r} references undeclared link(s): {missing_list}"
            )


def _find_root_link(link_names: list[str], child_links: set[str]) -> str:
    """Identify the root link (not a child of any joint).

    Args:
        link_names: All declared URDF link names, including visual-less links.
        child_links: Set of link names that appear as children in joints.

    Returns:
        Name of the root link, or "base" if none can be determined.
    """
    if link_names is None:
        raise ValueError("link_names must be provided")
    for link_name in link_names:
        if link_name not in child_links:
            return link_name
    return link_names[0] if link_names else "base"


def _parse_urdf(urdf_content: str) -> URDFModelResponse:
    """Parse a URDF XML string into a URDFModelResponse.

    Args:
        urdf_content: Raw URDF XML string.

    Returns:
        Parsed model data. ``model_name`` is always populated (defaults to
        ``"unknown"`` when the ``<robot>`` element has no ``name`` attribute).

    Raises:
        ValueError: If ``urdf_content`` is not a string, is empty, or the URDF
            cannot be parsed.
    """
    if not isinstance(urdf_content, str):
        raise ValueError("URDF content must be a string")
    if urdf_content == "":
        raise ValueError("URDF content must be a non-empty string")
    try:
        root = ElementTree.fromstring(urdf_content)
    except ElementTree.ParseError as e:
        raise ValueError(f"Invalid URDF XML: {e}") from e

    model_name = root.get("name", "unknown")
    materials = _parse_urdf_materials(root)
    link_names = _parse_urdf_link_names(root)
    links = _parse_urdf_links(root, materials)
    joints, child_links = _parse_urdf_joints(root)
    _validate_urdf_joint_links(joints, set(link_names))
    root_link = _find_root_link(link_names, child_links)

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
        models = discover_models()
        return ModelListResponse(models=models)
    except (
        RuntimeError,
        TypeError,
        AttributeError,
        OSError,
        ValueError,
    ) as exc:
        if logger:
            logger.exception("Error listing models")
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(exc)}"
        ) from exc


@router.get("/models/{model_name}/urdf", response_model=URDFModelResponse)
async def get_model_urdf(  # noqa: C901
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
    models = discover_models()
    model_entry = None

    for m in models:
        if m["name"] == model_name:
            model_entry = m
            break

    if model_entry is None:
        # Fall back to an exact basename match (names may be disambiguated
        # with a "parent_dir/name" prefix in discover_models()). Require an
        # unambiguous single match so resolution is deterministic.
        basename_matches = [
            m for m in models if m["name"].rsplit("/", 1)[-1] == model_name
        ]
        if len(basename_matches) == 1:
            model_entry = basename_matches[0]
        elif len(basename_matches) > 1:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' is ambiguous. "
                f"Candidates: {[m['name'] for m in basename_matches]}",
            )

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
        model_roots = [
            (root / model_dir)
            for model_dir in _MODEL_DIRS
            if (root / model_dir).exists()
        ]
        try:
            filepath = resolve_contained_path(filepath, model_roots)
        except HTTPException as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_entry['path']}",
            ) from exc
        urdf_content = filepath.read_text(encoding="utf-8")
        result = _parse_urdf(urdf_content)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse URDF: {str(exc)}"
        ) from exc
    except ImportError as exc:
        if logger:
            logger.exception("Error loading model %s", model_name)
        raise HTTPException(
            status_code=500, detail=f"Failed to load model: {str(exc)}"
        ) from exc
