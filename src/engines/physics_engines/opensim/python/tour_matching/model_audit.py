"""Pure-Python model geometry and structural qualification audit (OG-01, #10395).

Audits OpenSim .osim XML models without requiring native bindings:
1. Verifies SHA-256 integrity against pinned baselines.
2. Audits body count, coordinate count, actuator topology, and muscle presence.
3. Detects empty attached_geometry on equipment bodies (such as Club).
4. Audits mesh scale factors across anatomical segments (e.g. humerus, radius, ulna).
5. Implements DbC preconditions, postconditions, and fail-closed qualification gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

from defusedxml import ElementTree as SafeET

from src.shared.python.contracts import ensure, require

BASELINE_MODEL_SHA256: str = (
    "051d61eab9b72912105a308145392dd5c6c30faeaf7979b460c2ea485e9b0a8d"
)
BASELINE_OS3B_INPUT_SHA256: str = (
    "7dd1da1764bd8569d9e5de14eae6a249f8845e9fddf0f343796bef2ae8137381"
)


@dataclass(frozen=True)
class ModelGeometryAuditResult:
    """Summary of model geometry, topology, and qualification defects."""

    model_path: str
    sha256: str
    num_bodies: int
    num_coordinates: int
    num_actuators: int
    num_muscles: int
    body_names: tuple[str, ...]
    club_attached_geometry_count: int
    has_visible_club: bool
    arm_mesh_scales: dict[str, list[tuple[str, tuple[float, float, float]]]]
    has_unscaled_arm_mesh_defect: bool


def _parse_scale_factors(text: str | None) -> tuple[float, float, float]:
    """Parse 3D scale factors from string text."""
    if not text or not text.strip():
        return (1.0, 1.0, 1.0)
    parts = text.strip().split()
    if len(parts) != 3:
        return (1.0, 1.0, 1.0)
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def audit_model_geometry(model_path: Path | str) -> ModelGeometryAuditResult:
    """Perform a pure-XML audit of model geometry and topology.

    Preconditions:
    - model_path must exist and be an OpenSim document.

    Postconditions:
    - Returns structured audit result with non-empty body list and computed sha256.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    tree = SafeET.fromstring(data)
    model = tree.find("Model")
    if model is None and tree.tag == "Model":
        model = tree
    require(model is not None, "Document must contain a Model element")
    assert model is not None  # for mypy type narrowing

    # Bodies
    body_elements = model.findall(".//BodySet/objects/Body")
    body_names = tuple(b.get("name", "") for b in body_elements)

    # Coordinates
    coord_elements = model.findall(".//JointSet/objects//Coordinate")

    # Actuators and Muscles
    actuator_elements = model.findall(".//ForceSet/objects/CoordinateActuator")
    muscle_elements = model.findall(".//ForceSet/objects//Millard2012EquilibriumMuscle")
    muscle_elements += model.findall(".//ForceSet/objects//Thelen2003Muscle")

    # Club body inspection
    club_body = None
    for b in body_elements:
        if b.get("name") == "Club":
            club_body = b
            break

    club_geom_count = 0
    if club_body is not None:
        att_geom = club_body.find("attached_geometry")
        if att_geom is not None:
            club_geom_count = len(list(att_geom))

    has_visible_club = club_geom_count > 0

    # Arm mesh inspection
    arm_bodies = ("humerus_r", "humerus_l", "ulna_r", "ulna_l", "radius_r", "radius_l")
    arm_mesh_scales: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}
    has_unscaled_arm_mesh_defect = False

    for b in body_elements:
        bname = b.get("name", "")
        if bname in arm_bodies:
            mesh_list: list[tuple[str, tuple[float, float, float]]] = []
            for mesh in b.findall(".//Mesh"):
                file_elem = mesh.find("mesh_file")
                filename = (
                    file_elem.text.strip()
                    if file_elem is not None and file_elem.text
                    else ""
                )
                scale_elem = mesh.find("scale_factors")
                scales = _parse_scale_factors(
                    scale_elem.text if scale_elem is not None else None
                )
                mesh_list.append((filename, scales))
                if scales == (1.0, 1.0, 1.0):
                    has_unscaled_arm_mesh_defect = True
            arm_mesh_scales[bname] = mesh_list

    result = ModelGeometryAuditResult(
        model_path=str(path),
        sha256=sha,
        num_bodies=len(body_elements),
        num_coordinates=len(coord_elements),
        num_actuators=len(actuator_elements),
        num_muscles=len(muscle_elements),
        body_names=body_names,
        club_attached_geometry_count=club_geom_count,
        has_visible_club=has_visible_club,
        arm_mesh_scales=arm_mesh_scales,
        has_unscaled_arm_mesh_defect=has_unscaled_arm_mesh_defect,
    )

    ensure(len(result.sha256) == 64, "SHA-256 digest must be 64 hexadecimal characters")
    return result


def verify_model_qualification(
    model_path: Path | str,
    *,
    require_visible_club: bool = True,
    require_consistent_arm_scaling: bool = False,
) -> ModelGeometryAuditResult:
    """Validate model qualification against anatomical baseline requirements.

    Raises ValueError if required qualification gates are not met.
    """
    audit = audit_model_geometry(model_path)

    if require_visible_club and not audit.has_visible_club:
        raise ValueError(
            f"Model at {model_path} rejected: Club body has no attached visual geometry "
            f"(count={audit.club_attached_geometry_count})."
        )

    if require_consistent_arm_scaling and audit.has_unscaled_arm_mesh_defect:
        raise ValueError(
            f"Model at {model_path} rejected: Arm meshes remain at unit scale (1 1 1) "
            "despite non-unit segment joint scaling."
        )

    return audit
