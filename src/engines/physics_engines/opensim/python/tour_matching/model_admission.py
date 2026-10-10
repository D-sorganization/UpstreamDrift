"""Pinned OpenSim asset observations, never anatomical qualification (#11819)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
from pathlib import Path
import re
from typing import Any, Mapping

from defusedxml.expatbuilder import parseString

from src.engines.physics_engines.opensim.python.tour_matching.full_swing_tracking import (
    validate_model_checkpoint,
)
from src.engines.physics_engines.opensim.python.tour_matching._native_model_inventory import (
    observe_native_model,
)

_REQUIRED = (
    "source-license-and-ancestry",
    "external-dependency-closure",
    "anatomical-coverage-and-capacity",
    "muscle-parameter-and-path-validation",
    "full-state-initialization-and-replay",
    "contact-and-grip-qualification",
)


@dataclass(frozen=True)
class MuscleModelAssetInventory:
    """Structural evidence envelope; it has no qualified/ready verdict.

    Nested facts are observations, not an authenticated execution certificate.
    Native initialization does not imply equilibrium or admissible replay state.
    """

    source_sha256: str
    status: str
    serialized: dict[str, Any]
    resources: tuple[dict[str, Any], ...]
    native: dict[str, Any] | None
    required_evidence: tuple[str, ...]
    declared_regions: dict[str, tuple[str, ...]]
    declared_coordinate_roles: dict[str, tuple[str, ...]]
    diagnostic: str = ""
    scientific_status: str = "unqualified"
    schema_version: str = "opensim-muscle-asset-inventory/1"

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible observed facts without promoting their status."""
        return asdict(self)


def _text(node: Any, name: str) -> str | None:
    for child in node.childNodes:
        if child.nodeType == child.ELEMENT_NODE and child.tagName == name:
            return "".join(
                part.data
                for part in child.childNodes
                if part.nodeType in (part.TEXT_NODE, part.CDATA_SECTION_NODE)
            ).strip()
    return None


def _serialized_facts(data: bytes) -> tuple[Any, dict[str, Any]]:
    # OpenSim emits C++ scoped tags (HuntCrossleyForce::ContactParameters).
    # Disable namespace interpretation, not DTD/entity/external protections.
    document = parseString(data, namespaces=False, forbid_dtd=True)
    models = document.getElementsByTagName("Model")
    if len(models) != 1:
        raise ValueError("OpenSim document must contain a Model")
    model = models[0]
    elements = model.getElementsByTagName("*")
    # XML class suffixes are explicitly declarations, not runtime inheritance.
    muscles = [
        {
            "name": node.getAttribute("name"),
            "class": node.tagName,
            "ignore_tendon_compliance": _text(node, "ignore_tendon_compliance"),
            "ignore_activation_dynamics": _text(node, "ignore_activation_dynamics"),
            "path_frame_declarations": tuple(
                point.firstChild.data
                for point in node.getElementsByTagName("socket_parent_frame")
                if point.firstChild
            ),
        }
        for node in elements
        if node.tagName.endswith("Muscle")
    ]
    facts = {
        "model_name": model.getAttribute("name"),
        "classification_policy": "XML suffix declarations; native inheritance unverified",
        "muscle_like_declarations": tuple(muscles),
        "actuator_like_declarations": tuple(
            {"name": node.getAttribute("name"), "class": node.tagName}
            for node in elements
            if node.tagName.endswith("Actuator")
        ),
        "coordinates": tuple(
            {
                "name": node.getAttribute("name"),
                "locked": _text(node, "locked"),
                "prescribed": _text(node, "prescribed"),
            }
            for node in model.getElementsByTagName("Coordinate")
        ),
        "constraint_like_declarations": tuple(
            {"name": node.getAttribute("name"), "class": node.tagName}
            for node in elements
            if node.tagName.endswith("Constraint")
        ),
    }
    return model, facts


def _resource_facts(model: Any, directory: Path) -> tuple[dict[str, Any], ...]:
    """Observe local mesh candidates; never claim complete native dependency use."""
    rows = []
    for class_name, field, role in (
        ("Mesh", "mesh_file", "visual"),
        ("ContactMesh", "filename", "contact"),
    ):
        for node in model.getElementsByTagName(class_name):
            reference = _text(node, field)
            if not reference or reference.lower() == "unassigned":
                continue
            candidates = (directory / reference, directory / "Geometry" / reference)
            found = next((p for p in candidates if p.is_file()), None)
            rows.append(
                {
                    "reference": reference,
                    "role": role,
                    "status": "local-candidate-unverified-use" if found else "missing",
                    "sha256": hashlib.sha256(found.read_bytes()).hexdigest()
                    if found
                    else None,
                }
            )
    return tuple(rows)


def _validate_mapping(mapping: Mapping[str, tuple[str, ...]]) -> None:
    for role, paths in mapping.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError("declared role must be a nonempty string")
        if not isinstance(paths, tuple) or not paths:
            raise ValueError("declared paths must be a nonempty tuple")
        if any(not isinstance(p, str) or not p.startswith("/") for p in paths):
            raise ValueError("declared paths must be absolute native component paths")
        if len(set(paths)) != len(paths):
            raise ValueError("declared paths must be unique within each role")


def audit_muscle_model_asset(
    model_path: str | Path,
    expected_sha256: str,
    *,
    run_native: bool = True,
    region_frames: Mapping[str, tuple[str, ...]] | None = None,
    coordinate_roles: Mapping[str, tuple[str, ...]] | None = None,
) -> MuscleModelAssetInventory:
    """Observe pinned source and, optionally, initialize a fresh native model.

    Maps assign caller-declared regions/roles to exact native paths. Attachment
    coverage does not prove muscle action, joint crossing, strength or anatomy.
    No model conversion, parameter fitting, equilibration or integration occurs.
    Missing runtime/load failures preserve XML facts with unavailable evidence.
    Hash changes and invalid declared paths fail rather than emit an inventory.
    Only local mesh candidates are resolved; full resource closure stays required.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must be a full SHA-256 hex digest")
    if not isinstance(run_native, bool):
        raise TypeError("run_native must be bool")
    regions, roles = dict(region_frames or {}), dict(coordinate_roles or {})
    _validate_mapping(regions)
    _validate_mapping(roles)
    path = Path(model_path).resolve()
    digest = validate_model_checkpoint(path, expected_sha256)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError("model source changed during admission")
    xml, serialized = _serialized_facts(source)
    resources = _resource_facts(xml, path.parent)
    native, diagnostic, status = None, "", "serialized-only"
    if run_native:
        try:
            osim = import_module("opensim")
        except (ImportError, OSError) as exc:
            status, diagnostic = "native-unavailable", str(exc)
        else:
            try:
                model = osim.Model(str(path))
                model.finalizeConnections()
                state = model.initSystem()
            except RuntimeError as exc:
                status, diagnostic = "native-load-failed", str(exc)
            else:
                native = observe_native_model(osim, model, state, regions, roles)
                status = "native-initialized-structural-only"
    validate_model_checkpoint(path, digest)
    required = (
        _REQUIRED if native is not None else ("native-initialization", *_REQUIRED)
    )
    if native is not None and native["unregistered_muscle_paths"]:
        required = ("native-muscle-registration", *required)
    return MuscleModelAssetInventory(
        source_sha256=digest,
        status=status,
        serialized=serialized,
        resources=resources,
        native=native,
        required_evidence=required,
        declared_regions=regions,
        declared_coordinate_roles=roles,
        diagnostic=diagnostic,
    )
