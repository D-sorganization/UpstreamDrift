"""OpenSim ``.osim`` loader for the first-party Model Explorer.

The shared ``model_generation`` package is vendored from Tools in this
repository, so this loader intentionally lives under ``src.tools`` while
returning the same ``ParsedModel`` contract used by URDF/MJCF loaders.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import defusedxml.ElementTree as DefusedET
from model_generation.canonical_model import CanonicalModel
from model_generation.converters.urdf_parser import ParsedModel
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
    Origin,
)

_GROUND_LINK_NAME = "ground"
_GROUND_MASS = 1e-6
_MIN_INERTIA = 1e-9
_MAX_BODIES = 5000
_MAX_JOINTS = 10000
_JOINT_TYPES = {
    "BallJoint",
    "CustomJoint",
    "FreeJoint",
    "PinJoint",
    "SliderJoint",
    "WeldJoint",
}


@dataclass(frozen=True)
class _FrameRef:
    body: str
    origin: Origin


class OsimLoader:
    """Parse OpenSim model XML into ``ParsedModel`` and ``CanonicalModel``."""

    def load(self, source: str | Path, read_only: bool = False) -> ParsedModel:
        """Load an OpenSim ``.osim`` file path or XML string.

        Args:
            source: Path to a ``.osim`` file, or an XML string beginning with ``<``.
            read_only: Whether the returned model should be marked read-only.

        Returns:
            ``ParsedModel`` compatible with the model-generation editor stack.
        """
        xml_string, source_path = self._read_source(source)
        root = self._parse_xml(xml_string)
        model_elem = self._model_element(root)
        warnings: list[str] = []
        links = self._parse_links(model_elem, warnings)
        joints = self._parse_joints(model_elem, {link.name for link in links}, warnings)
        self._record_unconverted_sets(model_elem, warnings)

        return ParsedModel(
            name=model_elem.get("name", "unnamed_opensim_model"),
            links=links,
            joints=joints,
            materials={},
            original_xml=xml_string,
            source_path=source_path,
            warnings=warnings,
            read_only=read_only,
        )

    def load_canonical(self, source: str | Path) -> CanonicalModel:
        """Load an OpenSim model and validate it as a canonical model."""
        parsed = self.load(source)
        canonical = CanonicalModel.from_core(
            name=parsed.name,
            links=parsed.links,
            joints=parsed.joints,
            metadata={
                "source_format": "opensim-osim",
                "source_path": str(parsed.source_path) if parsed.source_path else None,
                "warnings": list(parsed.warnings),
            },
        )
        canonical.require_valid(strict=False)
        return canonical

    def to_urdf(self, source: str | Path) -> str:
        """Load an OpenSim file and emit URDF XML for existing visualizers."""
        return self.load(source).to_urdf(pretty_print=True)

    def _read_source(self, source: str | Path) -> tuple[str, Path | None]:
        if source is None:
            raise ValueError("source must be provided")
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8"), source
        if source.strip().startswith("<"):
            return source, None
        path = Path(source)
        return path.read_text(encoding="utf-8"), path

    def _parse_xml(self, xml_string: str) -> ET.Element:
        try:
            return DefusedET.fromstring(xml_string)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid OpenSim XML: {exc}") from exc

    def _model_element(self, root: ET.Element) -> ET.Element:
        root_tag = _tag(root)
        if root_tag == "Model":
            return root
        if root_tag != "OpenSimDocument":
            raise ValueError(f"Expected OpenSimDocument root element, got {root_tag!r}")
        model = _child(root, "Model")
        if model is None:
            raise ValueError("OpenSimDocument is missing a Model element")
        return model

    def _parse_links(self, model_elem: ET.Element, warnings: list[str]) -> list[Link]:
        bodies = list(_objects_in_set(model_elem, "BodySet"))
        if len(bodies) > _MAX_BODIES:
            raise ValueError(f"OpenSim model has too many bodies: {len(bodies)}")
        links = [_ground_link()]
        for body in bodies:
            if _tag(body) != "Body":
                continue
            links.append(_parse_body(body, warnings))
        return links

    def _parse_joints(
        self,
        model_elem: ET.Element,
        link_names: set[str],
        warnings: list[str],
    ) -> list[Joint]:
        joint_elems = [j for j in _objects_in_set(model_elem, "JointSet") if _tag(j)]
        if len(joint_elems) > _MAX_JOINTS:
            raise ValueError(f"OpenSim model has too many joints: {len(joint_elems)}")

        joints: list[Joint] = []
        for joint_elem in joint_elems:
            joint_tag = _tag(joint_elem)
            if joint_tag not in _JOINT_TYPES:
                warnings.append(f"Unsupported joint element {joint_tag!r} skipped")
                continue
            joints.append(_parse_joint(joint_elem, link_names, warnings))
        return joints

    def _record_unconverted_sets(
        self, model_elem: ET.Element, warnings: list[str]
    ) -> None:
        for set_name, label in (
            ("ForceSet", "force or muscle"),
            ("ConstraintSet", "constraint"),
            ("MarkerSet", "marker"),
        ):
            count = sum(1 for _ in _objects_in_set(model_elem, set_name))
            if count:
                warnings.append(
                    f"{set_name} contains {count} {label} element(s) not converted"
                )


def _tag(elem: ET.Element) -> str:
    return elem.tag.rsplit("}", 1)[-1]


def _child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in list(elem):
        if _tag(child) == name:
            return child
    return None


def _children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(elem) if _tag(child) == name]


def _descendants(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in elem.iter() if child is not elem and _tag(child) == name]


def _text(elem: ET.Element, name: str, default: str = "") -> str:
    child = _child(elem, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _float_text(elem: ET.Element, name: str, default: float) -> float:
    raw = _text(elem, name)
    return float(raw) if raw else default


def _vec_text(
    elem: ET.Element,
    name: str,
    default: tuple[float, ...],
    *,
    length: int,
) -> tuple[float, ...]:
    raw = _text(elem, name)
    if not raw:
        return default
    values = tuple(float(part) for part in raw.split())
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    return values


def _objects_in_set(model_elem: ET.Element, set_name: str) -> Iterable[ET.Element]:
    set_elem = _child(model_elem, set_name)
    if set_elem is None:
        return ()
    objects = _child(set_elem, "objects")
    return list(objects) if objects is not None else list(set_elem)


def _ground_link() -> Link:
    return Link(
        name=_GROUND_LINK_NAME,
        inertia=Inertia(
            ixx=_MIN_INERTIA,
            iyy=_MIN_INERTIA,
            izz=_MIN_INERTIA,
            mass=_GROUND_MASS,
        ),
    )


def _parse_body(body: ET.Element, warnings: list[str]) -> Link:
    name = body.get("name")
    if not name:
        raise ValueError("OpenSim Body is missing a name")
    inertia = _parse_body_inertia(body, name, warnings)
    visual = _parse_body_visual(body)
    return Link(name=name, inertia=inertia, visual_geometry=visual)


def _parse_body_inertia(
    body: ET.Element, body_name: str, warnings: list[str]
) -> Inertia:
    mass = _float_text(body, "mass", 1.0)
    if mass <= 0.0:
        warnings.append(
            f"Body '{body_name}' has non-positive mass {mass}; floored for validation"
        )
        mass = _GROUND_MASS
    center = _vec_text(body, "mass_center", (0.0, 0.0, 0.0), length=3)  # type: ignore[assignment]
    values = _vec_text(body, "inertia", (0.1, 0.1, 0.1, 0.0, 0.0, 0.0), length=6)
    diag = tuple(max(value, _MIN_INERTIA) for value in values[:3])
    if diag != values[:3]:
        warnings.append(
            f"Body '{body_name}' has non-positive inertia diagonal; floored"
        )
    return Inertia(
        ixx=diag[0],
        iyy=diag[1],
        izz=diag[2],
        ixy=values[3],
        ixz=values[4],
        iyz=values[5],
        mass=mass,
        center_of_mass=center,  # type: ignore[arg-type]
    )


def _parse_body_visual(body: ET.Element) -> Geometry | None:
    mesh = _first_mesh(body)
    if mesh is not None:
        filename = _text(mesh, "mesh_file")
        if filename:
            scale = _vec_text(mesh, "scale_factors", (1.0, 1.0, 1.0), length=3)  # type: ignore[assignment]
            return Geometry(
                geometry_type=GeometryType.MESH,
                mesh_filename=filename,
                mesh_scale=scale,  # type: ignore[arg-type]
            )
    visible = _child(body, "VisibleObject")
    if visible is not None:
        geometry_files = _text(visible, "geometry_files")
        first_file = geometry_files.split()[0] if geometry_files else ""
        if first_file:
            return Geometry(geometry_type=GeometryType.MESH, mesh_filename=first_file)
    return None


def _first_mesh(body: ET.Element) -> ET.Element | None:
    meshes = _descendants(body, "Mesh")
    return meshes[0] if meshes else None


def _parse_joint(
    joint_elem: ET.Element,
    link_names: set[str],
    warnings: list[str],
) -> Joint:
    name = joint_elem.get("name")
    if not name:
        raise ValueError("OpenSim joint is missing a name")
    frame_refs = _frame_refs(joint_elem)
    parent, child, origin = _joint_endpoints(joint_elem, frame_refs, link_names)
    joint_type, axis, limits = _joint_motion(joint_elem, warnings)
    if _tag(joint_elem) == "CustomJoint":
        warnings.append(
            f"CustomJoint '{name}' approximated as {joint_type.value} from coordinates"
        )
    return Joint(
        name=name,
        joint_type=joint_type,
        parent=parent,
        child=child,
        origin=origin,
        axis=axis,
        limits=limits,
    )


def _frame_refs(joint_elem: ET.Element) -> dict[str, _FrameRef]:
    refs: dict[str, _FrameRef] = {}
    frames = _child(joint_elem, "frames")
    if frames is None:
        return refs
    for frame in _children(frames, "PhysicalOffsetFrame"):
        name = frame.get("name")
        if not name:
            continue
        body = _body_from_path(_text(frame, "socket_parent"))
        refs[name] = _FrameRef(
            body=body,
            origin=Origin(
                xyz=_vec_text(frame, "translation", (0.0, 0.0, 0.0), length=3),  # type: ignore[arg-type]
                rpy=_vec_text(frame, "orientation", (0.0, 0.0, 0.0), length=3),  # type: ignore[arg-type]
            ),
        )
    return refs


def _joint_endpoints(
    joint_elem: ET.Element,
    frame_refs: dict[str, _FrameRef],
    link_names: set[str],
) -> tuple[str, str, Origin]:
    parent_body = _text(joint_elem, "parent_body")
    child_body = _text(joint_elem, "body") or _text(joint_elem, "child_body")
    if parent_body and child_body:
        return (
            _normalize_body_name(parent_body),
            _normalize_body_name(child_body),
            Origin(
                xyz=_vec_text(
                    joint_elem, "location_in_parent", (0.0, 0.0, 0.0), length=3
                ),  # type: ignore[arg-type]
                rpy=_vec_text(
                    joint_elem, "orientation_in_parent", (0.0, 0.0, 0.0), length=3
                ),  # type: ignore[arg-type]
            ),
        )

    parent_frame = _text(joint_elem, "socket_parent_frame")
    child_frame = _text(joint_elem, "socket_child_frame")
    parent_ref = _resolve_frame(parent_frame, frame_refs, link_names)
    child_ref = _resolve_frame(child_frame, frame_refs, link_names)
    return parent_ref.body, child_ref.body, parent_ref.origin


def _resolve_frame(
    frame_name: str,
    frame_refs: dict[str, _FrameRef],
    link_names: set[str],
) -> _FrameRef:
    if frame_name in frame_refs:
        return frame_refs[frame_name]
    body = _body_from_path(frame_name)
    if body in link_names or body == _GROUND_LINK_NAME:
        return _FrameRef(body=body, origin=Origin())
    return _FrameRef(body=body, origin=Origin())


def _body_from_path(path: str) -> str:
    if not path or path == "..":
        return _GROUND_LINK_NAME
    normalized = path.strip()
    if normalized in {"/ground", "ground"}:
        return _GROUND_LINK_NAME
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2 and parts[-2].lower() == "bodyset":
        return parts[-1]
    return _normalize_body_name(parts[-1] if parts else normalized)


def _normalize_body_name(name: str) -> str:
    return _GROUND_LINK_NAME if name == "ground" else name


def _joint_motion(
    joint_elem: ET.Element,
    warnings: list[str],
) -> tuple[JointType, tuple[float, float, float], JointLimits | None]:
    tag = _tag(joint_elem)
    if tag == "PinJoint":
        return JointType.REVOLUTE, (0.0, 0.0, 1.0), _coordinate_limits(joint_elem)
    if tag == "SliderJoint":
        return JointType.PRISMATIC, (1.0, 0.0, 0.0), _coordinate_limits(joint_elem)
    if tag == "BallJoint":
        return JointType.GIMBAL, (0.0, 0.0, 1.0), _coordinate_limits(joint_elem)
    if tag == "FreeJoint":
        return JointType.FLOATING, (0.0, 0.0, 1.0), None
    if tag == "WeldJoint":
        return JointType.FIXED, (0.0, 0.0, 1.0), None
    return _custom_joint_motion(joint_elem, warnings)


def _custom_joint_motion(
    joint_elem: ET.Element,
    warnings: list[str],
) -> tuple[JointType, tuple[float, float, float], JointLimits | None]:
    coordinates = _coordinates(joint_elem)
    if len(coordinates) > 1:
        warnings.append(
            f"CustomJoint '{joint_elem.get('name', '<unnamed>')}' has "
            f"{len(coordinates)} coordinates; approximating as gimbal"
        )
        return JointType.GIMBAL, (0.0, 0.0, 1.0), _coordinate_limits(joint_elem)
    motion_type = _text(coordinates[0], "motion_type") if coordinates else ""
    if motion_type == "translational":
        return JointType.PRISMATIC, (1.0, 0.0, 0.0), _coordinate_limits(joint_elem)
    if coordinates:
        return JointType.REVOLUTE, (0.0, 0.0, 1.0), _coordinate_limits(joint_elem)
    return JointType.FIXED, (0.0, 0.0, 1.0), None


def _coordinate_limits(joint_elem: ET.Element) -> JointLimits | None:
    coordinates = _coordinates(joint_elem)
    if not coordinates:
        return None
    range_text = _text(coordinates[0], "range")
    if not range_text:
        return None
    lower, upper = (float(value) for value in range_text.split()[:2])
    return JointLimits(lower=lower, upper=upper)


def _coordinates(joint_elem: ET.Element) -> list[ET.Element]:
    direct = _child(joint_elem, "coordinates")
    if direct is not None:
        return _children(direct, "Coordinate")
    coordinate_set = _child(joint_elem, "CoordinateSet")
    if coordinate_set is None:
        return []
    objects = _child(coordinate_set, "objects")
    return _children(objects if objects is not None else coordinate_set, "Coordinate")
