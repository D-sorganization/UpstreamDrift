"""SDFormat loader for Drake-native model browsing and composition."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as ET
from defusedxml.ElementTree import ParseError

from src.shared.python.logging_pkg.logger_utils import get_logger
from src.shared.python.model_generation.canonical_model import CanonicalModel
from src.shared.python.model_generation.core.composite_joints import (
    expand_gimbal_joint,
    expand_universal_joint,
)
from src.shared.python.model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)

logger = get_logger(__name__)

_Element: TypeAlias = Element
_DEFAULT_INERTIA = Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)
_SUPPORTED_JOINT_TYPES = {
    "fixed": JointType.FIXED,
    "revolute": JointType.REVOLUTE,
    "continuous": JointType.CONTINUOUS,
    "prismatic": JointType.PRISMATIC,
    "ball": JointType.GIMBAL,
    "universal": JointType.UNIVERSAL,
}


class SdfLoader:
    """Parse the useful subset of SDFormat into a canonical model."""

    def load(self, path: str | Path) -> CanonicalModel:
        """Load an SDF file from disk.

        Args:
            path: Path to an ``.sdf`` file.

        Returns:
            Canonical model containing links, joints, inertials, and first
            visual/collision geometry per link.

        Raises:
            ValueError: If the path is missing, malformed, or violates topology
                contracts needed for model browsing and composition.
        """
        sdf_path = Path(path)
        if not sdf_path.is_file():
            raise ValueError(f"SDF file does not exist: {sdf_path}")
        try:
            root = ET.parse(sdf_path).getroot()
        except (OSError, ParseError) as exc:
            raise ValueError(f"Failed to parse SDF file {sdf_path}: {exc}") from exc
        return self._load_root(root, source_path=sdf_path)

    def load_string(
        self, content: str, *, source_path: Path | None = None
    ) -> CanonicalModel:
        """Load SDF XML content from memory."""
        if not content.strip():
            raise ValueError("SDF content must be a non-empty string")
        try:
            root = ET.fromstring(content)
        except ParseError as exc:
            raise ValueError(f"Failed to parse SDF content: {exc}") from exc
        return self._load_root(root, source_path=source_path)

    def _load_root(self, root: _Element, *, source_path: Path | None) -> CanonicalModel:
        if _local_name(root.tag) != "sdf":
            raise ValueError("SDF root element must be <sdf>")
        models = _children(root, "model")
        if not models:
            raise ValueError("SDF document must contain at least one <model>")
        if len(models) > 1:
            logger.warning(
                "SDF document contains multiple <model> elements; loading first model"
            )
        model = models[0]
        model_name = model.get("name", "").strip()
        if not model_name:
            raise ValueError("SDF <model> must have a non-empty name")

        frame_poses: dict[str, Origin] = {}
        links: list[Link] = []
        for link_element in _children(model, "link"):
            name = link_element.get("name", "").strip()
            if name:
                frame_poses[name] = _pose_from_element(
                    _child(link_element, "pose"), frame_poses
                )
            links.append(self._parse_link(link_element, frame_poses))
        link_names = {link.name for link in links}
        if len(link_names) != len(links):
            raise ValueError("SDF model contains duplicate link names")

        joints: list[Joint] = []
        extra_links: list[Link] = []
        for joint_element in _children(model, "joint"):
            joint = self._parse_joint(joint_element, link_names, frame_poses)
            if joint.joint_type == JointType.GIMBAL:
                expanded_links, expanded_joints = expand_gimbal_joint(joint)
                extra_links.extend(expanded_links)
                joints.extend(expanded_joints)
            elif joint.joint_type == JointType.UNIVERSAL:
                expanded_links, expanded_joints = expand_universal_joint(joint)
                extra_links.extend(expanded_links)
                joints.extend(expanded_joints)
            else:
                joints.append(joint)

        metadata = {
            "source_format": "sdf",
            "sdf_version": root.get("version", ""),
        }
        if source_path is not None:
            metadata["source_path"] = str(source_path)
        canonical = CanonicalModel.from_core(
            name=model_name,
            links=[*links, *extra_links],
            joints=joints,
            metadata=metadata,
        )
        canonical.require_valid()
        return canonical

    def _parse_link(
        self, link_element: _Element, frame_poses: dict[str, Origin]
    ) -> Link:
        name = link_element.get("name", "").strip()
        if not name:
            raise ValueError("SDF <link> must have a non-empty name")
        link_origin = _pose_from_element(_child(link_element, "pose"), frame_poses)
        visual = _first_child(link_element, "visual")
        collision = _first_child(link_element, "collision")
        return Link(
            name=name,
            inertia=_parse_inertia(_first_child(link_element, "inertial")),
            visual_geometry=_parse_geometry(_first_child(visual, "geometry")),
            visual_origin=_pose_from_element(
                _child(visual, "pose"), frame_poses, default=link_origin
            ),
            visual_material=_parse_material(_first_child(visual, "material")),
            collision_geometry=_parse_geometry(_first_child(collision, "geometry")),
            collision_origin=_pose_from_element(
                _child(collision, "pose"), frame_poses, default=link_origin
            ),
        )

    def _parse_joint(
        self,
        joint_element: _Element,
        link_names: set[str],
        frame_poses: dict[str, Origin],
    ) -> Joint:
        name = joint_element.get("name", "").strip()
        if not name:
            raise ValueError("SDF <joint> must have a non-empty name")
        sdf_type = joint_element.get("type", "fixed").strip().lower()
        try:
            joint_type = _SUPPORTED_JOINT_TYPES[sdf_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported SDF joint type '{sdf_type}'") from exc

        parent = _required_text(joint_element, "parent", f"joint {name}")
        child = _required_text(joint_element, "child", f"joint {name}")
        if parent not in link_names:
            raise ValueError(f"joint {name} references unknown parent link '{parent}'")
        if child not in link_names:
            raise ValueError(f"joint {name} references unknown child link '{child}'")

        axis_element = _first_child(joint_element, "axis")
        return Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin=_pose_from_element(_child(joint_element, "pose"), frame_poses),
            axis=_parse_axis(axis_element),
            limits=_parse_limits(_first_child(axis_element, "limit")),
            dynamics=_parse_dynamics(_first_child(axis_element, "dynamics")),
        )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: _Element | None, name: str) -> list[_Element]:
    if element is None:
        return []
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(element: _Element | None, name: str) -> _Element | None:
    return next(iter(_children(element, name)), None)


def _child(element: _Element | None, name: str) -> _Element | None:
    return _first_child(element, name)


def _required_text(element: _Element, child_name: str, context: str) -> str:
    child = _child(element, child_name)
    text = "" if child is None or child.text is None else child.text.strip()
    if not text:
        raise ValueError(f"{context} requires <{child_name}>")
    return text


def _float_text(element: _Element | None, child_name: str, default: float) -> float:
    child = _child(element, child_name)
    if child is None or child.text is None or not child.text.strip():
        return default
    return float(child.text.strip())


def _vector_text(
    element: _Element | None,
    child_name: str,
    *,
    length: int,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    child = _child(element, child_name)
    if child is None or child.text is None or not child.text.strip():
        return default
    values = tuple(float(part) for part in child.text.split())
    if len(values) != length or any(not math.isfinite(value) for value in values):
        raise ValueError(f"<{child_name}> must contain {length} finite floats")
    return values


def _pose_from_element(
    pose_element: _Element | None,
    frames: dict[str, Origin],
    *,
    default: Origin | None = None,
) -> Origin:
    if (
        pose_element is None
        or pose_element.text is None
        or not pose_element.text.strip()
    ):
        return default or Origin()
    values = tuple(float(part) for part in pose_element.text.split())
    if len(values) != 6 or any(not math.isfinite(value) for value in values):
        raise ValueError("<pose> must contain 6 finite floats")
    origin = Origin(
        xyz=(values[0], values[1], values[2]),
        rpy=(values[3], values[4], values[5]),
    )
    relative_to = pose_element.get("relative_to", "").strip()
    if not relative_to:
        return origin
    base = frames.get(relative_to)
    if base is None:
        raise ValueError(f"<pose> references unknown relative_to frame '{relative_to}'")
    return Origin(
        xyz=(
            base.xyz[0] + origin.xyz[0],
            base.xyz[1] + origin.xyz[1],
            base.xyz[2] + origin.xyz[2],
        ),
        rpy=(
            base.rpy[0] + origin.rpy[0],
            base.rpy[1] + origin.rpy[1],
            base.rpy[2] + origin.rpy[2],
        ),
    )


def _parse_inertia(inertial: _Element | None) -> Inertia:
    if inertial is None:
        return _DEFAULT_INERTIA
    inertia_element = _first_child(inertial, "inertia")
    return Inertia(
        ixx=_float_text(inertia_element, "ixx", _DEFAULT_INERTIA.ixx),
        iyy=_float_text(inertia_element, "iyy", _DEFAULT_INERTIA.iyy),
        izz=_float_text(inertia_element, "izz", _DEFAULT_INERTIA.izz),
        ixy=_float_text(inertia_element, "ixy", 0.0),
        ixz=_float_text(inertia_element, "ixz", 0.0),
        iyz=_float_text(inertia_element, "iyz", 0.0),
        mass=_float_text(inertial, "mass", _DEFAULT_INERTIA.mass),
        center_of_mass=_pose_from_element(_child(inertial, "pose"), {}).xyz,
    )


def _parse_geometry(geometry: _Element | None) -> Geometry | None:
    if geometry is None:
        return None
    if (box := _first_child(geometry, "box")) is not None:
        size = _vector_text(box, "size", length=3, default=())
        return Geometry.box(size[0], size[1], size[2])
    if (sphere := _first_child(geometry, "sphere")) is not None:
        return Geometry.sphere(_float_text(sphere, "radius", 0.0))
    if (cylinder := _first_child(geometry, "cylinder")) is not None:
        return Geometry.cylinder(
            _float_text(cylinder, "radius", 0.0),
            _float_text(cylinder, "length", 0.0),
        )
    if (mesh := _first_child(geometry, "mesh")) is not None:
        uri = _required_text(mesh, "uri", "mesh geometry")
        scale = _vector_text(
            mesh,
            "scale",
            length=3,
            default=(1.0, 1.0, 1.0),
        )
        return Geometry.mesh(uri, scale=(scale[0], scale[1], scale[2]))
    return None


def _parse_material(material: _Element | None) -> Material | None:
    if material is None:
        return None
    diffuse = _vector_text(
        material,
        "diffuse",
        length=4,
        default=(0.8, 0.8, 0.8, 1.0),
    )
    return Material(name="sdf_material", color=diffuse)  # type: ignore[arg-type]


def _parse_axis(axis: _Element | None) -> tuple[float, float, float]:
    values = _vector_text(axis, "xyz", length=3, default=(0.0, 0.0, 1.0))
    return values[0], values[1], values[2]


def _parse_limits(limit: _Element | None) -> JointLimits | None:
    if limit is None:
        return None
    return JointLimits(
        lower=_float_text(limit, "lower", -math.inf),
        upper=_float_text(limit, "upper", math.inf),
        effort=_float_text(limit, "effort", 1000.0),
        velocity=_float_text(limit, "velocity", 10.0),
    )


def _parse_dynamics(dynamics: _Element | None) -> JointDynamics:
    if dynamics is None:
        return JointDynamics()
    return JointDynamics(
        damping=_float_text(dynamics, "damping", 0.5),
        friction=_float_text(dynamics, "friction", 0.0),
    )
