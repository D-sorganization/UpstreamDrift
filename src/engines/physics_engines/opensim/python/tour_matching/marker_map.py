"""Tour-capture marker labels mapped onto golf_humanoid.osim bodies.

The Rajagopal-derived golf humanoid has no separate head body, so the three
head markers ride on the torso exactly as they ride on the native Hub body;
that shared limitation must stay visible in every acceptance report. Knee
markers sit on the femur (lateral epicondyle), ankle markers on the tibia
(lateral malleolus) and toe markers on the calcaneus segment (metatarsal
heads), following the Rajagopal segment boundaries. Marker offsets within
each body are not defined here; they are calibrated in OS-3.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from defusedxml import ElementTree as SafeET

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    MARKER_VALIDITY_POLICY,
    tracked_labels,
)

GOLF_HUMANOID_MARKER_BODIES: MappingProxyType[str, str] = MappingProxyType(
    {
        "HeadTop": "torso",
        "HeadFront": "torso",
        "HeadSide": "torso",
        "BackTop": "torso",
        "BackLeft": "torso",
        "BackRight": "torso",
        "WaistLeft": "pelvis",
        "WaistRight": "pelvis",
        "WaistLBack": "pelvis",
        "WaistRBack": "pelvis",
        "LShoulderTop": "torso",
        "LShoulderBack": "torso",
        "LUArmHigh": "humerus_l",
        "LElbowOut": "humerus_l",
        "LWristTop": "radius_l",
        "RShoulderTop": "torso",
        "RShoulderBack": "torso",
        "RUArmHigh": "humerus_r",
        "RElbowOut": "humerus_r",
        "RWristTop": "radius_r",
        "LKneeOut": "femur_l",
        "LAnkleOut": "tibia_l",
        "LToeIn": "calcn_l",
        "LToeOut": "calcn_l",
        "RKneeOut": "femur_r",
        "RAnkleOut": "tibia_r",
        "RToeIn": "calcn_r",
        "RToeOut": "calcn_r",
        "Marker_2:2:1": "Club",
        "Marker_2:2:2": "Club",
        "Marker_2:2:3": "Club",
        "Marker_3:3:1": "Club",
        "Marker_3:3:2": "Club",
        "Marker_3:3:3": "Club",
    }
)

if set(GOLF_HUMANOID_MARKER_BODIES) != set(tracked_labels()):  # pragma: no cover
    raise RuntimeError("Marker-body map must cover exactly the tracked labels")


def body_for(label: str) -> str:
    """Return the golf_humanoid body carrying a tracked capture label."""
    if label in MARKER_SEGMENTS["unassigned"]:
        raise ValueError(f"Capture label has no anatomical role: {label}")
    try:
        return GOLF_HUMANOID_MARKER_BODIES[label]
    except KeyError as error:
        raise ValueError(f"Unknown capture label: {label}") from error


def labels_per_body() -> dict[str, tuple[str, ...]]:
    """Group tracked labels by body, preserving capture order."""
    grouped: dict[str, list[str]] = {}
    for label in tracked_labels():
        grouped.setdefault(GOLF_HUMANOID_MARKER_BODIES[label], []).append(label)
    return {body: tuple(labels) for body, labels in grouped.items()}


def model_body_names(osim_path: Path) -> set[str]:
    """Return the body names declared in an .osim file (safe XML parse)."""
    root = SafeET.parse(str(osim_path)).getroot()
    bodies = root.findall("Model/BodySet/objects/Body")
    names = {body.get("name", "") for body in bodies}
    if not names or "" in names:
        raise ValueError("Model declares no named bodies")
    return names


def bodies_missing_from(model_bodies: set[str]) -> tuple[str, ...]:
    """Return mapped bodies absent from the model, sorted."""
    return tuple(sorted(set(GOLF_HUMANOID_MARKER_BODIES.values()) - set(model_bodies)))


@precondition(lambda label, is_valid=True: isinstance(label, str), "label must be str")
@postcondition(lambda r: r >= 0.0, "weight must be non-negative")
def marker_weight(label: str, is_valid: bool = True) -> float:
    """Return OpenSim tracking weight for the marker label."""
    return MARKER_VALIDITY_POLICY.weight_for(label, is_valid=is_valid)


def marker_weights(is_valid: bool = True) -> dict[str, float]:
    """Return dictionary of marker weights for all tracked labels."""
    return {label: marker_weight(label, is_valid) for label in tracked_labels()}


ANTHRO_DOCUMENT_MARKER_BODIES: MappingProxyType[str, str] = MappingProxyType(
    {
        "HeadTop": "Head",
        "HeadFront": "Head",
        "HeadSide": "Head",
        "BackTop": "Spine",
        "BackLeft": "Spine",
        "BackRight": "Spine",
        "WaistLeft": "Hip",
        "WaistRight": "Hip",
        "WaistLBack": "Hip",
        "WaistRBack": "Hip",
        "LShoulderTop": "LScap",
        "LShoulderBack": "LScap",
        "LUArmHigh": "LS",
        "LElbowOut": "LS",
        "LWristTop": "LF",
        "RShoulderTop": "RScap",
        "RShoulderBack": "RScap",
        "RUArmHigh": "RS",
        "RElbowOut": "RS",
        "RWristTop": "RF",
        "LKneeOut": "femur_l",
        "LAnkleOut": "tibia_l",
        "LToeIn": "calcn_l",
        "LToeOut": "calcn_l",
        "RKneeOut": "femur_r",
        "RAnkleOut": "tibia_r",
        "RToeIn": "calcn_r",
        "RToeOut": "calcn_r",
        "Marker_2:2:1": "Clubhead",
        "Marker_2:2:2": "Clubhead",
        "Marker_2:2:3": "Clubhead",
        "Marker_3:3:1": "Clubhead",
        "Marker_3:3:2": "Clubhead",
        "Marker_3:3:3": "Clubhead",
    }
)


@precondition(
    lambda output_path: output_path is not None, "output_path must not be None"
)
@postcondition(lambda r: isinstance(r, Path) and r.is_file(), "Must write file")
def write_ik_tasks_xml(
    output_path: Path | str,
    weights: Mapping[str, float] | None = None,
) -> Path:
    """Serialize OpenSim 4.0 IKTaskSet XML with marker validity policy weights."""
    import xml.etree.ElementTree as ET

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    w_map = weights if weights is not None else marker_weights(is_valid=True)

    root = ET.Element("OpenSimDocument", attrib={"Version": "40000"})
    task_set = ET.SubElement(root, "IKTaskSet", attrib={"name": "IKTasks"})
    objects = ET.SubElement(task_set, "objects")
    ET.SubElement(task_set, "groups")

    for label in tracked_labels():
        task = ET.SubElement(objects, "IKMarkerTask", attrib={"name": label})
        ET.SubElement(task, "apply").text = "true"
        w = float(w_map.get(label, 1.0))
        ET.SubElement(task, "weight").text = format(w, ".6g")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    return out


def build_ik_task_set(
    weights: Mapping[str, float] | None = None,
) -> object:
    """Build OpenSim IKTaskSet object using native SDK if available."""
    import opensim  # type: ignore[import-not-found]

    w_map = weights if weights is not None else marker_weights(is_valid=True)
    tasks = opensim.IKTaskSet()
    for label in tracked_labels():
        task = opensim.IKMarkerTask()
        task.setName(label)
        task.setWeight(float(w_map.get(label, 1.0)))
        task.setApply(True)
        tasks.cloneAndAppend(task)
    return tasks
