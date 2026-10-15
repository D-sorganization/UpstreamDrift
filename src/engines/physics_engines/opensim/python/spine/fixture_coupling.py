"""Coupler constraints and lumped trunk muscles for the spine fixture (SPN-7b).

Augments the SPN-7a fixture tree (``fixture_model``) in place; pure
ElementTree, no OpenSim import.  Every number is INVENTED.

Coupling.  For a :class:`LinearCouplingLaw` ``theta_i = C_i @ phi`` the
generator adds

* a **massless, inertia-free driver body** ``spine_coupling_driver`` attached
  to ``pelvis`` by the CustomJoint ``spine_coupling_driver_jnt`` (identity
  offsets).  Its coordinates are the law's independent names, in order:
  ``phi_0..phi_2`` rotate about ``Z``, ``X``, ``Y`` (ISB ZXY axes) and
  ``phi_3..phi_5`` translate along ``X``, ``Y``, ``Z`` (so at most six
  independent coordinates).  Nothing hangs off the driver, so it never
  contributes to inverse dynamics or joint reaction; inventories exclude it by
  name (:data:`DRIVER_BODY`);
* one ``CoordinateCouplerConstraint`` per dependent level coordinate
  (``dependent = theta_i[row]``) whose ``coupled_coordinates_function`` is the
  multivariate ``LinearFunction`` with coefficients ``[C_i[row, :], 0]``.
  Cross-axis terms are therefore exact.  The constraints prescribe kinematics
  only: they are disabled for ID / SO / JointReaction (ADR-0055 item 6).

Laws that are not :class:`LinearCouplingLaw` raise ``TypeError``.

Muscles.  Eight bilateral ``Millard2012EquilibriumMuscle`` actuators (erector
spinae, rectus abdominis, external and internal oblique, ``_l``/``_r``) run
pelvis -> L3 via point -> thorax_proxy.  Frames: X anterior, Y cranial, Z
right, so right-side points have ``z > 0`` and the pair mirrors in the
sagittal (``z -> -z``) plane.  The parameters are invented and only meant to
give finite, bilaterally consistent moment arms for static-optimisation
tests: they carry no physiological claim.  Each tendon slack length is
``neutral path length - optimal fibre length`` (path length computed
analytically in :func:`neutral_path_lengths`), so every fibre sits at its
optimal length at the neutral pose and no muscle is slack there.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only; never parses untrusted XML
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.engines.physics_engines.opensim.python.full_body_osim import (
    _build_transform_axis,
    _format_numbers,
    _make_physical_offset_frame,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    _AXES,
    _COORD_RANGE_RAD,
    FixtureSpec,
    fixture_base_offset,
    fixture_level_names,
    fixture_neutral_transforms,
    fixture_terminal_offset,
)
from src.shared.python.spine.coupling import LinearCouplingLaw

__all__ = [
    "DRIVER_BODY",
    "DRIVER_JOINT",
    "MAX_INDEPENDENT_COORDINATES",
    "augment_fixture_tree",
    "fixture_muscle_names",
    "neutral_path_lengths",
]

DRIVER_BODY = "spine_coupling_driver"
DRIVER_JOINT = "spine_coupling_driver_jnt"
MAX_INDEPENDENT_COORDINATES = 6
_TRANSLATION_AXES = ("1 0 0", "0 1 0", "0 0 1")
_PELVIS = "pelvis"
_THORAX = "thorax_proxy"
_VIA_BODY = "L3"


@dataclass(frozen=True)
class _MuscleDef:
    stem: str
    force_n: float  # invented
    # (x, y, z) of the RIGHT-side points in the body frames; the left side
    # mirrors z.  Order: pelvis origin, L3 via, thorax_proxy insertion.
    points: tuple[tuple[float, float, float], ...]


_MUSCLES = (
    _MuscleDef(
        "erector_spinae",
        1500.0,
        ((-0.05, 0.08, 0.04), (-0.04, 0.02, 0.04), (-0.05, 0.05, 0.05)),
    ),
    _MuscleDef(
        "rectus_abdominis",
        900.0,
        ((0.06, 0.08, 0.03), (0.07, 0.02, 0.03), (0.06, 0.05, 0.03)),
    ),
    _MuscleDef(
        "external_oblique",
        700.0,
        ((0.03, 0.08, 0.10), (0.06, 0.02, 0.05), (0.03, 0.05, -0.03)),
    ),
    _MuscleDef(
        "internal_oblique",
        600.0,
        ((0.03, 0.08, -0.08), (0.06, 0.02, 0.01), (0.03, 0.05, 0.08)),
    ),
)
_OPTIMAL_FIBER_M = 0.12  # invented
_PENNATION_RAD = 0.0


def fixture_muscle_names() -> tuple[str, ...]:
    """The eight muscle names: each stem as ``_l`` then ``_r``."""
    return tuple(f"{m.stem}_{side}" for m in _MUSCLES for side in ("l", "r"))


def _neutral_body_transforms(spec: FixtureSpec) -> dict[str, np.ndarray]:
    """Neutral pose of each muscle-bearing body expressed in the pelvis frame."""
    poses = {_PELVIS: np.eye(4)}
    bodies = spec._bodies
    poses[bodies[1]] = fixture_base_offset(spec)
    neutral = fixture_neutral_transforms(spec)
    for names in fixture_level_names(spec):
        poses[names.superior_body] = poses[names.inferior_body] @ neutral[names.level]
    poses[_THORAX] = poses[bodies[-1]] @ fixture_terminal_offset(spec)
    return poses


def _path_length(mdef: _MuscleDef, side: str, poses: dict[str, np.ndarray]) -> float:
    sign = 1.0 if side == "r" else -1.0
    bodies = (_PELVIS, _VIA_BODY, _THORAX)
    world = [
        (poses[b] @ np.array([x, y, sign * z, 1.0]))[:3]
        for b, (x, y, z) in zip(bodies, mdef.points, strict=True)
    ]
    return float(
        sum(np.linalg.norm(b - a) for a, b in zip(world, world[1:], strict=False))
    )


def neutral_path_lengths(spec: FixtureSpec) -> dict[str, float]:
    """Analytic neutral muscle path length [m] per muscle name.

    Sum of straight segment lengths between the path points expressed in the
    pelvis frame at the neutral pose (all coordinates zero).  Pure Python;
    equals OpenSim's ``getLength`` at the default state to rounding.

    Raises:
        TypeError: ``spec`` is not a :class:`FixtureSpec`.
        ValueError: the spec has no ``L3`` body for the via point.
    """
    if not isinstance(spec, FixtureSpec):
        raise TypeError(f"spec must be a FixtureSpec, got {type(spec).__name__}")
    poses = _neutral_body_transforms(spec)
    if _VIA_BODY not in poses:
        raise ValueError(f"fixture spec has no {_VIA_BODY} body for the via point")
    return {
        f"{m.stem}_{side}": _path_length(m, side, poses)
        for m in _MUSCLES
        for side in ("l", "r")
    }


def _validate_law(law: object, spec: FixtureSpec) -> LinearCouplingLaw:
    if not isinstance(law, LinearCouplingLaw):
        raise TypeError(
            "coupling must be a LinearCouplingLaw (state-dependent and "
            f"polynomial laws are not representable), got {type(law).__name__}"
        )
    n_phi = len(law.independent)
    if n_phi > MAX_INDEPENDENT_COORDINATES:
        raise ValueError(
            f"a law may have at most {MAX_INDEPENDENT_COORDINATES} independent "
            f"coordinates for the driver joint, got {n_phi}"
        )
    stray = [lv.name for lv in law.levels if lv not in spec.levels]
    if stray:
        raise ValueError(f"law levels {stray} are not in the fixture spec levels")
    taken = {c for n in fixture_level_names(spec) for c in n.coordinates}
    clash = sorted(taken & set(law.independent))
    if clash:
        raise ValueError(f"independent names clash with level coordinates: {clash}")
    return law


def _add_driver_body(model: ET.Element) -> None:
    objects = model.find("BodySet/objects")
    body = ET.SubElement(objects, "Body", attrib={"name": DRIVER_BODY})
    ET.SubElement(body, "mass").text = "0"
    ET.SubElement(body, "mass_center").text = "0 0 0"
    ET.SubElement(body, "inertia").text = _format_numbers((0, 0, 0, 0, 0, 0))
    ET.SubElement(body, "attached_geometry")


def _add_driver_joint(model: ET.Element, independent: tuple[str, ...]) -> None:
    objects = model.find("JointSet/objects")
    joint = ET.SubElement(objects, "CustomJoint", attrib={"name": DRIVER_JOINT})
    pname, cname = f"{DRIVER_JOINT}_parent_offset", f"{DRIVER_JOINT}_child_offset"
    ET.SubElement(joint, "socket_parent_frame").text = pname
    ET.SubElement(joint, "socket_child_frame").text = cname
    coords = ET.SubElement(joint, "coordinates")
    for name in independent:
        coord = ET.SubElement(coords, "Coordinate", attrib={"name": name})
        ET.SubElement(coord, "default_value").text = "0"
        ET.SubElement(coord, "default_speed_value").text = "0"
        ET.SubElement(coord, "range").text = f"{-_COORD_RANGE_RAD} {_COORD_RANGE_RAD}"
        ET.SubElement(coord, "clamped").text = "false"
        ET.SubElement(coord, "locked").text = "false"
    frames = ET.SubElement(joint, "frames")
    frames.append(
        _make_physical_offset_frame(pname, f"/bodyset/{_PELVIS}", (0, 0, 0), (0, 0, 0))
    )
    frames.append(
        _make_physical_offset_frame(cname, f"/bodyset/{DRIVER_BODY}", (0, 0, 0))
    )
    transform = ET.SubElement(joint, "SpatialTransform")
    padded = [
        *independent,
        *([None] * (MAX_INDEPENDENT_COORDINATES - len(independent))),
    ]
    for idx, (axis, name) in enumerate(zip(_AXES, padded[:3], strict=True), start=1):
        _build_transform_axis(transform, "rotation", idx, axis, name)
    for idx, (axis, name) in enumerate(
        zip(_TRANSLATION_AXES, padded[3:], strict=True), start=1
    ):
        _build_transform_axis(transform, "translation", idx, axis, name)


def _add_couplers(model: ET.Element, law: LinearCouplingLaw, spec: FixtureSpec) -> None:
    constraints = ET.SubElement(model, "ConstraintSet", attrib={"name": "constraints"})
    objects = ET.SubElement(constraints, "objects")
    independent = " ".join(law.independent)
    for names in fixture_level_names(spec):
        if names.level not in law.matrices:
            continue
        matrix = law.matrices[names.level]
        for row, coord in enumerate(names.coordinates):
            item = ET.SubElement(
                objects,
                "CoordinateCouplerConstraint",
                attrib={"name": f"{coord}_coupler"},
            )
            ET.SubElement(item, "isEnforced").text = "true"
            ET.SubElement(item, "dependent_coordinate_name").text = coord
            ET.SubElement(item, "independent_coordinate_names").text = independent
            func = ET.SubElement(item, "coupled_coordinates_function")
            linear = ET.SubElement(func, "LinearFunction")
            ET.SubElement(linear, "coefficients").text = _format_numbers(
                (*(float(v) for v in matrix[row]), 0.0)
            )
            ET.SubElement(item, "scale_factor").text = "1"


def _add_muscle(
    forces: ET.Element, mdef: _MuscleDef, side: str, path_length: float
) -> None:
    sign = 1.0 if side == "r" else -1.0
    muscle = ET.SubElement(
        forces,
        "Millard2012EquilibriumMuscle",
        attrib={"name": f"{mdef.stem}_{side}"},
    )
    ET.SubElement(muscle, "appliesForce").text = "true"
    ET.SubElement(muscle, "max_isometric_force").text = format(mdef.force_n, ".17g")
    ET.SubElement(muscle, "optimal_fiber_length").text = format(
        _OPTIMAL_FIBER_M, ".17g"
    )
    ET.SubElement(muscle, "tendon_slack_length").text = format(
        path_length - _OPTIMAL_FIBER_M, ".17g"
    )
    ET.SubElement(muscle, "pennation_angle_at_optimal").text = format(
        _PENNATION_RAD, ".17g"
    )
    path = ET.SubElement(muscle, "GeometryPath", attrib={"name": "path"})
    pset = ET.SubElement(path, "PathPointSet")
    objects = ET.SubElement(pset, "objects")
    bodies = (_PELVIS, _VIA_BODY, _THORAX)
    labels = ("origin", "via_l3", "insertion")
    for body, label, (x, y, z) in zip(bodies, labels, mdef.points, strict=True):
        point = ET.SubElement(objects, "PathPoint", attrib={"name": label})
        ET.SubElement(point, "location").text = _format_numbers((x, y, sign * z))
        ET.SubElement(point, "socket_parent_frame").text = f"/bodyset/{body}"


def _add_muscles(model: ET.Element, spec: FixtureSpec) -> None:
    lengths = neutral_path_lengths(spec)
    forceset = ET.SubElement(model, "ForceSet", attrib={"name": "forceset"})
    objects = ET.SubElement(forceset, "objects")
    for mdef in _MUSCLES:
        for side in ("l", "r"):
            _add_muscle(objects, mdef, side, lengths[f"{mdef.stem}_{side}"])


#: Assembly accuracy for coupled fixtures; OpenSim's default (1e-9) leaves
#: coupler residuals at the 1e-9 acceptance tolerance of issue #12751.
ASSEMBLY_ACCURACY = 1e-12


def _set_assembly_accuracy(model: ET.Element) -> None:
    node = model.find("assembly_accuracy")
    if node is None:
        node = ET.SubElement(model, "assembly_accuracy")
    node.text = repr(ASSEMBLY_ACCURACY)


def augment_fixture_tree(
    root: ET.Element,
    spec: FixtureSpec,
    coupling: object | None,
    muscles: object,
) -> None:
    """Add the driver, couplers and/or muscles to a ``build_fixture_tree`` root.

    Preconditions: ``coupling`` is ``None`` or a :class:`LinearCouplingLaw`
    whose levels are in ``spec`` and whose independent names are unique, do not
    clash with level coordinates and number at most six; ``muscles`` is a bool.
    Postconditions: ``root`` gains (when requested) the massless driver body and
    joint, one coupler per dependent coordinate of each law level, and the
    eight muscles; nothing else changes.

    Raises:
        TypeError: wrong ``coupling`` or ``muscles`` type.
        ValueError: the law does not fit the fixture.
    """
    if not isinstance(muscles, bool):
        raise TypeError(f"muscles must be a bool, got {type(muscles).__name__}")
    law = None if coupling is None else _validate_law(coupling, spec)
    model = root.find("Model")
    if model is None:
        raise ValueError("root has no Model element")
    if law is not None:
        _set_assembly_accuracy(model)
        _add_driver_body(model)
        _add_driver_joint(model, law.independent)
        _add_couplers(model, law, spec)
    if muscles:
        _add_muscles(model, spec)
