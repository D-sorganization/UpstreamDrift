"""Synthetic segmental-spine OpenSim fixture generator (SPN-7a, #12750).

A CI-safe stand-in for the licensed spine models.  Every number here is
INVENTED (or a plain geometric convention); nothing is transcribed from a
SimTK ``.osim``.  Generation is pure ElementTree string building and needs no
OpenSim import; the XML helpers are shared with ``full_body_osim``.

Topology (caudal to cranial)::

    ground -> pelvis (FreeJoint) -> <first inferior body> (WeldJoint)
           -> ... one CustomJoint ``<LEVEL>_jnt`` per level ...
           -> thorax_proxy (WeldJoint)

For the default five lumbar levels the bodies are ``pelvis, sacrum, L5, L4,
L3, L2, L1, thorax_proxy``.  A body is named after the superior vertebra of
the level that ends in it; ``S1`` is called ``sacrum``.

Conventions (ADR-0055).  Frames: X anterior, Y cranial, Z right.  Level
``i`` joins an INFERIOR body to a SUPERIOR body; the transform of the superior
frame in the inferior frame is ``T_i = T_i0 @ T(R(theta_i), 0)`` with the ISB
ZXY intrinsic rotation ``R = Rz(FE) Rx(LB) Ry(AR)`` (CustomJoint
``rotation1/2/3`` about ``0 0 1``, ``1 0 0``, ``0 1 0``; translations are
constant zero).  The neutral offset ``T_i0 = [Rz(lordosis_i) | (0, L_inf, 0)]``
lives in the parent ``PhysicalOffsetFrame`` on the inferior body; the child
frame is the identity on the superior body.  ``L_inf`` is the inferior body's
``segment_length_m``.  The base weld is ``[Rz(sacral_slope) | (0, L_pelvis,
0)]`` (pelvis to first chain body) and the terminal weld is
``[Rz(thorax_tilt) | (0, L_last, 0)]`` (last vertebra to ``thorax_proxy``), so
the expected thorax-proxy transform in the pelvis frame is

    T_base @ T_1 @ ... @ T_n @ T_terminal

(caudal factor first; angles never add).  Units: metres, kilograms, radians
in the model; degrees in :class:`FixtureSpec`.  Rigid-body inertia is a
uniform cylinder of radius ``_RADIUS_M`` along Y, which is positive definite.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only; never parses untrusted XML
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.engines.physics_engines.opensim.python.full_body_osim import (
    _build_free_joint,
    _build_ground,
    _build_transform_axis,
    _format_numbers,
    _make_physical_offset_frame,
)
from src.shared.python.spine.coupling import LinearCouplingLaw
from src.shared.python.spine.levels import SpinalLevel

__all__ = [
    "FixtureLevelNames",
    "FixtureSpec",
    "build_fixture_osim",
    "build_fixture_tree",
    "fixture_base_offset",
    "fixture_body_names",
    "fixture_level_names",
    "fixture_neutral_transforms",
    "fixture_terminal_offset",
    "write_fixture_osim",
]

#: Invented per-level lordosis (deg, rotation about Z in the parent frame).
#: Range 0-10 deg, larger at the caudal levels, as an arbitrary smooth profile.
_DEFAULT_LORDOSIS_DEG: Mapping[SpinalLevel, float] = MappingProxyType(
    {
        SpinalLevel.L5_S1: 10.0,
        SpinalLevel.L4_L5: 8.0,
        SpinalLevel.L3_L4: 6.0,
        SpinalLevel.L2_L3: 4.0,
        SpinalLevel.L1_L2: 2.0,
    }
)
_DEFAULT_PELVIS_MASS_KG = 10.0  # invented
_DEFAULT_PELVIS_LENGTH_M = 0.10  # invented
_DEFAULT_SACRUM_MASS_KG = 1.5  # invented
_DEFAULT_SACRUM_LENGTH_M = 0.05  # invented
_DEFAULT_VERTEBRA_MASS_KG = 1.0  # invented
_DEFAULT_VERTEBRA_LENGTH_M = 0.04  # invented
_DEFAULT_THORAX_MASS_KG = 20.0  # invented
_THORAX_LENGTH_M = 0.25  # inertia only; invented
_RADIUS_M = 0.05  # inertia only; invented
_COORD_RANGE_RAD = 1.0  # keeps lateral bending far from the +-90 deg gimbal lock
_MAX_ANGLE_DEG = 89.0
_SUFFIXES = ("flex_ext", "lat_bend", "axial_rot")
_AXES = ("0 0 1", "1 0 0", "0 1 0")  # ISB ZXY: rotation1, rotation2, rotation3
_PELVIS = "pelvis"
_THORAX = "thorax_proxy"


def _body_name(vertebra: str) -> str:
    return "sacrum" if vertebra == "S1" else vertebra


def _chain_bodies(levels: tuple[SpinalLevel, ...]) -> tuple[str, ...]:
    """Mass/length-bearing bodies: pelvis, first inferior, then each superior."""
    return (
        _PELVIS,
        _body_name(levels[0].inferior),
        *(_body_name(lv.superior) for lv in levels),
    )


def _default_mass(body: str) -> float:
    if body == _PELVIS:
        return _DEFAULT_PELVIS_MASS_KG
    return _DEFAULT_SACRUM_MASS_KG if body == "sacrum" else _DEFAULT_VERTEBRA_MASS_KG


def _default_length(body: str) -> float:
    if body == _PELVIS:
        return _DEFAULT_PELVIS_LENGTH_M
    if body == "sacrum":
        return _DEFAULT_SACRUM_LENGTH_M
    return _DEFAULT_VERTEBRA_LENGTH_M


def _validate_levels(levels: object) -> tuple[SpinalLevel, ...]:
    if not isinstance(levels, tuple):
        raise TypeError(f"levels must be a tuple, got {type(levels).__name__}")
    for item in levels:
        if not isinstance(item, SpinalLevel):
            raise TypeError(f"levels must hold SpinalLevel, got {type(item).__name__}")
    if not levels:
        raise ValueError("levels must not be empty")
    ordinals = [item.ordinal for item in levels]
    if any(b - a != 1 for a, b in zip(ordinals, ordinals[1:], strict=False)):
        raise ValueError(
            "levels must be contiguous and caudal-first, got "
            f"{[item.name for item in levels]}"
        )
    return levels


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return out


def _check_keys(mapping: Mapping, expected: set, name: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing:
        raise ValueError(f"{name} is missing keys {sorted(map(str, missing))}")
    if extra:
        raise ValueError(f"{name} has unknown keys {sorted(map(str, extra))}")


def _angle_deg(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or abs(out) > _MAX_ANGLE_DEG:
        raise ValueError(f"{name} must be finite with |angle| <= 89 deg, got {value!r}")
    return out


@dataclass(frozen=True)
class FixtureSpec:
    """Parameters of the synthetic spine fixture (all values invented).

    Attributes:
        levels: contiguous caudal-first run of levels (default L5_S1..L1_L2).
        neutral_lordosis_deg: rotation about Z of each level's neutral offset,
            invented, default 10, 8, 6, 4, 2 deg caudal to cranial.  ``None``
            selects the defaults (for the default levels only).
        segment_mass_kg / segment_length_m: keyed by body name (``pelvis``, the
            first inferior body, then each superior vertebra).  ``None``
            selects invented defaults (pelvis 10 kg / 0.10 m, sacrum 1.5 kg /
            0.05 m, vertebrae 1 kg / 0.04 m).
        thorax_proxy_mass_kg: lumped mass of the rigid thorax proxy.
        sacral_slope_deg / thorax_tilt_deg: rotation about Z of the base weld
            (pelvis to first chain body) and terminal weld; non-zero so the
            composition test exercises the fixed offsets.

    Preconditions (``TypeError``/``ValueError``): levels contiguous, caudal
    first and non-empty; mapping keys exactly cover the levels / bodies;
    masses and lengths finite and > 0; angles finite with |angle| <= 89 deg.

    Postconditions: mappings are replaced by read-only copies.
    """

    levels: tuple[SpinalLevel, ...] = (
        SpinalLevel.L5_S1,
        SpinalLevel.L4_L5,
        SpinalLevel.L3_L4,
        SpinalLevel.L2_L3,
        SpinalLevel.L1_L2,
    )
    neutral_lordosis_deg: Mapping[SpinalLevel, float] | None = None
    segment_mass_kg: Mapping[str, float] | None = None
    segment_length_m: Mapping[str, float] | None = None
    thorax_proxy_mass_kg: float = _DEFAULT_THORAX_MASS_KG
    sacral_slope_deg: float = 30.0
    thorax_tilt_deg: float = 5.0
    _bodies: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        levels = _validate_levels(self.levels)
        bodies = _chain_bodies(levels)
        lordosis = self.neutral_lordosis_deg
        if lordosis is None:
            lordosis = {lv: _DEFAULT_LORDOSIS_DEG.get(lv, 0.0) for lv in levels}
        masses = self.segment_mass_kg
        if masses is None:
            masses = {b: _default_mass(b) for b in bodies}
        lengths = self.segment_length_m
        if lengths is None:
            lengths = {b: _default_length(b) for b in bodies}
        _check_keys(lordosis, set(levels), "neutral_lordosis_deg")
        _check_keys(masses, set(bodies), "segment_mass_kg")
        _check_keys(lengths, set(bodies), "segment_length_m")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(
            self,
            "neutral_lordosis_deg",
            MappingProxyType(
                {lv: _angle_deg(lordosis[lv], f"lordosis[{lv.name}]") for lv in levels}
            ),
        )
        object.__setattr__(
            self,
            "segment_mass_kg",
            MappingProxyType(
                {b: _finite_positive(masses[b], f"mass[{b}]") for b in bodies}
            ),
        )
        object.__setattr__(
            self,
            "segment_length_m",
            MappingProxyType(
                {b: _finite_positive(lengths[b], f"length[{b}]") for b in bodies}
            ),
        )
        object.__setattr__(
            self,
            "thorax_proxy_mass_kg",
            _finite_positive(self.thorax_proxy_mass_kg, "thorax_proxy_mass_kg"),
        )
        object.__setattr__(
            self, "sacral_slope_deg", _angle_deg(self.sacral_slope_deg, "slope")
        )
        object.__setattr__(
            self, "thorax_tilt_deg", _angle_deg(self.thorax_tilt_deg, "tilt")
        )
        object.__setattr__(self, "_bodies", bodies)


@dataclass(frozen=True)
class FixtureLevelNames:
    """Joint, coordinate and body names of one fixture level."""

    level: SpinalLevel
    joint: str
    flex_ext: str
    lat_bend: str
    axial_rot: str
    inferior_body: str
    superior_body: str

    @property
    def coordinates(self) -> tuple[str, str, str]:
        """Coordinate names in ISB ZXY order (FE, LB, AR)."""
        return (self.flex_ext, self.lat_bend, self.axial_rot)


def fixture_body_names(spec: FixtureSpec) -> tuple[str, ...]:
    """Body names caudal to cranial, ending in ``thorax_proxy``."""
    return (*spec._bodies, _THORAX)


def fixture_level_names(spec: FixtureSpec) -> tuple[FixtureLevelNames, ...]:
    """Joint/coordinate/body names per level, caudal to cranial.

    Joint ``<LEVEL>_jnt``; coordinates ``<level>_flex_ext``, ``_lat_bend``,
    ``_axial_rot`` with the lowercase level name.
    """
    out = []
    for level in spec.levels:
        stem = level.name.lower()
        out.append(
            FixtureLevelNames(
                level=level,
                joint=f"{level.name}_jnt",
                flex_ext=f"{stem}_flex_ext",
                lat_bend=f"{stem}_lat_bend",
                axial_rot=f"{stem}_axial_rot",
                inferior_body=_body_name(level.inferior),
                superior_body=_body_name(level.superior),
            )
        )
    return tuple(out)


def _rigid_z(angle_deg: float, length_m: float) -> np.ndarray:
    """``[Rz(angle) | (0, length, 0)]`` as a (4, 4) rigid transform."""
    a = math.radians(angle_deg)
    out = np.eye(4)
    out[:2, :2] = ((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a)))
    out[1, 3] = length_m
    return out


def fixture_neutral_transforms(spec: FixtureSpec) -> dict[SpinalLevel, np.ndarray]:
    """Neutral offset ``T_i0`` (superior in inferior frame) per level.

    Postcondition: each value is a (4, 4) rigid transform; use it as the
    ``neutral`` of :class:`~src.shared.python.spine.segmental.LevelTransform`.
    """
    return {
        names.level: _rigid_z(
            spec.neutral_lordosis_deg[names.level],
            spec.segment_length_m[names.inferior_body],
        )
        for names in fixture_level_names(spec)
    }


def fixture_base_offset(spec: FixtureSpec) -> np.ndarray:
    """Fixed transform of the first chain body (e.g. sacrum) in the pelvis frame."""
    return _rigid_z(spec.sacral_slope_deg, spec.segment_length_m[_PELVIS])


def fixture_terminal_offset(spec: FixtureSpec) -> np.ndarray:
    """Fixed transform of ``thorax_proxy`` in the last vertebra frame."""
    return _rigid_z(spec.thorax_tilt_deg, spec.segment_length_m[spec._bodies[-1]])


def _euler_z(mat: np.ndarray) -> tuple[float, float, float]:
    return (0.0, 0.0, math.atan2(mat[1, 0], mat[0, 0]))


def _add_body(objects: ET.Element, name: str, mass: float, length: float) -> None:
    body = ET.SubElement(objects, "Body", attrib={"name": name})
    ET.SubElement(body, "mass").text = format(mass, ".17g")
    ET.SubElement(body, "mass_center").text = _format_numbers((0.0, length / 2, 0.0))
    ixx = mass * (3 * _RADIUS_M**2 + length**2) / 12.0
    iyy = mass * _RADIUS_M**2 / 2.0
    ET.SubElement(body, "inertia").text = _format_numbers((ixx, iyy, ixx, 0, 0, 0))
    ET.SubElement(body, "attached_geometry")


def _build_bodyset(model: ET.Element, spec: FixtureSpec) -> None:
    bodyset = ET.SubElement(model, "BodySet", attrib={"name": "bodyset"})
    objects = ET.SubElement(bodyset, "objects")
    for body in spec._bodies:
        _add_body(
            objects, body, spec.segment_mass_kg[body], spec.segment_length_m[body]
        )
    _add_body(objects, _THORAX, spec.thorax_proxy_mass_kg, _THORAX_LENGTH_M)


def _add_weld(
    objects: ET.Element, name: str, parent: str, child: str, offset: np.ndarray
) -> None:
    joint = ET.SubElement(objects, "WeldJoint", attrib={"name": name})
    pname, cname = f"{name}_parent_offset", f"{name}_child_offset"
    ET.SubElement(joint, "socket_parent_frame").text = pname
    ET.SubElement(joint, "socket_child_frame").text = cname
    frames = ET.SubElement(joint, "frames")
    frames.append(
        _make_physical_offset_frame(
            pname, f"/bodyset/{parent}", offset[:3, 3], _euler_z(offset)
        )
    )
    frames.append(_make_physical_offset_frame(cname, f"/bodyset/{child}", (0, 0, 0)))


def _add_level_joint(
    objects: ET.Element, names: FixtureLevelNames, neutral: np.ndarray
) -> None:
    joint = ET.SubElement(objects, "CustomJoint", attrib={"name": names.joint})
    pname, cname = f"{names.joint}_parent_offset", f"{names.joint}_child_offset"
    ET.SubElement(joint, "socket_parent_frame").text = pname
    ET.SubElement(joint, "socket_child_frame").text = cname
    coords = ET.SubElement(joint, "coordinates")
    for coord_name in names.coordinates:
        coord = ET.SubElement(coords, "Coordinate", attrib={"name": coord_name})
        ET.SubElement(coord, "default_value").text = "0"
        ET.SubElement(coord, "default_speed_value").text = "0"
        ET.SubElement(coord, "range").text = f"{-_COORD_RANGE_RAD} {_COORD_RANGE_RAD}"
        ET.SubElement(coord, "clamped").text = "false"
        ET.SubElement(coord, "locked").text = "false"
    frames = ET.SubElement(joint, "frames")
    frames.append(
        _make_physical_offset_frame(
            pname,
            f"/bodyset/{names.inferior_body}",
            neutral[:3, 3],
            _euler_z(neutral),
        )
    )
    frames.append(
        _make_physical_offset_frame(cname, f"/bodyset/{names.superior_body}", (0, 0, 0))
    )
    transform = ET.SubElement(joint, "SpatialTransform")
    for idx, (axis, coord_name) in enumerate(
        zip(_AXES, names.coordinates, strict=True), start=1
    ):
        _build_transform_axis(transform, "rotation", idx, axis, coord_name)
    for idx, axis in enumerate(("1 0 0", "0 1 0", "0 0 1"), start=1):
        _build_transform_axis(transform, "translation", idx, axis, None)


def _build_jointset(model: ET.Element, spec: FixtureSpec) -> None:
    jointset = ET.SubElement(model, "JointSet", attrib={"name": "jointset"})
    objects = ET.SubElement(jointset, "objects")
    prims = [
        {"primitive": p, "coordinate": f"pelvis_{c}"}
        for p, c in (
            ("Rx", "rx"),
            ("Ry", "ry"),
            ("Rz", "rz"),
            ("Px", "tx"),
            ("Py", "ty"),
            ("Pz", "tz"),
        )
    ]
    _build_free_joint(
        objects,
        {
            "primitives": prims,
            "parent": "world",
            "child": _PELVIS,
            "parent_to_base": np.eye(4),
            "child_to_follower": np.eye(4),
        },
        "ground_pelvis",
    )
    _add_weld(
        objects,
        f"{_PELVIS}_{spec._bodies[1]}_weld",
        _PELVIS,
        spec._bodies[1],
        fixture_base_offset(spec),
    )
    neutral = fixture_neutral_transforms(spec)
    for names in fixture_level_names(spec):
        _add_level_joint(objects, names, neutral[names.level])
    last = spec._bodies[-1]
    _add_weld(
        objects, f"{last}_{_THORAX}_weld", last, _THORAX, fixture_terminal_offset(spec)
    )


def build_fixture_tree(spec: FixtureSpec) -> ET.Element:
    """Return the uncoupled, muscle-free ``OpenSimDocument`` element for ``spec``.

    Shared by :func:`build_fixture_osim` and ``fixture_coupling`` (SPN-7b),
    which augments the tree in place.  Raises ``TypeError`` for a non-spec.
    """
    if not isinstance(spec, FixtureSpec):
        raise TypeError(f"spec must be a FixtureSpec, got {type(spec).__name__}")
    root = ET.Element("OpenSimDocument", attrib={"Version": "40000"})
    model = ET.SubElement(root, "Model", attrib={"name": "segmental_spine_fixture"})
    _build_ground(model)
    ET.SubElement(model, "gravity").text = "0 -9.80665 0"
    ET.SubElement(model, "length_units").text = "m"
    ET.SubElement(model, "force_units").text = "N"
    _build_bodyset(model, spec)
    _build_jointset(model, spec)
    return root


def build_fixture_osim(
    spec: FixtureSpec,
    coupling: LinearCouplingLaw | None = None,
    muscles: bool = False,
) -> str:
    """Return the ``.osim`` XML text (OpenSim 4 format) for ``spec``.

    Args:
        spec: the fixture parameters.
        coupling: optional :class:`LinearCouplingLaw`; adds a massless driver
            body with the law's independent coordinates and one
            ``CoordinateCouplerConstraint`` per dependent level coordinate
            (SPN-7b, see ``fixture_coupling``).
        muscles: if true, add 8 bilateral lumped trunk muscles (SPN-7b).

    Preconditions: ``spec`` is a :class:`FixtureSpec`; ``muscles`` is a bool.
    Postconditions: the text is deterministic for equal arguments, has one
    ``CustomJoint`` per level in caudal-to-cranial order with ISB ZXY rotation
    axes, and positive mass and positive-definite inertia on every fixture
    body (the optional driver body is the only massless one).  With neither
    option there are no constraints, driver or muscles.

    Raises:
        TypeError: if ``spec`` is not a :class:`FixtureSpec`, ``muscles`` is
            not a bool or ``coupling`` is not a :class:`LinearCouplingLaw`.
        ValueError: if the law does not fit the fixture.
    """
    root = build_fixture_tree(spec)
    if coupling is not None or muscles is not False:
        from src.engines.physics_engines.opensim.python.spine.fixture_coupling import (
            augment_fixture_tree,
        )

        augment_fixture_tree(root, spec, coupling, muscles)
    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def write_fixture_osim(
    spec: FixtureSpec,
    path: str | Path,
    coupling: LinearCouplingLaw | None = None,
    muscles: bool = False,
) -> Path:
    """Write the fixture to ``path`` (must end in ``.osim``) and return it.

    ``coupling`` and ``muscles`` are forwarded to :func:`build_fixture_osim`.
    Parent directories are created.  Raises ``ValueError`` for another suffix.
    """
    out = Path(path)
    if out.suffix != ".osim":
        raise ValueError(f"fixture path must end in .osim, got {out.name!r}")
    text = build_fixture_osim(spec, coupling, muscles)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
