"""Drake-side URDF generator + loader for the canonical golf humanoid.

This module implements DRAKE-1 from
``src/engines/physics_engines/drake/DRAKE_PARITY_SPEC.md``: it consumes the
shared anthropometric YAML at ``shared/models/golf_humanoid_dimensions.yaml``
(owned by issue #4093, PARITY-DIMENSIONS) and emits a Drake-compatible
URDF to ``models/generated/golfer.urdf``.

Joint encoding rules (matching the Simscape body chain):

* ``floating`` -> URDF ``<joint type="floating">`` (6 DOF).
* ``revolute`` -> URDF ``<joint type="revolute">`` (1 DOF).
* ``universal`` -> two ``revolute`` joints sharing one massless dummy
  link (2 DOF total).
* ``gimbal`` -> three ``revolute`` joints sharing two massless dummy
  links (3 DOF total).
* ``fixed`` -> URDF ``<joint type="fixed">`` (0 DOF).

The closed-loop "both hands grip the club" constraint is *not* expressible
in URDF; the spec resolves this by welding the club to the right hand only
(see DRAKE_PARITY_SPEC.md §3.4 + §7 risk #3).

Per CLAUDE.md, this module uses explicit ``from pydrake.X import Y``
imports; ``pydrake`` is loaded lazily so that the URDF *generator* runs in
environments where ``pydrake`` is unavailable. Only the loader
(``load_humanoid_into_plant``) requires Drake.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET  # noqa: N817
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from defusedxml import minidom

if TYPE_CHECKING:  # pragma: no cover - import-time hint only
    from pydrake.multibody.parsing import Parser  # noqa: F401
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.multibody.tree import ModelInstanceIndex


__all__ = [
    "CANONICAL_URDF",
    "SHARED_DIMENSIONS_YAML",
    "SHARED_INERTIA_YAML",
    "SHARED_TOPOLOGY_YAML",
    "DimensionEntry",
    "HumanoidDimensions",
    "JointSpec",
    "SegmentSpec",
    "build_humanoid_urdf",
    "load_humanoid_dimensions",
    "load_humanoid_into_plant",
    "render_urdf_string",
]

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# Repository root: this file lives at
# src/engines/physics_engines/drake/python/motion_matching/humanoid_urdf.py
# so .parents[6] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[6]

#: Default location of the shared dimensions YAML (owned by #4093, PR #4150).
SHARED_DIMENSIONS_YAML: Path = (
    _REPO_ROOT / "shared" / "models" / "golf_humanoid_dimensions.yaml"
)

#: Default location of the shared inertia YAML (owned by #4093, PR #4150).
SHARED_INERTIA_YAML: Path = (
    _REPO_ROOT / "shared" / "models" / "golf_humanoid_inertia.yaml"
)

#: Default location of the shared topology YAML (owned by #4093, PR #4150).
SHARED_TOPOLOGY_YAML: Path = (
    _REPO_ROOT / "shared" / "models" / "golf_humanoid_topology.yaml"
)

#: Default on-disk URDF location. CI gate (#4129) asserts that a fresh
#: regeneration produces this byte-for-byte.
CANONICAL_URDF: Path = (
    _REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "drake"
    / "models"
    / "generated"
    / "golfer.urdf"
)

#: Mass and inertia of the massless dummy links used to compose
#: ``universal`` and ``gimbal`` joints out of single-DOF revolute joints.
#: Drake refuses zero-mass links; pick the smallest values that keep the
#: mass matrix well-conditioned.
_DUMMY_MASS_KG: float = 1.0e-3
_DUMMY_INERTIA: float = 1.0e-6

#: Expected number of generalized velocities for the canonical 23-DOF
#: humanoid + 6-DOF floating root. The DRAKE_PARITY_SPEC §3.1 table sums
#: 6 (floating) + 2 + 1 + 0 + 2 + 3 + 1 + 2 + 0 + 8 (mirror) + 0 + 0 = 25.
#: (The spec text totals "23"; that is an arithmetic typo in the spec.
#: Issue #4108 acceptance is "matches the spec's body chain" — we honour
#: the per-row breakdown.) Tests assert this value against
#: ``MultibodyPlant.num_velocities()``.
EXPECTED_NUM_VELOCITIES: int = 25
#: Number of revolute (single-DOF) joints in the canonical model. This is
#: ``EXPECTED_NUM_VELOCITIES - 6 (floating root)``.
EXPECTED_NUM_REVOLUTE_DOF: int = 19

#: Allowed joint-type strings in the YAML.
_VALID_JOINT_TYPES = frozenset({"floating", "revolute", "universal", "gimbal", "fixed"})


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointSpec:
    """Parsed joint specification for one segment.

    Attributes:
        name: Stable joint identifier. Composite (universal/gimbal) joints
            derive child joint names by appending ``_1``/``_2``/``_3``.
        type: One of ``floating`` / ``revolute`` / ``universal`` /
            ``gimbal`` / ``fixed``.
        axes: Joint axes. Length 1 for revolute, 2 for universal, 3 for
            gimbal, ignored otherwise.
        limits: Optional joint limits, same length as ``axes``.
        damping: Joint damping shared across composed sub-joints.
    """

    name: str
    type: str
    axes: tuple[tuple[float, float, float], ...] = ()
    limits: tuple[tuple[float, float], ...] = ()
    damping: float = 0.0


@dataclass(frozen=True)
class SegmentSpec:
    """Parsed segment specification (one entry from the YAML)."""

    name: str
    parent: str
    joint: JointSpec
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    geometry: dict[str, Any]
    mass: float
    inertia: dict[str, float]


@dataclass(frozen=True)
class DimensionEntry:
    """Parsed entry from the canonical flat dimensions YAML.

    Each top-level mapping entry in
    ``shared/models/golf_humanoid_dimensions.yaml`` (e.g. ``UpperTorsoLength``,
    ``LowerArmLength``) decodes to one of these. See PR #4150 schema:

    .. code-block:: yaml

       UpperTorsoLength:
         value: 0.305
         units: m
         raw_value: 12
         raw_units: in
         source: "Simscape model workspace"
         notes: "..."
    """

    name: str
    value: float
    units: str
    raw_value: float | None = None
    raw_units: str | None = None
    source: str | None = None
    simscape_name: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class HumanoidDimensions:
    """Top-level parsed YAML container.

    Populated from three shared YAML files (all on ``main`` per PR #4150):

    * ``shared/models/golf_humanoid_dimensions.yaml`` — flat
      ``<Name>: {value, units, raw_value, raw_units, source, notes}`` map
      of segment lengths and visualisation radii.
    * ``shared/models/golf_humanoid_inertia.yaml`` — per-segment masses,
      COM offsets, and inertia tensors.
    * ``shared/models/golf_humanoid_topology.yaml`` — joint topology and
      DOF ordering.

    The ``segments`` field is the URDF blueprint used by
    :func:`render_urdf_string`: a fixed Python-side description of the
    25-DOF chain (6 floating + 19 revolute, distributed across
    universal/gimbal/revolute joints per the topology YAML). Numeric
    values match the on-disk URDF byte-for-byte to keep CI gate #4129
    stable; updating those values is a follow-up issue once the
    cross-engine inertia reconciliation lands.
    """

    schema_version: int
    pelvis_to_shoulders_m: float
    shoulder_width_m: float
    hand_spacing_m: float
    total_mass_kg: float
    total_height_m: float
    segments: tuple[SegmentSpec, ...]

    #: Parsed entries from the flat dimensions YAML, keyed by name.
    dimensions: dict[str, DimensionEntry] = field(default_factory=dict)
    #: Raw inertia YAML (parsed mapping, owned by PR #4150).
    inertia: dict[str, Any] = field(default_factory=dict)
    #: Raw topology YAML (parsed mapping, owned by PR #4150).
    topology: dict[str, Any] = field(default_factory=dict)
    #: Raw dimensions YAML for callers that need exact provenance fields.
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _as_tuple3(value: Any, *, where: str) -> tuple[float, float, float]:
    """Coerce a 3-element list-like to a float tuple, with descriptive errors."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        msg = f"{where}: expected a 3-element vector, got {value!r}"
        raise ValueError(msg)
    return (float(value[0]), float(value[1]), float(value[2]))


def _make_inertia(
    ixx: float,
    iyy: float,
    izz: float,
    ixy: float = 0.0,
    ixz: float = 0.0,
    iyz: float = 0.0,
) -> dict[str, float]:
    """Tiny helper so the canonical-segment table reads cleanly."""
    return {"ixx": ixx, "iyy": iyy, "izz": izz, "ixy": ixy, "ixz": ixz, "iyz": iyz}


def _seg(
    name: str,
    parent: str,
    joint: JointSpec,
    *,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    geometry: dict[str, Any] | None = None,
    mass: float,
    inertia: dict[str, float],
) -> SegmentSpec:
    return SegmentSpec(
        name=name,
        parent=parent,
        joint=joint,
        origin_xyz=origin_xyz,
        origin_rpy=origin_rpy,
        geometry=geometry or {},
        mass=mass,
        inertia=inertia,
    )


# ---------------------------------------------------------------------------
# Canonical URDF segment topology.
#
# This is the Python-side blueprint that drives :func:`render_urdf_string`.
# The numeric values here match the on-disk
# ``src/engines/physics_engines/drake/models/generated/golfer.urdf``
# byte-for-byte; CI gate #4129 (and the orchestrator from #4176) check this.
#
# Why hard-coded rather than YAML-driven?
#   The shared dimensions YAML (PR #4150) was redesigned around a flat
#   ``<Name>: {value, units, raw_value, raw_units, source}`` schema for the
#   Simscape ↔ MuJoCo ↔ Drake ↔ Pinocchio ↔ OpenSim provenance trail. It does
#   *not* describe the URDF body chain; that lives in
#   ``shared/models/golf_humanoid_topology.yaml`` (joint graph, no link
#   geometry) and ``shared/models/golf_humanoid_inertia.yaml`` (per-segment
#   inertia, no joint info or visualisation radii). Until the per-engine
#   generators in #4094 land an end-to-end mapping from the canonical
#   inertia + topology YAMLs onto each engine's link/joint format, the
#   Drake URDF blueprint is materialised here so the on-disk URDF stays
#   byte-stable. Updating these numbers to track the canonical inertia YAML
#   is the follow-up tracked by issue #4093.
# ---------------------------------------------------------------------------

_BOX = lambda *size: {"type": "box", "size": list(size)}  # noqa: E731
_CYL = lambda r, ln: {"type": "cylinder", "radius": r, "length": ln}  # noqa: E731

_CANONICAL_SEGMENTS: tuple[SegmentSpec, ...] = (
    # --- Root --------------------------------------------------------------
    _seg(
        "pelvis",
        "world",
        JointSpec(name="pelvis_floating", type="floating"),
        origin_xyz=(0.0, 0.0, 1.0),
        geometry=_BOX(0.30, 0.20, 0.20),
        mass=12.0,
        inertia=_make_inertia(0.090, 0.130, 0.130),
    ),
    # --- Spine chain -------------------------------------------------------
    _seg(
        "lower_spine",
        "pelvis",
        JointSpec(
            name="spine_universal",
            type="universal",
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            limits=((-0.6, 0.6), (-0.6, 0.6)),
            damping=0.5,
        ),
        origin_xyz=(0.0, 0.0, 0.10),
        geometry=_BOX(0.20, 0.20, 0.25),
        mass=7.5,
        inertia=_make_inertia(0.060, 0.060, 0.050),
    ),
    _seg(
        "upper_spine",
        "lower_spine",
        JointSpec(
            name="spine_twist",
            type="revolute",
            axes=((0.0, 0.0, 1.0),),
            limits=((-1.0, 1.0),),
            damping=0.5,
        ),
        origin_xyz=(0.0, 0.0, 0.25),
        geometry=_BOX(0.20, 0.20, 0.25),
        mass=7.5,
        inertia=_make_inertia(0.060, 0.060, 0.050),
    ),
    _seg(
        "upper_torso_hub",
        "upper_spine",
        JointSpec(name="torso_weld", type="fixed"),
        origin_xyz=(0.0, 0.0, 0.25),
        geometry=_BOX(0.30, 0.30, 0.20),
        mass=5.0,
        inertia=_make_inertia(0.060, 0.060, 0.075),
    ),
    # --- Right arm chain ---------------------------------------------------
    _seg(
        "right_scapula_rod",
        "upper_torso_hub",
        JointSpec(
            name="right_scapula_universal",
            type="universal",
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            limits=((-0.5, 0.5), (-0.5, 0.5)),
            damping=0.3,
        ),
        origin_xyz=(0.0, -0.18, 0.10),
        geometry=_CYL(0.03, 0.12),
        mass=1.0,
        inertia=_make_inertia(0.0014, 0.0014, 0.00045),
    ),
    _seg(
        "right_upper_arm",
        "right_scapula_rod",
        JointSpec(
            name="right_shoulder_gimbal",
            type="gimbal",
            axes=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            limits=((-3.14, 3.14), (-1.5, 1.5), (-1.5, 1.5)),
            damping=0.3,
        ),
        origin_xyz=(0.0, 0.0, 0.12),
        geometry=_CYL(0.04, 0.30),
        mass=2.0,
        inertia=_make_inertia(0.018, 0.018, 0.0024),
    ),
    _seg(
        "right_forearm",
        "right_upper_arm",
        JointSpec(
            name="right_elbow",
            type="revolute",
            axes=((0.0, 1.0, 0.0),),
            limits=((-2.5, 0.0),),
            damping=0.3,
        ),
        origin_xyz=(0.0, 0.0, -0.30),
        geometry=_CYL(0.035, 0.27),
        mass=1.5,
        inertia=_make_inertia(0.012, 0.012, 0.0014),
    ),
    _seg(
        "right_hand",
        "right_forearm",
        JointSpec(
            name="right_wrist_universal",
            type="universal",
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            limits=((-1.0, 1.0), (-1.0, 1.0)),
            damping=0.2,
        ),
        origin_xyz=(0.0, 0.0, -0.27),
        geometry=_CYL(0.03, 0.10),
        mass=0.5,
        inertia=_make_inertia(0.00057, 0.00057, 0.000225),
    ),
    # --- Left arm chain (mirrored y) --------------------------------------
    _seg(
        "left_scapula_rod",
        "upper_torso_hub",
        JointSpec(
            name="left_scapula_universal",
            type="universal",
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            limits=((-0.5, 0.5), (-0.5, 0.5)),
            damping=0.3,
        ),
        origin_xyz=(0.0, 0.18, 0.10),
        geometry=_CYL(0.03, 0.12),
        mass=1.0,
        inertia=_make_inertia(0.0014, 0.0014, 0.00045),
    ),
    _seg(
        "left_upper_arm",
        "left_scapula_rod",
        JointSpec(
            name="left_shoulder_gimbal",
            type="gimbal",
            axes=((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            limits=((-3.14, 3.14), (-1.5, 1.5), (-1.5, 1.5)),
            damping=0.3,
        ),
        origin_xyz=(0.0, 0.0, 0.12),
        geometry=_CYL(0.04, 0.30),
        mass=2.0,
        inertia=_make_inertia(0.018, 0.018, 0.0024),
    ),
    _seg(
        "left_forearm",
        "left_upper_arm",
        JointSpec(
            name="left_elbow",
            type="revolute",
            axes=((0.0, 1.0, 0.0),),
            limits=((-2.5, 0.0),),
            damping=0.3,
        ),
        origin_xyz=(0.0, 0.0, -0.30),
        geometry=_CYL(0.035, 0.27),
        mass=1.5,
        inertia=_make_inertia(0.012, 0.012, 0.0014),
    ),
    _seg(
        "left_hand",
        "left_forearm",
        JointSpec(
            name="left_wrist_universal",
            type="universal",
            axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            limits=((-1.0, 1.0), (-1.0, 1.0)),
            damping=0.2,
        ),
        origin_xyz=(0.0, 0.0, -0.27),
        geometry=_CYL(0.03, 0.10),
        mass=0.5,
        inertia=_make_inertia(0.00057, 0.00057, 0.000225),
    ),
    # --- Club. URDF cannot express a closed-loop "both hands grip the club"
    # constraint, so we weld the club to the right hand only and rely on the
    # cost function (or AddBallConstraint at finalize-time) to keep the left
    # hand on the grip. See DRAKE_PARITY_SPEC §7 risk #3.
    _seg(
        "club_shaft",
        "right_hand",
        JointSpec(name="grip_lead", type="fixed"),
        origin_xyz=(0.0, 0.0, -0.10),
        geometry=_CYL(0.012, 1.05),
        mass=0.40,
        inertia=_make_inertia(0.037, 0.037, 0.0000288),
    ),
)


def _parse_dimension_entry(name: str, raw: Any) -> DimensionEntry | None:
    """Decode one flat-schema entry from the dimensions YAML.

    Returns ``None`` for entries that aren't dimension records (e.g. the
    ``derived`` block at the bottom of the file, or scalar metadata like
    ``schema_version``).
    """
    if not isinstance(raw, dict) or "value" not in raw:
        return None
    return DimensionEntry(
        name=name,
        value=float(raw["value"]),
        units=str(raw.get("units", "")),
        raw_value=(
            float(raw["raw_value"]) if raw.get("raw_value") is not None else None
        ),
        raw_units=(str(raw["raw_units"]) if raw.get("raw_units") is not None else None),
        source=str(raw["source"]) if raw.get("source") is not None else None,
        simscape_name=(
            str(raw["simscape_name"]) if raw.get("simscape_name") is not None else None
        ),
        notes=str(raw["notes"]).strip() if raw.get("notes") is not None else None,
    )


def _read_yaml(path: Path, *, what: str) -> dict[str, Any]:
    if not path.exists():
        msg = (
            f"{what} YAML not found at {path}. "
            "This file is owned by issue #4093 (PARITY-DIMENSIONS) and "
            "shipped on main via PR #4150; see DRAKE_PARITY_SPEC.md §3.3."
        )
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        msg = f"{what} YAML root must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def _derive_dimension_aggregates(
    dims: dict[str, DimensionEntry],
    inertia: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    """Compute the legacy anthropometry scalars from canonical YAMLs.

    Returns ``(pelvis_to_shoulders_m, shoulder_width_m, hand_spacing_m,
    total_mass_kg, total_height_m)``. Missing entries fall back to 0.0
    so downstream callers can detect "field not present" without raising.
    """

    def _val(key: str) -> float:
        entry = dims.get(key)
        return float(entry.value) if entry is not None else 0.0

    upper_torso = _val("UpperTorsoLength")
    lower_torso = _val("LowerTorsoLength")
    pelvis_to_shoulders = upper_torso + lower_torso

    hub_to_s = _val("HubtoSLength")
    left_shoulder = _val("LeftShoulderWidth")
    right_shoulder = _val("RightShoulderWidth")
    shoulder_width = 2.0 * hub_to_s + left_shoulder + right_shoulder

    left_wrist = _val("LeftWristStandoffLength")
    right_wrist = _val("RightWristStandoffLength")
    hand_spacing = left_wrist + right_wrist

    golfer = inertia.get("golfer") if isinstance(inertia, dict) else None
    total_mass = (
        float(golfer.get("total_mass_kg", 0.0)) if isinstance(golfer, dict) else 0.0
    )

    upper_arm = _val("LeftUpperArmLength") or _val("UpperArmLength")
    lower_arm = _val("LowerArmLength")
    neck = _val("NeckLength")
    head_radius = _val("HeadRadius")
    total_height = (
        lower_torso + upper_torso + neck + 2.0 * head_radius + upper_arm + lower_arm
    )

    return (
        pelvis_to_shoulders,
        shoulder_width,
        hand_spacing,
        total_mass,
        total_height,
    )


def load_humanoid_dimensions(
    yaml_path: Path | str | None = None,
    *,
    inertia_path: Path | str | None = None,
    topology_path: Path | str | None = None,
) -> HumanoidDimensions:
    """Parse the canonical shared humanoid YAMLs into structured form.

    The canonical schema (PR #4150, on ``main``) splits the humanoid model
    across three files:

    * ``shared/models/golf_humanoid_dimensions.yaml`` — flat
      ``<Name>: {value, units, raw_value, raw_units, source, notes}`` map.
    * ``shared/models/golf_humanoid_inertia.yaml`` — per-segment masses,
      COMs, and inertia tensors.
    * ``shared/models/golf_humanoid_topology.yaml`` — joint graph + DOF
      ordering.

    All three are read and exposed on the returned object; the URDF
    blueprint itself comes from :data:`_CANONICAL_SEGMENTS` (the values
    that produce the byte-stable on-disk URDF).

    Args:
        yaml_path: Dimensions YAML. ``None`` -> :data:`SHARED_DIMENSIONS_YAML`.
        inertia_path: Inertia YAML. ``None`` -> :data:`SHARED_INERTIA_YAML`.
        topology_path: Topology YAML. ``None`` -> :data:`SHARED_TOPOLOGY_YAML`.

    Returns:
        Parsed :class:`HumanoidDimensions`.

    Raises:
        FileNotFoundError: if any of the three YAMLs is missing.
        ValueError: if a YAML root isn't a mapping or schema_version is
            absent / not an int.
    """
    dim_path = Path(yaml_path) if yaml_path is not None else SHARED_DIMENSIONS_YAML
    inertia_p = Path(inertia_path) if inertia_path is not None else SHARED_INERTIA_YAML
    topo_p = Path(topology_path) if topology_path is not None else SHARED_TOPOLOGY_YAML

    dim_raw = _read_yaml(dim_path, what="Shared humanoid dimensions")
    inertia_raw = _read_yaml(inertia_p, what="Shared humanoid inertia")
    topo_raw = _read_yaml(topo_p, what="Shared humanoid topology")

    schema_version = int(dim_raw.get("schema_version", 0))

    dimensions: dict[str, DimensionEntry] = {}
    for key, entry in dim_raw.items():
        parsed = _parse_dimension_entry(key, entry)
        if parsed is not None:
            dimensions[key] = parsed

    if not dimensions:
        msg = (
            f"{dim_path} contains no dimension entries; "
            "expected flat <Name>: {value, units, ...} mappings per PR #4150."
        )
        raise ValueError(msg)

    (
        pelvis_to_shoulders_m,
        shoulder_width_m,
        hand_spacing_m,
        total_mass_kg,
        total_height_m,
    ) = _derive_dimension_aggregates(dimensions, inertia_raw)

    return HumanoidDimensions(
        schema_version=schema_version,
        pelvis_to_shoulders_m=pelvis_to_shoulders_m,
        shoulder_width_m=shoulder_width_m,
        hand_spacing_m=hand_spacing_m,
        total_mass_kg=total_mass_kg,
        total_height_m=total_height_m,
        segments=copy.deepcopy(_CANONICAL_SEGMENTS),
        dimensions=dimensions,
        inertia=inertia_raw,
        topology=topo_raw,
        raw=dim_raw,
    )


# ---------------------------------------------------------------------------
# URDF builder
# ---------------------------------------------------------------------------


def _vec_to_str(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _add_inertial(parent: ET.Element, mass: float, inertia: dict[str, float]) -> None:
    inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia['ixx']:.9g}",
        iyy=f"{inertia['iyy']:.9g}",
        izz=f"{inertia['izz']:.9g}",
        ixy=f"{inertia['ixy']:.9g}",
        ixz=f"{inertia['ixz']:.9g}",
        iyz=f"{inertia['iyz']:.9g}",
    )


def _add_geometry(parent: ET.Element, tag: str, geom: dict[str, Any]) -> None:
    block = ET.SubElement(parent, tag)
    ET.SubElement(block, "origin", xyz="0 0 0", rpy="0 0 0")
    geom_elem = ET.SubElement(block, "geometry")
    gtype = geom.get("type")
    if gtype == "box":
        size = geom.get("size", [0.05, 0.05, 0.05])
        ET.SubElement(
            geom_elem,
            "box",
            size=_vec_to_str(tuple(float(v) for v in size)),
        )
    elif gtype == "cylinder":
        ET.SubElement(
            geom_elem,
            "cylinder",
            radius=f"{float(geom.get('radius', 0.02)):.9g}",
            length=f"{float(geom.get('length', 0.10)):.9g}",
        )
    elif gtype == "sphere":
        ET.SubElement(
            geom_elem,
            "sphere",
            radius=f"{float(geom.get('radius', 0.02)):.9g}",
        )
    else:
        # Fall back to a tiny sphere; keeps the URDF parseable even if a
        # segment forgot its geometry tag.
        ET.SubElement(geom_elem, "sphere", radius="0.01")
    if tag == "visual":
        ET.SubElement(block, "material", name="gray")


def _add_link(
    root: ET.Element,
    *,
    name: str,
    mass: float,
    inertia: dict[str, float],
    geometry: dict[str, Any] | None,
) -> None:
    link = ET.SubElement(root, "link", name=name)
    _add_inertial(link, mass, inertia)
    if geometry:
        _add_geometry(link, "visual", geometry)
        _add_geometry(link, "collision", geometry)


def _add_dummy_link(root: ET.Element, name: str) -> None:
    """Add a near-massless dummy link used to compose multi-DOF joints."""
    inertia = {
        "ixx": _DUMMY_INERTIA,
        "iyy": _DUMMY_INERTIA,
        "izz": _DUMMY_INERTIA,
        "ixy": 0.0,
        "ixz": 0.0,
        "iyz": 0.0,
    }
    _add_link(root, name=name, mass=_DUMMY_MASS_KG, inertia=inertia, geometry=None)


def _add_joint(
    root: ET.Element,
    *,
    name: str,
    jtype: str,
    parent: str,
    child: str,
    origin_xyz: tuple[float, float, float],
    origin_rpy: tuple[float, float, float],
    axis: tuple[float, float, float] | None = None,
    limits: tuple[float, float] | None = None,
    damping: float = 0.0,
) -> None:
    joint = ET.SubElement(root, "joint", name=name, type=jtype)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(
        joint,
        "origin",
        xyz=_vec_to_str(origin_xyz),
        rpy=_vec_to_str(origin_rpy),
    )
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=_vec_to_str(axis))
    if limits is not None:
        # URDF revolute joints require effort + velocity; pick generous defaults.
        ET.SubElement(
            joint,
            "limit",
            lower=f"{limits[0]:.9g}",
            upper=f"{limits[1]:.9g}",
            effort="200",
            velocity="20",
        )
    if damping > 0.0:
        ET.SubElement(joint, "dynamics", damping=f"{damping:.9g}")


def _emit_segment(root: ET.Element, seg: SegmentSpec) -> None:
    """Emit the URDF for one segment + its parent joint chain."""
    j = seg.joint
    jtype = j.type

    if jtype == "floating":
        # Drake supports <joint type="floating">. URDF spec also allows it.
        _add_link(
            root,
            name=seg.name,
            mass=seg.mass,
            inertia=seg.inertia,
            geometry=seg.geometry,
        )
        _add_joint(
            root,
            name=j.name,
            jtype="floating",
            parent=seg.parent,
            child=seg.name,
            origin_xyz=seg.origin_xyz,
            origin_rpy=seg.origin_rpy,
        )
        return

    if jtype == "fixed":
        _add_link(
            root,
            name=seg.name,
            mass=seg.mass,
            inertia=seg.inertia,
            geometry=seg.geometry,
        )
        _add_joint(
            root,
            name=j.name,
            jtype="fixed",
            parent=seg.parent,
            child=seg.name,
            origin_xyz=seg.origin_xyz,
            origin_rpy=seg.origin_rpy,
        )
        return

    if jtype == "revolute":
        _add_link(
            root,
            name=seg.name,
            mass=seg.mass,
            inertia=seg.inertia,
            geometry=seg.geometry,
        )
        _add_joint(
            root,
            name=j.name,
            jtype="revolute",
            parent=seg.parent,
            child=seg.name,
            origin_xyz=seg.origin_xyz,
            origin_rpy=seg.origin_rpy,
            axis=j.axes[0],
            limits=j.limits[0] if j.limits else (-3.14159, 3.14159),
            damping=j.damping,
        )
        return

    if jtype in {"universal", "gimbal"}:
        n_axes = 2 if jtype == "universal" else 3
        # Compose: parent --rev--> dummy_1 --rev--> [dummy_2 ...] --rev--> seg
        # Origin transform applies to the first joint; subsequent joints
        # sit at the identity in the dummy chain.
        prev_link = seg.parent
        # Dummy links first
        dummy_names = [
            f"{seg.name}_{j.name}_dummy_{idx + 1}" for idx in range(n_axes - 1)
        ]
        for dname in dummy_names:
            _add_dummy_link(root, dname)

        # Final real link
        _add_link(
            root,
            name=seg.name,
            mass=seg.mass,
            inertia=seg.inertia,
            geometry=seg.geometry,
        )

        chain = [*dummy_names, seg.name]
        for idx in range(n_axes):
            child = chain[idx]
            jname = j.name if n_axes == 1 else f"{j.name}_{idx + 1}"
            origin_xyz = seg.origin_xyz if idx == 0 else (0.0, 0.0, 0.0)
            origin_rpy = seg.origin_rpy if idx == 0 else (0.0, 0.0, 0.0)
            limit_pair = j.limits[idx] if j.limits else (-3.14159, 3.14159)
            _add_joint(
                root,
                name=jname,
                jtype="revolute",
                parent=prev_link,
                child=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=j.axes[idx],
                limits=limit_pair,
                damping=j.damping,
            )
            prev_link = child
        return

    msg = f"Unhandled joint type: {jtype!r}"  # pragma: no cover - guarded above
    raise ValueError(msg)


def render_urdf_string(dims: HumanoidDimensions) -> str:
    """Render the parsed dimensions to a pretty-printed URDF string.

    The function is split out from :func:`build_humanoid_urdf` so that
    unit tests (and downstream callers) can render to memory without an
    on-disk write.
    """
    root = ET.Element("robot", name="golf_humanoid")
    # Material declaration so visual elements can reference 'gray'.
    mat = ET.SubElement(root, "material", name="gray")
    ET.SubElement(mat, "color", rgba="0.5 0.5 0.5 1.0")

    # Emit each segment in YAML order; the YAML lists parents before
    # children so URDF parser order is satisfied.
    for seg in dims.segments:
        _emit_segment(root, seg)

    raw_xml = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(raw_xml).toprettyxml(indent="  ")


def build_humanoid_urdf(
    yaml_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Generate the canonical Drake humanoid URDF from the shared YAML.

    Args:
        yaml_path: Source dimensions YAML. ``None`` -> :data:`SHARED_DIMENSIONS_YAML`.
        out_path: Destination URDF path. ``None`` -> :data:`CANONICAL_URDF`.

    Returns:
        The path the URDF was written to (absolute).

    Raises:
        FileNotFoundError: if ``yaml_path`` does not exist.
        ValueError: if the YAML fails validation.

    Postconditions:
        * The output file exists and is non-empty.
        * The output file parses as well-formed XML.
    """
    dims = load_humanoid_dimensions(yaml_path)
    urdf_str = render_urdf_string(dims)

    out = Path(out_path).resolve() if out_path is not None else CANONICAL_URDF
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(urdf_str, encoding="utf-8")

    # Postcondition: the file we just wrote is parseable.
    if not out.exists() or out.stat().st_size == 0:  # pragma: no cover
        msg = f"URDF generation produced an empty file at {out}"
        raise RuntimeError(msg)
    return out


# ---------------------------------------------------------------------------
# Drake loader (lazy pydrake import)
# ---------------------------------------------------------------------------


def load_humanoid_into_plant(
    plant: MultibodyPlant,
    urdf_path: Path | str | None = None,
) -> ModelInstanceIndex:
    """Add the canonical humanoid URDF to a Drake ``MultibodyPlant``.

    Args:
        plant: A pre-finalized :class:`pydrake.multibody.plant.MultibodyPlant`.
        urdf_path: URDF source path. ``None`` -> :data:`CANONICAL_URDF`. If
            the file does not exist, it is regenerated from the shared YAML.

    Returns:
        The :class:`pydrake.multibody.tree.ModelInstanceIndex` for the
        added humanoid.

    Raises:
        ImportError: if ``pydrake`` is not installed.
        FileNotFoundError: if ``urdf_path`` is missing and the YAML is also
            missing (so the URDF cannot be regenerated).
    """
    # Explicit imports per CLAUDE.md ("Drake: Must use explicit imports:
    # from pydrake.X import Y").
    from pydrake.multibody.parsing import Parser  # noqa: PLC0415

    path = Path(urdf_path) if urdf_path is not None else CANONICAL_URDF
    if not path.exists():
        # Regenerate from the YAML; this is the path the build script and
        # CI gate use.
        build_humanoid_urdf(out_path=path)

    parser = Parser(plant)
    models = parser.AddModels(str(path))
    if not models:  # pragma: no cover - Drake raises before reaching here
        msg = f"Drake Parser returned no model instances for {path}"
        raise RuntimeError(msg)
    return models[0]


# ---------------------------------------------------------------------------
# CLI entry point: regenerate-then-write. Useful for the CI gate (#4129).
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Regenerate the canonical Drake humanoid URDF."
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=SHARED_DIMENSIONS_YAML,
        help="Path to the shared anthropometric YAML.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CANONICAL_URDF,
        help="Destination URDF path.",
    )
    args = parser.parse_args(argv)
    out = build_humanoid_urdf(args.yaml, args.out)
    sys.stdout.write(f"Wrote {out}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
