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

#: Default location of the shared dimensions YAML (owned by #4093).
SHARED_DIMENSIONS_YAML: Path = (
    _REPO_ROOT / "shared" / "models" / "golf_humanoid_dimensions.yaml"
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
class HumanoidDimensions:
    """Top-level parsed YAML container."""

    schema_version: int
    pelvis_to_shoulders_m: float
    shoulder_width_m: float
    hand_spacing_m: float
    total_mass_kg: float
    total_height_m: float
    segments: tuple[SegmentSpec, ...]

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


def _parse_joint(raw: dict[str, Any], *, segment_name: str) -> JointSpec:
    if "name" not in raw or "type" not in raw:
        msg = (
            f"segment {segment_name!r}: joint entry must specify both 'name' and 'type'"
        )
        raise ValueError(msg)
    jtype = str(raw["type"])
    if jtype not in _VALID_JOINT_TYPES:
        msg = (
            f"segment {segment_name!r}: unknown joint type {jtype!r} "
            f"(allowed: {sorted(_VALID_JOINT_TYPES)})"
        )
        raise ValueError(msg)

    axes: tuple[tuple[float, float, float], ...] = ()
    if jtype == "revolute":
        if "axis" not in raw:
            msg = f"segment {segment_name!r}: revolute joint requires 'axis'"
            raise ValueError(msg)
        axes = (_as_tuple3(raw["axis"], where=f"{segment_name}.axis"),)
    elif jtype in {"universal", "gimbal"}:
        if "axes" not in raw:
            msg = (
                f"segment {segment_name!r}: {jtype} joint requires 'axes' "
                f"(list of 3-vectors)"
            )
            raise ValueError(msg)
        axes_raw = raw["axes"]
        expected = 2 if jtype == "universal" else 3
        if not isinstance(axes_raw, list) or len(axes_raw) != expected:
            msg = (
                f"segment {segment_name!r}: {jtype} joint requires exactly "
                f"{expected} axes, got {axes_raw!r}"
            )
            raise ValueError(msg)
        axes = tuple(
            _as_tuple3(a, where=f"{segment_name}.axes[{i}]")
            for i, a in enumerate(axes_raw)
        )

    limits: tuple[tuple[float, float], ...] = ()
    raw_limits = raw.get("limits")
    if raw_limits is not None:
        if jtype == "revolute":
            limits = ((float(raw_limits[0]), float(raw_limits[1])),)
        elif jtype in {"universal", "gimbal"}:
            limits = tuple((float(pair[0]), float(pair[1])) for pair in raw_limits)

    return JointSpec(
        name=str(raw["name"]),
        type=jtype,
        axes=axes,
        limits=limits,
        damping=float(raw.get("damping", 0.0)),
    )


def _parse_segment(raw: dict[str, Any]) -> SegmentSpec:
    if "name" not in raw or "parent" not in raw or "joint" not in raw:
        msg = (
            "segment entry must specify 'name', 'parent', and 'joint'; "
            f"got {sorted(raw)!r}"
        )
        raise ValueError(msg)

    name = str(raw["name"])
    origin = raw.get("origin", {})
    origin_xyz = _as_tuple3(
        origin.get("xyz", [0.0, 0.0, 0.0]), where=f"{name}.origin.xyz"
    )
    origin_rpy = _as_tuple3(
        origin.get("rpy", [0.0, 0.0, 0.0]), where=f"{name}.origin.rpy"
    )

    geometry = raw.get("geometry", {})
    if not isinstance(geometry, dict):
        msg = f"segment {name!r}: geometry must be a mapping"
        raise ValueError(msg)

    mass = float(raw.get("mass", 0.0))
    if mass <= 0.0:
        msg = f"segment {name!r}: mass must be > 0, got {mass!r}"
        raise ValueError(msg)

    inertia = raw.get("inertia", {})
    if not isinstance(inertia, dict):
        msg = f"segment {name!r}: inertia must be a mapping"
        raise ValueError(msg)
    needed = {"ixx", "iyy", "izz", "ixy", "ixz", "iyz"}
    missing = needed - set(inertia)
    if missing:
        msg = f"segment {name!r}: inertia is missing keys {sorted(missing)}"
        raise ValueError(msg)

    return SegmentSpec(
        name=name,
        parent=str(raw["parent"]),
        joint=_parse_joint(raw["joint"], segment_name=name),
        origin_xyz=origin_xyz,
        origin_rpy=origin_rpy,
        geometry=geometry,
        mass=mass,
        inertia={k: float(v) for k, v in inertia.items()},
    )


def load_humanoid_dimensions(
    yaml_path: Path | str | None = None,
) -> HumanoidDimensions:
    """Parse the shared humanoid dimensions YAML into structured form.

    Args:
        yaml_path: Path to the YAML. ``None`` resolves to
            :data:`SHARED_DIMENSIONS_YAML`.

    Returns:
        Parsed :class:`HumanoidDimensions`.

    Raises:
        FileNotFoundError: if the YAML is missing.
        ValueError: if a required field is absent or malformed.
    """
    path = Path(yaml_path) if yaml_path is not None else SHARED_DIMENSIONS_YAML
    if not path.exists():
        msg = (
            f"Shared humanoid dimensions YAML not found at {path}. "
            "This file is owned by issue #4093 (PARITY-DIMENSIONS); "
            "see DRAKE_PARITY_SPEC.md §3.3."
        )
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        msg = f"YAML root must be a mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    schema_version = int(raw.get("schema_version", 0))
    anthro = raw.get("anthropometry", {})
    if not isinstance(anthro, dict):
        msg = "YAML must define an 'anthropometry' mapping"
        raise ValueError(msg)

    segments_raw = raw.get("segments", [])
    if not isinstance(segments_raw, list) or not segments_raw:
        msg = "YAML must define a non-empty 'segments' list"
        raise ValueError(msg)

    segments = tuple(_parse_segment(s) for s in segments_raw)

    return HumanoidDimensions(
        schema_version=schema_version,
        pelvis_to_shoulders_m=float(anthro.get("pelvis_to_shoulders_m", 0.0)),
        shoulder_width_m=float(anthro.get("shoulder_width_m", 0.0)),
        hand_spacing_m=float(anthro.get("hand_spacing_m", 0.0)),
        total_mass_kg=float(anthro.get("total_mass_kg", 0.0)),
        total_height_m=float(anthro.get("total_height_m", 0.0)),
        segments=segments,
        raw=raw,
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
