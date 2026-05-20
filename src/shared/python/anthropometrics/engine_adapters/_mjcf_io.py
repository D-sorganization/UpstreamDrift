"""Shared MJCF emit / parse for MyoSuite (MuJoCo) adapter.

MJCF is MuJoCo's native XML format. It expresses inertial
properties on a ``<body>`` via two child elements::

    <body name="seg">
      <inertial pos="cx cy cz" mass="m" fullinertia="ixx iyy izz ixy ixz iyz"/>
    </body>

MuJoCo's XML parser is strict and rejects unknown namespaces or
attributes on the ``<mujoco>`` root, so we **cannot** embed the
canonical anthropometric metadata as a namespaced sidecar element
the way the URDF emitter does. Instead, the writer emits a
sidecar JSON file alongside the ``.xml`` (sharing its stem) that
carries the per-segment metadata required to reconstruct the full
:class:`SubjectAnthropometrics`. The MJCF itself remains
mujoco-loadable; the sidecar lets the reader recover canonical
metadata losslessly.
"""

from __future__ import annotations

import json
import defusedxml.ElementTree as ET  # noqa: S314  # Security: defusedxml prevents XML attacks
from pathlib import Path
from typing import Any

import numpy as np

from .._subject_anthropometrics import SubjectAnthropometrics
from ..segment_properties import SegmentProperties

_FLOAT_FMT = repr


# --------------------------------------------------------------------------- #
# Path helpers.                                                               #
# --------------------------------------------------------------------------- #
def _sidecar_path(mjcf_path: Path) -> Path:
    return mjcf_path.with_suffix(mjcf_path.suffix + ".meta.json")


# --------------------------------------------------------------------------- #
# Writer.                                                                     #
# --------------------------------------------------------------------------- #
def write_mjcf_subject(anthro: SubjectAnthropometrics, output_path: Path) -> None:
    """Write *anthro* as an MJCF document plus a sidecar JSON.

    The sidecar JSON is written next to *output_path* with the
    suffix ``<mjcf>.meta.json`` (e.g. ``subject.xml`` →
    ``subject.xml.meta.json``).
    """
    if not isinstance(anthro, SubjectAnthropometrics):
        raise TypeError(
            f"anthro must be a SubjectAnthropometrics, got {type(anthro).__name__}"
        )
    output_path = Path(output_path)
    mujoco = ET.Element("mujoco", {"model": anthro.subject_id})
    worldbody = ET.SubElement(mujoco, "worldbody")

    seg_meta_payload: list[dict[str, Any]] = []
    for seg_name, props in anthro.segments:
        body = ET.SubElement(worldbody, "body", {"name": seg_name})
        cx, cy, cz = (float(v) for v in props.com_xyz_m.tolist())
        tensor = props.inertia_tensor
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": (f"{_FLOAT_FMT(cx)} {_FLOAT_FMT(cy)} {_FLOAT_FMT(cz)}"),
                "mass": _FLOAT_FMT(float(props.mass_kg)),
                # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
                "fullinertia": " ".join(
                    _FLOAT_FMT(float(v))
                    for v in (
                        tensor[0, 0],
                        tensor[1, 1],
                        tensor[2, 2],
                        tensor[0, 1],
                        tensor[0, 2],
                        tensor[1, 2],
                    )
                ),
            },
        )
        seg_meta_payload.append(
            {
                "name": seg_name,
                "body_part_id": props.body_part_id,
                "length_m": float(props.length_m),
                "proximal_marker": props.proximal_marker,
                "distal_marker": props.distal_marker,
                "source_method": props.source_method,
                "source_subject_height_m": float(props.source_subject_height_m),
                "source_subject_mass_kg": float(props.source_subject_mass_kg),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mujoco).write(output_path, encoding="utf-8", xml_declaration=True)

    sidecar = {
        "schema_version": 1,
        "subject_id": anthro.subject_id,
        "height_m": float(anthro.height_m),
        "mass_kg": float(anthro.mass_kg),
        "sex": anthro.sex,
        "source_method": anthro.source_method,
        "age_years": (None if anthro.age_years is None else float(anthro.age_years)),
        "segments": seg_meta_payload,
    }
    _sidecar_path(output_path).write_text(
        json.dumps(sidecar, indent=2, allow_nan=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Reader.                                                                     #
# --------------------------------------------------------------------------- #
def read_mjcf_subject(input_path: Path) -> SubjectAnthropometrics:
    """Reverse of :func:`write_mjcf_subject`."""
    input_path = Path(input_path)
    sidecar_path = _sidecar_path(input_path)
    if not sidecar_path.exists():
        raise ValueError(
            f"MJCF subject sidecar JSON not found at {sidecar_path} (the "
            "MyoSuite/MJCF reader requires the .meta.json companion written "
            "by the matching writer)"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    tree = ET.parse(str(input_path))
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(
            f"expected MJCF root <mujoco>, got <{root.tag}> in {input_path}"
        )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF subject file is missing <worldbody> in {input_path}")

    inertial_by_name: dict[str, dict[str, str]] = {}
    for body in worldbody.findall("body"):
        seg_name = body.get("name")
        if seg_name is None:
            raise ValueError(f"<body> missing required name in {input_path}")
        inertial = body.find("inertial")
        if inertial is None:
            raise ValueError(
                f"<body name={seg_name!r}> missing <inertial> in {input_path}"
            )
        inertial_by_name[seg_name] = inertial.attrib

    segments: list[tuple[str, SegmentProperties]] = []
    for seg_meta in sidecar["segments"]:
        seg_name = seg_meta["name"]
        if seg_name not in inertial_by_name:
            raise ValueError(
                f"sidecar references segment {seg_name!r} not present "
                f"in MJCF {input_path}"
            )
        attrs = inertial_by_name[seg_name]
        pos_parts = attrs.get("pos", "0 0 0").split()
        if len(pos_parts) != 3:
            raise ValueError(
                f"<inertial pos=...> must have 3 components for {seg_name!r}"
            )
        com = np.asarray([float(p) for p in pos_parts], dtype=float)
        if "mass" not in attrs:
            raise ValueError(f"<inertial> for {seg_name!r} missing required 'mass'")
        if "fullinertia" not in attrs:
            raise ValueError(
                f"<inertial> for {seg_name!r} missing required 'fullinertia'"
            )
        fi_parts = attrs["fullinertia"].split()
        if len(fi_parts) != 6:
            raise ValueError(
                f"<inertial fullinertia=...> for {seg_name!r} requires 6 "
                f"components, got {attrs['fullinertia']!r}"
            )
        ixx, iyy, izz, ixy, ixz, iyz = (float(p) for p in fi_parts)
        tensor = np.array(
            [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
            dtype=float,
        )
        props = SegmentProperties(
            name=seg_name,
            body_part_id=seg_meta["body_part_id"],
            length_m=float(seg_meta["length_m"]),
            proximal_marker=seg_meta["proximal_marker"],
            distal_marker=seg_meta["distal_marker"],
            mass_kg=float(attrs["mass"]),
            com_xyz_m=com,
            inertia_tensor=tensor,
            source_method=seg_meta["source_method"],
            source_subject_height_m=float(seg_meta["source_subject_height_m"]),
            source_subject_mass_kg=float(seg_meta["source_subject_mass_kg"]),
        )
        segments.append((seg_name, props))

    if not segments:
        raise ValueError(f"MJCF subject file has no <body> elements in {input_path}")

    return SubjectAnthropometrics(
        subject_id=sidecar["subject_id"],
        height_m=float(sidecar["height_m"]),
        mass_kg=float(sidecar["mass_kg"]),
        segments=tuple(segments),
        source_method=sidecar["source_method"],
        age_years=(
            None if sidecar.get("age_years") is None else float(sidecar["age_years"])
        ),
        sex=sidecar.get("sex", "unspecified"),
    )
