"""Shared URDF emit / parse logic for URDF-consuming engine adapters.

URDF is the native exchange format for several physics engines
(Drake, Pinocchio) and an accepted ingestion format for others
(MyoSuite via mujoco_to_urdf flows). To avoid duplicating the
serialisation logic in every adapter we collect it here and reuse
the existing low-level :func:`write_urdf_inertial` /
:func:`read_urdf_inertial` helpers.

The on-disk schema is::

    <robot name="<subject_id>">
        <ud:metadata
            schema_version="1"
            subject_id="..."
            height_m="..." mass_kg="..." sex="..."
            age_years="..."             # optional
            source_method="..."
            xmlns:ud="..."/>
        <link name="<segment_name>">
            <ud:segment
                body_part_id="..." length_m="..."
                proximal_marker="..." distal_marker="..."
                source_method="..."
                source_subject_height_m="..."
                source_subject_mass_kg="..."/>
            <inertial> ... </inertial>
        </link>
        ...
    </robot>

The ``<ud:metadata>`` and ``<ud:segment>`` sidecar elements are in
a custom XML namespace so other URDF consumers ignore them while
this package's reader recovers the full :class:`SubjectAnthropometrics`
without information loss.
"""

from __future__ import annotations

# stdlib ElementTree builds the URDF document (Element/SubElement/write/
# register_namespace); defusedxml parses untrusted input and refuses XXE /
# entity expansion (issue #6927). defusedxml does not expose the builder API,
# so building and parsing use separate imports.
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # build-only; parse via DefusedET
from pathlib import Path

import defusedxml.ElementTree as DefusedET

from .._subject_anthropometrics import SubjectAnthropometrics
from ..readers.urdf_inertial import read_urdf_inertial
from ..segment_properties import SegmentProperties
from ..writers.urdf_inertial import write_urdf_inertial

UD_NS = "https://upstream-drift.dev/anthropometrics"
"""Custom XML namespace for the sidecar metadata elements."""

_FLOAT_FMT = repr  # Python's shortest round-trip float repr.


# --------------------------------------------------------------------------- #
# Writer.                                                                     #
# --------------------------------------------------------------------------- #
def write_urdf_subject(anthro: SubjectAnthropometrics, output_path: Path) -> None:
    """Write *anthro* as a URDF document to *output_path*.

    The file is well-formed XML with the ``ud:`` namespace declared
    on the root ``<robot>`` element. Every ``<link>`` carries a
    ``<ud:segment>`` sidecar with the canonical metadata plus the
    standard URDF ``<inertial>`` block produced by
    :func:`write_urdf_inertial`.
    """
    if not isinstance(anthro, SubjectAnthropometrics):
        raise TypeError(
            f"anthro must be a SubjectAnthropometrics, got {type(anthro).__name__}"
        )
    output_path = Path(output_path)

    ET.register_namespace("ud", UD_NS)
    robot = ET.Element("robot", {"name": anthro.subject_id})

    meta_attrs = {
        "schema_version": "1",
        "subject_id": anthro.subject_id,
        "height_m": _FLOAT_FMT(float(anthro.height_m)),
        "mass_kg": _FLOAT_FMT(float(anthro.mass_kg)),
        "sex": anthro.sex,
        "source_method": anthro.source_method,
    }
    if anthro.age_years is not None:
        meta_attrs["age_years"] = _FLOAT_FMT(float(anthro.age_years))
    ET.SubElement(robot, f"{{{UD_NS}}}metadata", meta_attrs)

    for seg_name, props in anthro.segments:
        link = ET.SubElement(robot, "link", {"name": seg_name})
        seg_attrs = {
            "body_part_id": props.body_part_id,
            "length_m": _FLOAT_FMT(float(props.length_m)),
            "source_method": props.source_method,
            "source_subject_height_m": _FLOAT_FMT(float(props.source_subject_height_m)),
            "source_subject_mass_kg": _FLOAT_FMT(float(props.source_subject_mass_kg)),
        }
        if props.proximal_marker is not None:
            seg_attrs["proximal_marker"] = props.proximal_marker
        if props.distal_marker is not None:
            seg_attrs["distal_marker"] = props.distal_marker
        ET.SubElement(link, f"{{{UD_NS}}}segment", seg_attrs)
        link.append(write_urdf_inertial(props))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------------------- #
# Reader.                                                                     #
# --------------------------------------------------------------------------- #
def read_urdf_subject(input_path: Path) -> SubjectAnthropometrics:
    """Reverse of :func:`write_urdf_subject`."""
    input_path = Path(input_path)
    tree = DefusedET.parse(str(input_path))
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError(
            f"expected URDF root <robot>, got <{robot.tag}> in {input_path}"
        )

    meta = robot.find(f"{{{UD_NS}}}metadata")
    if meta is None:
        raise ValueError(
            f"URDF subject file is missing required <ud:metadata> in {input_path}"
        )

    subject_id = _required_attr(meta, "subject_id", "<ud:metadata>")
    height_m = float(_required_attr(meta, "height_m", "<ud:metadata>"))
    mass_kg = float(_required_attr(meta, "mass_kg", "<ud:metadata>"))
    sex = meta.get("sex", "unspecified")
    source_method = _required_attr(meta, "source_method", "<ud:metadata>")
    age_raw = meta.get("age_years")
    age_years = float(age_raw) if age_raw is not None else None

    segments: list[tuple[str, SegmentProperties]] = []
    for link in robot.findall("link"):
        seg_name = _required_attr(link, "name", "<link>")
        seg_meta = link.find(f"{{{UD_NS}}}segment")
        if seg_meta is None:
            raise ValueError(
                f"<link name={seg_name!r}> is missing required <ud:segment> "
                f"sidecar in {input_path}"
            )
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(
                f"<link name={seg_name!r}> is missing required <inertial> "
                f"in {input_path}"
            )
        props = read_urdf_inertial(
            inertial,
            name=seg_name,
            body_part_id=_required_attr(seg_meta, "body_part_id", "<ud:segment>"),
            length_m=float(_required_attr(seg_meta, "length_m", "<ud:segment>")),
            proximal_marker=seg_meta.get("proximal_marker"),
            distal_marker=seg_meta.get("distal_marker"),
            source_method=_required_attr(seg_meta, "source_method", "<ud:segment>"),
            source_subject_height_m=float(
                _required_attr(seg_meta, "source_subject_height_m", "<ud:segment>")
            ),
            source_subject_mass_kg=float(
                _required_attr(seg_meta, "source_subject_mass_kg", "<ud:segment>")
            ),
        )
        segments.append((seg_name, props))

    if not segments:
        raise ValueError(f"URDF subject file has no <link> elements in {input_path}")

    return SubjectAnthropometrics(
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        segments=tuple(segments),
        source_method=source_method,
        age_years=age_years,
        sex=sex,
    )


# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #
def _required_attr(elem: ET.Element, attr: str, label: str) -> str:
    value = elem.get(attr)
    if value is None:
        raise ValueError(f"{label} is missing required attribute {attr!r}")
    return value
