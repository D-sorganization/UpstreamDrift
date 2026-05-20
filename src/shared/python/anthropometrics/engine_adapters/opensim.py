"""OpenSim :class:`EngineAdapter` — emits a minimal ``.osim`` model.

OpenSim (https://opensim.stanford.edu) describes a subject as an
XML ``<OpenSimDocument>`` containing a ``<Model>`` with a
``<BodySet>`` of ``<Body>`` elements. Each ``<Body>`` carries::

    <mass>m</mass>
    <mass_center>cx cy cz</mass_center>
    <inertia>ixx iyy izz ixy ixz iyz</inertia>

The adapter writes exactly that schema plus a custom
``<UDSubjectMetadata>`` and per-body ``<UDSegmentMetadata>``
sidecar carrying the canonical anthropometric metadata so
:meth:`import_back` recovers a complete
:class:`SubjectAnthropometrics`.

OpenSim itself ignores unknown XML elements at load time, so the
sidecars do not interfere with native ``opensim.Model(path)``
ingestion.
"""

from __future__ import annotations

import defusedxml.ElementTree as ET  # noqa: S314  # Security: defusedxml prevents XML attacks
from pathlib import Path

import numpy as np

from .._subject_anthropometrics import SubjectAnthropometrics
from ..segment_properties import SegmentProperties

_FLOAT_FMT = repr


class OpenSimAdapter:
    """Round-trip a :class:`SubjectAnthropometrics` through a ``.osim`` file."""

    engine_name: str = "opensim"

    def export(
        self, anthropometrics: SubjectAnthropometrics, output_path: Path
    ) -> None:
        """Serialise *anthropometrics* to *output_path* as a minimal OpenSim model."""
        if not isinstance(anthropometrics, SubjectAnthropometrics):
            raise TypeError(
                "anthropometrics must be a SubjectAnthropometrics, got "
                f"{type(anthropometrics).__name__}"
            )
        output_path = Path(output_path)

        document = ET.Element("OpenSimDocument", {"Version": "40500"})
        model = ET.SubElement(document, "Model", {"name": anthropometrics.subject_id})

        meta_attrs = {
            "schema_version": "1",
            "subject_id": anthropometrics.subject_id,
            "height_m": _FLOAT_FMT(float(anthropometrics.height_m)),
            "mass_kg": _FLOAT_FMT(float(anthropometrics.mass_kg)),
            "sex": anthropometrics.sex,
            "source_method": anthropometrics.source_method,
        }
        if anthropometrics.age_years is not None:
            meta_attrs["age_years"] = _FLOAT_FMT(float(anthropometrics.age_years))
        ET.SubElement(model, "UDSubjectMetadata", meta_attrs)

        body_set = ET.SubElement(model, "BodySet")
        objects = ET.SubElement(body_set, "objects")
        for seg_name, props in anthropometrics.segments:
            body = ET.SubElement(objects, "Body", {"name": seg_name})
            ET.SubElement(body, "mass").text = _FLOAT_FMT(float(props.mass_kg))
            cx, cy, cz = (float(v) for v in props.com_xyz_m.tolist())
            ET.SubElement(
                body, "mass_center"
            ).text = f"{_FLOAT_FMT(cx)} {_FLOAT_FMT(cy)} {_FLOAT_FMT(cz)}"
            t = props.inertia_tensor
            ET.SubElement(body, "inertia").text = " ".join(
                _FLOAT_FMT(float(v))
                for v in (t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2])
            )
            seg_attrs = {
                "body_part_id": props.body_part_id,
                "length_m": _FLOAT_FMT(float(props.length_m)),
                "source_method": props.source_method,
                "source_subject_height_m": _FLOAT_FMT(
                    float(props.source_subject_height_m)
                ),
                "source_subject_mass_kg": _FLOAT_FMT(
                    float(props.source_subject_mass_kg)
                ),
            }
            if props.proximal_marker is not None:
                seg_attrs["proximal_marker"] = props.proximal_marker
            if props.distal_marker is not None:
                seg_attrs["distal_marker"] = props.distal_marker
            ET.SubElement(body, "UDSegmentMetadata", seg_attrs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(document).write(
            output_path, encoding="utf-8", xml_declaration=True
        )

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        """Reverse of :meth:`export`."""
        input_path = Path(input_path)
        tree = ET.parse(str(input_path))
        document = tree.getroot()
        if document.tag != "OpenSimDocument":
            raise ValueError(
                f"expected <OpenSimDocument> root, got <{document.tag}> in {input_path}"
            )
        model = document.find("Model")
        if model is None:
            raise ValueError(f"OpenSim document missing <Model> in {input_path}")
        meta = model.find("UDSubjectMetadata")
        if meta is None:
            raise ValueError(
                f"OpenSim document missing <UDSubjectMetadata> sidecar in {input_path}"
            )
        subject_id = _required_attr(meta, "subject_id", "<UDSubjectMetadata>")
        height_m = float(_required_attr(meta, "height_m", "<UDSubjectMetadata>"))
        mass_kg = float(_required_attr(meta, "mass_kg", "<UDSubjectMetadata>"))
        sex = meta.get("sex", "unspecified")
        source_method = _required_attr(meta, "source_method", "<UDSubjectMetadata>")
        age_raw = meta.get("age_years")
        age_years = float(age_raw) if age_raw is not None else None

        body_set = model.find("BodySet")
        if body_set is None:
            raise ValueError(f"OpenSim model missing <BodySet> in {input_path}")
        objects = body_set.find("objects")
        if objects is None:
            raise ValueError(f"OpenSim <BodySet> missing <objects> in {input_path}")

        segments: list[tuple[str, SegmentProperties]] = []
        for body in objects.findall("Body"):
            seg_name = _required_attr(body, "name", "<Body>")
            mass_text = _required_text(body, "mass", "<Body>")
            com_text = _required_text(body, "mass_center", "<Body>")
            inertia_text = _required_text(body, "inertia", "<Body>")
            seg_meta = body.find("UDSegmentMetadata")
            if seg_meta is None:
                raise ValueError(
                    f"<Body name={seg_name!r}> missing <UDSegmentMetadata>"
                )

            com_parts = com_text.split()
            if len(com_parts) != 3:
                raise ValueError(
                    f"<mass_center> requires 3 components, got {com_text!r}"
                )
            inertia_parts = inertia_text.split()
            if len(inertia_parts) != 6:
                raise ValueError(
                    f"<inertia> requires 6 components, got {inertia_text!r}"
                )
            ixx, iyy, izz, ixy, ixz, iyz = (float(p) for p in inertia_parts)
            tensor = np.array(
                [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
                dtype=float,
            )
            props = SegmentProperties(
                name=seg_name,
                body_part_id=_required_attr(
                    seg_meta, "body_part_id", "<UDSegmentMetadata>"
                ),
                length_m=float(
                    _required_attr(seg_meta, "length_m", "<UDSegmentMetadata>")
                ),
                proximal_marker=seg_meta.get("proximal_marker"),
                distal_marker=seg_meta.get("distal_marker"),
                mass_kg=float(mass_text),
                com_xyz_m=np.asarray([float(p) for p in com_parts], dtype=float),
                inertia_tensor=tensor,
                source_method=_required_attr(
                    seg_meta, "source_method", "<UDSegmentMetadata>"
                ),
                source_subject_height_m=float(
                    _required_attr(
                        seg_meta,
                        "source_subject_height_m",
                        "<UDSegmentMetadata>",
                    )
                ),
                source_subject_mass_kg=float(
                    _required_attr(
                        seg_meta,
                        "source_subject_mass_kg",
                        "<UDSegmentMetadata>",
                    )
                ),
            )
            segments.append((seg_name, props))

        if not segments:
            raise ValueError(f"OpenSim BodySet has no <Body> elements in {input_path}")

        return SubjectAnthropometrics(
            subject_id=subject_id,
            height_m=height_m,
            mass_kg=mass_kg,
            segments=tuple(segments),
            source_method=source_method,
            age_years=age_years,
            sex=sex,
        )


def _required_attr(elem: ET.Element, attr: str, label: str) -> str:
    value = elem.get(attr)
    if value is None:
        raise ValueError(f"{label} missing required attribute {attr!r}")
    return value


def _required_text(parent: ET.Element, child_tag: str, label: str) -> str:
    child = parent.find(child_tag)
    if child is None or child.text is None or not child.text.strip():
        raise ValueError(f"{label} missing required <{child_tag}> child text")
    return child.text.strip()
