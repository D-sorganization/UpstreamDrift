"""Reader for OpenSim ``<Body>`` XML elements.

Companion to :func:`anthropometrics.writers.write_osim_body`. Parses
the three OpenSim inertial children plus the auxiliary
``<UDMetadata>`` block emitted by the writer to reconstruct a full
:class:`anthropometrics.SegmentProperties`.

OpenSim ``<Body>`` schema understood by this reader:

* ``<mass>kg</mass>`` — required scalar.
* ``<mass_center>x y z</mass_center>`` — required 3-vector.
* ``<inertia>Ixx Iyy Izz Ixy Ixz Iyz</inertia>`` — required 6-tuple
  expanded into a symmetric 3x3 tensor.
* ``<UDMetadata>...</UDMetadata>`` — required for round-trip with
  :func:`anthropometrics.writers.write_osim_body`. Carries the
  segment metadata that OpenSim's native schema cannot represent.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np

from ..segment_properties import SegmentProperties

if TYPE_CHECKING:
    from .._types import FloatArray


_METADATA_TAG = "UDMetadata"
_REQUIRED_OPENSIM_TAGS: tuple[str, ...] = ("mass", "mass_center", "inertia")
_REQUIRED_METADATA_TAGS: tuple[str, ...] = (
    "body_part_id",
    "length_m",
    "source_method",
    "source_subject_height_m",
    "source_subject_mass_kg",
)


def read_osim_body(elem: ET.Element) -> SegmentProperties:
    """Return the :class:`SegmentProperties` encoded by *elem*.

    Args:
        elem: An ``xml.etree.ElementTree.Element`` whose tag is
            ``"Body"`` (case-sensitive). Must contain the three
            OpenSim inertial children plus the ``UDMetadata``
            block produced by
            :func:`anthropometrics.writers.write_osim_body`.

    Returns:
        A fully-validated :class:`SegmentProperties`.

    Raises:
        TypeError: When *elem* is not an XML ``Element``.
        ValueError: When *elem* is not a ``<Body>`` element, when a
            required child is missing, or when a numeric field is
            malformed.
    """
    if not isinstance(elem, ET.Element):
        raise TypeError(
            "read_osim_body requires an xml.etree.ElementTree.Element, "
            f"got {type(elem).__name__}"
        )
    if elem.tag != "Body":
        raise ValueError(f"read_osim_body expected <Body> root, got <{elem.tag}>")

    name = elem.attrib.get("name", "").strip()
    if not name:
        raise ValueError("OpenSim <Body> is missing a non-empty 'name' attribute")

    for tag in _REQUIRED_OPENSIM_TAGS:
        if elem.find(tag) is None:
            raise ValueError(f"OpenSim <Body name='{name}'> missing <{tag}>")

    mass_kg = _parse_scalar(elem, "mass")
    com_xyz_m = _parse_vector(elem, "mass_center", expected_count=3)
    inertia_tensor = _parse_inertia(elem, "inertia")

    metadata = elem.find(_METADATA_TAG)
    if metadata is None:
        raise ValueError(
            f"OpenSim <Body name='{name}'> missing <{_METADATA_TAG}>; "
            "auxiliary metadata is required for round-trip with write_osim_body"
        )
    for tag in _REQUIRED_METADATA_TAGS:
        if metadata.find(tag) is None:
            raise ValueError(
                f"<{_METADATA_TAG}> in <Body name='{name}'> missing <{tag}>"
            )

    body_part_id = _required_text(metadata, "body_part_id")
    length_m = _parse_scalar(metadata, "length_m")
    source_method = _required_text(metadata, "source_method")
    source_subject_height_m = _parse_scalar(metadata, "source_subject_height_m")
    source_subject_mass_kg = _parse_scalar(metadata, "source_subject_mass_kg")
    proximal_marker = _optional_text(metadata, "proximal_marker")
    distal_marker = _optional_text(metadata, "distal_marker")

    return SegmentProperties(
        name=name,
        body_part_id=body_part_id,
        length_m=length_m,
        proximal_marker=proximal_marker,
        distal_marker=distal_marker,
        mass_kg=mass_kg,
        com_xyz_m=com_xyz_m,
        inertia_tensor=inertia_tensor,
        source_method=source_method,
        source_subject_height_m=source_subject_height_m,
        source_subject_mass_kg=source_subject_mass_kg,
    )


# --------------------------------------------------------------------------- #
# Parsing helpers.                                                            #
# --------------------------------------------------------------------------- #
def _element_text(parent: ET.Element, tag: str) -> str:
    """Return the stripped text of ``parent/tag`` or raise ``ValueError``."""
    child = parent.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"<{tag}> child of <{parent.tag}> has no text content")
    text = child.text.strip()
    if not text:
        raise ValueError(f"<{tag}> child of <{parent.tag}> has empty text content")
    return text


def _parse_scalar(parent: ET.Element, tag: str) -> float:
    """Parse a single floating-point scalar from ``parent/tag``."""
    text = _element_text(parent, tag)
    tokens = text.split()
    if len(tokens) != 1:
        raise ValueError(
            f"<{tag}> in <{parent.tag}> expected one float, got {len(tokens)}"
        )
    return _to_float(tokens[0], tag)


def _parse_vector(
    parent: ET.Element,
    tag: str,
    *,
    expected_count: int,
) -> FloatArray:
    """Parse ``expected_count`` floats from ``parent/tag`` into an ndarray."""
    text = _element_text(parent, tag)
    tokens = text.split()
    if len(tokens) != expected_count:
        raise ValueError(
            f"<{tag}> in <{parent.tag}> expected {expected_count} floats, "
            f"got {len(tokens)}"
        )
    return np.array([_to_float(t, tag) for t in tokens], dtype=float)


def _parse_inertia(parent: ET.Element, tag: str) -> FloatArray:
    """Parse the OpenSim ``Ixx Iyy Izz Ixy Ixz Iyz`` 6-tuple into a 3x3 tensor."""
    components = _parse_vector(parent, tag, expected_count=6)
    ixx, iyy, izz, ixy, ixz, iyz = (float(component) for component in components)
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


def _required_text(parent: ET.Element, tag: str) -> str:
    """Return the stripped text of ``parent/tag`` (must be non-empty)."""
    return _element_text(parent, tag)


def _optional_text(parent: ET.Element, tag: str) -> str | None:
    """Return the stripped text of ``parent/tag`` or ``None`` if absent."""
    child = parent.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _to_float(token: str, label: str) -> float:
    """Parse *token* as a finite float, raising on malformed input."""
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(f"<{label}> contained non-numeric token {token!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"<{label}> contained non-finite value {value!r}")
    return value
