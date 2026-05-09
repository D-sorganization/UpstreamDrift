"""Reader for MuJoCo MJCF ``<body><inertial>`` elements.

MuJoCo's MJCF schema attaches inertial parameters to a rigid body via
a single ``<inertial>`` child element with three required attributes
and a choice of two equivalent inertia representations:

* ``pos="x y z"`` — position of the centre-of-mass in the body frame
  (metres).
* ``mass="kg"`` — segment mass in kilograms.
* ``diaginertia="Ix Iy Iz"`` — three principal moments along the body
  frame's x/y/z axes. **Off-diagonal terms are implicitly zero.**
* ``fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"`` — six unique entries of
  the symmetric inertia tensor expressed in the body frame.

Exactly one of ``diaginertia`` or ``fullinertia`` must be present.
This reader accepts either form and reconstructs a full 3x3 symmetric
inertia tensor that satisfies every invariant declared on
:class:`anthropometrics.SegmentProperties`.

Reader-side responsibilities
----------------------------
* Parse only the contract fields. Vendor extensions
  (``ud:body_part_id`` namespaced attributes set by the writer) are
  read when present so write -> read round-trips are loss-less.
* Treat sentinel attributes (``ud:source_method``, ``ud:length_m``,
  ``ud:source_subject_height_m``, ``ud:source_subject_mass_kg``,
  ``ud:proximal_marker``, ``ud:distal_marker``, ``ud:body_part_id``,
  ``ud:name``) as the canonical anthropometric metadata MJCF cannot
  natively express.
* Fail loud on missing required attributes — silent default values
  would mask upstream pipeline bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np

from ..segment_properties import SegmentProperties

if TYPE_CHECKING:
    from .._types import FloatArray

# --------------------------------------------------------------------------- #
# Vendor-extension attribute names — kept in one place so the reader and the  #
# writer cannot drift apart.                                                  #
# --------------------------------------------------------------------------- #
_UD_NAME = "ud:name"
_UD_BODY_PART_ID = "ud:body_part_id"
_UD_LENGTH_M = "ud:length_m"
_UD_SOURCE_METHOD = "ud:source_method"
_UD_SOURCE_HEIGHT_M = "ud:source_subject_height_m"
_UD_SOURCE_MASS_KG = "ud:source_subject_mass_kg"
_UD_PROXIMAL_MARKER = "ud:proximal_marker"
_UD_DISTAL_MARKER = "ud:distal_marker"


def read_mjcf_body(elem: ET.Element) -> SegmentProperties:
    """Build a :class:`SegmentProperties` from a MJCF ``<body>`` element.

    Args:
        elem: Either the ``<body>`` element wrapping an ``<inertial>``
            child or the ``<inertial>`` element directly. Both shapes
            are accepted because callers commonly hold a reference to
            either depending on how they parsed the source document.

    Returns:
        A fully validated :class:`SegmentProperties`. Off-diagonal
        terms of the inertia tensor are zero when the MJCF source used
        ``diaginertia``; otherwise they are taken verbatim from
        ``fullinertia``.

    Raises:
        ValueError: When *elem* is neither a ``<body>`` nor an
            ``<inertial>`` element, when the ``<inertial>`` child is
            absent, when a required attribute is missing, when neither
            (or both) of ``diaginertia`` / ``fullinertia`` is present,
            or when any of the canonical anthropometric invariants on
            :class:`SegmentProperties` is violated.
    """
    inertial = _resolve_inertial(elem)
    body = elem if elem.tag == "body" else None

    pos = _parse_floats(inertial, "pos", expected=3)
    mass = _parse_scalar(inertial, "mass")
    inertia_tensor = _parse_inertia(inertial)

    name = _read_name(body, inertial)
    body_part_id = _read_optional_attr(inertial, _UD_BODY_PART_ID, default=name)
    source_method = _read_optional_attr(inertial, _UD_SOURCE_METHOD, default="mjcf")

    length_m = _read_positive_optional(inertial, _UD_LENGTH_M, default=1.0)
    source_height_m = _read_positive_optional(
        inertial, _UD_SOURCE_HEIGHT_M, default=1.0
    )
    source_mass_kg = _read_positive_optional(inertial, _UD_SOURCE_MASS_KG, default=mass)

    proximal_marker = inertial.get(_UD_PROXIMAL_MARKER) or None
    distal_marker = inertial.get(_UD_DISTAL_MARKER) or None

    return SegmentProperties(
        name=name,
        body_part_id=body_part_id,
        length_m=length_m,
        proximal_marker=proximal_marker,
        distal_marker=distal_marker,
        mass_kg=mass,
        com_xyz_m=pos,
        inertia_tensor=inertia_tensor,
        source_method=source_method,
        source_subject_height_m=source_height_m,
        source_subject_mass_kg=source_mass_kg,
    )


# --------------------------------------------------------------------------- #
# Internals.                                                                  #
# --------------------------------------------------------------------------- #
def _resolve_inertial(elem: ET.Element) -> ET.Element:
    """Return the ``<inertial>`` element regardless of which level *elem* is."""
    if elem.tag == "inertial":
        return elem
    if elem.tag == "body":
        inertial = elem.find("inertial")
        if inertial is None:
            raise ValueError(
                "MJCF <body> element has no <inertial> child; cannot build "
                "SegmentProperties"
            )
        return inertial
    raise ValueError(f"expected <body> or <inertial> element, got <{elem.tag}>")


def _read_name(body: ET.Element | None, inertial: ET.Element) -> str:
    """Resolve the segment name from <body name=...> or vendor fallback."""
    if body is not None:
        body_name = body.get("name")
        if body_name and body_name.strip():
            return body_name.strip()
    fallback = inertial.get(_UD_NAME)
    if fallback and fallback.strip():
        return fallback.strip()
    raise ValueError(
        "MJCF body has no name attribute and no ud:name vendor fallback; "
        "SegmentProperties.name must be a non-empty string"
    )


def _read_optional_attr(elem: ET.Element, attr: str, *, default: str) -> str:
    """Return the trimmed attribute or the supplied default."""
    raw = elem.get(attr)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped or default


def _read_positive_optional(elem: ET.Element, attr: str, *, default: float) -> float:
    """Parse a positive float attribute, falling back to *default*."""
    raw = elem.get(attr)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"MJCF <inertial> attribute {attr!r} must be a float, got {raw!r}"
        ) from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"MJCF <inertial> attribute {attr!r} must be positive finite, got {value!r}"
        )
    return value


def _parse_scalar(elem: ET.Element, attr: str) -> float:
    """Parse a required positive scalar attribute."""
    raw = elem.get(attr)
    if raw is None:
        raise ValueError(f"MJCF <inertial> missing required attribute {attr!r}")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"MJCF <inertial> attribute {attr!r} must be a float, got {raw!r}"
        ) from error
    return value


def _parse_floats(elem: ET.Element, attr: str, *, expected: int) -> FloatArray:
    """Parse a whitespace-separated list of *expected* floats."""
    raw = elem.get(attr)
    if raw is None:
        raise ValueError(f"MJCF <inertial> missing required attribute {attr!r}")
    tokens = raw.split()
    if len(tokens) != expected:
        raise ValueError(
            f"MJCF <inertial> attribute {attr!r} must contain {expected} "
            f"floats, got {len(tokens)} ({raw!r})"
        )
    try:
        values = [float(token) for token in tokens]
    except ValueError as error:
        raise ValueError(
            f"MJCF <inertial> attribute {attr!r} contains non-float token ({raw!r})"
        ) from error
    return np.asarray(values, dtype=float)


def _parse_inertia(elem: ET.Element) -> FloatArray:
    """Reconstruct a 3x3 symmetric inertia tensor from MJCF attributes.

    The MJCF schema requires exactly one of ``diaginertia`` or
    ``fullinertia``; this helper enforces that with a clear error
    message when both or neither are present.
    """
    diag_raw = elem.get("diaginertia")
    full_raw = elem.get("fullinertia")
    if diag_raw is None and full_raw is None:
        raise ValueError(
            "MJCF <inertial> requires one of 'diaginertia' or 'fullinertia'"
        )
    if diag_raw is not None and full_raw is not None:
        raise ValueError(
            "MJCF <inertial> must not declare both 'diaginertia' and "
            "'fullinertia'; choose one."
        )

    if full_raw is not None:
        full = _parse_floats(elem, "fullinertia", expected=6)
        ixx, iyy, izz, ixy, ixz, iyz = full.tolist()
        tensor = np.array(
            [
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ],
            dtype=float,
        )
        return tensor

    diag = _parse_floats(elem, "diaginertia", expected=3)
    return np.diag(diag).astype(float)
