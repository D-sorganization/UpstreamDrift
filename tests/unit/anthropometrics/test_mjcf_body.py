"""Round-trip tests for MJCF ``<body><inertial>`` reader/writer.

The contract under test:

* ``write_mjcf_body(props)`` returns an ``ET.Element`` rooted at
  ``<body>`` with a single ``<inertial>`` child.
* ``read_mjcf_body(elem)`` reconstructs a :class:`SegmentProperties`
  identical to the original within ``rtol=1e-9, atol=1e-12``.
* The reader accepts both ``diaginertia`` and ``fullinertia`` forms
  (writer always emits ``fullinertia``; tests synthesise hand-rolled
  XML for the ``diaginertia`` path to prove input compatibility).
* The reader fails loud on malformed input.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import pytest

from anthropometrics import (
    SegmentProperties,
    read_mjcf_body,
    write_mjcf_body,
)

ROUND_TRIP_RTOL = 1e-9
ROUND_TRIP_ATOL = 1e-12


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
def _diag_props() -> SegmentProperties:
    """A SegmentProperties whose inertia tensor is purely diagonal."""
    return SegmentProperties(
        name="upper_arm_left",
        body_part_id="upper_arm_left",
        length_m=0.32,
        proximal_marker="LSHO",
        distal_marker="LELB",
        mass_kg=2.05,
        com_xyz_m=np.array([0.012, -0.003, 0.142], dtype=float),
        inertia_tensor=np.diag([0.0123456789, 0.0234567891, 0.0145678912]),
        source_method="de_leva",
        source_subject_height_m=1.83,
        source_subject_mass_kg=82.0,
    )


def _full_props() -> SegmentProperties:
    """A SegmentProperties whose inertia tensor has non-zero off-diagonals."""
    # Construct a symmetric positive-definite tensor that satisfies the
    # principal-moment triangle inequality after eigen-decomposition.
    base = np.diag([0.045, 0.041, 0.0095])
    off = np.array(
        [
            [0.0, 0.00031, -0.00017],
            [0.00031, 0.0, 0.00022],
            [-0.00017, 0.00022, 0.0],
        ],
        dtype=float,
    )
    tensor = base + off
    return SegmentProperties(
        name="torso",
        body_part_id="torso_segment",  # different on purpose -> ud:body_part_id used
        length_m=0.51,
        proximal_marker=None,
        distal_marker=None,
        mass_kg=34.5,
        com_xyz_m=np.array([0.0, 0.0, 0.27], dtype=float),
        inertia_tensor=tensor,
        source_method="dempster",
        source_subject_height_m=1.78,
        source_subject_mass_kg=78.0,
    )


def _assert_props_round_trip(a: SegmentProperties, b: SegmentProperties) -> None:
    """Field-by-field equivalence within the contracted tolerance."""
    assert a.name == b.name
    assert a.body_part_id == b.body_part_id
    assert a.proximal_marker == b.proximal_marker
    assert a.distal_marker == b.distal_marker
    assert a.source_method == b.source_method
    assert a.length_m == pytest.approx(
        b.length_m, rel=ROUND_TRIP_RTOL, abs=ROUND_TRIP_ATOL
    )
    assert a.mass_kg == pytest.approx(
        b.mass_kg, rel=ROUND_TRIP_RTOL, abs=ROUND_TRIP_ATOL
    )
    assert a.source_subject_height_m == pytest.approx(
        b.source_subject_height_m, rel=ROUND_TRIP_RTOL, abs=ROUND_TRIP_ATOL
    )
    assert a.source_subject_mass_kg == pytest.approx(
        b.source_subject_mass_kg, rel=ROUND_TRIP_RTOL, abs=ROUND_TRIP_ATOL
    )
    np.testing.assert_allclose(
        a.com_xyz_m, b.com_xyz_m, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL
    )
    np.testing.assert_allclose(
        a.inertia_tensor, b.inertia_tensor, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL
    )


# --------------------------------------------------------------------------- #
# Round-trip tests.                                                           #
# --------------------------------------------------------------------------- #
def test_round_trip_diagonal_inertia() -> None:
    original = _diag_props()
    elem = write_mjcf_body(original)
    restored = read_mjcf_body(elem)
    _assert_props_round_trip(original, restored)


def test_round_trip_full_inertia() -> None:
    original = _full_props()
    elem = write_mjcf_body(original)
    restored = read_mjcf_body(elem)
    _assert_props_round_trip(original, restored)


def test_writer_emits_fullinertia() -> None:
    """The writer always uses fullinertia to preserve off-diagonal terms."""
    elem = write_mjcf_body(_full_props())
    inertial = elem.find("inertial")
    assert inertial is not None
    assert "fullinertia" in inertial.attrib
    assert "diaginertia" not in inertial.attrib


def test_reader_accepts_diaginertia_input() -> None:
    """Hand-rolled MJCF with diaginertia round-trips through write+read."""
    diag = [0.0123456789, 0.0234567891, 0.0145678912]
    body = ET.Element("body", attrib={"name": "forearm_right"})
    ET.SubElement(
        body,
        "inertial",
        attrib={
            "pos": "0.01 0.0 0.12",
            "mass": "1.25",
            "diaginertia": " ".join(repr(v) for v in diag),
            "ud:length_m": "0.27",
            "ud:source_method": "de_leva",
            "ud:source_subject_height_m": "1.83",
            "ud:source_subject_mass_kg": "82.0",
        },
    )

    restored = read_mjcf_body(body)
    np.testing.assert_allclose(
        restored.inertia_tensor,
        np.diag(diag),
        rtol=ROUND_TRIP_RTOL,
        atol=ROUND_TRIP_ATOL,
    )

    # And re-writing then re-reading still round-trips.
    second_elem = write_mjcf_body(restored)
    second = read_mjcf_body(second_elem)
    _assert_props_round_trip(restored, second)


def test_diaginertia_and_fullinertia_paths_agree_when_offdiag_zero() -> None:
    """When off-diagonals are zero either form must reconstruct identically."""
    diag_values = [0.011, 0.013, 0.012]

    via_diag = ET.Element("body", attrib={"name": "head"})
    ET.SubElement(
        via_diag,
        "inertial",
        attrib={
            "pos": "0.0 0.0 0.08",
            "mass": "4.5",
            "diaginertia": " ".join(repr(v) for v in diag_values),
        },
    )

    via_full = ET.Element("body", attrib={"name": "head"})
    ET.SubElement(
        via_full,
        "inertial",
        attrib={
            "pos": "0.0 0.0 0.08",
            "mass": "4.5",
            "fullinertia": (" ".join(repr(v) for v in diag_values + [0.0, 0.0, 0.0])),
        },
    )

    a = read_mjcf_body(via_diag)
    b = read_mjcf_body(via_full)
    _assert_props_round_trip(a, b)


def test_reader_accepts_inertial_element_directly() -> None:
    """Callers that already drilled down to <inertial> should still work."""
    body = write_mjcf_body(_diag_props())
    inertial = body.find("inertial")
    assert inertial is not None
    inertial.set("ud:name", "upper_arm_left")  # ensure name is recoverable
    restored = read_mjcf_body(inertial)
    assert restored.name == "upper_arm_left"


# --------------------------------------------------------------------------- #
# Failure modes.                                                              #
# --------------------------------------------------------------------------- #
def test_reader_rejects_unknown_element() -> None:
    with pytest.raises(ValueError, match="expected <body> or <inertial>"):
        read_mjcf_body(ET.Element("worldbody"))


def test_reader_rejects_body_without_inertial_child() -> None:
    with pytest.raises(ValueError, match="no <inertial> child"):
        read_mjcf_body(ET.Element("body", attrib={"name": "x"}))


def test_reader_rejects_missing_inertia_attributes() -> None:
    body = ET.Element("body", attrib={"name": "x"})
    ET.SubElement(body, "inertial", attrib={"pos": "0 0 0", "mass": "1.0"})
    with pytest.raises(ValueError, match="diaginertia"):
        read_mjcf_body(body)


def test_reader_rejects_both_inertia_forms() -> None:
    body = ET.Element("body", attrib={"name": "x"})
    ET.SubElement(
        body,
        "inertial",
        attrib={
            "pos": "0 0 0",
            "mass": "1.0",
            "diaginertia": "0.1 0.1 0.1",
            "fullinertia": "0.1 0.1 0.1 0 0 0",
        },
    )
    with pytest.raises(ValueError, match="must not declare both"):
        read_mjcf_body(body)


def test_reader_rejects_missing_required_attribute() -> None:
    body = ET.Element("body", attrib={"name": "x"})
    ET.SubElement(
        body,
        "inertial",
        attrib={"mass": "1.0", "diaginertia": "0.1 0.1 0.1"},
    )
    with pytest.raises(ValueError, match="missing required attribute 'pos'"):
        read_mjcf_body(body)


def test_reader_rejects_wrong_token_count() -> None:
    body = ET.Element("body", attrib={"name": "x"})
    ET.SubElement(
        body,
        "inertial",
        attrib={"pos": "0 0", "mass": "1.0", "diaginertia": "0.1 0.1 0.1"},
    )
    with pytest.raises(ValueError, match="must contain 3 floats"):
        read_mjcf_body(body)


def test_reader_rejects_non_float_token() -> None:
    body = ET.Element("body", attrib={"name": "x"})
    ET.SubElement(
        body,
        "inertial",
        attrib={"pos": "0 0 nope", "mass": "1.0", "diaginertia": "0.1 0.1 0.1"},
    )
    with pytest.raises(ValueError, match="non-float token"):
        read_mjcf_body(body)


def test_reader_rejects_non_float_mass() -> None:
    body = ET.Element("body", attrib={"name": "x"})
    ET.SubElement(
        body,
        "inertial",
        attrib={"pos": "0 0 0", "mass": "kg", "diaginertia": "0.1 0.1 0.1"},
    )
    with pytest.raises(ValueError, match="'mass' must be a float"):
        read_mjcf_body(body)


def test_reader_requires_name() -> None:
    body = ET.Element("body")
    ET.SubElement(
        body,
        "inertial",
        attrib={"pos": "0 0 0", "mass": "1.0", "diaginertia": "0.1 0.1 0.1"},
    )
    with pytest.raises(ValueError, match="no name attribute"):
        read_mjcf_body(body)


def test_reader_uses_ud_name_fallback() -> None:
    body = ET.Element("body")
    ET.SubElement(
        body,
        "inertial",
        attrib={
            "pos": "0 0 0",
            "mass": "1.0",
            "diaginertia": "0.1 0.1 0.1",
            "ud:name": "torso",
        },
    )
    props = read_mjcf_body(body)
    assert props.name == "torso"


def test_reader_rejects_invalid_optional_float() -> None:
    body = ET.Element("body", attrib={"name": "torso"})
    ET.SubElement(
        body,
        "inertial",
        attrib={
            "pos": "0 0 0",
            "mass": "1.0",
            "diaginertia": "0.1 0.1 0.1",
            "ud:length_m": "abc",
        },
    )
    with pytest.raises(ValueError, match="must be a float"):
        read_mjcf_body(body)


def test_reader_rejects_non_positive_optional() -> None:
    body = ET.Element("body", attrib={"name": "torso"})
    ET.SubElement(
        body,
        "inertial",
        attrib={
            "pos": "0 0 0",
            "mass": "1.0",
            "diaginertia": "0.1 0.1 0.1",
            "ud:length_m": "-0.3",
        },
    )
    with pytest.raises(ValueError, match="positive finite"):
        read_mjcf_body(body)


def test_writer_omits_marker_attrs_when_none() -> None:
    """ud:proximal_marker / ud:distal_marker absent when the field is None."""
    elem = write_mjcf_body(_full_props())
    inertial = elem.find("inertial")
    assert inertial is not None
    assert "ud:proximal_marker" not in inertial.attrib
    assert "ud:distal_marker" not in inertial.attrib


def test_writer_round_trip_with_markers() -> None:
    elem = write_mjcf_body(_diag_props())
    inertial = elem.find("inertial")
    assert inertial is not None
    assert inertial.attrib["ud:proximal_marker"] == "LSHO"
    assert inertial.attrib["ud:distal_marker"] == "LELB"
