"""Unit tests for ``anthropometrics.{readers,writers}.osim_body``.

Coverage targets:

* Round-trip identity for representative ``SegmentProperties``
  fixtures (with and without optional marker fields, with a
  non-diagonal inertia tensor).
* Direct serialisation shape — every required OpenSim child
  element is present with the documented contents.
* Error handling — wrong root tag, missing required children,
  malformed numeric content, missing ``UDMetadata`` block, and
  non-:class:`SegmentProperties` argument to the writer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

from anthropometrics import SegmentProperties
from anthropometrics.readers.osim_body import read_osim_body
from anthropometrics.writers.osim_body import write_osim_body


# --------------------------------------------------------------------------- #
# Fixture helpers.                                                            #
# --------------------------------------------------------------------------- #
def _diag_inertia(ix: float, iy: float, iz: float) -> np.ndarray:
    """Return a diagonal inertia tensor with the given principal moments."""
    return np.diag([ix, iy, iz]).astype(float)


def _make_segment(**overrides: Any) -> SegmentProperties:
    """Return a default-valid :class:`SegmentProperties`, with overrides."""
    defaults: dict[str, Any] = {
        "name": "upper_arm_left",
        "body_part_id": "upper_arm",
        "length_m": 0.30,
        "proximal_marker": "L_SHO",
        "distal_marker": "L_ELB",
        "mass_kg": 2.0,
        "com_xyz_m": np.array([0.15, 0.0, 0.0]),
        "inertia_tensor": _diag_inertia(0.02, 0.02, 0.005),
        "source_method": "de_leva",
        "source_subject_height_m": 1.80,
        "source_subject_mass_kg": 75.0,
    }
    defaults.update(overrides)
    return SegmentProperties(**defaults)


def _assert_segments_equal(a: SegmentProperties, b: SegmentProperties) -> None:
    """Compare two SegmentProperties at rtol=1e-9, atol=1e-12."""
    assert a.name == b.name
    assert a.body_part_id == b.body_part_id
    assert a.proximal_marker == b.proximal_marker
    assert a.distal_marker == b.distal_marker
    assert a.source_method == b.source_method
    assert np.isclose(a.length_m, b.length_m, rtol=1e-9, atol=1e-12)
    assert np.isclose(a.mass_kg, b.mass_kg, rtol=1e-9, atol=1e-12)
    assert np.isclose(
        a.source_subject_height_m,
        b.source_subject_height_m,
        rtol=1e-9,
        atol=1e-12,
    )
    assert np.isclose(
        a.source_subject_mass_kg,
        b.source_subject_mass_kg,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(a.com_xyz_m, b.com_xyz_m, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        a.inertia_tensor, b.inertia_tensor, rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# Round-trip.                                                                 #
# --------------------------------------------------------------------------- #
def test_round_trip_diagonal_inertia() -> None:
    """write -> read on the canonical fixture is bit-identical."""
    original = _make_segment()
    elem = write_osim_body(original)
    recovered = read_osim_body(elem)
    _assert_segments_equal(original, recovered)


def test_round_trip_without_markers() -> None:
    """Optional marker fields stay ``None`` after a write/read round-trip."""
    original = _make_segment(proximal_marker=None, distal_marker=None)
    elem = write_osim_body(original)
    recovered = read_osim_body(elem)
    assert recovered.proximal_marker is None
    assert recovered.distal_marker is None
    _assert_segments_equal(original, recovered)


def test_round_trip_full_inertia_tensor() -> None:
    """A non-diagonal but physically valid tensor round-trips exactly."""
    base = _diag_inertia(0.05, 0.04, 0.03)
    off = np.array(
        [
            [0.0, 1.0e-4, 2.0e-4],
            [1.0e-4, 0.0, 3.0e-4],
            [2.0e-4, 3.0e-4, 0.0],
        ]
    )
    inertia = base + off
    com = np.array([0.05, -0.02, 0.01])
    original = _make_segment(
        name="torso",
        body_part_id="torso",
        length_m=0.5,
        proximal_marker="C7",
        distal_marker="L5",
        mass_kg=25.0,
        com_xyz_m=com,
        inertia_tensor=inertia,
    )
    elem = write_osim_body(original)
    recovered = read_osim_body(elem)
    _assert_segments_equal(original, recovered)


# --------------------------------------------------------------------------- #
# Writer — element shape.                                                     #
# --------------------------------------------------------------------------- #
def test_writer_emits_required_opensim_elements() -> None:
    """Verify the OpenSim-native fragment matches the documented schema."""
    seg = _make_segment(
        mass_kg=1.5,
        com_xyz_m=np.array([0.1, 0.2, 0.3]),
        inertia_tensor=_diag_inertia(0.011, 0.012, 0.013),
    )
    elem = write_osim_body(seg)

    assert elem.tag == "Body"
    assert elem.attrib == {"name": "upper_arm_left"}

    mass = elem.find("mass")
    assert mass is not None
    assert mass.text is not None and float(mass.text) == pytest.approx(1.5)

    mass_center = elem.find("mass_center")
    assert mass_center is not None
    assert mass_center.text is not None
    assert [float(t) for t in mass_center.text.split()] == [0.1, 0.2, 0.3]

    inertia = elem.find("inertia")
    assert inertia is not None
    assert inertia.text is not None
    parsed = [float(t) for t in inertia.text.split()]
    assert parsed == [0.011, 0.012, 0.013, 0.0, 0.0, 0.0]


def test_writer_omits_marker_subelements_when_none() -> None:
    """``UDMetadata`` skips optional marker tags when their values are ``None``."""
    seg = _make_segment(proximal_marker=None, distal_marker=None)
    elem = write_osim_body(seg)
    metadata = elem.find("UDMetadata")
    assert metadata is not None
    assert metadata.find("proximal_marker") is None
    assert metadata.find("distal_marker") is None


def test_writer_rejects_non_segment_properties() -> None:
    """The writer is type-checked at runtime."""
    with pytest.raises(TypeError, match="SegmentProperties"):
        write_osim_body("not-a-segment")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Reader — error paths.                                                       #
# --------------------------------------------------------------------------- #
def test_reader_rejects_non_element() -> None:
    """Non-Element input raises ``TypeError`` with a clear message."""
    with pytest.raises(TypeError, match="Element"):
        read_osim_body("<Body/>")  # type: ignore[arg-type]


def test_reader_rejects_wrong_root_tag() -> None:
    """Reader fails fast when the root tag is not ``<Body>``."""
    bad = ET.Element("Joint", attrib={"name": "elbow"})
    with pytest.raises(ValueError, match="<Joint>"):
        read_osim_body(bad)


def test_reader_rejects_missing_name_attribute() -> None:
    """Reader rejects a ``<Body>`` without a populated ``name`` attribute."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    elem.attrib.pop("name")
    with pytest.raises(ValueError, match="name"):
        read_osim_body(elem)


@pytest.mark.parametrize("missing_tag", ["mass", "mass_center", "inertia"])
def test_reader_rejects_missing_opensim_child(missing_tag: str) -> None:
    """Each required OpenSim child triggers a clear error when absent."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    target = elem.find(missing_tag)
    assert target is not None
    elem.remove(target)
    with pytest.raises(ValueError, match=missing_tag):
        read_osim_body(elem)


def test_reader_rejects_missing_metadata_block() -> None:
    """The ``UDMetadata`` block is mandatory for round-trip restoration."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    metadata = elem.find("UDMetadata")
    assert metadata is not None
    elem.remove(metadata)
    with pytest.raises(ValueError, match="UDMetadata"):
        read_osim_body(elem)


@pytest.mark.parametrize(
    "missing_tag",
    [
        "body_part_id",
        "length_m",
        "source_method",
        "source_subject_height_m",
        "source_subject_mass_kg",
    ],
)
def test_reader_rejects_missing_metadata_field(missing_tag: str) -> None:
    """Required UDMetadata children each surface a tag-specific error."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    metadata = elem.find("UDMetadata")
    assert metadata is not None
    target = metadata.find(missing_tag)
    assert target is not None
    metadata.remove(target)
    with pytest.raises(ValueError, match=missing_tag):
        read_osim_body(elem)


def test_reader_rejects_inertia_with_wrong_token_count() -> None:
    """OpenSim ``<inertia>`` must contain exactly six space-separated floats."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    inertia = elem.find("inertia")
    assert inertia is not None
    inertia.text = "0.1 0.2 0.3"  # only three tokens
    with pytest.raises(ValueError, match="expected 6 floats"):
        read_osim_body(elem)


def test_reader_rejects_mass_center_with_wrong_token_count() -> None:
    """``<mass_center>`` must contain exactly three floats."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    mc = elem.find("mass_center")
    assert mc is not None
    mc.text = "1.0 2.0"  # only two tokens
    with pytest.raises(ValueError, match="expected 3 floats"):
        read_osim_body(elem)


def test_reader_rejects_non_numeric_mass() -> None:
    """Non-numeric content in a numeric field raises a clear error."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    mass = elem.find("mass")
    assert mass is not None
    mass.text = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric"):
        read_osim_body(elem)


def test_reader_rejects_non_finite_mass() -> None:
    """``inf`` or ``nan`` in a numeric field raises a clear error."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    mass = elem.find("mass")
    assert mass is not None
    mass.text = "inf"
    with pytest.raises(ValueError, match="non-finite"):
        read_osim_body(elem)


def test_reader_rejects_empty_text_field() -> None:
    """Empty text in a required scalar element raises ``ValueError``."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    mass = elem.find("mass")
    assert mass is not None
    mass.text = "   "
    with pytest.raises(ValueError, match="empty text"):
        read_osim_body(elem)


def test_reader_rejects_scalar_with_extra_tokens() -> None:
    """A scalar tag must contain exactly one float."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    mass = elem.find("mass")
    assert mass is not None
    mass.text = "1.0 2.0"
    with pytest.raises(ValueError, match="one float"):
        read_osim_body(elem)


def test_reader_rejects_empty_self_closed_element() -> None:
    """A self-closed numeric element (``<mass/>``) reports no text content."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    mass = elem.find("mass")
    assert mass is not None
    mass.text = None
    with pytest.raises(ValueError, match="no text content"):
        read_osim_body(elem)


def test_reader_handles_optional_marker_text_blank() -> None:
    """A blank optional marker tag is treated identically to absence."""
    seg = _make_segment()
    elem = write_osim_body(seg)
    metadata = elem.find("UDMetadata")
    assert metadata is not None
    proximal = metadata.find("proximal_marker")
    assert proximal is not None
    proximal.text = "   "
    recovered = read_osim_body(elem)
    assert recovered.proximal_marker is None
