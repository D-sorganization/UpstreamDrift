"""Cross-engine round-trip tests for anthropometric adapters.

Issue #4819 -- every available adapter (URDF, OpenSim ``<Body>``,
JSON persistence) must round-trip a canonical
:class:`SubjectAnthropometrics` losslessly. The MJCF adapter is
not yet on ``main``; the test guards itself behind an ``importlib``
probe so it activates automatically once the writer / reader land.
"""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from anthropometrics import (
    SegmentProperties,
    SubjectAnthropometrics,
    load_subject,
    save_subject,
)
from anthropometrics.estimators import DeLevaEstimator
from anthropometrics.readers import read_osim_body, read_urdf_inertial
from anthropometrics.writers import write_osim_body, write_urdf_inertial

_RTOL = 1e-9
_ATOL = 1e-12


@pytest.fixture(scope="module")
def reference_subject() -> SubjectAnthropometrics:
    """A realistic male subject; covers head + every limb segment."""
    return DeLevaEstimator().estimate(
        subject_id="roundtrip_subject",
        height_m=1.78,
        mass_kg=75.0,
        sex="M",
        age_years=32.0,
    )


def _assert_segments_close(
    expected: SegmentProperties, actual: SegmentProperties
) -> None:
    """Assert the numeric fields of two segments match within tolerance."""
    assert actual.mass_kg == pytest.approx(expected.mass_kg, rel=_RTOL, abs=_ATOL)
    assert actual.length_m == pytest.approx(expected.length_m, rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        actual.com_xyz_m, expected.com_xyz_m, rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        actual.inertia_tensor, expected.inertia_tensor, rtol=_RTOL, atol=_ATOL
    )


def test_urdf_round_trip_every_segment(
    reference_subject: SubjectAnthropometrics,
) -> None:
    """URDF write -> read recovers identical inertials within 1e-9."""
    for name, props in reference_subject.segments:
        elem = write_urdf_inertial(props)
        roundtripped = ET.fromstring(ET.tostring(elem))
        recovered = read_urdf_inertial(
            roundtripped,
            name=props.name,
            body_part_id=props.body_part_id,
            length_m=props.length_m,
            source_method=props.source_method,
            source_subject_height_m=props.source_subject_height_m,
            source_subject_mass_kg=props.source_subject_mass_kg,
        )
        _assert_segments_close(props, recovered)
        assert recovered.name == name


def test_osim_round_trip_every_segment(
    reference_subject: SubjectAnthropometrics,
) -> None:
    """OpenSim ``<Body>`` write -> read recovers identical metadata."""
    for _name, props in reference_subject.segments:
        elem = write_osim_body(props)
        recovered = read_osim_body(ET.fromstring(ET.tostring(elem)))
        _assert_segments_close(props, recovered)
        assert recovered.name == props.name
        assert recovered.body_part_id == props.body_part_id
        assert recovered.source_method == props.source_method


def test_json_persistence_round_trip(
    reference_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """JSON save -> load recovers an identical SubjectAnthropometrics."""
    out = tmp_path / "subject.json"
    save_subject(reference_subject, out)
    loaded = load_subject(out)

    assert loaded.subject_id == reference_subject.subject_id
    assert loaded.source_method == reference_subject.source_method
    assert loaded.sex == reference_subject.sex
    assert loaded.age_years == reference_subject.age_years
    assert loaded.height_m == pytest.approx(
        reference_subject.height_m, rel=_RTOL, abs=_ATOL
    )
    assert loaded.mass_kg == pytest.approx(
        reference_subject.mass_kg, rel=_RTOL, abs=_ATOL
    )
    assert len(loaded.segments) == len(reference_subject.segments)
    for (lname, lprops), (ename, eprops) in zip(
        loaded.segments, reference_subject.segments, strict=True
    ):
        assert lname == ename
        _assert_segments_close(eprops, lprops)


def _try_import_mjcf_pair():
    """Return ``(write, read)`` for the MJCF adapter, or ``None`` if absent.

    Only a *missing* adapter module returns ``None``. If the adapter module
    exists but fails to import for some other reason (e.g. an internal import
    typo or a missing dependency inside the adapter itself), the original
    error propagates so the round-trip coverage cannot silently regress.
    """
    try:
        writer_mod = importlib.import_module("anthropometrics.writers.mjcf_body")
    except ModuleNotFoundError as exc:
        # Only skip when *this* adapter module is the missing one. A
        # ModuleNotFoundError raised from inside the adapter (i.e. its own
        # imports) must surface as a real failure.
        if exc.name in {
            "anthropometrics.writers.mjcf_body",
            "anthropometrics.writers",
            "anthropometrics",
        }:
            return None
        raise
    try:
        reader_mod = importlib.import_module("anthropometrics.readers.mjcf_body")
    except ModuleNotFoundError as exc:
        if exc.name in {
            "anthropometrics.readers.mjcf_body",
            "anthropometrics.readers",
            "anthropometrics",
        }:
            return None
        raise
    write = getattr(writer_mod, "write_mjcf_body", None)
    read = getattr(reader_mod, "read_mjcf_body", None)
    if write is None or read is None:
        return None
    return write, read


def test_mjcf_round_trip_when_adapter_available(
    reference_subject: SubjectAnthropometrics,
) -> None:
    """If an MJCF adapter ships, exercise the same round-trip contract."""
    pair = _try_import_mjcf_pair()
    if pair is None:
        pytest.skip("MJCF adapter not present on this branch")
    write, read = pair
    for _name, props in reference_subject.segments:
        elem = write(props)
        recovered = read(elem)
        _assert_segments_close(props, recovered)


def test_urdf_rejects_missing_mass(reference_subject: SubjectAnthropometrics) -> None:
    """A URDF ``<inertial>`` missing ``<mass>`` raises ``ValueError``."""
    _, props = reference_subject.segments[0]
    elem = write_urdf_inertial(props)
    elem.remove(elem.find("mass"))
    with pytest.raises(ValueError, match="mass"):
        read_urdf_inertial(elem)
