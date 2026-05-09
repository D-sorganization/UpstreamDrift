"""End-to-end pipeline tests for the anthropometrics package.

Issue #4819 -- synthetic subject -> estimator -> XML adapter ->
reload must reconstruct a :class:`SubjectAnthropometrics`
indistinguishable from the source. Also exercises the optional
C3D subject-info reader when ``data/C3D_TA_Driver.c3d`` is present
and ``ezc3d`` is installed.
"""

from __future__ import annotations

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
from anthropometrics.estimators import (
    DeLevaEstimator,
    DempsterEstimator,
    ZatsiorskyEstimator,
)
from anthropometrics.readers import read_osim_body, read_urdf_inertial
from anthropometrics.writers import write_osim_body, write_urdf_inertial

_REPO_ROOT = Path(__file__).resolve().parents[3]
_C3D_FIXTURE = _REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


def _rebuild_via_urdf(record: SubjectAnthropometrics) -> SubjectAnthropometrics:
    """Round-trip *record* segment-by-segment through URDF and rebuild."""
    rebuilt: list[tuple[str, SegmentProperties]] = []
    for name, props in record.segments:
        xml_text = ET.tostring(write_urdf_inertial(props))
        recovered = read_urdf_inertial(
            ET.fromstring(xml_text),
            name=props.name,
            body_part_id=props.body_part_id,
            length_m=props.length_m,
            source_method=props.source_method,
            source_subject_height_m=props.source_subject_height_m,
            source_subject_mass_kg=props.source_subject_mass_kg,
        )
        rebuilt.append((name, recovered))
    return SubjectAnthropometrics(
        subject_id=record.subject_id,
        height_m=record.height_m,
        mass_kg=record.mass_kg,
        segments=tuple(rebuilt),
        source_method=record.source_method,
        age_years=record.age_years,
        sex=record.sex,
    )


def _rebuild_via_osim(record: SubjectAnthropometrics) -> SubjectAnthropometrics:
    """Round-trip *record* through OpenSim ``<Body>`` XML."""
    rebuilt: list[tuple[str, SegmentProperties]] = []
    for name, props in record.segments:
        xml_text = ET.tostring(write_osim_body(props))
        recovered = read_osim_body(ET.fromstring(xml_text))
        rebuilt.append((name, recovered))
    return SubjectAnthropometrics(
        subject_id=record.subject_id,
        height_m=record.height_m,
        mass_kg=record.mass_kg,
        segments=tuple(rebuilt),
        source_method=record.source_method,
        age_years=record.age_years,
        sex=record.sex,
    )


@pytest.mark.parametrize(
    "estimator_cls",
    [DeLevaEstimator, DempsterEstimator, ZatsiorskyEstimator],
)
def test_synthetic_subject_through_urdf_roundtrip(estimator_cls) -> None:
    """Estimator -> URDF -> reload preserves every segment exactly."""
    source = estimator_cls().estimate(
        subject_id="e2e_urdf",
        height_m=1.80,
        mass_kg=72.5,
        sex="M",
    )
    rebuilt = _rebuild_via_urdf(source)
    for (n0, p0), (n1, p1) in zip(source.segments, rebuilt.segments, strict=True):
        assert n0 == n1
        assert p1.mass_kg == pytest.approx(p0.mass_kg, rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(
            p1.inertia_tensor, p0.inertia_tensor, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(p1.com_xyz_m, p0.com_xyz_m, rtol=1e-9, atol=1e-12)


def test_synthetic_subject_through_osim_then_json(tmp_path: Path) -> None:
    """Full estimator -> osim XML -> json -> load chain is lossless."""
    source = DeLevaEstimator().estimate(
        subject_id="e2e_osim_json",
        height_m=1.74,
        mass_kg=68.0,
        sex="F",
    )
    via_osim = _rebuild_via_osim(source)
    out = tmp_path / "rebuilt.json"
    save_subject(via_osim, out)
    loaded = load_subject(out)

    assert loaded.subject_id == source.subject_id
    assert loaded.sex == source.sex
    assert loaded.height_m == pytest.approx(source.height_m, rel=1e-9)
    for (sn, sp), (ln, lp) in zip(source.segments, loaded.segments, strict=True):
        assert sn == ln
        assert lp.mass_kg == pytest.approx(sp.mass_kg, rel=1e-9, abs=1e-12)
        np.testing.assert_allclose(
            lp.inertia_tensor, sp.inertia_tensor, rtol=1e-9, atol=1e-12
        )


def test_pipeline_inertia_is_physically_realisable() -> None:
    """All eigenvalues positive and triangle inequality holds throughout."""
    record = ZatsiorskyEstimator().estimate(
        subject_id="phys",
        height_m=1.80,
        mass_kg=85.0,
        sex="M",
    )
    rebuilt = _rebuild_via_urdf(record)
    for _name, props in rebuilt.segments:
        eigs = np.linalg.eigvalsh(props.inertia_tensor)
        assert np.all(eigs > 0), f"non-positive eigenvalues: {eigs}"
        ix, iy, iz = sorted(float(e) for e in eigs)
        # Triangle inequality on principal moments -- a hard physical
        # constraint enforced by the SegmentProperties contract.
        assert ix + iy + 1e-9 >= iz


def test_pipeline_mass_closure_preserved_through_urdf() -> None:
    """Sum of segment masses is preserved through the URDF round-trip."""
    record = DeLevaEstimator().estimate(
        subject_id="mass_through_urdf",
        height_m=1.78,
        mass_kg=80.0,
        sex="M",
    )
    pre = sum(p.mass_kg for _, p in record.segments)
    rebuilt = _rebuild_via_urdf(record)
    post = sum(p.mass_kg for _, p in rebuilt.segments)
    assert post == pytest.approx(pre, rel=1e-12, abs=1e-12)
    assert post == pytest.approx(80.0, rel=1e-2)


def test_c3d_pipeline_against_bundled_fixture() -> None:
    """End-to-end against ``data/C3D_TA_Driver.c3d`` when available.

    Reads subject metadata from the bundled C3D, drives the de Leva
    estimator with it, and asserts mass-closure plus positive
    inertia eigenvalues across every segment.
    """
    if not _C3D_FIXTURE.exists():
        pytest.skip(f"C3D fixture not bundled: {_C3D_FIXTURE}")
    pytest.importorskip("ezc3d", reason="C3D pipeline test needs ezc3d")
    from anthropometrics import read_c3d_subject_metadata

    meta = read_c3d_subject_metadata(_C3D_FIXTURE)
    height = meta.height_m if meta.height_m is not None else 1.78
    mass = meta.mass_kg if meta.mass_kg is not None else 75.0
    sex_value = meta.sex.value if meta.sex is not None else "unspecified"
    record = DeLevaEstimator().estimate(
        subject_id=meta.subject_id or "c3d_subject",
        height_m=height,
        mass_kg=mass,
        sex=sex_value,
        age_years=meta.age_years,
    )
    total_mass = sum(p.mass_kg for _, p in record.segments)
    assert total_mass == pytest.approx(mass, rel=1e-2)
    for _, props in record.segments:
        eigs = np.linalg.eigvalsh(props.inertia_tensor)
        assert np.all(eigs > 0)
