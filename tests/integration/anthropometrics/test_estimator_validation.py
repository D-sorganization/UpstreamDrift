"""Validate each estimator against PUBLISHED canonical ratio tables.

Issue #4819 -- for each estimator wrapper, assert that the values it
emits for a published reference subject match the original paper's
table within a tight tolerance. The hard-coded constants below come
straight from the source publications and are cited inline.

The unit tier for each estimator already covers shape and
construction invariants. This integration suite locks the *numeric
contract*: if anyone edits the bundled ratio JSON or the de Leva
dataclass and the published values drift, this file fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from anthropometrics.estimators import (
    DeLevaEstimator,
    DempsterEstimator,
    ZatsiorskyEstimator,
)

# Published reference subjects: de Leva (1996) Table 4 (men, n=100,
# Italian athletes, mean height 1.741 m, mean mass 73.0 kg) and
# Table 5 (women, n=15, mean height 1.735 m, mean mass 61.9 kg).
_DE_LEVA_MALE_HEIGHT_M = 1.741
_DE_LEVA_MALE_MASS_KG = 73.0
_DE_LEVA_FEMALE_HEIGHT_M = 1.735
_DE_LEVA_FEMALE_MASS_KG = 61.9

PUBLISHED_TOL = 1e-3


def _segment_dict(record):
    """Return ``{name: SegmentProperties}`` for ergonomic lookup."""
    return dict(record.segments)


# de Leva 1996 -- Table 4 (men) and Table 5 (women).
@pytest.mark.parametrize(
    ("segment", "mass_ratio", "length_ratio", "com_proximal_ratio"),
    [
        # de Leva (1996), J. Biomech. 29(9), Table 4, male reference.
        ("upper_arm", 0.0271, 0.186, 0.5772),
        ("forearm", 0.0162, 0.146, 0.4574),
        ("thigh", 0.1416, 0.245, 0.4095),
        ("shin", 0.0433, 0.246, 0.4459),
    ],
)
def test_de_leva_male_published_ratios(
    segment: str,
    mass_ratio: float,
    length_ratio: float,
    com_proximal_ratio: float,
) -> None:
    """De Leva male reference values reproduce within 1e-3.

    Citation: de Leva (1996), J. Biomech. 29(9), Table 4. The check
    is on the *derived* segment, not the raw ratio table -- so it
    transitively verifies the estimator algebra as well.
    """
    record = DeLevaEstimator().estimate(
        subject_id="dl_male",
        height_m=_DE_LEVA_MALE_HEIGHT_M,
        mass_kg=_DE_LEVA_MALE_MASS_KG,
        sex="M",
    )
    seg = _segment_dict(record)[f"left_{segment}"]
    expected_length = _DE_LEVA_MALE_HEIGHT_M * length_ratio
    assert seg.length_m == pytest.approx(expected_length, abs=PUBLISHED_TOL)
    assert seg.com_xyz_m[2] == pytest.approx(
        expected_length * com_proximal_ratio, abs=PUBLISHED_TOL
    )
    # Mass is renormalised so the per-subject sum closes -- but the
    # ratio between two segments still equals the published ratio.
    ref_seg = _segment_dict(record)["left_thigh"]
    ref_ratio = 0.1416  # de Leva (1996) Table 4.
    measured_ratio = seg.mass_kg / ref_seg.mass_kg
    expected_ratio = mass_ratio / ref_ratio
    assert measured_ratio == pytest.approx(expected_ratio, rel=PUBLISHED_TOL)


def test_de_leva_female_pelvis_published() -> None:
    """De Leva (1996) Table 5 (women) -- pelvis ratios reproduce."""
    record = DeLevaEstimator().estimate(
        subject_id="dl_female",
        height_m=_DE_LEVA_FEMALE_HEIGHT_M,
        mass_kg=_DE_LEVA_FEMALE_MASS_KG,
        sex="F",
    )
    seg = _segment_dict(record)["pelvis"]
    # Published female pelvis length ratio = 0.078 (de Leva 1996 T5).
    assert seg.length_m == pytest.approx(
        _DE_LEVA_FEMALE_HEIGHT_M * 0.078, abs=PUBLISHED_TOL
    )


# Dempster 1955 -- WADC TR-55-159, segment ratio table.
@pytest.mark.parametrize(
    ("anatomical", "length_ratio", "com_proximal_ratio"),
    [
        # Dempster (1955), WADC TR-55-159 segment ratio table.
        ("left_upper_arm", 0.186, 0.436),
        ("left_forearm", 0.146, 0.430),
        ("left_thigh", 0.245, 0.433),
        ("left_shin", 0.246, 0.433),
        ("left_foot", 0.152, 0.429),
    ],
)
def test_dempster_published_ratios(
    anatomical: str,
    length_ratio: float,
    com_proximal_ratio: float,
) -> None:
    """Reproduce Dempster (1955) length and CoM ratios within 1e-3."""
    height_m = 1.78
    mass_kg = 75.0
    record = DempsterEstimator().estimate(
        subject_id="dempster_ref",
        height_m=height_m,
        mass_kg=mass_kg,
        sex="M",
    )
    seg = _segment_dict(record)[anatomical]
    assert seg.length_m == pytest.approx(height_m * length_ratio, abs=PUBLISHED_TOL)
    assert seg.com_xyz_m[2] == pytest.approx(
        seg.length_m * com_proximal_ratio, abs=PUBLISHED_TOL
    )


def test_dempster_gyration_radii_diagonal_inertia() -> None:
    """Dempster (1955) thigh gyration radii produce expected inertia.

    Published radii (k_x = k_y = 0.323, k_z = 0.149); the derived
    inertia is m * (k * L)^2 along each principal axis.
    """
    height_m = 1.78
    mass_kg = 75.0
    record = DempsterEstimator().estimate(
        subject_id="dempster_gyr",
        height_m=height_m,
        mass_kg=mass_kg,
        sex="M",
    )
    thigh = _segment_dict(record)["left_thigh"]
    length = thigh.length_m
    mass = thigh.mass_kg
    expected_xx = mass * (0.323 * length) ** 2
    expected_zz = mass * (0.149 * length) ** 2
    assert thigh.inertia_tensor[0, 0] == pytest.approx(expected_xx, rel=PUBLISHED_TOL)
    assert thigh.inertia_tensor[1, 1] == pytest.approx(expected_xx, rel=PUBLISHED_TOL)
    assert thigh.inertia_tensor[2, 2] == pytest.approx(expected_zz, rel=PUBLISHED_TOL)
    np.testing.assert_allclose(
        thigh.inertia_tensor - np.diag(np.diag(thigh.inertia_tensor)),
        np.zeros((3, 3)),
        atol=1e-12,
    )


# Zatsiorsky-Seluyanov 1985 -- Biomechanics IX-B regression table.
@pytest.mark.parametrize(
    ("anatomical", "length_ratio", "com_proximal_ratio"),
    [
        # Zatsiorsky & Seluyanov (1985), Biomechanics IX-B p. 233.
        ("left_upper_arm", 0.186, 0.5772),
        ("left_forearm", 0.146, 0.4574),
        ("left_thigh", 0.245, 0.4095),
        ("left_shin", 0.246, 0.4459),
    ],
)
def test_zatsiorsky_published_ratios(
    anatomical: str,
    length_ratio: float,
    com_proximal_ratio: float,
) -> None:
    """Reproduce Zatsiorsky-Seluyanov (1985) ratios within 1e-3."""
    height_m = 1.74
    mass_kg = 73.0
    record = ZatsiorskyEstimator().estimate(
        subject_id="zat_ref",
        height_m=height_m,
        mass_kg=mass_kg,
        sex="M",
    )
    seg = _segment_dict(record)[anatomical]
    assert seg.length_m == pytest.approx(height_m * length_ratio, abs=PUBLISHED_TOL)
    assert seg.com_xyz_m[2] == pytest.approx(
        seg.length_m * com_proximal_ratio, abs=PUBLISHED_TOL
    )


# Mass closure -- every estimator must conserve total subject mass.
@pytest.mark.parametrize(
    "estimator_cls",
    [DeLevaEstimator, DempsterEstimator, ZatsiorskyEstimator],
)
def test_total_segment_mass_closes_to_subject_mass(estimator_cls) -> None:
    """Sum of segment masses equals subject mass within 1%."""
    mass_kg = 80.0
    record = estimator_cls().estimate(
        subject_id="mass_closure",
        height_m=1.80,
        mass_kg=mass_kg,
        sex="M",
    )
    total = sum(props.mass_kg for _, props in record.segments)
    assert total == pytest.approx(mass_kg, rel=1e-2)


@pytest.mark.parametrize(
    "estimator_cls",
    [DeLevaEstimator, DempsterEstimator, ZatsiorskyEstimator],
)
def test_inertia_eigenvalues_strictly_positive(estimator_cls) -> None:
    """Every emitted inertia tensor must have strictly positive eigenvalues."""
    record = estimator_cls().estimate(
        subject_id="eigtest",
        height_m=1.75,
        mass_kg=70.0,
        sex="M",
    )
    for _name, props in record.segments:
        eigs = np.linalg.eigvalsh(props.inertia_tensor)
        assert np.all(eigs > 0), f"non-positive eigenvalues on {props.name}: {eigs}"
