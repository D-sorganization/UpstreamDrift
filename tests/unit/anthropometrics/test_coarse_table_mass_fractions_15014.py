"""Pin Dempster (1955) and Zatsiorsky-Seluyanov (1985) limb mass fractions (#15014).

The bundled JSON files store eight published segment *classes* per table.
:mod:`anthropometrics.estimators._humanoid_expansion` expands those classes onto
the 21-segment humanoid graph without erroneous renormalisation (the pre-#15036
bug assigned the whole trunk fraction to each trunk partition).

These tests assert bilateral limb segments recover the published class fractions
at de Leva (1996) male reference stature (1.741 m, 73 kg).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from anthropometrics.estimators import DempsterEstimator, ZatsiorskyEstimator

PUBLISHED_TOL = 1e-3
_REFERENCE_HEIGHT_M = 1.741
_REFERENCE_MASS_KG = 73.0

# Dempster (1955) WADC TR-55-159, Table III segment mass / total body mass.
_DEMPSTER_CLASS_MASS: dict[str, float] = {
    "head": 0.081,
    "trunk": 0.497,
    "upper_arm": 0.028,
    "forearm": 0.016,
    "hand": 0.006,
    "thigh": 0.10,
    "shank": 0.0465,
    "foot": 0.0145,
}

# Zatsiorsky & Seluyanov (1985) Biomechanics IX-B regression table (living
# subjects; not cadaver-derived). de Leva (1996) cites the adjusted set.
_ZATSIORSKY_CLASS_MASS: dict[str, float] = {
    "head": 0.0694,
    "trunk": 0.4346,
    "upper_arm": 0.0271,
    "forearm": 0.0162,
    "hand": 0.0061,
    "thigh": 0.1416,
    "shank": 0.0433,
    "foot": 0.0137,
}

_LIMB_ANATOMICAL = (
    ("left_upper_arm", "upper_arm"),
    ("left_forearm", "forearm"),
    ("left_hand", "hand"),
    ("left_thigh", "thigh"),
    ("left_shin", "shank"),
    ("left_foot", "foot"),
)


@pytest.mark.parametrize(
    ("anatomical", "class_id", "published_fraction"),
    [
        *((seg, cls, _DEMPSTER_CLASS_MASS[cls]) for seg, cls in _LIMB_ANATOMICAL),
    ],
    ids=[f"dempster-{cls}" for _, cls in _LIMB_ANATOMICAL],
)
def test_dempster_limb_mass_fractions_match_wadc_1955(
    anatomical: str, class_id: str, published_fraction: float
) -> None:
    record = DempsterEstimator().estimate(
        subject_id="dempster-mass-pin",
        height_m=_REFERENCE_HEIGHT_M,
        mass_kg=_REFERENCE_MASS_KG,
        sex="M",
    )
    by_name = dict(record.segments)
    assert by_name[anatomical].mass_kg == pytest.approx(
        _REFERENCE_MASS_KG * published_fraction, abs=PUBLISHED_TOL
    )


@pytest.mark.parametrize(
    ("anatomical", "class_id", "published_fraction"),
    [
        *((seg, cls, _ZATSIORSKY_CLASS_MASS[cls]) for seg, cls in _LIMB_ANATOMICAL),
    ],
    ids=[f"zatsiorsky-{cls}" for _, cls in _LIMB_ANATOMICAL],
)
def test_zatsiorsky_limb_mass_fractions_match_biomechanics_ix_b(
    anatomical: str, class_id: str, published_fraction: float
) -> None:
    record = ZatsiorskyEstimator().estimate(
        subject_id="zatsiorsky-mass-pin",
        height_m=_REFERENCE_HEIGHT_M,
        mass_kg=_REFERENCE_MASS_KG,
        sex="M",
    )
    by_name = dict(record.segments)
    assert by_name[anatomical].mass_kg == pytest.approx(
        _REFERENCE_MASS_KG * published_fraction, abs=PUBLISHED_TOL
    )


def test_dempster_total_mass_closes_without_renormalization() -> None:
    """Expansion must preserve input body mass (no 0.364 scale factor)."""
    record = DempsterEstimator().estimate(
        subject_id="dempster-total",
        height_m=_REFERENCE_HEIGHT_M,
        mass_kg=_REFERENCE_MASS_KG,
        sex="M",
    )
    total = sum(props.mass_kg for _, props in record.segments)
    assert total == pytest.approx(_REFERENCE_MASS_KG, abs=PUBLISHED_TOL)
