"""ANSUR-backed hand size prior (#13760)."""

from __future__ import annotations

import math

import pytest

from src.shared.python.anthropometrics._types import Sex
from src.shared.python import hand_size

# ANSUR II public CSVs (handlength/stature, handbreadth/handlength), per-subject
# ratios. Datasets sha256 0547aea0... (male n=4082) and ed7e800a... (female n=1986).
# Values recomputed 2026-10-25 from the byte-identical GitHub mirror of PSU files.
ANSUR_II_MALE_HL_ST_MEAN = 0.1101
ANSUR_II_MALE_HL_ST_SD = 0.0042
ANSUR_II_FEMALE_HL_ST_MEAN = 0.1112
ANSUR_II_FEMALE_HL_ST_SD = 0.0047
ANSUR_II_MALE_BL_LEN_MEAN = 0.4572
ANSUR_II_MALE_BL_LEN_SD = 0.0211
ANSUR_II_FEMALE_BL_LEN_MEAN = 0.4322
ANSUR_II_FEMALE_BL_LEN_SD = 0.0210
ANSUR_II_MALE_BL_LEN_P5 = 0.4233
ANSUR_II_MALE_BL_LEN_P95 = 0.4921
ANSUR_II_FEMALE_BL_LEN_P5 = 0.3977
ANSUR_II_FEMALE_BL_LEN_P95 = 0.4671


@pytest.mark.unit
def test_hand_length_fractions_match_the_ansur_ii_means() -> None:
    assert (
        pytest.approx(ANSUR_II_MALE_HL_ST_MEAN, abs=5e-4)
        == hand_size.HAND_LENGTH_STATURE_FRACTION_MALE
    )
    assert (
        pytest.approx(ANSUR_II_FEMALE_HL_ST_MEAN, abs=5e-4)
        == hand_size.HAND_LENGTH_STATURE_FRACTION_FEMALE
    )
    assert (
        abs(hand_size.HAND_LENGTH_STATURE_FRACTION_MALE - ANSUR_II_MALE_HL_ST_MEAN)
        < ANSUR_II_MALE_HL_ST_SD
    )
    assert (
        abs(hand_size.HAND_LENGTH_STATURE_FRACTION_FEMALE - ANSUR_II_FEMALE_HL_ST_MEAN)
        < ANSUR_II_FEMALE_HL_ST_SD
    )


@pytest.mark.unit
def test_breadth_over_length_matches_the_ansur_ii_means() -> None:
    assert (
        pytest.approx(ANSUR_II_MALE_BL_LEN_MEAN, abs=5e-4)
        == hand_size.BREADTH_OVER_LENGTH_MALE
    )
    assert (
        pytest.approx(ANSUR_II_FEMALE_BL_LEN_MEAN, abs=5e-4)
        == hand_size.BREADTH_OVER_LENGTH_FEMALE
    )
    assert (
        abs(hand_size.BREADTH_OVER_LENGTH_MALE - ANSUR_II_MALE_BL_LEN_MEAN)
        < ANSUR_II_MALE_BL_LEN_SD
    )
    assert (
        abs(hand_size.BREADTH_OVER_LENGTH_FEMALE - ANSUR_II_FEMALE_BL_LEN_MEAN)
        < ANSUR_II_FEMALE_BL_LEN_SD
    )
    lo, hi = hand_size.BREADTH_LENGTH_RATIO_RANGE
    assert math.isclose(lo, 0.4089, abs_tol=5e-4)
    assert math.isclose(hi, 0.4888, abs_tol=5e-4)


@pytest.mark.unit
def test_resolve_hand_length_and_breadth_from_stature() -> None:
    stature = 1.80
    length_m = hand_size.resolve_hand_length_m(stature, Sex.MALE.value)
    breadth_m = hand_size.resolve_hand_breadth_m(length_m, Sex.MALE.value)
    assert length_m == pytest.approx(
        stature * hand_size.HAND_LENGTH_STATURE_FRACTION_MALE
    )
    assert breadth_m == pytest.approx(length_m * hand_size.BREADTH_OVER_LENGTH_MALE)


@pytest.mark.unit
def test_explicit_overrides_preserved_and_validated() -> None:
    length = hand_size.resolve_hand_length_m(1.75, override_m=0.20)
    breadth = hand_size.resolve_hand_breadth_m(length, override_m=0.09)
    assert length == 0.20
    assert breadth == 0.09


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, float("nan"), 0.13, 0.25])
def test_hand_length_dbc_rejects_invalid(bad: float | bool) -> None:
    with pytest.raises((TypeError, ValueError)):
        hand_size.validate_hand_length_m(bad)


@pytest.mark.unit
def test_physics_constants_hand_length_matches_reference() -> None:
    from src.shared.python.core.physics_constants import HAND_LENGTH_M

    assert float(HAND_LENGTH_M) == pytest.approx(hand_size.REFERENCE_HAND_LENGTH_M)


@pytest.mark.unit
def test_breadth_ratio_dbc_rejects_out_of_band() -> None:
    with pytest.raises(ValueError, match="hand_breadth_m/hand_length_m"):
        hand_size.validate_breadth_length_ratio(0.05, 0.19)
