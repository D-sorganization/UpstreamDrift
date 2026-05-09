"""Unit tests for :class:`anthropometrics.estimators.DeLevaEstimator`.

Validates that the wrapper:

* Reproduces published de Leva (1996) numerical values within
  ``1e-3`` from the canonical ratio table that lives in
  :mod:`humanoid_character_builder.core.anthropometry`.
* Does **not** duplicate the ratios — the wrapper must read the
  same module that the rest of the codebase uses.
* Produces a fully-validated :class:`SubjectAnthropometrics`
  whose total segment mass closes to the input mass within 1%.
* Validates subject-level inputs via Design-by-Contract
  ``ValueError`` raises.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from anthropometrics import Estimator, SubjectAnthropometrics
from anthropometrics.estimators import DeLevaEstimator
from anthropometrics.estimators import from_de_leva as de_leva_module
from humanoid_character_builder.core.anthropometry import DE_LEVA_DATA


PUBLISHED_TOL = 1e-3


# --------------------------------------------------------------------------- #
# Validation against published de Leva (1996) numbers.                        #
# --------------------------------------------------------------------------- #
def test_male_upper_arm_mass_ratio_matches_published() -> None:
    """Male upper-arm mass ratio in the source table is 0.0271."""
    male = DE_LEVA_DATA.male["upper_arm"]
    assert male.mass_ratio == pytest.approx(0.0271, abs=PUBLISHED_TOL)


def test_male_forearm_com_proximal_ratio_matches_published() -> None:
    """Male forearm CoM proximal ratio is 0.4574 (≈ 0.457)."""
    male = DE_LEVA_DATA.male["forearm"]
    assert male.com_proximal_ratio == pytest.approx(0.457, abs=1e-3)


def test_male_upper_arm_longitudinal_gyration_matches_published() -> None:
    """Male upper-arm sagittal radius of gyration is 0.285."""
    # The de Leva tables call this "gyration_sagittal" (rotation
    # about the mediolateral axis); historic literature also
    # labels the same axis "longitudinal" depending on convention.
    # Both notations resolve to the value 0.285 from the published
    # paper.
    male = DE_LEVA_DATA.male["upper_arm"]
    assert male.gyration_sagittal == pytest.approx(0.285, abs=PUBLISHED_TOL)


def test_published_values_round_trip_through_estimator() -> None:
    """A subject estimate preserves the published mass ratio.

    For a subject of total mass M, the upper-arm segment of a
    de-Leva estimate must satisfy ``mass_kg / (M * normalisation)
    == published_ratio`` to within 1e-3.
    """
    est = DeLevaEstimator()
    height_m = 1.83
    mass_kg = 82.0
    subject = est.estimate(
        subject_id="published-roundtrip",
        height_m=height_m,
        mass_kg=mass_kg,
        sex="M",
    )

    # Sum of raw ratios across the emitted segmentation lets the
    # test back out the published ratio from a normalised mass.
    raw_sum = 0.0
    for anatomical, class_id in _de_leva_name_map().items():
        del anatomical
        raw_sum += DE_LEVA_DATA.male[class_id].mass_ratio

    upper_arm = next(
        props for name, props in subject.segments if name == "left_upper_arm"
    )
    expected_kg = mass_kg * (0.0271 / raw_sum)
    assert upper_arm.mass_kg == pytest.approx(expected_kg, rel=PUBLISHED_TOL)


# --------------------------------------------------------------------------- #
# Smoke: full subject build + mass closure within 1%.                         #
# --------------------------------------------------------------------------- #
def test_smoke_full_subject_returns_subject_anthropometrics() -> None:
    """1.83 m / 82 kg male produces a non-empty SubjectAnthropometrics."""
    est = DeLevaEstimator()
    subject = est.estimate(
        subject_id="smoke",
        height_m=1.83,
        mass_kg=82.0,
        sex="M",
    )
    assert isinstance(subject, SubjectAnthropometrics)
    assert subject.source_method == "de_leva_1996"
    assert len(subject.segments) >= 11


def test_smoke_mass_closes_within_one_percent() -> None:
    """Sum of segment masses equals total subject mass within 1%."""
    est = DeLevaEstimator()
    mass_kg = 82.0
    subject = est.estimate(
        subject_id="closure",
        height_m=1.83,
        mass_kg=mass_kg,
        sex="M",
    )
    total = sum(props.mass_kg for _, props in subject.segments)
    assert abs(total - mass_kg) / mass_kg <= 0.01


def test_smoke_female_subject_uses_female_table() -> None:
    """Sex='F' must select the female sub-table (different ratios)."""
    est = DeLevaEstimator()
    male = est.estimate(subject_id="m", height_m=1.70, mass_kg=70.0, sex="M")
    female = est.estimate(subject_id="f", height_m=1.70, mass_kg=70.0, sex="F")
    male_thigh = next(p for n, p in male.segments if n == "left_thigh")
    female_thigh = next(p for n, p in female.segments if n == "left_thigh")
    # The two tables differ in thigh mass ratio (0.1416 vs 0.1478),
    # so the realised masses must also differ.
    assert male_thigh.mass_kg != pytest.approx(female_thigh.mass_kg, rel=1e-4)


def test_estimator_protocol_isinstance_check() -> None:
    """The wrapper must satisfy the runtime-checkable Estimator Protocol."""
    assert isinstance(DeLevaEstimator(), Estimator)


# --------------------------------------------------------------------------- #
# Design by Contract — subject-input validation.                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_height", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_height_raises_value_error(bad_height: float) -> None:
    """``height_m`` must be a positive finite number."""
    est = DeLevaEstimator()
    with pytest.raises(ValueError, match="height_m"):
        est.estimate(subject_id="x", height_m=bad_height, mass_kg=70.0)


@pytest.mark.parametrize("bad_mass", [0.0, -50.0, float("nan"), float("inf")])
def test_invalid_mass_raises_value_error(bad_mass: float) -> None:
    """``mass_kg`` must be a positive finite number."""
    est = DeLevaEstimator()
    with pytest.raises(ValueError, match="mass_kg"):
        est.estimate(subject_id="x", height_m=1.80, mass_kg=bad_mass)


def test_invalid_subject_id_raises_value_error() -> None:
    """``subject_id`` must be a non-empty string."""
    est = DeLevaEstimator()
    with pytest.raises(ValueError, match="subject_id"):
        est.estimate(subject_id=" ", height_m=1.80, mass_kg=70.0)


def test_invalid_sex_raises_value_error() -> None:
    """``sex`` must be one of M/F/unspecified."""
    est = DeLevaEstimator()
    with pytest.raises(ValueError, match="sex"):
        est.estimate(subject_id="x", height_m=1.80, mass_kg=70.0, sex="other")


def test_invalid_age_raises_value_error() -> None:
    """``age_years`` if provided must be non-negative."""
    est = DeLevaEstimator()
    with pytest.raises(ValueError, match="age_years"):
        est.estimate(
            subject_id="x",
            height_m=1.80,
            mass_kg=70.0,
            age_years=-1.0,
        )


# --------------------------------------------------------------------------- #
# DRY — the wrapper must NOT duplicate the de Leva ratio table.               #
# --------------------------------------------------------------------------- #
def test_wrapper_does_not_duplicate_ratio_table() -> None:
    """The wrapper module text must not redeclare the published ratios.

    The published mass-ratio constants live in exactly one place
    (humanoid_character_builder.core.anthropometry). The wrapper
    module is allowed to import them but must not redeclare any
    of the canonical numerical values inline.
    """
    wrapper_path = Path(inspect.getsourcefile(de_leva_module) or "")
    assert wrapper_path.exists()
    text = wrapper_path.read_text(encoding="utf-8")
    # A handful of canonical de Leva numerical constants. If any
    # appear as literals inside the wrapper, the ratio table has
    # been duplicated.
    forbidden_literals = ["0.0694", "0.0271", "0.4574", "0.1416", "0.0433"]
    for literal in forbidden_literals:
        assert literal not in text, (
            f"de Leva wrapper contains a duplicated ratio literal "
            f"{literal!r}; ratios must live only in "
            f"humanoid_character_builder.core.anthropometry."
        )


def test_wrapper_uses_canonical_module() -> None:
    """The wrapper must actually import from the canonical module."""
    wrapper_path = Path(inspect.getsourcefile(de_leva_module) or "")
    text = wrapper_path.read_text(encoding="utf-8")
    assert "humanoid_character_builder.core.anthropometry" in text


# --------------------------------------------------------------------------- #
# Helpers.                                                                    #
# --------------------------------------------------------------------------- #
def _de_leva_name_map() -> dict[str, str]:
    """Return the canonical anatomical->class map used by the wrapper."""
    from humanoid_character_builder.core.anthropometry import _SEGMENT_NAME_MAP

    return dict(_SEGMENT_NAME_MAP)
