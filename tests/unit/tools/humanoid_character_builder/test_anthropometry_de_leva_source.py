"""The humanoid de Leva table derives from whole Table 4 rows and sums to one (#13010).

The table used to be typed by hand. Its mass fractions summed to 1.068 (male)
and 1.071 (female) because the neck and an oversized trunk were added on top of
de Leva's head-and-neck and trunk, and ``estimate_segment_masses`` rescaled the
result to hide it. Several rows paired one endpoint row's length with another's
CM position or radii.
"""

from __future__ import annotations

import pytest

from humanoid_character_builder.core import anthropometry
from src.shared.python.body_segment_parameters import de_leva_1996 as table4

pytestmark = pytest.mark.unit

GENDER_FACTORS = (0.0, 0.3, 0.5, 1.0)
SEX_TABLES = (
    ("female", anthropometry.DE_LEVA_DATA.female),
    ("male", anthropometry.DE_LEVA_DATA.male),
)


def _composite_sum(gender_factor: float) -> float:
    return sum(
        anthropometry.get_segment_mass_ratio(name, gender_factor)
        for name in anthropometry._SEGMENT_NAME_MAP
    )


@pytest.mark.parametrize("gender_factor", GENDER_FACTORS)
def test_segment_masses_sum_to_the_body_without_rescaling(gender_factor: float) -> None:
    assert _composite_sum(gender_factor) == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("gender_factor", GENDER_FACTORS)
def test_estimated_masses_are_the_table_fractions(gender_factor: float) -> None:
    masses = anthropometry.estimate_segment_masses(80.0, gender_factor)
    for name, mass in masses.items():
        expected = 80.0 * anthropometry.get_segment_mass_ratio(name, gender_factor)
        assert mass == pytest.approx(expected, rel=1e-12), name
    assert sum(masses.values()) == pytest.approx(80.0, rel=2e-4)


def test_de_leva_segments_are_whole_table4_rows() -> None:
    """Mass (with the virtual segments carved from it), CM and radii share one row."""
    for sex, table in SEX_TABLES:
        for segment, row_id in anthropometry.DE_LEVA_SOURCE_ROWS.items():
            data = table[segment]
            row = table4.table4_row(row_id, sex)
            carved = sum(
                share * anthropometry.virtual_segment_count(virtual)
                for virtual, (parent, share) in anthropometry.VIRTUAL_SEGMENTS.items()
                if parent == segment
            )
            radii = (
                data.gyration_sagittal,
                data.gyration_transverse,
                data.gyration_longitudinal,
            )
            assert (
                table4.match_row(
                    sex, data.mass_ratio + carved, data.com_proximal_ratio, radii
                )
                == row_id
            ), (sex, segment)


def test_thigh_mass_is_de_leva_for_an_80_kg_male() -> None:
    masses = anthropometry.estimate_segment_masses(80.0, 1.0)
    assert masses["left_thigh"] == pytest.approx(0.1416 * 80.0, rel=1e-12)
    assert masses["head"] + masses["neck"] == pytest.approx(0.0694 * 80.0, rel=1e-12)


def test_a_table_that_does_not_sum_to_one_is_refused(monkeypatch) -> None:
    broken = dict(anthropometry.DE_LEVA_DATA.male)
    broken["thigh"] = table4_thigh = anthropometry.SegmentAnthropometry(
        **{**vars(broken["thigh"]), "mass_ratio": 0.20}
    )
    assert table4_thigh.mass_ratio == 0.20
    monkeypatch.setattr(anthropometry.DE_LEVA_DATA, "male", broken)
    with pytest.raises(ValueError, match="sum"):
        anthropometry.estimate_segment_masses(80.0, 1.0)


def test_estimator_reports_unscaled_de_leva_masses() -> None:
    from anthropometrics.estimators import DeLevaEstimator

    for sex in ("M", "F"):
        subject = DeLevaEstimator().estimate(
            subject_id="s", height_m=1.8, mass_kg=80.0, sex=sex
        )
        segments = dict(subject.segments)
        gender_factor = 1.0 if sex == "M" else 0.0
        for name, props in segments.items():
            expected = 80.0 * anthropometry.get_segment_mass_ratio(name, gender_factor)
            assert props.mass_kg == pytest.approx(expected, rel=1e-12), (sex, name)
