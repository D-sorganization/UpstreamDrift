"""Sampled sole traces distinguish ground-first and ball-first iron strikes."""

import math

import pytest

from src.shared.python.physics.iron_ground_contact import analyze_iron_ground_contact

pytestmark = pytest.mark.unit


def test_interpolated_entry_before_ball_is_fat() -> None:
    result = analyze_iron_ground_contact([0, 2, 4], [0.01, -0.01, 0.01], 2.5)
    assert result.first_contact_time_s == pytest.approx(1)
    assert result.pre_ball_contact_duration_s == pytest.approx(1.5)
    assert result.max_depth_m == pytest.approx(0.01)
    assert result.is_fat_shot


@pytest.mark.parametrize("ball_time", [0.5, 1.0])
def test_ball_before_or_at_entry_is_not_fat(ball_time: float) -> None:
    result = analyze_iron_ground_contact([0, 2, 4], [0.01, -0.01, 0.01], ball_time)
    assert not result.is_fat_shot
    assert result.pre_ball_contact_duration_s == 0


@pytest.mark.parametrize("heights", [[0.01, 0, 0.01], [0.01, 0.02, 0.01]])
def test_tangent_or_airborne_trace_has_no_penetrating_contact(heights: list) -> None:
    result = analyze_iron_ground_contact([0, 1, 2], heights, 1)
    assert result.first_contact_time_s is None
    assert not result.is_fat_shot
    assert result.max_depth_m == result.pre_ball_contact_duration_s == 0


def test_separate_contacts_exclude_airborne_gap_and_clip_at_ball() -> None:
    result = analyze_iron_ground_contact(
        [0, 1, 2, 4, 6], [0.01, -0.01, 0.01, -0.01, 0.01], 4.5
    )
    assert result.first_contact_time_s == pytest.approx(0.5)
    assert result.pre_ball_contact_duration_s == pytest.approx(2.5)
    assert result.is_fat_shot


def test_contact_finished_before_ball_still_counts_as_fat() -> None:
    result = analyze_iron_ground_contact([0, 1, 2, 4], [0, -0.02, 0, 0.01], 3)
    assert result.first_contact_time_s == 0
    assert result.pre_ball_contact_duration_s == 2
    assert result.is_fat_shot


def test_record_ending_in_turf_reports_only_observed_duration() -> None:
    result = analyze_iron_ground_contact([10, 12], [0.01, -0.01], 12)
    assert result.first_contact_time_s == 11
    assert result.pre_ball_contact_duration_s == 1


def test_large_finite_heights_do_not_overflow_crossing_interpolation() -> None:
    result = analyze_iron_ground_contact([0, 2], [1e308, -1e308], 2)
    assert result.first_contact_time_s == 1
    assert result.pre_ball_contact_duration_s == 1


@pytest.mark.parametrize(
    ("times", "heights", "ball_time"),
    [
        ([], [], 0),
        ([0], [0], 0),
        ([0, 1], [0], 0.5),
        ([0, 0], [0, -0.01], 0),
        ([1, 0], [0, -0.01], 0.5),
        ([0, math.inf], [0, 0], 0.5),
        ([-1e308, 1e308], [0, 0], 0),
        ([0, 1], [0, math.nan], 0.5),
        ([0, 1], [-0.01, 0], 0.5),
        ([0, 1], [0, 0], -0.1),
        ([0, 1], [0, 0], 1.1),
        ([0, 1], [0, 0], math.nan),
    ],
)
def test_invalid_or_incomplete_entry_trace_is_rejected(
    times: list, heights: list, ball_time: float
) -> None:
    with pytest.raises(ValueError):
        analyze_iron_ground_contact(times, heights, ball_time)
