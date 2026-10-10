"""Saved SG effects and uncertainty retain numerical precision in the GUI."""

import re

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize(
    ("estimate", "low", "high"),
    [
        (0.004597062764879343, 0.004531382987542545, 0.004658536119213675),
        (-0.00392726220708177, -0.003979629136146112, -0.003872946949009742),
        (-0.003284521591238346, -0.0033180608278863878, -0.003246621472447612),
        (0.5000015, 0.500001, 0.500002),
        (0.0000746156, -0.0000710405, 0.000218961),
    ],
)
def test_saved_interval_preserves_values_and_nonzero_width(estimate, low, high):
    try:
        from src.tools.shot_pattern_analysis.matched_gui import (
            format_matched_course_scoring,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(f"Optional GUI dependencies unavailable: {exc}")
    text = format_matched_course_scoring(
        {
            "status": "available",
            "paired_n": 10000,
            "total_n": 10000,
            "paired_benefit_vs_straight": {
                "Draw": {"estimate": estimate, "lower_95": low, "upper_95": high}
            },
        }
    )
    match = re.search(r"Draw: ([^ ]+) \[([^,]+), ([^\]]+)\] strokes", text)
    assert match is not None
    shown = tuple(float(value) for value in match.groups())
    assert shown == pytest.approx((estimate, low, high), rel=0.01, abs=0)
    assert shown[1] < shown[2]
    assert shown[2] - shown[1] == pytest.approx(high - low, rel=0.01, abs=0)


@pytest.mark.parametrize("value", [0.0015, -0.004597062764879343, 0.0000746156])
def test_small_mean_retains_numeric_accuracy(value):
    try:
        from src.tools.shot_pattern_analysis.matched_gui import (
            format_matched_course_scoring,
        )
    except (ImportError, OSError) as exc:
        pytest.skip(f"Optional GUI dependencies unavailable: {exc}")
    text = format_matched_course_scoring(
        {
            "status": "available",
            "patterns": {"Straight": {"mean_strokes_gained": value}},
        }
    )
    match = re.search(r"Mean Strokes Gained: Straight: ([^;]+);", text)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(value, rel=0.01, abs=0)
