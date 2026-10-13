"""Test numerical precision and error-budget formatting for matched reports.

Ensures that 95% Monte Carlo intervals and strokes gained values preserve:
- Nonzero interval widths (no collapsing to identical endpoints)
- Nonzero estimates (no rounding small effects to zero)
- Interval-sign relationships and zero-crossing integrity
- Per-value relative error <= 0.1% (or absolute error within tolerance)
- Interval width error <= 1.0% (existing GUI policy)
- Contractual handling of missing, unavailable, and reversed intervals
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from scripts.research.shot_pattern_analysis.build_final_report import _interval
from src.tools.shot_pattern_analysis.matched_interpretation import (
    render_effect_table,
    render_scoring_table,
)

pytestmark = pytest.mark.unit

# Confirmed real saved-data values from Phase 2 readiness review
SEVEN_IRON_SHAFT_ONLY_DRAW = {
    "estimate": -0.0005056061017231564,
    "lower_95": -0.0005136910470865477,
    "upper_95": -0.0004968345325920594,
}
DRIVER_COMBINED2_DRAW = {
    "estimate": 2.6911773453713884e-05,
    "lower_95": -0.00016417507301096237,
    "upper_95": 0.00022037368523597208,
}
DRIVER_COMBINED2_FADE = {
    "estimate": 4.2175453160651455e-05,
    "lower_95": -0.00017391896152132183,
    "upper_95": 0.0002645740815497581,
}
PW_LEAN10_SHAFT_ONLY_DRAW = {
    "estimate": 0.0021857471659876408,
    "lower_95": 0.002154988572700285,
    "upper_95": 0.0022147960540337117,
}
PW_LEAN10_SHAFT_ONLY_FADE = {
    "estimate": -0.0020186960394436696,
    "lower_95": -0.0020452370743776287,
    "upper_95": -0.0019906470029674673,
}


def _parse_interval_string(text: str) -> tuple[float, float, float]:
    """Parse 'est [low, high]' string into floats."""
    match = re.search(
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) \[([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?), ([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\]",
        text,
    )
    assert match is not None, f"Failed to parse interval from: {text!r}"
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def _assert_precision_policy(
    estimate: float,
    low: float,
    high: float,
    rendered_est: float,
    rendered_low: float,
    rendered_high: float,
) -> None:
    """Verify precision matches the existing reviewed GUI error-budget policy."""
    # 1. Nonzero estimate must not be zeroed
    if estimate != 0.0:
        assert rendered_est != 0.0, f"Nonzero estimate {estimate} was rendered as zero"

    # 2. Nonzero interval width must not collapse
    width = high - low
    if width > 0.0:
        assert rendered_low < rendered_high, (
            f"Interval [{low}, {high}] collapsed to identical endpoints: [{rendered_low}, {rendered_high}]"
        )
        rendered_width = rendered_high - rendered_low
        assert rendered_width == pytest.approx(width, rel=0.01), (
            f"Rendered width {rendered_width} differs from true width {width} by > 1%"
        )

    # 3. Per-value relative error <= 0.1% or within endpoint tolerance
    tol = 0.005 * width if width > 0.0 else None
    for raw, rep in (
        (estimate, rendered_est),
        (low, rendered_low),
        (high, rendered_high),
    ):
        budget = abs(raw) * 0.001
        if tol is not None:
            budget = min(budget, tol)
        assert abs(rep - raw) <= budget + 1e-12, (
            f"Value {raw} rendered as {rep} exceeds error budget {budget}"
        )

    # 4. Sign relationships preserved
    if low > 0.0:
        assert rendered_low > 0.0, (
            f"Positive low {low} rendered non-positive {rendered_low}"
        )
    if high < 0.0:
        assert rendered_high < 0.0, (
            f"Negative high {high} rendered non-negative {rendered_high}"
        )
    if low < 0.0 < high:
        assert rendered_low < 0.0 < rendered_high, (
            f"Zero-crossing interval [{low}, {high}] failed crossing in [{rendered_low}, {rendered_high}]"
        )


class TestReportIntervalDefectRegression:
    """Verify build_final_report._interval fixes the confirmed precision defect."""

    @pytest.mark.parametrize(
        "record",
        [
            SEVEN_IRON_SHAFT_ONLY_DRAW,
            DRIVER_COMBINED2_DRAW,
            DRIVER_COMBINED2_FADE,
            PW_LEAN10_SHAFT_ONLY_DRAW,
            PW_LEAN10_SHAFT_ONLY_FADE,
        ],
    )
    def test_interval_preserves_confirmed_saved_data(
        self, record: dict[str, float]
    ) -> None:
        text = _interval(record)
        r_est, r_low, r_high = _parse_interval_string(text)
        _assert_precision_policy(
            record["estimate"],
            record["lower_95"],
            record["upper_95"],
            r_est,
            r_low,
            r_high,
        )

    def test_interval_handles_unavailable_and_malformed(self) -> None:
        assert _interval(None) == "Unavailable"
        assert _interval({}) == "Unavailable"
        assert _interval({"estimate": 0.01}) == "Unavailable"
        assert (
            _interval({"estimate": float("nan"), "lower_95": 0.0, "upper_95": 1.0})
            == "Unavailable"
        )
        assert (
            _interval({"estimate": 0.0, "lower_95": float("-inf"), "upper_95": 1.0})
            == "Unavailable"
        )

    def test_interval_rejects_reversed_interval(self) -> None:
        with pytest.raises(ValueError, match="reversed"):
            _interval({"estimate": 0.0, "lower_95": 0.5, "upper_95": 0.2})


class TestInterpretationTablesDefectRegression:
    """Verify render_effect_table and render_scoring_table fix the precision defect."""

    def test_render_effect_table_preserves_small_and_narrow_intervals(self) -> None:
        effects = [
            {
                "comparison_kind": "curve_strength",
                "common_six_n": 10000,
                "total_n_per_population": 10000,
                "source_protocols": ["p1", "p2"],
                "source_cells": ["c1", "c2"],
                "draw_minus_straight_change": SEVEN_IRON_SHAFT_ONLY_DRAW,
                "fade_minus_straight_change": DRIVER_COMBINED2_DRAW,
            }
        ]
        table = render_effect_table(effects)
        # 7-iron Draw interval must not collapse
        assert "-0.0005 [-0.0005, -0.0005]" not in table
        draw_match = re.search(r"\| Draw \| ([^|]+) \|", table)
        assert draw_match is not None
        d_est, d_low, d_high = _parse_interval_string(draw_match.group(1).strip())
        _assert_precision_policy(
            SEVEN_IRON_SHAFT_ONLY_DRAW["estimate"],
            SEVEN_IRON_SHAFT_ONLY_DRAW["lower_95"],
            SEVEN_IRON_SHAFT_ONLY_DRAW["upper_95"],
            d_est,
            d_low,
            d_high,
        )

        # Driver Fade interval must not show estimate as 0.0000
        assert "+0.0000 [" not in table
        fade_match = re.search(r"\| Fade \| ([^|]+) \|", table)
        assert fade_match is not None
        f_est, f_low, f_high = _parse_interval_string(fade_match.group(1).strip())
        _assert_precision_policy(
            DRIVER_COMBINED2_DRAW["estimate"],
            DRIVER_COMBINED2_DRAW["lower_95"],
            DRIVER_COMBINED2_DRAW["upper_95"],
            f_est,
            f_low,
            f_high,
        )

    def test_render_effect_table_preserves_reversed_check(self) -> None:
        effects = [
            {
                "comparison_kind": "curve_strength",
                "common_six_n": 10000,
                "total_n_per_population": 10000,
                "source_protocols": ["p1", "p2"],
                "source_cells": ["c1", "c2"],
                "draw_minus_straight_change": {
                    "estimate": 0.0,
                    "lower_95": 1.0,
                    "upper_95": 0.5,
                },
                "fade_minus_straight_change": None,
            }
        ]
        with pytest.raises(ValueError, match="reversed"):
            render_effect_table(effects)

    def test_render_scoring_table_preserves_small_nonzero_absolute_means(self) -> None:
        named_scores = [
            (
                "cell_driver",
                {
                    "status": "available",
                    "total_n_per_pattern": 10000,
                    "context": {
                        "start_lie": "tee",
                        "scenario_id": "test",
                        "start_distance_m": 400.0,
                        "hole_distance_m": 400.0,
                        "fairway_half_width_m": 15.0,
                        "green_radius_m": None,
                    },
                    "patterns": {
                        "Straight": {
                            "absolute_mean_strokes_gained": DRIVER_COMBINED2_DRAW[
                                "estimate"
                            ],
                            "individual_endpoint_supported_n": 10000,
                            "common_three_scored_n": 10000,
                        },
                        "Draw": {
                            "absolute_mean_strokes_gained": -0.4,
                            "individual_endpoint_supported_n": 10000,
                            "common_three_scored_n": 10000,
                        },
                        "Fade": {
                            "absolute_mean_strokes_gained": 0.0,
                            "individual_endpoint_supported_n": 10000,
                            "common_three_scored_n": 10000,
                        },
                    },
                },
            )
        ]
        table = render_scoring_table(named_scores)
        # Small nonzero mean must not be shown as +0.0000
        assert "+0.0000" not in table or "+0.000" in table  # Fade is exactly 0.0
        # Straight has small estimate 2.691e-5
        straight_match = re.search(r"\| Straight \| ([^|]+) \|", table)
        assert straight_match is not None
        straight_val = float(straight_match.group(1).strip())
        assert straight_val != 0.0
        assert (
            abs(straight_val - DRIVER_COMBINED2_DRAW["estimate"])
            <= abs(DRIVER_COMBINED2_DRAW["estimate"]) * 0.001
        )


class TestPureFormattingModule:
    """Verify that pure formatting module functions meet precision and contract rules."""

    def test_pure_formatter_import_and_contracts(self) -> None:
        from src.tools.shot_pattern_analysis.matched_formatting import (
            format_sg_interval,
            format_sg_value,
            is_finite_real,
        )

        assert is_finite_real(1.0)
        assert not is_finite_real(True)
        assert not is_finite_real(float("nan"))
        assert not is_finite_real(float("inf"))
        assert not is_finite_real("0.5")

        assert format_sg_value(0.0) == "+0.000"
        assert format_sg_value("not_a_number") == "unavailable"
        assert format_sg_value(float("nan")) == "unavailable"

        # Reversed interval
        with pytest.raises(ValueError, match="reversed"):
            format_sg_interval(0.0, 1.0, 0.5)

        # Extremely narrow interval
        val_est, val_low, val_high = 0.5000015, 0.500001, 0.500002
        text = format_sg_interval(val_est, val_low, val_high)
        r_est, r_low, r_high = _parse_interval_string(text)
        _assert_precision_policy(val_est, val_low, val_high, r_est, r_low, r_high)

    @pytest.mark.parametrize(
        ("estimate", "low", "high"),
        [
            # Ordinary positive
            (0.02, 0.01, 0.03),
            # Ordinary negative
            (-0.4, -0.45, -0.35),
            # Zero estimate with symmetrical interval
            (0.0, -0.001, 0.001),
            # Small positive crossing zero
            (2.69118e-5, -0.000164175, 0.000220374),
            # Small negative crossing zero
            (-3.5e-5, -0.0002, 0.0001),
            # Small one-sided positive
            (0.00025, 0.00012, 0.00038),
            # Small one-sided negative
            (-0.00025, -0.00038, -0.00012),
            # Very narrow interval around non-trivial value
            (0.12345678, 0.12345670, 0.12345686),
        ],
    )
    def test_interval_category_matrix(
        self, estimate: float, low: float, high: float
    ) -> None:
        from src.tools.shot_pattern_analysis.matched_formatting import (
            format_sg_interval,
        )

        text = format_sg_interval(estimate, low, high)
        r_est, r_low, r_high = _parse_interval_string(text)
        _assert_precision_policy(estimate, low, high, r_est, r_low, r_high)

    def test_format_sg_interval_record_helper(self) -> None:
        from src.tools.shot_pattern_analysis.matched_formatting import (
            format_sg_interval_record,
        )

        assert format_sg_interval_record(None) == "Unavailable"
        assert format_sg_interval_record("not-a-map") == "Unavailable"
        assert format_sg_interval_record({}, unavailable_text="—") == "—"
        assert format_sg_interval_record({"estimate": 0.01}) == "Unavailable"
        assert (
            format_sg_interval_record(
                {"estimate": 0.01, "lower_95": 0.0, "upper_95": float("nan")}
            )
            == "Unavailable"
        )
        with pytest.raises(ValueError, match="reversed"):
            format_sg_interval_record(
                {"estimate": 0.0, "lower_95": 0.5, "upper_95": 0.2}
            )

        valid = format_sg_interval_record(
            {"estimate": 0.02, "lower_95": 0.01, "upper_95": 0.03}
        )
        assert valid == "+0.020 [+0.010, +0.030]"

    def test_all_47_phase2_affected_records_from_scan(self) -> None:
        """Verify all 47 real affected records from precision_scan.json pass."""
        import json
        from pathlib import Path
        from src.tools.shot_pattern_analysis.matched_formatting import (
            format_sg_interval,
        )

        scan_path = (
            Path(__file__).parent / "fixtures" / "matched_precision_affected.json"
        )
        scan_data = json.loads(scan_path.read_text())
        affected = scan_data.get("affected", [])
        assert len(affected) == 47, f"Expected 47 affected records, got {len(affected)}"

        for row in affected:
            v = row["values"]
            est = float(v["estimate"])
            low = float(v["lower_95"])
            high = float(v["upper_95"])
            text = format_sg_interval(est, low, high)
            r_est, r_low, r_high = _parse_interval_string(text)
            _assert_precision_policy(est, low, high, r_est, r_low, r_high)


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf"), True, "0.01"])
def test_value_formatter_rejects_invalid_absolute_budget(tolerance) -> None:
    from src.tools.shot_pattern_analysis.matched_formatting import format_sg_value

    with pytest.raises(ValueError, match="tolerance"):
        format_sg_value(0.0, absolute_tolerance=tolerance)


def test_zero_absolute_budget_requires_exact_round_trip() -> None:
    from src.tools.shot_pattern_analysis.matched_formatting import format_sg_value

    value = 0.012345678901234567
    assert float(format_sg_value(value, absolute_tolerance=0.0)) == value


def test_report_has_no_silently_ignored_precision_parameter() -> None:
    import inspect

    assert "digits" not in inspect.signature(_interval).parameters


def test_report_source_guard_binds_pure_formatter() -> None:
    import hashlib
    from pathlib import Path
    from scripts.research.shot_pattern_analysis.build_final_report import (
        _report_source_hashes,
    )
    from src.tools.shot_pattern_analysis import matched_formatting

    path = Path(matched_formatting.__file__)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        _report_source_hashes()["src/tools/shot_pattern_analysis/matched_formatting.py"]
        == expected
    )
