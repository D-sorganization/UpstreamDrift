"""ADR-0046 G0 drift gate: strokes gained, UD stack vs vendored Tools stack.

Measured on ``adr0046_cross_stack_session_v1.json`` (160 clean shots, one
expected-strokes baseline, both stacks reading the same DataFrame).

AGREE — asserted exactly
    * Baseline table digest. ``strokes_gained_types.baseline_table_sha256``
      (UD) and ``launch_monitor_strokes_gained_baseline.baseline_table_hash``
      (Tools) return the identical digest
      ``188a6eafa9eebd8a0f4c9ba288d858ad359e35999ba2706989c75d349f509925``
      for the same states. The two baseline wire formats are interchangeable.
    * Included row count: 160 == 160.
    * Per-shot SG value: max |UD - Tools| = 4.44e-16 over 160 rows. UD walks
      the sorted stratum by hand; Tools calls ``numpy.interp``. Same answer to
      float noise.
    * Session mean SG: 0.80592372152815683 on both, bit-for-bit (delta 0.0).

RESOLVED — was a divergence, now an agreement
    D1. Malformed rows. **Resolved by ADR-0048 decision G1-D3**, "the canonical
        error posture is exclude-and-audit". Both stacks now exclude the row,
        record it against the same ``reason_code``, degrade ``status`` to
        ``"partial"``, and return an unchanged mean. Tools previously raised
        ``ValueError`` and lost the whole session for 3 of the 4 failure modes;
        for the 4th (blank lie/context/target) it silently dropped the row with
        no exclusion record at all — which G1-D3 called "the worst outcome
        available". This gate no longer measures a divergence here; it measures
        that the resolution holds on both sides, and it fails if either stack
        regresses to raising or to silence.

DIFFER — documented and pinned below
    D2. Uncertainty. UD reports sd/standard error/Student-t CI plus a combined
        benchmark standard error. Tools' result carries no uncertainty at all:
        the Tools loader parses ``standard_error`` off the baseline and then
        discards it in ``_expected``. The Tools field set pinned below **grew
        by three** with G1-D3 (``status``, ``excluded_rows``, ``exclusions``)
        and ``mean`` widened to optional; that addition is the audit surface,
        not uncertainty, so D2 itself is unchanged and still open.
    D3. Grouped SG. UD computes per-player/session/club estimates in-process.
        Tools' local calculator has no grouping; ``TrustedSummaryRequest``
        only builds a *payload to send to UpstreamDrift*, so today the Tools
        stack cannot compute grouped SG at all.
    D4. Longitudinal SG trend. UD fits per-player slope/R2/p. Tools' SG module
        has none.
    D5. Estimand note (intra-UD, recorded for the G1 port plan): UD's SG
        longitudinal fit is **shot-level** (sample_count 40 per player = 5
        sessions x 8 shots), while UD's own ``longitudinal.py`` aggregates to
        20 player-session cells first and warns against pseudo-replication.
        The canonical layer must pick one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.shared.python.launch_monitor.strokes_gained import (
    analyze_source_backed_strokes_gained,
)
from src.shared.python.launch_monitor.strokes_gained_types import (
    CourseStateColumnsV1,
    ExpectedStrokesBaselineV2,
    GroupingDimensionV1,
    LongitudinalDimensionV1,
    StrokesGainedRequestV1,
    baseline_table_sha256,
)
from tests.integration.launch_monitor_drift.conftest import (
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from rate_of_closure.launch_monitor_strokes_gained import (  # noqa: E402
    SourceBackedStrokesGainedRequest,
    calculate_source_backed_strokes_gained,
)
from rate_of_closure.launch_monitor_strokes_gained_baseline import (  # noqa: E402
    baseline_table_hash,
    load_strokes_gained_baseline,
)

BASELINE_TABLE_SHA256 = (
    "188a6eafa9eebd8a0f4c9ba288d858ad359e35999ba2706989c75d349f509925"
)
EXPECTED_INCLUDED_ROWS = 160
PINNED_MEAN_STROKES_GAINED = 0.80592372152815683
MAX_PER_ROW_DELTA = 1e-12

# D2 pins: values only the UD stack produces.
UD_ONLY_STANDARD_DEVIATION = 0.3109790626188181
UD_ONLY_STANDARD_ERROR = 0.024585053562489797
UD_ONLY_CI_LOWER = 0.757368333851554
UD_ONLY_CI_UPPER = 0.8544791092047597
UD_ONLY_BENCHMARK_STANDARD_ERROR_MEAN = 0.010509224804888823

# D3 pins: grouped means only the UD stack produces.
UD_ONLY_PLAYER_MEANS = {
    "P1": 0.80845404044011371,
    "P2": 0.85473334756906527,
    "P3": 0.87813372759235198,
    "P4": 0.68237377051109649,
}

# D1 pins (RESOLVED per ADR-0048 G1-D3): the single ``reason_code`` **both**
# stacks must record for one malformed row appended to 160 clean. Before the
# resolution this mapped each case to a UD reason code *and* a divergent Tools
# behaviour ("raises" for three cases, "silently_drops" for the fourth); the
# Tools half now agrees, so one code per case is the whole contract.
DEGENERATE_EXPECTATIONS = {
    "outside_baseline_start": "outside_baseline",
    "missing_course_state": "missing_course_state",
    "negative_finish_distance": "invalid_distance",
    "unknown_stratum": "outside_baseline",
}


def _ud_request() -> StrokesGainedRequestV1:
    return StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie",
            context_column="start_context",
            target_column="target",
            distance_column="start_distance_yards",
            distance_unit="yd",
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie",
            context_column="finish_context",
            target_column="target",
            distance_column="finish_distance_metres",
            distance_unit="m",
        ),
        shot_id_column="shot_id",
        confidence_level=0.95,
        min_samples=3,
        summaries=(
            GroupingDimensionV1(
                dimension="player",
                column="player_id",
                trust_level="pseudonymous_stable",
                evidence="Synthetic ADR-0046 G0 fixture pseudonym.",
            ),
        ),
        longitudinal=LongitudinalDimensionV1(
            order_column="session_order",
            order_unit="session",
            group_column="player_id",
            group_dimension="player",
            trust_level="pseudonymous_stable",
            evidence="Synthetic ADR-0046 G0 fixture session ordinal.",
        ),
    )


def _tools_request() -> SourceBackedStrokesGainedRequest:
    return SourceBackedStrokesGainedRequest(
        before_lie_column="start_lie",
        before_context_column="start_context",
        before_target_column="target",
        before_distance_column="start_distance_yards",
        after_lie_column="finish_lie",
        after_context_column="finish_context",
        after_target_column="target",
        after_distance_column="finish_distance_metres",
        before_distance_unit="yd",
        after_distance_unit="m",
    )


@pytest.fixture(scope="module")
def ud_result(session_frame: pd.DataFrame, baseline_document: dict[str, Any]):
    baseline = ExpectedStrokesBaselineV2.model_validate(baseline_document)
    return analyze_source_backed_strokes_gained(session_frame, baseline, _ud_request())


@pytest.fixture(scope="module")
def tools_result(session_frame: pd.DataFrame, baseline_path: Path):
    baseline = load_strokes_gained_baseline(baseline_path)
    return calculate_source_backed_strokes_gained(
        session_frame, baseline, _tools_request()
    )


def test_baseline_table_digest_agrees_across_stacks(
    baseline_document: dict[str, Any],
) -> None:
    """AGREE: both stacks canonicalise the same states to the same digest."""
    ud_digest = baseline_table_sha256(baseline_document["states"])
    tools_digest = baseline_table_hash(baseline_document["states"])

    assert ud_digest == tools_digest == BASELINE_TABLE_SHA256
    assert baseline_document["table_sha256"] == BASELINE_TABLE_SHA256


def test_included_row_count_agrees(ud_result, tools_result) -> None:
    """AGREE: both stacks admit all 160 clean shots."""
    assert ud_result.exclusions.included_row_count == EXPECTED_INCLUDED_ROWS
    assert len(tools_result.values) == EXPECTED_INCLUDED_ROWS
    assert ud_result.status == "available"


def test_per_shot_strokes_gained_agrees_to_float_noise(ud_result, tools_result) -> None:
    """AGREE: manual interpolation (UD) and numpy.interp (Tools) coincide."""
    ud_values = [row.strokes_gained for row in ud_result.row_results]
    tools_values = list(tools_result.values)

    assert len(ud_values) == len(tools_values)
    deltas = [
        abs(left - right) for left, right in zip(ud_values, tools_values, strict=True)
    ]
    assert max(deltas) < MAX_PER_ROW_DELTA, (
        f"per-shot SG drift {max(deltas):.3e} exceeds {MAX_PER_ROW_DELTA:.0e}"
    )


def test_session_mean_strokes_gained_agrees_exactly(ud_result, tools_result) -> None:
    """AGREE: the session mean is bit-for-bit identical across stacks."""
    assert ud_result.value_summary.mean == pytest.approx(
        PINNED_MEAN_STROKES_GAINED, rel=1e-12
    )
    assert tools_result.mean == pytest.approx(PINNED_MEAN_STROKES_GAINED, rel=1e-12)
    assert ud_result.value_summary.mean - tools_result.mean == 0.0


def test_divergence_d2_only_ud_reports_strokes_gained_uncertainty(
    ud_result, tools_result
) -> None:
    """DIFFER (D2): Tools reports no SG uncertainty at all.

    UD: sd 0.31097906, se 0.02458505, 95% t CI [0.75736833, 0.85447911],
    combined benchmark standard error 0.01050922.
    Tools: the result dataclass has no uncertainty field; the baseline's
    ``standard_error`` column is parsed and then dropped in ``_expected``.
    """
    summary = ud_result.value_summary
    assert summary.standard_deviation == pytest.approx(
        UD_ONLY_STANDARD_DEVIATION, rel=1e-12
    )
    assert summary.standard_error == pytest.approx(UD_ONLY_STANDARD_ERROR, rel=1e-12)
    assert summary.confidence_interval is not None
    assert summary.confidence_interval.lower == pytest.approx(
        UD_ONLY_CI_LOWER, rel=1e-12
    )
    assert summary.confidence_interval.upper == pytest.approx(
        UD_ONLY_CI_UPPER, rel=1e-12
    )
    assert ud_result.uncertainty.benchmark_standard_error_mean == pytest.approx(
        UD_ONLY_BENCHMARK_STANDARD_ERROR_MEAN, rel=1e-12
    )

    tools_fields = set(type(tools_result).__dataclass_fields__)
    assert tools_fields == {
        "metric_name",
        "unit",
        "values",
        "mean",
        "baseline_id",
        "baseline_version",
        "source_url",
        "license",
        "table_sha256",
        "backing_rows",
        "formula",
        # Added by ADR-0048 G1-D3's audit surface. Additive: the eleven fields
        # above are all still present.
        "status",
        "excluded_rows",
        "exclusions",
    }
    assert not tools_fields & {
        "standard_deviation",
        "standard_error",
        "confidence_interval",
        "uncertainty",
    }
    backing_fields = set(type(tools_result.backing_rows[0]).__dataclass_fields__)
    assert "standard_error" not in backing_fields


def test_divergence_d3_only_ud_computes_grouped_strokes_gained(
    ud_result, tools_result
) -> None:
    """DIFFER (D3): the Tools stack cannot compute grouped SG locally."""
    grouped = {
        summary.group_value: summary.estimate.mean
        for summary in ud_result.group_summaries
        if summary.dimension == "player"
    }
    assert set(grouped) == set(UD_ONLY_PLAYER_MEANS)
    for player, expected in UD_ONLY_PLAYER_MEANS.items():
        assert grouped[player] == pytest.approx(expected, rel=1e-12)

    assert not hasattr(tools_result, "group_summaries")


def test_divergence_d4_d5_longitudinal_strokes_gained_estimand(
    ud_result, tools_result
) -> None:
    """DIFFER (D4/D5): only UD fits an SG trend, and it fits shot-level.

    Each UD per-player fit uses sample_count 40 (every shot), not the 20
    player-session cells that UD's own ``longitudinal.py`` insists on. The
    canonical layer has to choose one estimand.
    """
    trends = {item.group_value: item for item in ud_result.longitudinal_summaries}
    assert set(trends) == {"P1", "P2", "P3", "P4"}
    assert all(item.sample_count == 40 for item in trends.values())
    assert trends["P4"].slope == pytest.approx(0.075881035543697128, rel=1e-9)
    assert trends["P4"].r_squared == pytest.approx(0.15450437016457175, rel=1e-9)

    assert not hasattr(tools_result, "longitudinal_summaries")


@pytest.mark.parametrize("case", sorted(DEGENERATE_EXPECTATIONS))
def test_resolved_d1_both_stacks_exclude_and_audit_a_malformed_row(
    case: str,
    session_frame: pd.DataFrame,
    degenerate_frame: pd.DataFrame,
    baseline_document: dict[str, Any],
    baseline_path: Path,
) -> None:
    """RESOLVED (D1, ADR-0048 G1-D3): both stacks exclude, audit, and continue.

    Appending exactly one malformed row to the 160 clean shots:

    ==========================  =====================  =====================
    case                        UD                     Tools
    ==========================  =====================  =====================
    outside_baseline_start      excluded, partial      excluded, partial
    missing_course_state        excluded, partial      excluded, partial
    negative_finish_distance    excluded, partial      excluded, partial
    unknown_stratum             excluded, partial      excluded, partial
    ==========================  =====================  =====================

    Before G1-D3 the Tools column read "ValueError, no result" for the first,
    third and fourth cases and "dropped, no record" for the second. Both
    stacks now agree on the ``reason_code`` as well as the outcome, and both
    means are unchanged at ``PINNED_MEAN_STROKES_GAINED`` — the 160 good rows
    survive one bad row on both sides.

    This test fails if either stack regresses to raising *or* to silence.
    """
    reason_code = DEGENERATE_EXPECTATIONS[case]
    row = degenerate_frame.loc[degenerate_frame["degenerate_case"] == case].drop(
        columns=["degenerate_case"]
    )
    mixed = pd.concat([session_frame, row], ignore_index=True)

    ud = analyze_source_backed_strokes_gained(
        mixed,
        ExpectedStrokesBaselineV2.model_validate(baseline_document),
        _ud_request(),
    )
    assert ud.status == "partial"
    assert ud.exclusions.included_row_count == EXPECTED_INCLUDED_ROWS
    assert ud.exclusions.by_reason == {reason_code: 1}
    assert ud.value_summary.mean == pytest.approx(PINNED_MEAN_STROKES_GAINED, rel=1e-12)

    baseline = load_strokes_gained_baseline(baseline_path)
    tools = calculate_source_backed_strokes_gained(mixed, baseline, _tools_request())

    assert tools.status == "partial"
    assert len(tools.values) == EXPECTED_INCLUDED_ROWS
    assert tools.mean == pytest.approx(PINNED_MEAN_STROKES_GAINED, rel=1e-12)

    # The audit surface itself, not merely the surviving mean.
    assert tools.exclusions.by_reason == {reason_code: 1}
    assert tools.exclusions.input_row_count == EXPECTED_INCLUDED_ROWS + 1
    assert tools.exclusions.included_row_count == EXPECTED_INCLUDED_ROWS
    assert tools.exclusions.total_excluded == 1
    assert len(tools.excluded_rows) == 1
    assert tools.excluded_rows[0].reason_code == reason_code
    assert tools.excluded_rows[0].source_index == EXPECTED_INCLUDED_ROWS
    assert tools.excluded_rows[0].message

    # Cross-stack agreement, asserted directly rather than case by case.
    assert tools.exclusions.by_reason == ud.exclusions.by_reason
    assert tools.status == ud.status
    assert tools.exclusions.included_row_count == ud.exclusions.included_row_count
    assert tools.exclusions.total_excluded == ud.exclusions.total_excluded
    assert [excluded.reason_code for excluded in tools.excluded_rows] == [
        excluded.reason_code for excluded in ud.excluded_rows
    ]
    assert [excluded.source_index for excluded in tools.excluded_rows] == [
        excluded.source_index for excluded in ud.excluded_rows
    ]


def test_resolved_d1_neither_stack_drops_a_row_in_silence(
    session_frame: pd.DataFrame,
    degenerate_frame: pd.DataFrame,
    baseline_document: dict[str, Any],
    baseline_path: Path,
) -> None:
    """RESOLVED (D1): every supplied row is accounted for on both sides.

    G1-D3 prohibits a silent drop outright, and the pre-resolution Tools stack
    committed exactly that on the ``missing_course_state`` case. With all four
    malformed rows appended at once, both stacks must report
    ``input == included + excluded`` and must name all four causes.
    """
    malformed = degenerate_frame.drop(columns=["degenerate_case"])
    mixed = pd.concat([session_frame, malformed], ignore_index=True)
    expected_by_reason = {
        "outside_baseline": 2,
        "missing_course_state": 1,
        "invalid_distance": 1,
    }

    ud = analyze_source_backed_strokes_gained(
        mixed,
        ExpectedStrokesBaselineV2.model_validate(baseline_document),
        _ud_request(),
    )
    tools = calculate_source_backed_strokes_gained(
        mixed, load_strokes_gained_baseline(baseline_path), _tools_request()
    )

    for summary in (ud.exclusions, tools.exclusions):
        assert summary.input_row_count == len(mixed)
        assert summary.included_row_count == EXPECTED_INCLUDED_ROWS
        assert summary.total_excluded == len(degenerate_frame)
        assert summary.included_row_count + summary.total_excluded == len(mixed)
        assert summary.by_reason == expected_by_reason

    assert ud.status == tools.status == "partial"
    assert len(tools.excluded_rows) == len(ud.excluded_rows) == len(degenerate_frame)
    assert tools.mean == pytest.approx(PINNED_MEAN_STROKES_GAINED, rel=1e-12)
    assert ud.value_summary.mean == pytest.approx(PINNED_MEAN_STROKES_GAINED, rel=1e-12)
