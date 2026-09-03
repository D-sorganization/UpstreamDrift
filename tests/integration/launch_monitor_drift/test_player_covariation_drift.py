"""ADR-0046 G0.1 drift gate: player covariation, UD stack vs vendored Tools.

ADR-0048 row 2 of the five ``needs-decision`` rows, and the largest previously
unmeasured surface in the platform. ADR-0046 lists player covariation as a
UD-only capability; it is not. Both stacks implement the same design — pooled,
within-player (group-mean-centred), between-player (player means), per-player
estimates, and a Fisher-z fixed/DerSimonian-Laird random-effects synthesis —
split across the same three modules.

    UD    ``player_covariation.py`` (336) + ``player_covariation_core.py``
          (427) + ``player_covariation_types.py`` (335) = 1,098 lines.
    Tools ``rate_of_closure/player_covariation.py`` (371) +
          ``_player_covariation_scan.py`` (100) + ``_player_covariation_types.py``
          (99) = 570 lines.

AGREE — 52 shared scalars compared, 51 identical inside UD's declared
12-decimal reporting quantum
    * Every per-player estimate for P1-P4 (Pearson, Spearman, OLS slope and
      intercept, r-squared, and both Fisher-z bounds), all four with n=40 and
      equal 0.25/0.25 fixed and random meta weights.
    * Pooled (n=160, r 0.060093306233), within-player (r -0.089711605547) and
      between-player (n=4, r 0.820163413566) point estimates.
    * The whole meta-analysis: fixed and random effect r -0.093447506928 with
      interval [-0.24945259306, 0.067285280773], ``tau_squared`` 0.0,
      ``q_statistic`` 0.574044790862 and ``i_squared_pct`` 0.0.
    * Both stacks raise the aggregation-reversal warning on this fixture
      (pooled +0.060 against within-player -0.090).
    * The six-pair exploratory scan returns the same pairs in the same rank
      order with the same random-effect r, ``i_squared_pct`` and
      ``direction_consistency`` (1.0 / 0.75 / 0.75 / 0.5 / 0.5 / 0.25); the
      unrestricted scan finds 15 pairs on both sides.
    * A three-shot player is excluded by both (UD ``insufficient_samples``,
      Tools ``status="insufficient_samples"``) and neither counts it toward the
      four meta contributors.

DIFFER — documented and pinned below
    D21. **The reporting quantum does not survive UD's summation reordering.**
         UD rounds every public float to 12 decimals; 51 of 52 shared scalars
         then match Tools exactly at that precision. ``q_statistic`` does not:
         UD reports 0.574044790862 and Tools 0.5740447908612423, which rounds
         to ...861. UD's ``np.vdot`` accumulation differs from Tools'
         ``np.sum(w * d**2)`` in the last bits and the difference lands across
         a rounding boundary. Max |UD - Tools| over all 52 scalars is
         7.577272143066693e-13. The same gap widens in ``i_squared_pct``, which
         is derived from ``q_statistic``: on the scan's fourth pair
         (``carry_distance_metres`` x ``session_order``) UD reports
         74.480825075496 and Tools 74.48082507549292, an absolute difference of
         3.083755473198835e-12 — three times the reporting quantum, though only
         4.14e-14 in relative terms. UD also normalises the within-player
         intercept to 0.0 where Tools reports order-1e-15 accumulation noise
         (2.9491001136044283e-15 on Windows, 2.8086667748613606e-15 on the
         Linux CI runner). UD's rounded values are pinned to the bit; Tools'
         raw values are pinned to a few ulps, because they are the ones that
         move with the BLAS.
    D22. **Tools reports a between-player Fisher interval that UD withholds.**
         With four player means Tools returns [-0.6655142653044201,
         0.9960866924324187] — a Fisher-z interval on n-3 = 1 degree of
         freedom. UD sets ``include_interval=False`` for that scope and returns
         ``None``.
    D23. **Units are inferred from column names in Tools and resolved from the
         canonical registry in UD.** Tools' suffix heuristic labels
         ``start_distance_yards`` as ``"s"`` (seconds) and ``session_order`` as
         ``"s"``; UD returns ``canonical_unit="unknown"`` with
         ``authority="unknown"`` and refuses to guess.
    D24. **A player with zero pairwise-complete rows.** UD keeps the player in
         ``per_player`` with ``state="unavailable"``,
         ``reason_code="insufficient_samples"`` and downgrades the whole result
         to ``status="partial"``. Tools drops the player from the table
         entirely, so the caller cannot tell the player was ever present.
    D25. **Request validation.** Tools accepts ``confidence_level=0.2``
         (returning a 20% interval [0.03992526547944134, 0.080212402987504]
         labelled no differently from a 95% one) and accepts
         ``player_column == x_column`` (160 "players" of one shot each). UD
         rejects both at model construction. UD caps a scan at 20 columns;
         Tools accepted 21 columns and 210 pairs.
    D26. **Identity trust and row retention.** UD refuses to run at all without
         an explicitly attested ``PlayerIdentityV2``; Tools has no identity
         concept. Tools' result carries a 160x6 ``backing_data`` frame holding
         the raw x/y values; UD's ``lineage`` carries content-addressed backing
         records instead. UD's result exposes 19 fields to Tools' 11, the
         difference being the evidence layer (availability, missingness,
         uncertainty, lineage, identity, provenance, claims, status,
         contract/analysis kind).
    D27. **Exclusion accounting.** A blank player identifier is booked by UD as
         ``excluded_by_reason={"blank_player_identity": 1}``; Tools reports it
         as "1 rows were excluded for missing or non-finite values", which
         names the wrong cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.tools.launch_monitor_model.contract_v2 import (
    AnalysisContextV2,
    PlayerIdentityV2,
)
from src.tools.launch_monitor_model.player_covariation import (
    analyze_player_covariation_v1,
    scan_player_covariation_v1,
)
from src.tools.launch_monitor_model.player_covariation_types import (
    PlayerCovariationRequestV1,
    PlayerCovariationScanRequestV1,
)
from tests.integration.launch_monitor_drift.conftest import (
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from rate_of_closure._player_covariation_types import (  # noqa: E402
    MIN_FISHER_SAMPLES,
    CovariationRequest,
    PairScanRequest,
)
from rate_of_closure.player_covariation import (  # noqa: E402
    analyze_player_covariation,
    scan_covariation_pairs,
)

X_COLUMN = "start_distance_yards"
Y_COLUMN = "carry_distance_metres"
PLAYER_COLUMN = "player_id"
SCAN_COLUMNS = (
    "start_distance_yards",
    "carry_distance_metres",
    "lateral_carry_metres",
    "session_order",
)

EXPECTED_SAMPLE_COUNT = 160
EXPECTED_PLAYER_COUNT = 4
EXPECTED_PER_PLAYER_SAMPLES = 40

# AGREE pins, UD side (Tools' raw values agree inside the 12-decimal quantum).
UD_PER_PLAYER_PEARSON = {
    "P1": -0.000937129848,
    "P2": -0.073982232796,
    "P3": -0.158714297045,
    "P4": -0.138858573218,
}
UD_POOLED_PEARSON = 0.060093306233
UD_WITHIN_PEARSON = -0.089711605547
UD_BETWEEN_PEARSON = 0.820163413566
UD_BETWEEN_SAMPLE_COUNT = 4
UD_META_EFFECT_R = -0.093447506928
UD_META_CI = (-0.24945259306, 0.067285280773)
UD_META_TAU_SQUARED = 0.0
UD_META_I_SQUARED_PCT = 0.0
EXPECTED_SCAN_PAIR_COUNT = 6
EXPECTED_UNRESTRICTED_SCAN_PAIR_COUNT = 15
EXPECTED_DIRECTION_CONSISTENCY = (1.0, 0.75, 0.75, 0.5, 0.5, 0.25)

# D21 pins.
REPORTING_QUANTUM = 1e-12
UD_Q_STATISTIC = 0.574044790862
TOOLS_Q_STATISTIC = 0.5740447908612423
MAX_ABSOLUTE_DELTA = 7.577272143066693e-13
TOOLS_WITHIN_INTERCEPT = 2.9491001136044283e-15
HETEROGENEOUS_SCAN_RANK = 4
HETEROGENEOUS_SCAN_PAIR = ("carry_distance_metres", "session_order")
UD_SCAN_I_SQUARED_PCT = 74.480825075496
TOOLS_SCAN_I_SQUARED_PCT = 74.48082507549292
SCAN_I_SQUARED_ABSOLUTE_DELTA = 3.083755473198835e-12

# D22 pins.
TOOLS_ONLY_BETWEEN_CI = (-0.6655142653044201, 0.9960866924324187)

# D25 pins.
TOOLS_LOW_CONFIDENCE_POOLED_CI = (0.03992526547944134, 0.080212402987504)
UD_SCAN_COLUMN_CAP = 20

# D26 pins.
UD_RESULT_FIELD_COUNT = 19
TOOLS_RESULT_FIELD_COUNT = 11
TOOLS_BACKING_DATA_SHAPE = (160, 6)


@pytest.fixture(scope="module")
def context() -> AnalysisContextV2:
    """The trusted-identity assertion UD requires before it will run."""
    return AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column=PLAYER_COLUMN,
            evidence="ADR-0046 cross-stack fixture declares player_id per shot",
        )
    )


def _ud_request(**overrides: object) -> PlayerCovariationRequestV1:
    settings: dict[str, object] = {
        "x_column": X_COLUMN,
        "y_column": Y_COLUMN,
        "player_column": PLAYER_COLUMN,
    }
    settings.update(overrides)
    return PlayerCovariationRequestV1(**settings)  # type: ignore[arg-type]


def _tools_request() -> CovariationRequest:
    return CovariationRequest(X_COLUMN, Y_COLUMN, PLAYER_COLUMN)


@pytest.fixture(scope="module")
def ud_result(session_frame: pd.DataFrame, context: AnalysisContextV2):
    return analyze_player_covariation_v1(session_frame, _ud_request(), context=context)


@pytest.fixture(scope="module")
def tools_result(session_frame: pd.DataFrame):
    return analyze_player_covariation(session_frame, _tools_request())


def _shared_scalars(ud_result, tools_result) -> list[tuple[str, float, float]]:
    """Every scalar both stacks report for the same selected pair."""
    pairs: list[tuple[str, float, float]] = []
    tools_players = tools_result.per_player.set_index("player_id")
    estimate_fields = (
        "pearson_r",
        "spearman_r",
        "slope",
        "intercept",
        "r_squared",
        "ci_lower",
        "ci_upper",
    )
    for item in ud_result.per_player:
        row = tools_players.loc[item.player_id]
        for field in estimate_fields:
            pairs.append(
                (
                    f"per_player[{item.player_id}].{field}",
                    getattr(item.estimate, field),
                    float(row[field]),
                )
            )
    for label, ud_scope, tools_scope in (
        ("pooled", ud_result.pooled, tools_result.pooled),
        ("within_player", ud_result.within_player, tools_result.within_player),
        ("between_player", ud_result.between_player, tools_result.between_player),
    ):
        for field in estimate_fields[:5]:
            pairs.append(
                (
                    f"{label}.{field}",
                    getattr(ud_scope, field),
                    getattr(tools_scope, field),
                )
            )
    for field in (
        "fixed_effect_r",
        "fixed_ci_lower",
        "fixed_ci_upper",
        "random_effect_r",
        "random_ci_lower",
        "random_ci_upper",
        "tau_squared",
        "q_statistic",
        "i_squared_pct",
    ):
        pairs.append(
            (
                f"meta.{field}",
                getattr(ud_result.meta_analysis, field),
                getattr(tools_result.meta_analysis, field),
            )
        )
    return pairs


def test_per_player_estimates_agree(ud_result, tools_result) -> None:
    """AGREE: four players, same n, same estimates, same meta weights."""
    tools_players = tools_result.per_player.set_index("player_id")
    assert len(ud_result.per_player) == EXPECTED_PLAYER_COUNT
    assert len(tools_players) == EXPECTED_PLAYER_COUNT

    for item in ud_result.per_player:
        row = tools_players.loc[item.player_id]
        estimate = item.estimate
        assert estimate.state == "available"
        assert row["status"] == "ok"
        assert estimate.sample_count == EXPECTED_PER_PLAYER_SAMPLES
        assert int(row["sample_count"]) == EXPECTED_PER_PLAYER_SAMPLES
        assert estimate.pearson_r == pytest.approx(
            UD_PER_PLAYER_PEARSON[item.player_id], abs=REPORTING_QUANTUM
        )
        assert estimate.pearson_r == pytest.approx(
            float(row["pearson_r"]), abs=REPORTING_QUANTUM
        )
        assert item.fixed_weight == pytest.approx(float(row["fixed_weight"]))
        assert item.random_weight == pytest.approx(float(row["random_weight"]))
        assert item.fixed_weight == pytest.approx(0.25)


def test_pooled_within_and_between_scopes_agree(ud_result, tools_result) -> None:
    """AGREE: the three aggregation scopes match on both point estimates."""
    assert ud_result.pooled.sample_count == tools_result.pooled.sample_count
    assert ud_result.pooled.sample_count == EXPECTED_SAMPLE_COUNT
    assert ud_result.pooled.group_count == tools_result.pooled.group_count
    assert ud_result.pooled.group_count == EXPECTED_PLAYER_COUNT
    assert ud_result.pooled.pearson_r == pytest.approx(
        UD_POOLED_PEARSON, abs=REPORTING_QUANTUM
    )
    assert ud_result.pooled.pearson_r == pytest.approx(
        tools_result.pooled.pearson_r, abs=REPORTING_QUANTUM
    )

    assert ud_result.within_player.pearson_r == pytest.approx(
        UD_WITHIN_PEARSON, abs=REPORTING_QUANTUM
    )
    assert ud_result.within_player.pearson_r == pytest.approx(
        tools_result.within_player.pearson_r, abs=REPORTING_QUANTUM
    )

    assert ud_result.between_player.sample_count == UD_BETWEEN_SAMPLE_COUNT
    assert tools_result.between_player.sample_count == UD_BETWEEN_SAMPLE_COUNT
    assert ud_result.between_player.pearson_r == pytest.approx(
        UD_BETWEEN_PEARSON, abs=REPORTING_QUANTUM
    )
    assert ud_result.between_player.pearson_r == pytest.approx(
        tools_result.between_player.pearson_r, abs=REPORTING_QUANTUM
    )


def test_meta_analysis_agrees(ud_result, tools_result) -> None:
    """AGREE: the Fisher-z fixed/DerSimonian-Laird synthesis is the same."""
    ud_meta = ud_result.meta_analysis
    tools_meta = tools_result.meta_analysis

    assert ud_meta.contributor_count == tools_meta.contributor_count
    assert ud_meta.contributor_count == EXPECTED_PLAYER_COUNT
    assert ud_meta.total_sample_count == tools_meta.total_sample_count
    assert ud_meta.total_sample_count == EXPECTED_SAMPLE_COUNT

    for field, expected in (
        ("fixed_effect_r", UD_META_EFFECT_R),
        ("random_effect_r", UD_META_EFFECT_R),
        ("fixed_ci_lower", UD_META_CI[0]),
        ("fixed_ci_upper", UD_META_CI[1]),
        ("random_ci_lower", UD_META_CI[0]),
        ("random_ci_upper", UD_META_CI[1]),
        ("tau_squared", UD_META_TAU_SQUARED),
        ("i_squared_pct", UD_META_I_SQUARED_PCT),
    ):
        ud_value = getattr(ud_meta, field)
        assert ud_value == pytest.approx(expected, abs=REPORTING_QUANTUM)
        assert ud_value == pytest.approx(
            getattr(tools_meta, field), abs=REPORTING_QUANTUM
        )
    # tau^2 = 0 collapses random effects onto fixed effects in both stacks.
    assert ud_meta.fixed_effect_r == ud_meta.random_effect_r
    assert tools_meta.fixed_effect_r == tools_meta.random_effect_r


def test_aggregation_reversal_is_flagged_by_both(ud_result, tools_result) -> None:
    """AGREE: pooled +0.060 against within-player -0.090 warns on both sides."""
    reversal = "Possible aggregation reversal"
    assert any(reversal in warning for warning in ud_result.warnings)
    assert any(reversal in warning for warning in tools_result.warnings)
    assert np.sign(ud_result.pooled.pearson_r) != np.sign(
        ud_result.within_player.pearson_r
    )


def test_pair_scan_ranking_agrees(
    session_frame: pd.DataFrame, context: AnalysisContextV2
) -> None:
    """AGREE: same pairs, same order, same effects and consistency scores."""
    ud_scan = scan_player_covariation_v1(
        session_frame,
        PlayerCovariationScanRequestV1(
            player_column=PLAYER_COLUMN, numeric_columns=SCAN_COLUMNS
        ),
        context=context,
    )
    tools_scan = scan_covariation_pairs(
        session_frame,
        PairScanRequest(player_column=PLAYER_COLUMN, numeric_columns=SCAN_COLUMNS),
    )

    assert ud_scan.pair_count == EXPECTED_SCAN_PAIR_COUNT
    assert len(tools_scan.ranking) == EXPECTED_SCAN_PAIR_COUNT
    assert ud_scan.status == "available"

    for index, ud_pair in enumerate(ud_scan.ranking):
        tools_pair = tools_scan.ranking.iloc[index]
        assert ud_pair.rank == index + 1
        assert (ud_pair.x_column, ud_pair.y_column) == (
            tools_pair["x_column"],
            tools_pair["y_column"],
        )
        assert ud_pair.random_effect_r == pytest.approx(
            float(tools_pair["random_effect_r"]), abs=REPORTING_QUANTUM
        )
        assert ud_pair.fixed_effect_r == pytest.approx(
            float(tools_pair["fixed_effect_r"]), abs=REPORTING_QUANTUM
        )
        # Relative, not absolute: D21's accumulation gap is visible here.
        assert ud_pair.i_squared_pct == pytest.approx(
            float(tools_pair["i_squared_pct"]), rel=1e-12
        )
        assert ud_pair.direction_consistency == pytest.approx(
            EXPECTED_DIRECTION_CONSISTENCY[index]
        )
        assert ud_pair.direction_consistency == pytest.approx(
            float(tools_pair["direction_consistency"])
        )

    # Unrestricted selection finds the same six numeric columns on both sides.
    ud_all = scan_player_covariation_v1(
        session_frame,
        PlayerCovariationScanRequestV1(player_column=PLAYER_COLUMN),
        context=context,
    )
    tools_all = scan_covariation_pairs(
        session_frame, PairScanRequest(player_column=PLAYER_COLUMN)
    )
    assert ud_all.pair_count == EXPECTED_UNRESTRICTED_SCAN_PAIR_COUNT
    assert len(tools_all.ranking) == EXPECTED_UNRESTRICTED_SCAN_PAIR_COUNT


def test_per_player_sample_floor_agrees(
    session_frame: pd.DataFrame, context: AnalysisContextV2
) -> None:
    """AGREE: a three-shot player is excluded by both, not silently pooled."""
    frame = pd.concat(
        [
            session_frame,
            pd.DataFrame(
                [
                    {**session_frame.iloc[index].to_dict(), PLAYER_COLUMN: "P5"}
                    for index in range(3)
                ]
            ),
        ],
        ignore_index=True,
    )

    ud = analyze_player_covariation_v1(frame, _ud_request(), context=context)
    tools = analyze_player_covariation(frame, _tools_request())

    ud_p5 = next(item for item in ud.per_player if item.player_id == "P5")
    tools_p5 = tools.per_player.set_index("player_id").loc["P5"]
    assert ud_p5.estimate.state == "unavailable"
    assert ud_p5.estimate.reason_code == "insufficient_samples"
    assert ud_p5.estimate.sample_count == 3
    assert tools_p5["status"] == "insufficient_samples"
    assert int(tools_p5["sample_count"]) == 3
    assert ud.meta_analysis.contributor_count == EXPECTED_PLAYER_COUNT
    assert tools.meta_analysis.contributor_count == EXPECTED_PLAYER_COUNT
    assert MIN_FISHER_SAMPLES == 4


def test_divergence_d21_reporting_quantum_and_summation_order(
    ud_result, tools_result
) -> None:
    """DIFFER (D21): 51 of 52 scalars round-trip; ``q_statistic`` does not."""
    pairs = _shared_scalars(ud_result, tools_result)
    assert len(pairs) == 52

    deltas = {name: abs(ud - tools) for name, ud, tools in pairs}
    worst_name = max(deltas, key=lambda key: deltas[key])
    assert worst_name == "meta.q_statistic"
    assert deltas[worst_name] == pytest.approx(MAX_ABSOLUTE_DELTA, rel=1e-3, abs=0.0)
    assert deltas[worst_name] < REPORTING_QUANTUM

    mismatched = [name for name, ud, tools in pairs if round(tools, 12) != ud]
    assert mismatched == ["meta.q_statistic"]
    # UD's value is exact because ``_reported_float`` rounds it; Tools' is not,
    # so it is pinned to a few ulps rather than to the bit.
    assert ud_result.meta_analysis.q_statistic == UD_Q_STATISTIC
    tools_q_statistic = tools_result.meta_analysis.q_statistic
    assert tools_q_statistic == pytest.approx(TOOLS_Q_STATISTIC, rel=1e-14, abs=0.0)
    assert round(tools_q_statistic, 12) != UD_Q_STATISTIC

    # UD also normalises a residual-scale intercept to a clean zero. Tools'
    # value is BLAS-dependent accumulation noise, so only its scale is pinned.
    assert ud_result.within_player.intercept == 0.0
    tools_intercept = tools_result.within_player.intercept
    assert tools_intercept != 0.0
    assert abs(tools_intercept) < REPORTING_QUANTUM
    assert abs(tools_intercept) == pytest.approx(
        abs(TOOLS_WITHIN_INTERCEPT), rel=0.5, abs=0.0
    )
    assert round(tools_intercept, 12) == 0.0


def test_divergence_d21_accumulation_gap_widens_in_derived_heterogeneity(
    session_frame: pd.DataFrame, context: AnalysisContextV2
) -> None:
    """DIFFER (D21): ``i_squared_pct`` is derived from ``q_statistic`` and
    inherits its accumulation gap, at 4x the 12-decimal quantum."""
    ud_scan = scan_player_covariation_v1(
        session_frame,
        PlayerCovariationScanRequestV1(
            player_column=PLAYER_COLUMN, numeric_columns=SCAN_COLUMNS
        ),
        context=context,
    )
    tools_scan = scan_covariation_pairs(
        session_frame,
        PairScanRequest(player_column=PLAYER_COLUMN, numeric_columns=SCAN_COLUMNS),
    )

    ud_pair = ud_scan.ranking[HETEROGENEOUS_SCAN_RANK - 1]
    tools_pair = tools_scan.ranking.iloc[HETEROGENEOUS_SCAN_RANK - 1]
    assert (ud_pair.x_column, ud_pair.y_column) == HETEROGENEOUS_SCAN_PAIR

    tools_i_squared = float(tools_pair["i_squared_pct"])
    assert ud_pair.i_squared_pct == UD_SCAN_I_SQUARED_PCT
    assert tools_i_squared == pytest.approx(
        TOOLS_SCAN_I_SQUARED_PCT, rel=1e-14, abs=0.0
    )
    delta = abs(UD_SCAN_I_SQUARED_PCT - tools_i_squared)
    assert delta == pytest.approx(SCAN_I_SQUARED_ABSOLUTE_DELTA, rel=1e-3, abs=0.0)
    assert delta > REPORTING_QUANTUM
    assert delta / tools_i_squared < 1e-13
    assert round(tools_i_squared, 12) != UD_SCAN_I_SQUARED_PCT


def test_divergence_d22_between_player_interval_exists_only_in_tools(
    ud_result, tools_result
) -> None:
    """DIFFER (D22): a Fisher interval on four player means, on one side only."""
    assert ud_result.between_player.ci_lower is None
    assert ud_result.between_player.ci_upper is None
    assert tools_result.between_player.ci_lower == pytest.approx(
        TOOLS_ONLY_BETWEEN_CI[0], rel=1e-12
    )
    assert tools_result.between_player.ci_upper == pytest.approx(
        TOOLS_ONLY_BETWEEN_CI[1], rel=1e-12
    )
    # Both stacks do withhold the within-player interval.
    assert ud_result.within_player.ci_lower is None
    assert tools_result.within_player.ci_lower is None
    assert ud_result.uncertainty.within_player_interval == "unavailable-clustered"


def test_divergence_d23_unit_resolution_differs(ud_result, tools_result) -> None:
    """DIFFER (D23): a name-suffix guess against the canonical registry."""
    assert tools_result.units == {"x": "s", "y": "m"}
    assert ud_result.units[X_COLUMN].canonical_unit == "unknown"
    assert ud_result.units[X_COLUMN].authority == "unknown"
    assert ud_result.units[Y_COLUMN].canonical_unit == "unknown"
    assert ud_result.units[Y_COLUMN].authority == "unknown"


def test_divergence_d24_player_with_no_complete_rows(
    session_frame: pd.DataFrame, context: AnalysisContextV2
) -> None:
    """DIFFER (D24): UD books the player as excluded; Tools erases them."""
    ghost = pd.concat(
        [
            session_frame,
            pd.DataFrame(
                [
                    {
                        **session_frame.iloc[0].to_dict(),
                        PLAYER_COLUMN: "P5",
                        X_COLUMN: np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    ud = analyze_player_covariation_v1(ghost, _ud_request(), context=context)
    tools = analyze_player_covariation(ghost, _tools_request())

    assert [item.player_id for item in ud.per_player] == ["P1", "P2", "P3", "P4", "P5"]
    ghost_player = next(item for item in ud.per_player if item.player_id == "P5")
    assert ghost_player.estimate.state == "unavailable"
    assert ghost_player.estimate.reason_code == "insufficient_samples"
    assert ghost_player.estimate.sample_count == 0
    assert ud.status == "partial"
    assert ud.missingness.excluded_by_reason == {
        "blank_player_identity": 0,
        "pairwise_incomplete": 1,
    }
    assert ud.missingness.excluded_player_count_by_reason == {"insufficient_samples": 1}

    assert list(tools.per_player["player_id"]) == ["P1", "P2", "P3", "P4"]
    assert "P5" not in set(tools.per_player["player_id"])


def test_divergence_d25_request_validation_differs(
    session_frame: pd.DataFrame, context: AnalysisContextV2
) -> None:
    """DIFFER (D25): Tools runs three requests UD rejects outright."""
    with pytest.raises(ValidationError):
        _ud_request(confidence_level=0.2)
    with pytest.raises(ValidationError):
        _ud_request(player_column=X_COLUMN)

    low_confidence = analyze_player_covariation(
        session_frame,
        CovariationRequest(X_COLUMN, Y_COLUMN, PLAYER_COLUMN, MIN_FISHER_SAMPLES, 0.2),
    )
    assert low_confidence.pooled.ci_lower == pytest.approx(
        TOOLS_LOW_CONFIDENCE_POOLED_CI[0], rel=1e-12
    )
    assert low_confidence.pooled.ci_upper == pytest.approx(
        TOOLS_LOW_CONFIDENCE_POOLED_CI[1], rel=1e-12
    )

    self_identified = analyze_player_covariation(
        session_frame, CovariationRequest(X_COLUMN, Y_COLUMN, X_COLUMN)
    )
    assert self_identified.pooled.group_count == EXPECTED_SAMPLE_COUNT

    # 210 pairs, so keep the row count small; the cap is what is under test.
    wide = session_frame.head(80).copy()
    for index in range(20):
        wide[f"extra_{index:02d}"] = np.linspace(0.0, 1.0, len(wide)) + float(index)
    columns = tuple(
        sorted(column for column in wide.columns if wide[column].dtype.kind == "f")
    )[: UD_SCAN_COLUMN_CAP + 1]
    assert len(columns) == UD_SCAN_COLUMN_CAP + 1

    with pytest.raises(ValidationError):
        scan_player_covariation_v1(
            wide,
            PlayerCovariationScanRequestV1(
                player_column=PLAYER_COLUMN, numeric_columns=columns
            ),
            context=context,
        )
    oversized = scan_covariation_pairs(
        wide, PairScanRequest(player_column=PLAYER_COLUMN, numeric_columns=columns)
    )
    assert len(oversized.ranking) == 210


def test_divergence_d26_identity_trust_and_row_retention(
    session_frame: pd.DataFrame, ud_result, tools_result
) -> None:
    """DIFFER (D26): an identity gate and a row-free result on one side only."""
    with pytest.raises(ValueError, match="explicit trusted player identity"):
        analyze_player_covariation_v1(session_frame, _ud_request())

    # Tools has no identity concept and ran unconditionally.
    assert tools_result.pooled.sample_count == EXPECTED_SAMPLE_COUNT

    assert tools_result.backing_data.shape == TOOLS_BACKING_DATA_SHAPE
    assert set(tools_result.backing_data.columns) == {
        "source_index",
        "player_id",
        "x",
        "y",
        "centered_x",
        "centered_y",
    }
    assert not hasattr(ud_result, "backing_data")
    assert len(ud_result.lineage.backing_records) == EXPECTED_SAMPLE_COUNT

    assert len(type(ud_result).model_fields) == UD_RESULT_FIELD_COUNT
    assert len(type(tools_result).__dataclass_fields__) == TOOLS_RESULT_FIELD_COUNT
    ud_only = set(type(ud_result).model_fields) - set(
        type(tools_result).__dataclass_fields__
    )
    assert ud_only == {
        "analysis_kind",
        "availability",
        "claims",
        "contract_version",
        "lineage",
        "missingness",
        "player_identity",
        "status",
        "uncertainty",
        "vendor_provenance",
    }
    assert set(type(tools_result).__dataclass_fields__) - set(
        type(ud_result).model_fields
    ) == {"backing_data", "method_description"}


def test_divergence_d27_exclusion_accounting_names_the_cause_only_in_ud(
    session_frame: pd.DataFrame, context: AnalysisContextV2
) -> None:
    """DIFFER (D27): Tools books a blank identity as a non-finite value."""
    blank = pd.concat(
        [
            session_frame,
            pd.DataFrame([{**session_frame.iloc[0].to_dict(), PLAYER_COLUMN: "  "}]),
        ],
        ignore_index=True,
    )

    ud = analyze_player_covariation_v1(blank, _ud_request(), context=context)
    tools = analyze_player_covariation(blank, _tools_request())

    assert ud.pooled.sample_count == EXPECTED_SAMPLE_COUNT
    assert tools.pooled.sample_count == EXPECTED_SAMPLE_COUNT
    assert ud.missingness.input_row_count == EXPECTED_SAMPLE_COUNT + 1
    assert ud.missingness.excluded_by_reason == {
        "blank_player_identity": 1,
        "pairwise_incomplete": 0,
    }
    assert ud.missingness.non_finite_by_variable == {X_COLUMN: 0, Y_COLUMN: 0}

    assert any(
        "excluded for missing or non-finite values" in warning
        for warning in tools.warnings
    )
    assert not hasattr(tools, "missingness")
