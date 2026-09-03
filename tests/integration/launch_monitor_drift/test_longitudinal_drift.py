"""ADR-0046 G0 drift gate: longitudinal trend, UD vs vendored Tools stack.

Metric under test: ``proximity_yards`` (the finish distance to the hole,
lower-is-better), on the same 160-shot session — 4 players x 5 sessions x 8
shots. Both stacks collapse shots to one cell per player-session before
fitting, and both refuse to run without explicit identity attestation.

AGREE — asserted exactly
    * Session cell count: 20 == 20, with identical (player, session) keys.
    * Session cell means: max |UD - Tools| = 0.0 across all 20 cells.
    * Per-player OLS slope on those cells: max |UD - Tools| = 0.0.
      P1 -0.6333284530839894, P2 -0.6566383830927385,
      P3 +0.2922907370953632, P4 -1.1044500082020998 (yd/session).

RESOLVED — G1-D1's named-method pair (ADR-0048)
    D10. **RESOLVED.** The pooled estimate used to be a *different estimator*
         in each stack, and on this fixture the two disagreed about statistical
         significance: UD's player-FE OLS with cluster-robust standard errors
         returned -0.52553153 on [-1.57630, +0.52524] (crosses zero,
         p = 0.20969656) while Tools' inverse-variance DerSimonian-Laird
         random effects returned -0.52827898 on [-1.01454, -0.04202] (does
         not). Point estimates within 0.52%, UD's interval 2.16x wider — the
         same data reading "no detectable trend" one side and "improving" the
         other.

         ADR-0048 Decision **G1-D1** ruled that neither is "the" right answer
         at k = 4 and preserved both as named, provenance-carrying options,
         exactly as ADR-0045 preserved the two putting roll models. The
         canonical layer this repository now consumes carries the pair:
         ``ud-cluster-robust-fe/1`` (UD's estimator, arithmetic unchanged) and
         ``dl-random-effects/1`` (the ``rate_of_closure`` estimator). Both
         gates below still run, but they no longer measure a *divergence*:
         ``pooled_method`` selects which estimator answers, the result names
         it, and ``dl-random-effects/1`` reproduces every Tools number
         bit-for-bit. Every value pinned as "Tools-only" is now also pinned as
         a canonical output, at delta exactly 0.0.
    D11. **RESOLVED.** Per-player SE/CI/p/R2/first-to-last change were Tools-
         only; ``LongitudinalPlayerAssociationV1`` carried the point estimate
         and a direction label alone and "cannot express per-player slope
         uncertainty at all". G1-D1's stated consequence closed that gap in
         the same change, and the six fields now agree with Tools exactly.
    D12. **RESOLVED.** tau^2 0.15941371, Q 9.79986169, I^2 69.38732305% and
         ``improvement_probability`` 0.98338660 were Tools-only outputs. They
         are the ``dl-random-effects/1`` half of the pair and are reported —
         and only reported — when that estimator is the one selected;
         ``ud-cluster-robust-fe/1`` still refuses to invent them, and the
         contract's own validator enforces that.

DIFFER — documented and pinned below
    D13. Cluster floors. UD's contract hard-floors
         ``minimum_player_clusters`` at 4 (pydantic ``ge=4``); Tools synthesises
         a population effect from 2 contributors.
    D14. Failure posture. UD returns a structured ``status="unavailable"``
         result with a ``reason_code``; Tools raises ``ValueError``. Same
         input, one auditable result vs one exception.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from shared.python.launch_monitor.contract_v2 import (
    AnalysisContextV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
)
from shared.python.launch_monitor.longitudinal import (
    analyze_longitudinal_sessions,
)
from shared.python.launch_monitor.longitudinal_types import (
    LongitudinalSessionRequestV1,
    PooledMethod,
)
from tests.integration.launch_monitor_drift.conftest import (
    YARDS_PER_METRE,
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from rate_of_closure.launch_monitor_longitudinal import (  # noqa: E402
    LongitudinalRequest,
    analyze_longitudinal_performance,
)

METRIC = "proximity_yards"
EXPECTED_SESSION_CELLS = 20
EXPECTED_PLAYERS = ("P1", "P2", "P3", "P4")

PINNED_PLAYER_SLOPES = {
    "P1": -0.6333284530839894,
    "P2": -0.6566383830927385,
    "P3": 0.2922907370953632,
    "P4": -1.1044500082020998,
}

# D10 pins. Both halves of G1-D1's named-method pair, pinned side by side:
# ``ud-cluster-robust-fe/1`` keeps UD's numbers unchanged, and
# ``dl-random-effects/1`` reproduces the ``rate_of_closure`` numbers below.
UD_POOLED_SLOPE = -0.5255315268208663
UD_POOLED_STANDARD_ERROR = 0.3301766523964166
UD_POOLED_CI_LOW = -1.5763009943307855
UD_POOLED_CI_HIGH = 0.5252379406890527
UD_POOLED_P_VALUE = 0.20969656193018768
TOOLS_RANDOM_SLOPE = -0.5282789828979909
TOOLS_RANDOM_CI_LOW = -1.0145384362562389
TOOLS_RANDOM_CI_HIGH = -0.04201952953974292
TOOLS_FIXED_SLOPE = -0.6399797407097567
TOOLS_FIXED_CI_LOW = -0.7473117002746321
TOOLS_FIXED_CI_HIGH = -0.5326477811448813
POOLED_SLOPE_RATIO_UD_OVER_TOOLS_RANDOM = 0.99479923266670045
POOLED_INTERVAL_WIDTH_RATIO = 2.1609234746039427

# D11/D12 pins. Pinned as Tools-only when G0 measured them; now pinned as
# canonical outputs too, at delta exactly 0.0 against the same values.
TOOLS_ONLY_P1_STANDARD_ERROR = 0.427793725141183
TOOLS_ONLY_P1_P_VALUE = 0.23532861203653252
TOOLS_ONLY_P1_R_SQUARED = 0.4221591266852809
TOOLS_ONLY_P1_FIRST_TO_LAST = -1.0598996609798768
TOOLS_ONLY_TAU_SQUARED = 0.1594137105940229
TOOLS_ONLY_Q_STATISTIC = 9.799861688653488
TOOLS_ONLY_I_SQUARED_PCT = 69.38732305300319
TOOLS_ONLY_IMPROVEMENT_PROBABILITY = 0.9833865960693259


@pytest.fixture(scope="module")
def metric_frame(session_frame: pd.DataFrame) -> pd.DataFrame:
    """Add the shared lower-is-better metric both stacks read by name."""
    frame = session_frame.copy()
    frame[METRIC] = frame["finish_distance_metres"] * YARDS_PER_METRE
    return frame


def _ud_context() -> AnalysisContextV2:
    return AnalysisContextV2(
        player_identity=PlayerIdentityV2(
            trust_level="pseudonymous_stable",
            identifier_column="player_id",
            evidence="Synthetic ADR-0046 G0 fixture pseudonym.",
        ),
        session_identity=SessionIdentityV2(
            trust_level="explicit_user_attested",
            identifier_column="session_id",
            evidence="Synthetic ADR-0046 G0 fixture session identifier.",
        ),
        order_evidence=OrderEvidenceV2(
            trust_level="explicit_user_attested",
            order_column="session_order",
            order_kind="ordinal",
            unit="session",
            evidence="Synthetic ADR-0046 G0 fixture session ordinal.",
        ),
    )


def _ud_request(
    pooled_method: str = "ud-cluster-robust-fe/1",
) -> LongitudinalSessionRequestV1:
    """Build the canonical request, naming which G1-D1 estimator answers."""
    return LongitudinalSessionRequestV1(
        metric=METRIC,
        direction="lower_is_better",
        session_aggregate="mean",
        minimum_sessions_per_player=3,
        minimum_player_clusters=4,
        confidence_level=0.95,
        pooled_method=cast(PooledMethod, pooled_method),
    )


def _tools_request() -> LongitudinalRequest:
    return LongitudinalRequest(
        metric_column=METRIC,
        session_column="session_id",
        session_order_column="session_order",
        player_column="player_id",
        player_identity_attested=True,
        session_identity_attested=True,
        higher_is_better=False,
        confidence_level=0.95,
        min_sessions=3,
    )


@pytest.fixture(scope="module")
def ud_result(metric_frame: pd.DataFrame):
    return analyze_longitudinal_sessions(
        metric_frame, _ud_request(), context=_ud_context()
    )


@pytest.fixture(scope="module")
def ud_random_effects_result(metric_frame: pd.DataFrame):
    """The same analysis under G1-D1's other named estimator."""
    return analyze_longitudinal_sessions(
        metric_frame, _ud_request("dl-random-effects/1"), context=_ud_context()
    )


@pytest.fixture(scope="module")
def tools_result(metric_frame: pd.DataFrame):
    return analyze_longitudinal_performance(metric_frame, _tools_request())


def test_session_cells_agree_exactly(ud_result, tools_result) -> None:
    """AGREE: identical player-session cells with identical means."""
    ud_cells = {
        (item.player_id, item.session_id): item.metric_value
        for item in ud_result.session_aggregates
    }
    tools_cells = {
        (point.player_id, point.session_id): point.mean
        for point in tools_result.session_points
    }

    assert ud_result.status == "available"
    assert ud_result.missingness.session_cell_count == EXPECTED_SESSION_CELLS
    assert len(tools_cells) == EXPECTED_SESSION_CELLS
    assert set(ud_cells) == set(tools_cells)
    deltas = [abs(ud_cells[key] - tools_cells[key]) for key in ud_cells]
    assert max(deltas) == 0.0, f"session-mean drift {max(deltas):.3e}"
    assert all(item.shot_count == 8 for item in ud_result.session_aggregates)


def test_per_player_slopes_agree_exactly(ud_result, tools_result) -> None:
    """AGREE: identical OLS slope per player on the identical cells."""
    ud_slopes = {
        item.player_id: item.estimate_per_order_unit
        for item in ud_result.player_associations
    }
    tools_slopes = {
        player.player_id: player.slope_per_session for player in tools_result.players
    }

    assert set(ud_slopes) == set(tools_slopes) == set(EXPECTED_PLAYERS)
    for player, expected in PINNED_PLAYER_SLOPES.items():
        assert ud_slopes[player] == pytest.approx(expected, rel=1e-12)
        assert tools_slopes[player] == pytest.approx(expected, rel=1e-12)
        assert ud_slopes[player] - tools_slopes[player] == 0.0


def test_resolved_d10_the_named_method_pair_carries_both_estimators(
    ud_result, ud_random_effects_result, tools_result
) -> None:
    """RESOLVED (D10): both estimators survive, each naming itself.

    G0 pinned this as the sharpest divergence in the longitudinal pair: the
    same four per-player slopes went in and opposite significance verdicts
    came out. ADR-0048 Decision **G1-D1** ruled that neither is "the" right
    answer at k = 4 and preserved both as named, provenance-carrying options.

    The pins below are the *same numbers* G0 recorded - none moved. What moved
    is which side of the gate they sit on: UD's cluster-robust estimate is
    still exactly UD's, and the ``rate_of_closure`` random-effects estimate is
    now also reachable from the canonical layer, at delta exactly ``0.0``.
    """
    pooled = ud_result.pooled_association
    population = tools_result.population
    assert pooled is not None

    # ``ud-cluster-robust-fe/1`` - UD's estimator, arithmetic untouched. The
    # identifier is the one G1-D1 named; the old free-text ``method`` string
    # ``player_fixed_effects_ols_clustered_by_player`` is gone by decision.
    assert pooled.method == "ud-cluster-robust-fe/1"
    assert pooled.estimate_per_order_unit == pytest.approx(UD_POOLED_SLOPE, rel=1e-9)
    assert pooled.standard_error == pytest.approx(UD_POOLED_STANDARD_ERROR, rel=1e-9)
    assert pooled.confidence_interval_low == pytest.approx(UD_POOLED_CI_LOW, rel=1e-9)
    assert pooled.confidence_interval_high == pytest.approx(UD_POOLED_CI_HIGH, rel=1e-9)
    assert pooled.p_value == pytest.approx(UD_POOLED_P_VALUE, rel=1e-9)

    # ``rate_of_closure`` is unchanged by the decision and still pins the same
    # random-effects and fixed-effects numbers it always did.
    assert population.random_effect_slope == pytest.approx(TOOLS_RANDOM_SLOPE, rel=1e-9)
    assert population.random_ci_lower == pytest.approx(TOOLS_RANDOM_CI_LOW, rel=1e-9)
    assert population.random_ci_upper == pytest.approx(TOOLS_RANDOM_CI_HIGH, rel=1e-9)
    assert population.fixed_effect_slope == pytest.approx(TOOLS_FIXED_SLOPE, rel=1e-9)
    assert population.fixed_ci_lower == pytest.approx(TOOLS_FIXED_CI_LOW, rel=1e-9)
    assert population.fixed_ci_upper == pytest.approx(TOOLS_FIXED_CI_HIGH, rel=1e-9)

    # ``dl-random-effects/1`` - the pair's other half, reproducing the
    # ``rate_of_closure`` estimate bit-for-bit rather than approximately.
    paired = ud_random_effects_result.pooled_association
    assert paired is not None
    assert paired.method == "dl-random-effects/1"
    assert paired.estimate_per_order_unit - population.random_effect_slope == 0.0
    assert paired.confidence_interval_low - population.random_ci_lower == 0.0
    assert paired.confidence_interval_high - population.random_ci_upper == 0.0

    # The two verdicts still disagree - that was never a defect to fix, and
    # G1-D1 exists precisely so a reader is told which one they are reading.
    assert pooled.confidence_interval_low < 0.0 < pooled.confidence_interval_high
    assert paired.confidence_interval_high < 0.0
    ratio = pooled.estimate_per_order_unit / paired.estimate_per_order_unit
    assert ratio == pytest.approx(POOLED_SLOPE_RATIO_UD_OVER_TOOLS_RANDOM, rel=1e-9)
    ud_width = pooled.confidence_interval_high - pooled.confidence_interval_low
    paired_width = paired.confidence_interval_high - paired.confidence_interval_low
    assert ud_width / paired_width == pytest.approx(
        POOLED_INTERVAL_WIDTH_RATIO, rel=1e-9
    )

    # A result never reports one estimator's number under the other's name:
    # the availability record names the estimator that produced it.
    pooled_availability = next(
        item
        for item in ud_result.availability
        if item.result_path == "pooled_association"
    )
    assert "clustered by player" in str(pooled_availability.message)
    paired_availability = next(
        item
        for item in ud_random_effects_result.availability
        if item.result_path == "pooled_association"
    )
    assert "DerSimonian-Laird" in str(paired_availability.message)


def test_resolved_d11_per_player_uncertainty_agrees_exactly(
    ud_result, tools_result
) -> None:
    """RESOLVED (D11): the six per-player uncertainty fields now agree at 0.0.

    G0 recorded that ``rate_of_closure`` "reports SE/CI/p/R2/first-to-last
    change per player" while UD's ``LongitudinalPlayerAssociationV1`` "carries
    the point estimate and a direction label only" and "cannot express
    per-player slope uncertainty at all". G1-D1's stated consequence closed
    that gap in the same change. The slope arithmetic is untouched - the
    ``test_per_player_slopes_agree_exactly`` pins above are unchanged - and the
    uncertainty ``linregress`` was already computing is now reported.
    """
    ud_fields = set(type(ud_result.player_associations[0]).model_fields)
    assert {
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_value",
        "r_squared",
        "first_to_last_change",
    } <= ud_fields
    # D11 adds; every field the record carried before is still there.
    assert {
        "player_id",
        "session_count",
        "estimate_per_order_unit",
        "direction",
        "state",
        "reason_code",
    } <= ud_fields

    ud_p1 = next(
        item for item in ud_result.player_associations if item.player_id == "P1"
    )
    p1 = next(player for player in tools_result.players if player.player_id == "P1")
    assert p1.standard_error == pytest.approx(TOOLS_ONLY_P1_STANDARD_ERROR, rel=1e-9)
    assert p1.p_value == pytest.approx(TOOLS_ONLY_P1_P_VALUE, rel=1e-9)
    assert p1.r_squared == pytest.approx(TOOLS_ONLY_P1_R_SQUARED, rel=1e-9)
    assert p1.first_to_last_change == pytest.approx(
        TOOLS_ONLY_P1_FIRST_TO_LAST, rel=1e-9
    )

    # The same four numbers, from the canonical layer, at delta exactly 0.0.
    assert ud_p1.standard_error is not None
    assert ud_p1.standard_error - p1.standard_error == 0.0
    assert ud_p1.p_value is not None and ud_p1.p_value - p1.p_value == 0.0
    assert ud_p1.r_squared is not None and ud_p1.r_squared - p1.r_squared == 0.0
    assert ud_p1.first_to_last_change is not None
    assert ud_p1.first_to_last_change - p1.first_to_last_change == 0.0


def test_resolved_d12_heterogeneity_is_reported_by_the_estimator_that_has_it(
    ud_result, ud_random_effects_result, tools_result
) -> None:
    """RESOLVED (D12): tau^2 / Q / I^2 / P(improvement) agree at delta 0.0.

    They are not a general capability of the contract, and G1-D1 is explicit
    about why: they are ``dl-random-effects/1`` outputs. Selecting the other
    estimator does not silently produce them - ``PooledAssociationV1``'s own
    validator refuses a ``ud-cluster-robust-fe/1`` result carrying a
    heterogeneity block, which is the "results from different estimators are
    never numerically compared without the names attached" rule made
    executable rather than documented.
    """
    population = tools_result.population
    assert population.tau_squared == pytest.approx(TOOLS_ONLY_TAU_SQUARED, rel=1e-9)
    assert population.q_statistic == pytest.approx(TOOLS_ONLY_Q_STATISTIC, rel=1e-9)
    assert population.i_squared_pct == pytest.approx(TOOLS_ONLY_I_SQUARED_PCT, rel=1e-9)
    assert population.improvement_probability == pytest.approx(
        TOOLS_ONLY_IMPROVEMENT_PROBABILITY, rel=1e-9
    )

    paired = ud_random_effects_result.pooled_association
    assert paired is not None
    assert paired.tau_squared is not None
    assert paired.tau_squared - population.tau_squared == 0.0
    assert paired.q_statistic is not None
    assert paired.q_statistic - population.q_statistic == 0.0
    assert paired.i_squared_pct is not None
    assert paired.i_squared_pct - population.i_squared_pct == 0.0
    assert paired.improvement_probability is not None
    assert paired.improvement_probability - population.improvement_probability == 0.0

    # The cluster-robust half reports none of them, and cannot be made to.
    pooled = ud_result.pooled_association
    assert pooled is not None
    assert pooled.tau_squared is None
    assert pooled.q_statistic is None
    assert pooled.i_squared_pct is None
    assert pooled.improvement_probability is None
    with pytest.raises(ValueError, match="does not estimate between-player"):
        type(pooled).model_validate(
            {
                **pooled.model_dump(),
                "tau_squared": TOOLS_ONLY_TAU_SQUARED,
                "q_statistic": TOOLS_ONLY_Q_STATISTIC,
                "i_squared_pct": TOOLS_ONLY_I_SQUARED_PCT,
            }
        )


def test_divergence_d13_cluster_floors_differ(metric_frame: pd.DataFrame) -> None:
    """DIFFER (D13): UD demands 4 player clusters; Tools synthesises from 2."""
    with pytest.raises(ValueError, match="greater than or equal to 4"):
        LongitudinalSessionRequestV1(metric=METRIC, minimum_player_clusters=2)

    two_players = metric_frame[metric_frame["player_id"].isin(["P1", "P2"])]
    ud = analyze_longitudinal_sessions(
        two_players, _ud_request(), context=_ud_context()
    )
    assert ud.status == "partial"
    assert ud.pooled_association is None
    reasons = {item.reason_code for item in ud.availability}
    assert "insufficient_player_clusters" in reasons

    tools = analyze_longitudinal_performance(two_players, _tools_request())
    assert tools.population.contributor_count == 2
    assert tools.population.random_effect_slope is not None


def test_divergence_d14_failure_posture_differs(metric_frame: pd.DataFrame) -> None:
    """DIFFER (D14): UD returns an auditable result; Tools raises.

    Both stacks reject a session whose rows disagree about the session's order
    value, but only UD's rejection is inspectable.
    """
    broken = metric_frame.copy()
    mask = broken["session_id"] == "P1-S2"
    broken.loc[broken.index[mask][:4], "session_order"] = 9.0

    ud = analyze_longitudinal_sessions(broken, _ud_request(), context=_ud_context())
    assert ud.status == "unavailable"
    assert ud.pooled_association is None
    assert ud.availability[0].reason_code == "nonconstant_session_order"
    assert ud.missingness.input_row_count == len(broken)

    with pytest.raises(ValueError, match="exactly one order"):
        analyze_longitudinal_performance(broken, _tools_request())


def test_both_stacks_require_explicit_identity_attestation(
    metric_frame: pd.DataFrame,
) -> None:
    """AGREE (posture): neither stack infers identity from layout."""
    unattested = analyze_longitudinal_sessions(
        metric_frame, _ud_request(), context=AnalysisContextV2()
    )
    assert unattested.status == "unavailable"
    assert unattested.availability[0].reason_code == "untrusted_player_identity"

    with pytest.raises(ValueError, match="identity must both be explicitly attested"):
        analyze_longitudinal_performance(
            metric_frame,
            LongitudinalRequest(
                metric_column=METRIC,
                session_column="session_id",
                session_order_column="session_order",
                player_column="player_id",
                player_identity_attested=False,
                session_identity_attested=True,
                higher_is_better=False,
            ),
        )
