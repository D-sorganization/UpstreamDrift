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

DIFFER — documented and pinned below
    D10. The pooled estimate is a **different estimator** in each stack, and on
         this fixture the two disagree about statistical significance:

         =============================  ============  =====================
         quantity                       UD            Tools
         =============================  ============  =====================
         method                         player-FE OLS  inverse-variance +
                                        + cluster-     DerSimonian-Laird
                                        robust SE      random effects
         pooled slope (yd/session)      -0.52553153   -0.52827898 (random)
                                                      -0.63997974 (fixed)
         95% interval                   [-1.57630,    [-1.01454,
                                          +0.52524]     -0.04202]
         crosses zero                   yes            no
         =============================  ============  =====================

         Point estimates are within 0.52% of each other (UD/Tools-random
         0.99479923); the UD interval is 2.16x wider, so the same data reads
         "no detectable trend" in UD and "improving, p<0.05-equivalent" in
         Tools. Tools additionally reports ``improvement_probability``
         0.98338660 for the same session.
    D11. Per-player uncertainty. Tools reports SE/CI/p/R2/first-to-last change
         per player; UD's ``LongitudinalPlayerAssociationV1`` carries the point
         estimate and a direction label only. **UD cannot express per-player
         slope uncertainty at all.**
    D12. Heterogeneity. Tools reports tau^2 0.15941371, Q 9.79986169,
         I^2 69.38732305%. UD has no heterogeneity output.
    D13. Cluster floors. UD's contract hard-floors
         ``minimum_player_clusters`` at 4 (pydantic ``ge=4``); Tools synthesises
         a population effect from 2 contributors.
    D14. Failure posture. UD returns a structured ``status="unavailable"``
         result with a ``reason_code``; Tools raises ``ValueError``. Same
         input, one auditable result vs one exception.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.shared.python.launch_monitor.contract_v2 import (
    AnalysisContextV2,
    OrderEvidenceV2,
    PlayerIdentityV2,
    SessionIdentityV2,
)
from src.shared.python.launch_monitor.longitudinal import (
    analyze_longitudinal_sessions,
)
from src.shared.python.launch_monitor.longitudinal_types import (
    LongitudinalSessionRequestV1,
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

# D10 pins.
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

# D11/D12 pins.
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


def _ud_request() -> LongitudinalSessionRequestV1:
    return LongitudinalSessionRequestV1(
        metric=METRIC,
        direction="lower_is_better",
        session_aggregate="mean",
        minimum_sessions_per_player=3,
        minimum_player_clusters=4,
        confidence_level=0.95,
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


def test_divergence_d10_pooled_estimators_disagree_on_significance(
    ud_result, tools_result
) -> None:
    """DIFFER (D10): same slopes in, opposite significance verdicts out."""
    pooled = ud_result.pooled_association
    population = tools_result.population
    assert pooled is not None
    assert pooled.method == "player_fixed_effects_ols_clustered_by_player"

    assert pooled.estimate_per_order_unit == pytest.approx(UD_POOLED_SLOPE, rel=1e-9)
    assert pooled.standard_error == pytest.approx(UD_POOLED_STANDARD_ERROR, rel=1e-9)
    assert pooled.confidence_interval_low == pytest.approx(UD_POOLED_CI_LOW, rel=1e-9)
    assert pooled.confidence_interval_high == pytest.approx(UD_POOLED_CI_HIGH, rel=1e-9)
    assert pooled.p_value == pytest.approx(UD_POOLED_P_VALUE, rel=1e-9)

    assert population.random_effect_slope == pytest.approx(TOOLS_RANDOM_SLOPE, rel=1e-9)
    assert population.random_ci_lower == pytest.approx(TOOLS_RANDOM_CI_LOW, rel=1e-9)
    assert population.random_ci_upper == pytest.approx(TOOLS_RANDOM_CI_HIGH, rel=1e-9)
    assert population.fixed_effect_slope == pytest.approx(TOOLS_FIXED_SLOPE, rel=1e-9)
    assert population.fixed_ci_lower == pytest.approx(TOOLS_FIXED_CI_LOW, rel=1e-9)
    assert population.fixed_ci_upper == pytest.approx(TOOLS_FIXED_CI_HIGH, rel=1e-9)

    ratio = pooled.estimate_per_order_unit / population.random_effect_slope
    assert ratio == pytest.approx(POOLED_SLOPE_RATIO_UD_OVER_TOOLS_RANDOM, rel=1e-9)

    ud_width = pooled.confidence_interval_high - pooled.confidence_interval_low
    tools_width = population.random_ci_upper - population.random_ci_lower
    assert ud_width / tools_width == pytest.approx(
        POOLED_INTERVAL_WIDTH_RATIO, rel=1e-9
    )

    # The headline drift: the intervals do not agree about zero.
    assert pooled.confidence_interval_low < 0.0 < pooled.confidence_interval_high
    assert population.random_ci_upper < 0.0


def test_divergence_d11_only_tools_reports_per_player_uncertainty(
    ud_result, tools_result
) -> None:
    """DIFFER (D11): UD's per-player record has no uncertainty fields."""
    ud_fields = set(type(ud_result.player_associations[0]).model_fields)
    assert ud_fields == {
        "player_id",
        "session_count",
        "estimate_per_order_unit",
        "direction",
        "state",
        "reason_code",
    }
    assert not ud_fields & {
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_value",
        "r_squared",
        "first_to_last_change",
    }

    p1 = next(player for player in tools_result.players if player.player_id == "P1")
    assert p1.standard_error == pytest.approx(TOOLS_ONLY_P1_STANDARD_ERROR, rel=1e-9)
    assert p1.p_value == pytest.approx(TOOLS_ONLY_P1_P_VALUE, rel=1e-9)
    assert p1.r_squared == pytest.approx(TOOLS_ONLY_P1_R_SQUARED, rel=1e-9)
    assert p1.first_to_last_change == pytest.approx(
        TOOLS_ONLY_P1_FIRST_TO_LAST, rel=1e-9
    )


def test_divergence_d12_only_tools_reports_heterogeneity(
    ud_result, tools_result
) -> None:
    """DIFFER (D12): tau^2 / Q / I^2 / improvement probability are Tools-only."""
    population = tools_result.population
    assert population.tau_squared == pytest.approx(TOOLS_ONLY_TAU_SQUARED, rel=1e-9)
    assert population.q_statistic == pytest.approx(TOOLS_ONLY_Q_STATISTIC, rel=1e-9)
    assert population.i_squared_pct == pytest.approx(TOOLS_ONLY_I_SQUARED_PCT, rel=1e-9)
    assert population.improvement_probability == pytest.approx(
        TOOLS_ONLY_IMPROVEMENT_PROBABILITY, rel=1e-9
    )

    pooled_fields = set(type(ud_result.pooled_association).model_fields)
    assert not pooled_fields & {
        "tau_squared",
        "q_statistic",
        "i_squared_pct",
        "improvement_probability",
    }


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
