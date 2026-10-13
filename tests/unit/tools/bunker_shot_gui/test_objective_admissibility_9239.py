"""A carry objective with no qualification cannot rank designs (#9239, #9545).

The workbench's A/B comparison ranks two soles on absolute carry error, and
carry rests on a sand-to-ball transfer no measurement has qualified. These
tests pin the rule the comparison now enforces: absent a qualification, no
**predictive** winner is ever named -- even when the carry numbers and their
bands happen to be finite and well separated. The exploratory diagnostics are
kept, labelled as exploratory, with the reason attached.

The grid cases #9239 asks to be tested separately are each here: an
all-unavailable grid, a constant/degenerate grid, a partially refused grid and
a genuinely qualified one.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from bunkershot3d.ball import ObjectiveDisposition
from bunkershot3d.study.comparison import compare_designs
from bunkershot3d.study.ranking import BandedRanking, rank_with_bands
from bunkershot3d.vandv.budget import (
    UncertaintyBudget,
    UncertaintyClass,
    UncertaintyTerm,
)
from src.tools.bunker_shot_gui.design import WedgeDesign
from src.tools.bunker_shot_gui.model import (
    COMPARISON_SEED,
    WorkbenchComparison,
    WorkbenchModel,
)
from src.tools.bunker_shot_gui.objective import (
    MIN_RANKABLE_CONDITIONS,
    SHIPPED_CARRY_DISPOSITION,
    CarryObjectiveStanding,
    carry_objective_standing,
)
from src.tools.bunker_shot_gui.report import comparison_report

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

WINDOW_HALF_WIDTH_M = 1.2
"""Half-width of the carry window the synthetic grids are judged against:
the workbench default, 10 % of a 12 m target."""

INSIDE_WINDOW_M = 0.5
"""A carry error comfortably inside the window."""

OUTSIDE_WINDOW_M = 10.0
"""A carry error the #9239 grid produces: 0.06-2.04 m carry against 12 m."""

NARROW_HALF_WIDTH = 0.1
"""Band half-width small enough that the two synthetic designs separate."""

QUALIFIED = CarryObjectiveStanding(disposition=ObjectiveDisposition.SUPPORTED)
"""What a future qualification (#9694) would hand the comparison."""


def _errors(first: list[float], second: list[float]) -> np.ndarray:
    """A ``(2, n)`` shared carry-error grid."""
    return np.array([first, second], dtype=np.float64)


def _budget(central: float) -> UncertaintyBudget:
    """A narrow one-term budget centred on ``central``."""
    return UncertaintyBudget(
        quantity="mean absolute carry error",
        central=central,
        terms=(
            UncertaintyTerm.symmetric(
                name="accelerated sand mass",
                uncertainty_class=UncertaintyClass.MODEL_FORM,
                half_width=NARROW_HALF_WIDTH,
                source="issue #8659",
            ),
        ),
    )


def _separated_banding() -> BandedRanking:
    """Two well separated bands: ``left`` is clearly lower (better)."""
    return rank_with_bands("left", _budget(1.0), "right", _budget(5.0))


def _comparison(
    objective: CarryObjectiveStanding | None, left=None, right=None
) -> WorkbenchComparison:
    """A comparison whose banded verdict is decided in ``left``'s favour."""
    errors = _errors([1.0, 1.1, 0.9], [5.0, 5.1, 4.9])
    stand_in = SimpleNamespace(design=SimpleNamespace(name="stand-in"))
    extra = {} if objective is None else {"objective": objective}
    return WorkbenchComparison(
        left=left if left is not None else stand_in,
        right=right if right is not None else stand_in,
        ranking=compare_designs(("left", "right"), errors, seed=COMPARISON_SEED),
        shared_points=errors.shape[1],
        banded=_separated_banding(),
        **extra,
    )


class TestStandingContract:
    """The value object cannot claim more than its disposition allows."""

    def test_the_shipped_disposition_is_uncalibrated(self) -> None:
        assert (
            SHIPPED_CARRY_DISPOSITION is ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED
        )

    def test_only_a_supported_objective_is_predictive(self) -> None:
        assert QUALIFIED.predictive
        for disposition in ObjectiveDisposition:
            if disposition is ObjectiveDisposition.SUPPORTED:
                continue
            standing = CarryObjectiveStanding(disposition, reasons=("why",))
            assert not standing.predictive

    def test_an_unavailable_objective_must_say_why(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            CarryObjectiveStanding(ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED)

    def test_a_supported_objective_carries_no_refusal_reason(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            CarryObjectiveStanding(ObjectiveDisposition.SUPPORTED, reasons=("x",))


class TestGridCases:
    """The four grids #9239 names, each judged on its own."""

    def test_an_uncalibrated_finite_grid_is_not_predictive(self) -> None:
        """Finite, separated, inside the window -- and still not predictive."""
        standing = carry_objective_standing(
            _errors([0.1, 0.2], [0.8, 0.9]),
            disposition=SHIPPED_CARRY_DISPOSITION,
            window_half_width_m=WINDOW_HALF_WIDTH_M,
        )
        assert standing.disposition is ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED
        assert not standing.predictive
        assert any("#9239" in reason for reason in standing.reasons)

    def test_an_empty_window_demotes_a_qualified_objective(self) -> None:
        """The #9239 grid: no condition carries anywhere near the target."""
        standing = carry_objective_standing(
            _errors([OUTSIDE_WINDOW_M] * 3, [OUTSIDE_WINDOW_M + 1.0] * 3),
            disposition=ObjectiveDisposition.SUPPORTED,
            window_half_width_m=WINDOW_HALF_WIDTH_M,
        )
        assert standing.disposition is ObjectiveDisposition.DEGENERATE_TARGET
        assert not standing.predictive
        assert any("empty" in reason for reason in standing.reasons)

    def test_an_identical_objective_demotes_a_qualified_objective(self) -> None:
        """A constant grid gives the ranking nothing to tell apart."""
        same = [INSIDE_WINDOW_M, INSIDE_WINDOW_M]
        standing = carry_objective_standing(
            _errors(same, same),
            disposition=ObjectiveDisposition.SUPPORTED,
            window_half_width_m=WINDOW_HALF_WIDTH_M,
        )
        assert standing.disposition is ObjectiveDisposition.DEGENERATE_TARGET
        assert any("identical" in reason for reason in standing.reasons)

    def test_degeneracy_is_reported_beside_the_missing_calibration(self) -> None:
        """Both reasons travel; the uncalibrated disposition is kept."""
        standing = carry_objective_standing(
            _errors([OUTSIDE_WINDOW_M] * 2, [OUTSIDE_WINDOW_M] * 2),
            disposition=SHIPPED_CARRY_DISPOSITION,
            window_half_width_m=WINDOW_HALF_WIDTH_M,
        )
        assert standing.disposition is ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED
        assert len(standing.reasons) == 3

    def test_a_qualified_partially_refused_grid_stays_supported(self) -> None:
        """Refused cells are already dropped from the shared grid; what
        remains is enough to rank on when it is qualified and informative."""
        standing = carry_objective_standing(
            _errors([0.1, 0.2], [0.8, 0.9]),
            disposition=ObjectiveDisposition.SUPPORTED,
            window_half_width_m=WINDOW_HALF_WIDTH_M,
        )
        assert standing == QUALIFIED

    def test_a_grid_too_small_to_rank_is_a_precondition_error(self) -> None:
        with pytest.raises(ValueError, match=str(MIN_RANKABLE_CONDITIONS)):
            carry_objective_standing(
                _errors([0.1], [0.2]),
                disposition=ObjectiveDisposition.SUPPORTED,
                window_half_width_m=WINDOW_HALF_WIDTH_M,
            )

    def test_a_non_finite_grid_is_a_precondition_error(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            carry_objective_standing(
                _errors([0.1, np.nan], [0.2, 0.3]),
                disposition=ObjectiveDisposition.SUPPORTED,
                window_half_width_m=WINDOW_HALF_WIDTH_M,
            )

    def test_a_non_positive_window_is_a_precondition_error(self) -> None:
        with pytest.raises(ValueError, match="window"):
            carry_objective_standing(
                _errors([0.1, 0.2], [0.3, 0.4]),
                disposition=ObjectiveDisposition.SUPPORTED,
                window_half_width_m=0.0,
            )


class TestComparisonNamesNoPredictiveWinner:
    """The comparison value object enforces the rule, not the presenter."""

    def test_the_default_objective_is_fail_closed(self) -> None:
        comparison = _comparison(objective=None)
        assert comparison.objective.disposition is SHIPPED_CARRY_DISPOSITION
        assert not comparison.predictive

    def test_a_decided_uncalibrated_ranking_names_no_winner(self) -> None:
        comparison = _comparison(objective=None)
        assert comparison.separated
        assert comparison.winner is None
        assert comparison.exploratory_leader == "left"

    def test_the_statement_says_it_is_exploratory(self) -> None:
        text = _comparison(objective=None).verdict_statement
        assert "EXPLORATORY" in text
        assert "#9239" in text

    def test_a_qualified_ranking_names_its_winner(self) -> None:
        comparison = _comparison(objective=QUALIFIED)
        assert comparison.predictive
        assert comparison.winner == "left"

    def test_an_all_unavailable_grid_names_no_winner(self) -> None:
        """No shared condition answered: no ranking, no winner, no leader."""
        stand_in = SimpleNamespace(design=SimpleNamespace(name="stand-in"))
        comparison = WorkbenchComparison(
            left=stand_in,
            right=stand_in,
            ranking_unavailable_reason="fewer than two delivery conditions",
            objective=QUALIFIED,
        )
        assert comparison.winner is None
        assert comparison.exploratory_leader is None
        assert not comparison.predictive


class TestReportSaysExploratory:
    """The presenter cannot print a predictive verdict the model refused."""

    @pytest.fixture
    def pair(self, nominal_evaluation):  # type: ignore[no-untyped-def]
        left = dataclasses.replace(nominal_evaluation, design=WedgeDesign(name="left"))
        right = dataclasses.replace(
            nominal_evaluation, design=WedgeDesign(name="right")
        )
        return left, right

    def test_an_uncalibrated_report_names_no_better_design(self, pair) -> None:  # type: ignore[no-untyped-def]
        text = comparison_report(_comparison(None, *pair))
        assert "left is better" not in text
        assert "EXPLORATORY" in text
        assert ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED.value in text

    def test_a_qualified_report_names_the_better_design(self, pair) -> None:  # type: ignore[no-untyped-def]
        text = comparison_report(_comparison(QUALIFIED, *pair))
        assert "left is better" in text
        assert ObjectiveDisposition.SUPPORTED.value in text


class TestTheModelAttachesTheStanding:
    """Against the real solver: the comparison is born uncalibrated."""

    def test_a_refused_pair_is_uncalibrated_and_unranked(
        self, model, firm_sand, quasi_static_swing
    ) -> None:  # type: ignore[no-untyped-def]
        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            quasi_static_swing,
        )
        assert comparison.objective.disposition is SHIPPED_CARRY_DISPOSITION
        assert comparison.winner is None

    def test_a_ranked_pair_is_never_predictive(
        self, model, firm_sand, tour_swing
    ) -> None:  # type: ignore[no-untyped-def]
        comparison = model.compare(
            WedgeDesign(name="left", marketed_bounce_deg=6.0),
            WedgeDesign(name="right", marketed_bounce_deg=13.0),
            firm_sand,
            tour_swing,
        )
        if comparison.banded is None:
            pytest.skip(
                "Banded comparison requires optional upstream_physics ball flight kernel"
            )
        assert not comparison.predictive
        assert comparison.winner is None
        assert comparison.objective.reasons

    def test_the_ranked_path_attaches_the_standing(
        self, coarse_settings, nominal_evaluation, firm_sand, tour_swing, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Kernel-free: the ranked branch, fed finite well separated grids.

        The evaluations are the real nominal one with only the carry grid
        replaced, so the bootstrap and bands run for real; the standing must
        still forbid a predictive winner because nothing is qualified.
        """
        target = coarse_settings.target_carry_m
        grids = {
            "left": np.full((2, 2), target - INSIDE_WINDOW_M),
            "right": np.full((2, 2), target - OUTSIDE_WINDOW_M),
        }
        grids["left"][0, 0] -= NARROW_HALF_WIDTH
        grids["right"][0, 0] -= NARROW_HALF_WIDTH

        def fake_evaluate(design, sand, swing):  # type: ignore[no-untyped-def]
            playability = dataclasses.replace(
                nominal_evaluation.playability,
                carry_m=grids[design.name],
                carry_verdict=nominal_evaluation.verdict,
            )
            return dataclasses.replace(
                nominal_evaluation, design=design, playability=playability
            )

        model = WorkbenchModel(coarse_settings)
        monkeypatch.setattr(model, "evaluate", fake_evaluate)
        comparison = model.compare(
            WedgeDesign(name="left"), WedgeDesign(name="right"), firm_sand, tour_swing
        )
        assert comparison.banded is not None
        assert comparison.objective.disposition is SHIPPED_CARRY_DISPOSITION
        assert not comparison.predictive
        assert comparison.winner is None
        assert comparison.exploratory_leader == "left"
