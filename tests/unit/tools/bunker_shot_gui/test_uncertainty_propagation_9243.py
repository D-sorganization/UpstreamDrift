"""The accelerated-mass interval reaches the design ranking (issue #9243).

Two things are proved here against the real solver, not a mock. First that the
interval issue #8659 opened at the divot **arrives** at the comparison surface
instead of collapsing to its central value on the way. Second, and this is the
one that matters, that carrying it there **removes a verdict the shipped
comparison used to produce**: two soles the bootstrap ranks confidently are
indistinguishable once the model-form band they were both computed under is
carried through.

Everything imported through ``src.tools.bunker_shot_gui.model`` on purpose.
The workbench imports ``bunkershot3d.*`` while these tests live under the
``src.*`` root, and the two roots produce *different class objects* for the
same source file -- so an enum imported the other way would fail every
identity comparison silently.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bunkershot3d.ball import SandDelivery
from bunkershot3d.solvers import (
    FidelityTier,
    ValidityVerdict,
    evaluate_envelope,
)
from bunkershot3d.study.ranking import rank_with_bands
from src.tools.bunker_shot_gui.design import (
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import (
    ConsistencyBand,
    RankingVerdict,
    ShotOutcome,
    UncertaintyClass,
    WorkbenchComparison,
    WorkbenchModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SEPARATING_SETTINGS = SolverSetup(
    n_profile_points=12,
    n_stations=5,
    playability_points=4,
    target_carry_m=12.0,
)
"""Coarse mesh, four stations per axis: the cheapest grid on which the
bootstrap alone still separates the two soles below, so the test can show the
band erasing a verdict rather than an already-absent one."""


def _verdict() -> ValidityVerdict:
    """A verdict of the kind an F0 greenside shot carries."""
    return evaluate_envelope(
        speed_m_s=25.0,
        feature_lengths_m={"clubhead": 0.06},
        grain_diameter_m=0.0005,
        element_size_m=0.002,
        dynamic_terms_active=True,
    )


@pytest.fixture(scope="module")
def banded_model() -> WorkbenchModel:
    """A workbench on the settings the separation test needs."""
    return WorkbenchModel(_SEPARATING_SETTINGS)


@pytest.fixture(scope="module")
def nominal_shot(banded_model: WorkbenchModel) -> ShotOutcome:
    """The archetypal greenside shot, run once for the whole module."""
    return banded_model.run_shot(
        WedgeDesign(name="nominal").geometry(),
        SandCondition().sand_state(),
        SwingSetup(),
    )


@pytest.fixture(scope="module")
def comparison(banded_model: WorkbenchModel) -> WorkbenchComparison:
    """Two genuinely different soles, compared once for the whole module."""
    return banded_model.compare(
        WedgeDesign(name="narrow-low", marketed_bounce_deg=6.0, sole_width_mm=20.0),
        WedgeDesign(name="wide-high", marketed_bounce_deg=20.0, sole_width_mm=26.0),
        SandCondition(),
        SwingSetup(),
    )


class TestCarryBandReachesTheShot:
    """Carry stops being a point at the first place it is computed."""

    def test_the_nominal_shot_carries_a_band(self, nominal_shot: ShotOutcome) -> None:
        """The interval survives the launch and flight models."""
        assert nominal_shot.carry_band is not None

    def test_the_band_is_centred_on_the_reported_carry(
        self, nominal_shot: ShotOutcome
    ) -> None:
        """One number and its band, never two different shots side by side."""
        assert nominal_shot.carry_band is not None
        assert nominal_shot.carry_band.central == pytest.approx(nominal_shot.carry_m)

    def test_the_band_is_wide(self, nominal_shot: ShotOutcome) -> None:
        """A 2.4x mass interval does not produce a decorative carry band.

        The threshold is loose on purpose: the point is that the width is of
        the order of the number itself, not that it takes a particular value
        no measurement supports.
        """
        assert nominal_shot.carry_band is not None
        assert nominal_shot.carry_band.relative_half_width > 0.3

    def test_a_refused_shot_carries_no_band(self, nominal_shot: ShotOutcome) -> None:
        """ADR-0032's refusal rule extends to the band, not just the number."""
        with pytest.raises(ValueError, match="must not carry a force"):
            ShotOutcome(
                verdict=nominal_shot.verdict,
                fidelity_tier=FidelityTier.F0,
                refused=True,
                delivered=nominal_shot.delivered,
                carry_band=ConsistencyBand(0.5, 1.0, 2.0),
            )


class TestPropagationDirection:
    """Less sand sharing the same impulse throws the ball further."""

    def _delivery(self, mass_kg: float) -> SandDelivery:
        """A strike whose accelerated mass is pinned to one value."""
        return SandDelivery(
            impulse_n_s=2.9,
            displaced_mass_kg=mass_kg,
            contact_duration_s=0.006,
            entry_speed_m_s=25.0,
            exit_speed_m_s=15.0,
            bed_relative_density=0.6,
            verdict=_verdict(),
        )

    def test_the_lower_mass_edge_becomes_the_upper_carry_edge(
        self, banded_model: WorkbenchModel
    ) -> None:
        """The map is monotone decreasing, so the edges swap on the way."""
        geometry = WedgeDesign(name="nominal").geometry()
        swing = SwingSetup()
        banded = banded_model.carry_estimate(
            geometry,
            swing,
            replace(
                self._delivery(0.27),
                displaced_mass_bounds_kg=(0.18, 0.41),
            ),
        )
        light = banded_model.carry_estimate(geometry, swing, self._delivery(0.18))
        heavy = banded_model.carry_estimate(geometry, swing, self._delivery(0.41))
        assert banded.band is not None
        assert banded.band.upper == pytest.approx(light.carry_m)
        assert banded.band.lower == pytest.approx(heavy.carry_m)
        assert light.carry_m > heavy.carry_m

    def test_a_delivery_without_an_interval_yields_no_band(
        self, banded_model: WorkbenchModel
    ) -> None:
        """A point mass produces a point carry, and says so by returning None."""
        estimate = banded_model.carry_estimate(
            WedgeDesign(name="nominal").geometry(),
            SwingSetup(),
            self._delivery(0.27),
        )
        assert estimate.band is None

    def test_a_carry_band_around_the_wrong_number_is_refused(self) -> None:
        """The pairing is enforced, not merely conventional."""
        from src.tools.bunker_shot_gui.model import CarryEstimate

        with pytest.raises(ValueError, match="different claim"):
            CarryEstimate(
                carry_m=1.6,
                verdict=_verdict(),
                band=ConsistencyBand(0.5, 1.0, 2.0),
            )


class TestPlayabilityGridCarriesBands:
    """Every cell of the sweep, not just the nominal shot."""

    def test_the_grid_has_bands(self, comparison: WorkbenchComparison) -> None:
        """The sweep the ranking is built on is banded end to end."""
        assert comparison.left.playability.has_bands

    def test_every_banded_cell_brackets_its_own_carry(
        self, comparison: WorkbenchComparison
    ) -> None:
        """A cell reporting a carry outside its band is refused on build."""
        playability = comparison.left.playability
        banded = np.isfinite(playability.carry_lower_m) & np.isfinite(
            playability.carry_upper_m
        )
        assert banded.any()
        assert np.all(playability.carry_lower_m[banded] <= playability.carry_m[banded])
        assert np.all(playability.carry_m[banded] <= playability.carry_upper_m[banded])

    def test_the_band_grids_line_up_with_the_carry_grid(
        self, comparison: WorkbenchComparison
    ) -> None:
        """Shape agreement is a precondition, not a coincidence."""
        playability = comparison.left.playability
        assert playability.carry_lower_m.shape == playability.carry_m.shape
        assert playability.carry_upper_m.shape == playability.carry_m.shape


class TestTheComparisonRefusesToRank:
    """The central deliverable, against the real solver."""

    def test_two_different_soles_are_indistinguishable(
        self, comparison: WorkbenchComparison
    ) -> None:
        """Genuinely different designs, and the model still cannot separate them."""
        assert comparison.banded is not None
        assert comparison.banded.verdict is RankingVerdict.INDISTINGUISHABLE

    def test_no_winner_is_named(self, comparison: WorkbenchComparison) -> None:
        """A caller reading a name off the comparison gets ``None``."""
        assert comparison.winner is None

    def test_separated_reads_the_banded_verdict(
        self, comparison: WorkbenchComparison
    ) -> None:
        """The old property must not keep answering from the bootstrap alone."""
        assert not comparison.separated

    def test_the_bootstrap_alone_still_names_a_winner(
        self, comparison: WorkbenchComparison
    ) -> None:
        """This is the overclaim being removed, pinned so it cannot return."""
        assert comparison.ranking is not None
        assert comparison.ranking.best in {"narrow-low", "wide-high"}

    def test_the_band_is_what_erases_the_verdict(
        self, comparison: WorkbenchComparison
    ) -> None:
        """Strip the model-form term and the same two soles separate again.

        The whole PR in one assertion: the sampling spread the shipped
        comparison measured does distinguish these designs, and the
        accelerated-mass interval it never carried does not.
        """
        assert comparison.banded is not None
        sampling_only = [
            replace(
                budget,
                terms=tuple(
                    term
                    for term in budget.terms
                    if term.uncertainty_class is UncertaintyClass.SAMPLING
                ),
                unquantified=(),
            )
            for budget in comparison.banded.budgets
        ]
        thin = rank_with_bands(
            "narrow-low", sampling_only[0], "wide-high", sampling_only[1]
        )
        assert thin.verdict is RankingVerdict.A_BETTER
        assert comparison.banded.verdict is RankingVerdict.INDISTINGUISHABLE

    def test_the_overlap_depth_is_reported(
        self, comparison: WorkbenchComparison
    ) -> None:
        """How badly the bands overlap is more useful than that they do."""
        assert comparison.banded is not None
        assert comparison.banded.separation < 0.0


class TestWhatDominates:
    """A comparison decided by one assumption says which one."""

    def test_the_mass_interval_dominates(self, comparison: WorkbenchComparison) -> None:
        """Not the bootstrap, which is the only term that used to be reported."""
        assert comparison.banded is not None
        dominant = comparison.banded.dominant
        assert dominant is not None
        assert "accelerated sand mass" in dominant.term.name
        assert dominant.term.uncertainty_class is UncertaintyClass.MODEL_FORM

    def test_the_mass_interval_swamps_the_budget(
        self, comparison: WorkbenchComparison
    ) -> None:
        """Past the dominance threshold, the ranking is about one assumption."""
        assert comparison.banded is not None
        assert comparison.banded.dominant is not None
        assert comparison.banded.dominant.swamps

    def test_the_classes_are_reported_apart(
        self, comparison: WorkbenchComparison
    ) -> None:
        """Model-form width and sampling width are never one number."""
        assert comparison.banded is not None
        split = comparison.banded.budgets[0].by_class()
        assert UncertaintyClass.MODEL_FORM in split
        assert UncertaintyClass.SAMPLING in split
        assert (
            split[UncertaintyClass.MODEL_FORM].width
            > split[UncertaintyClass.SAMPLING].width
        )


class TestHonestyBoundary:
    """What the comparison is not allowed to imply."""

    def test_the_uncalibrated_transfer_efficiency_is_named(
        self, comparison: WorkbenchComparison
    ) -> None:
        """It scales the answer and nobody has bounded it (#8616)."""
        assert comparison.banded is not None
        names = {term.name for term in comparison.banded.unquantified}
        assert "ball momentum transfer efficiency" in names

    def test_the_missing_carry_gci_is_named_as_space_time(
        self, comparison: WorkbenchComparison
    ) -> None:
        """ADR-0033's Courant-fixed study must not read as a spatial u_h."""
        assert comparison.banded is not None
        reasons = " ".join(term.reason for term in comparison.banded.unquantified)
        assert "SPACE-TIME" in reasons
        assert "Courant" in reasons

    def test_no_verdict_here_is_defensible(
        self, comparison: WorkbenchComparison
    ) -> None:
        """Unsized model-form terms outstanding, so the band is a floor."""
        assert comparison.banded is not None
        assert not comparison.banded.defensible

    def test_the_statement_never_says_confidence(
        self, comparison: WorkbenchComparison
    ) -> None:
        """The one word this surface may not borrow."""
        text = comparison.verdict_statement.lower()
        assert "indistinguishable" in text
        assert "not a confidence interval" in text
        assert text.count("confidence") == 1
