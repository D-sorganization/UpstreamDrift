"""The credibility statement, and the gate that keeps it fresh (#8616).

Two jobs. The first is to assert the NASA-STD-7009B framing itself: every
factor reports an achieved level *and* the gap to the threshold its
intended use demands, and the factor that cannot honestly be self-scored
is left unscored rather than filled in.

The second is a freshness gate. ``docs/bunkershot3d/credibility.md``
carries three generated blocks between HTML comment markers, and each
must match what :mod:`bunkershot3d.vandv.credibility` produces right now.
A credibility statement that drifts away from the code it describes is
worse than none, because it is read as current.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from bunkershot3d.vandv import (
    CREDIBILITY_ASSESSMENT,
    MAX_CREDIBILITY_LEVEL,
    CredibilityFactor,
    FactorAssessment,
    VerificationError,
    credibility_table_markdown,
    domain_of_applicability,
    domain_table_markdown,
    envelope_exceedance,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

CREDIBILITY_DOC = (
    Path(__file__).resolve().parents[3] / "docs" / "bunkershot3d" / "credibility.md"
)


def _envelope_block() -> str:
    """The headline exceedance sentence, as the document publishes it."""
    return envelope_exceedance().describe()


#: Named blocks in ``credibility.md`` and the renderer each must match.
GENERATED_BLOCKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("envelope", _envelope_block),
    ("credibility-table", credibility_table_markdown),
    ("domain-table", domain_table_markdown),
)


def _generated_block(text: str, name: str) -> str:
    """Extract the block between ``<!-- generated:name -->`` markers."""
    opening = f"<!-- generated:{name} -->"
    closing = f"<!-- end:{name} -->"
    start = text.index(opening) + len(opening)
    return text[start : text.index(closing)].strip()


def _normalised(block: str) -> str:
    """Strip the formatting prettier owns, keeping the content the module owns.

    The repository's ``prettier`` pre-commit hook pads Markdown table cells to
    a common width and stretches the separator row to match. That is the
    hook's business, not the credibility statement's, so the freshness gate
    compares *cell contents* rather than column alignment. Everything that
    carries meaning -- every level, threshold, gap and bound -- still has to
    match the module exactly.
    """
    lines = []
    for raw in block.strip().splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        line = re.sub(r"-{2,}", "--", line)
        if line.startswith("|"):
            line = " | ".join(cell.strip() for cell in line.strip("|").split("|"))
        if line:
            lines.append(line)
    return "\n".join(lines)


class TestTheAssessmentIsHonest:
    """The framing, not the prose."""

    def test_every_factor_is_assessed_exactly_once(self) -> None:
        covered = [item.factor for item in CREDIBILITY_ASSESSMENT]
        assert sorted(covered) == sorted(CredibilityFactor)

    def test_validation_is_at_level_zero(self) -> None:
        """Not "limited", not "in progress". Zero, and three levels short.

        The only comparison that can be formed is noise-limited, and a
        noise-limited comparison carries no information about model error.
        """
        validation = _factor(CredibilityFactor.VALIDATION)
        assert validation.achieved_level == 0
        assert validation.gap == 3
        assert not validation.meets_threshold

    def test_use_history_is_at_level_zero(self) -> None:
        """The solver has never been used to make a design decision."""
        assert _factor(CredibilityFactor.USE_HISTORY).achieved_level == 0

    def test_only_one_factor_meets_its_threshold(self) -> None:
        met = [item for item in CREDIBILITY_ASSESSMENT if item.meets_threshold]
        assert [item.factor for item in met] == [CredibilityFactor.MS_MANAGEMENT]

    def test_people_qualifications_is_not_self_scored(self) -> None:
        """A team rating its own competence is not evidence."""
        people = _factor(CredibilityFactor.PEOPLE_QUALIFICATIONS)
        assert people.achieved_level is None
        assert not people.is_assessed
        assert people.gap is None
        assert people.level_text() == "not assessed"

    def test_an_unassessed_factor_never_counts_as_meeting_its_threshold(self) -> None:
        assert not _factor(CredibilityFactor.PEOPLE_QUALIFICATIONS).meets_threshold

    def test_every_factor_states_its_evidence_and_its_gap(self) -> None:
        for item in CREDIBILITY_ASSESSMENT:
            assert len(item.evidence) > 40, item.factor
            assert len(item.gap_statement) > 40, item.factor

    def test_the_gap_is_never_negative(self) -> None:
        """Exceeding a threshold is not a negative gap, it is a met one."""
        for item in CREDIBILITY_ASSESSMENT:
            assert item.gap is None or item.gap >= 0

    def test_a_level_outside_the_scale_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="must be an integer in 0-"):
            FactorAssessment(
                factor=CredibilityFactor.VERIFICATION,
                achieved_level=MAX_CREDIBILITY_LEVEL + 1,
                threshold_level=3,
                evidence="x" * 50,
                gap_statement="y" * 50,
            )

    def test_a_level_without_evidence_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="empty evidence"):
            FactorAssessment(
                factor=CredibilityFactor.VERIFICATION,
                achieved_level=4,
                threshold_level=3,
                evidence="   ",
                gap_statement="y" * 50,
            )


class TestEnvelopeExceedance:
    """The two headline numbers, computed from the solver's own constants."""

    def test_we_are_about_sixty_times_outside_the_stated_froude_limit(self) -> None:
        exceedance = envelope_exceedance()
        assert exceedance.froude == pytest.approx(25.24, abs=0.05)
        assert 60.0 < exceedance.froude_exceedance < 65.0

    def test_we_are_about_twenty_times_beyond_published_validation(self) -> None:
        assert 15.0 < envelope_exceedance().speed_exceedance < 20.0

    def test_the_headline_uses_the_most_flattering_feature_scale(self) -> None:
        """The 30 mm sole and 5 mm edge are worse, so 100 mm is not cherry-picked."""
        clubhead = envelope_exceedance(feature_length_m=0.100)
        sole = envelope_exceedance(feature_length_m=0.030)
        edge = envelope_exceedance(feature_length_m=0.005)
        assert clubhead.froude < sole.froude < edge.froude

    def test_a_non_positive_scale_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="must be positive"):
            envelope_exceedance(feature_length_m=0.0)


class TestDomainOfApplicability:
    """Only two factors have any published measurement, and neither fits."""

    def test_only_the_two_wivou_factors_have_a_measured_domain(self) -> None:
        assert sorted(domain_of_applicability()) == [
            "divot_depth_m",
            "entry_distance_behind_ball_m",
        ]

    def test_both_sweeps_extend_outside_the_measured_domain(self) -> None:
        for factor, overlap in domain_of_applicability().items():
            assert overlap.is_extrapolation, factor
            assert overlap.covered_fraction < 1.0

    def test_the_entry_sweep_starts_below_anything_ever_measured(self) -> None:
        """25 mm was the target players were given; 80 mm is where they entered."""
        overlap = domain_of_applicability()["entry_distance_behind_ball_m"]
        assert overlap.swept[0] < overlap.measured[0]
        assert overlap.covered_fraction == pytest.approx(0.56, abs=0.01)

    def test_the_divot_sweep_also_overruns_the_measurement(self) -> None:
        overlap = domain_of_applicability()["divot_depth_m"]
        assert overlap.covered_fraction == pytest.approx(0.675, abs=0.01)


class TestTheDocumentIsFresh:
    """CI keeps the published statement matching the code."""

    def test_the_document_exists(self) -> None:
        assert CREDIBILITY_DOC.is_file()

    @pytest.mark.parametrize("name", [name for name, _ in GENERATED_BLOCKS])
    def test_a_generated_block_matches_the_module(self, name: str) -> None:
        renderer = dict(GENERATED_BLOCKS)[name]
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        assert _normalised(_generated_block(text, name)) == _normalised(renderer()), (
            f"the {name!r} block in credibility.md is stale. Regenerate it from "
            "bunkershot3d.vandv rather than editing the block by hand."
        )

    def test_the_document_names_every_credibility_factor(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        for factor in CredibilityFactor:
            assert factor.title in text, factor

    def test_the_document_states_that_nothing_is_validated(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        assert "## What Is Validated" in text
        assert "noise-limited" in text
        assert "indeterminate" in text

    def test_the_document_names_the_two_uncalibrated_constants(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        assert "`lambda`" in text
        assert "`delta_h`" in text
        assert "uncalibrated" in text.lower()

    def test_the_document_lists_the_quantities_with_no_data(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        for phrase in (
            "ball launch angle",
            "ejecta mass",
            "clubhead deceleration",
            "coefficient of restitution",
        ):
            assert phrase in text, phrase

    def test_the_document_says_the_gap_is_not_a_search_failure(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        assert "not a search failure" in text

    def test_the_document_states_the_simple_addition_rule(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        assert "simple addition" in text
        assert "quadrature" in text

    def test_the_document_warns_that_u_num_is_not_a_physics_error_bar(self) -> None:
        text = CREDIBILITY_DOC.read_text(encoding="utf-8")
        assert "covers the numerics only" in text


def _factor(factor: CredibilityFactor) -> FactorAssessment:
    """Look up one factor's assessment."""
    for item in CREDIBILITY_ASSESSMENT:
        if item.factor is factor:
            return item
    raise AssertionError(f"{factor} is not in the assessment")
