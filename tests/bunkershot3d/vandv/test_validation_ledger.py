"""The validation ledger, and proof that a measurement moves the score (#8616).

Four things are asserted here, in this order, because each depends on the
one before it.

1. **The ledger is well formed.** Every factor the credibility statement
   scores has a ledger entry; every entry names what holds it at the level
   it sits at; and every entry that a *measurement* could move carries the
   minimum measurement that would move it, in a structure the code reads
   rather than in prose a reader has to trust.
2. **The assessment is derived from the ledger.** There is exactly one
   number, not two that can drift apart. The failure mode this designs
   against is a published table saying 0 and a roadmap implying 2.
3. **The empty ledger keeps the score at zero.** Writing a roadmap is not
   evidence. If building the apparatus moved the score, the apparatus
   would be the defect it was built to prevent.
4. **A measurement moves it.** A record fed through the intake path lifts
   the affected factor by exactly one level and flips the affected sand
   property to :attr:`ProvenanceBasis.MEASURED`. The fixture used is
   marked :attr:`MeasurementBasis.SYNTHETIC_FIXTURE` and **carries no
   value at all**, which is the only way to prove the apparatus works
   without inventing a plausible-looking number for someone to quote
   later.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from bunkershot3d.sand.provenance import ProvenanceBasis
from bunkershot3d.vandv.credibility import (
    CREDIBILITY_ASSESSMENT,
    MAX_CREDIBILITY_LEVEL,
    CredibilityFactor,
    FactorAssessment,
    credibility_assessment,
)
from bunkershot3d.vandv.exceptions import VerificationError
from bunkershot3d.vandv.ledger import (
    Blocker,
    EffortClass,
    LedgerEntry,
    LevelStep,
)
from bunkershot3d.vandv.measurement import (
    SYNTHETIC_SOURCE_MARKER,
    MeasurementBasis,
    MeasurementIntakeError,
    MeasurementRecord,
    MeasurementRegister,
)
from bunkershot3d.vandv.measurement_intake import (
    EVIDENTIAL_RANK,
    MEASUREMENTS_DIR,
    is_provenance_upgrade,
    load_measurement_document,
    provenance_updates,
    shipped_register,
)
from bunkershot3d.vandv.roadmap import (
    MEASUREMENT_SPECS,
    VALIDATION_LEDGER,
    leverage_table_markdown,
    measurement_spec_table_markdown,
    roadmap_table_markdown,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

ROADMAP_DOC = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "bunkershot3d"
    / "validation-roadmap.md"
)

#: The one spec whose measurement is a bench test on the material itself.
BULK_DENSITY = "bunker_sand_bulk_density_kg_m3"
REPOSE = "bunker_sand_angle_of_repose_deg"


def synthetic_record(spec_key: str) -> MeasurementRecord:
    """Build an obviously synthetic record that satisfies ``spec_key``.

    It carries **no value**: :class:`MeasurementRecord` refuses a synthetic
    record that supplies one. A fixture that cannot hold a number cannot be
    mistaken for a measurement, quoted out of context, or copied into a
    report. What it does carry is the *procedural* half of the acceptance
    criterion -- sample count and expanded uncertainty -- which is what the
    ledger actually gates on.
    """
    spec = MEASUREMENT_SPECS[spec_key]
    return MeasurementRecord(
        spec_key=spec_key,
        basis=MeasurementBasis.SYNTHETIC_FIXTURE,
        source=f"{SYNTHETIC_SOURCE_MARKER} tests/bunkershot3d/vandv",
        instrument="none: this record exists to exercise the intake path",
        conditions="none: this record exists to exercise the intake path",
        sample_count=spec.acceptance.min_samples,
        relative_expanded_uncertainty=(
            spec.acceptance.max_relative_expanded_uncertainty
        ),
        unit=spec.unit,
        value=None,
    )


class TestTheLedgerIsWellFormed:
    """Structure before content: the shape has to make drift impossible."""

    def test_every_credibility_factor_has_a_ledger_entry(self) -> None:
        covered = [entry.factor for entry in VALIDATION_LEDGER.entries]
        assert sorted(covered) == sorted(CredibilityFactor)

    def test_every_entry_names_what_holds_it_where_it_is(self) -> None:
        for entry in VALIDATION_LEDGER.entries:
            assert entry.blocker in set(Blocker), entry.factor
            assert len(entry.held_because) > 40, entry.factor

    def test_only_measurement_limited_entries_carry_steps(self) -> None:
        """A step is a promise that a measurement moves the level.

        Verification, robustness and M&S management are limited by analysis
        and process work, and use history accrues only by use. Attaching a
        measurement to any of them would be a lie in a machine-readable
        format, which is worse than a lie in prose because code acts on it.
        """
        for entry in VALIDATION_LEDGER.entries:
            has_steps = bool(entry.steps)
            assert has_steps is entry.is_measurement_limited, entry.factor

    def test_exactly_three_factors_are_measurement_limited(self) -> None:
        limited = [
            entry.factor
            for entry in VALIDATION_LEDGER.entries
            if entry.is_measurement_limited
        ]
        assert sorted(limited) == [
            CredibilityFactor.INPUT_PEDIGREE,
            CredibilityFactor.RESULTS_UNCERTAINTY,
            CredibilityFactor.VALIDATION,
        ]

    def test_every_step_names_at_least_one_measurement(self) -> None:
        for entry in VALIDATION_LEDGER.entries:
            for step in entry.steps:
                assert step.requires, (entry.factor, step.from_level)

    def test_every_required_spec_key_exists(self) -> None:
        for entry in VALIDATION_LEDGER.entries:
            for step in entry.steps:
                for key in step.requires:
                    assert key in MEASUREMENT_SPECS, key

    def test_steps_start_where_the_factor_actually_sits(self) -> None:
        """A roadmap that starts above the current level is a wish list."""
        for entry in VALIDATION_LEDGER.entries:
            if entry.steps:
                assert entry.steps[0].from_level == entry.held_level, entry.factor

    def test_steps_are_contiguous_and_ascend_one_level_at_a_time(self) -> None:
        for entry in VALIDATION_LEDGER.entries:
            for step in entry.steps:
                assert step.to_level == step.from_level + 1
            pairs = zip(entry.steps, entry.steps[1:], strict=False)
            for lower, upper in pairs:
                assert upper.from_level == lower.to_level

    def test_every_spec_is_reachable_from_some_step(self) -> None:
        """A spec nothing depends on is a measurement nobody needs."""
        required = {
            key
            for entry in VALIDATION_LEDGER.entries
            for step in entry.steps
            for key in step.requires
        }
        assert set(MEASUREMENT_SPECS) == required

    def test_every_spec_states_conditions_instrument_and_acceptance(self) -> None:
        for key, spec in MEASUREMENT_SPECS.items():
            assert len(spec.conditions) > 60, key
            assert len(spec.instrument_class) > 40, key
            assert len(spec.acceptance.statement) > 60, key
            assert spec.unit.strip(), key

    def test_provenance_targets_are_disjoint_across_specs(self) -> None:
        """Two specs claiming the same property is how a flip races itself."""
        seen: set[str] = set()
        for spec in MEASUREMENT_SPECS.values():
            overlap = seen & set(spec.provenance_keys)
            assert not overlap, overlap
            seen |= set(spec.provenance_keys)

    def test_a_step_naming_an_unknown_spec_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="unknown measurement"):
            VALIDATION_LEDGER.step_specs(
                LevelStep(
                    from_level=0,
                    to_level=1,
                    requires=("no_such_measurement",),
                    rationale="x" * 50,
                )
            )

    def test_an_entry_with_steps_but_no_measurement_blocker_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="only a measurement-limited"):
            LedgerEntry(
                factor=CredibilityFactor.USE_HISTORY,
                held_level=0,
                threshold_level=2,
                blocker=Blocker.USE,
                held_because="x" * 50,
                evidence="y" * 50,
                gap_statement="z" * 50,
                steps=(
                    LevelStep(
                        from_level=0,
                        to_level=1,
                        requires=(BULK_DENSITY,),
                        rationale="w" * 50,
                    ),
                ),
            )

    def test_a_measurement_limited_entry_without_steps_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="must name at least one"):
            LedgerEntry(
                factor=CredibilityFactor.VALIDATION,
                held_level=0,
                threshold_level=3,
                blocker=Blocker.MEASUREMENT,
                held_because="x" * 50,
                evidence="y" * 50,
                gap_statement="z" * 50,
                steps=(),
            )


class TestTheAssessmentIsDerivedFromTheLedger:
    """One number, not two. The drift this prevents is silent by nature."""

    def test_the_shipped_assessment_is_the_ledger_read_with_no_measurements(
        self,
    ) -> None:
        derived = credibility_assessment(MeasurementRegister(records=()))
        assert derived == CREDIBILITY_ASSESSMENT

    def test_every_derived_level_equals_the_level_the_ledger_holds(self) -> None:
        held = {e.factor: e.held_level for e in VALIDATION_LEDGER.entries}
        for item in CREDIBILITY_ASSESSMENT:
            assert item.achieved_level == held[item.factor], item.factor

    def test_the_thresholds_come_from_the_ledger_too(self) -> None:
        thresholds = {e.factor: e.threshold_level for e in VALIDATION_LEDGER.entries}
        for item in CREDIBILITY_ASSESSMENT:
            assert item.threshold_level == thresholds[item.factor], item.factor


class TestBuildingTheRoadmapDoesNotRaiseTheScore:
    """The constraint this whole change is built around.

    A roadmap is a statement of what is missing. If writing one moved the
    number, the number would be measuring documentation effort.
    """

    def test_the_shipped_register_holds_no_measurements(self) -> None:
        assert shipped_register().is_empty

    def test_validation_is_still_zero_of_four(self) -> None:
        validation = _factor(CredibilityFactor.VALIDATION)
        assert validation.achieved_level == 0
        assert validation.gap == 3

    def test_no_factor_moved_when_the_ledger_gained_its_steps(self) -> None:
        """The published table, pinned level by level, against the ledger."""
        published = {
            CredibilityFactor.VERIFICATION: 2,
            CredibilityFactor.VALIDATION: 0,
            CredibilityFactor.INPUT_PEDIGREE: 2,
            CredibilityFactor.RESULTS_UNCERTAINTY: 2,
            CredibilityFactor.RESULTS_ROBUSTNESS: 1,
            CredibilityFactor.USE_HISTORY: 0,
            CredibilityFactor.MS_MANAGEMENT: 3,
            CredibilityFactor.PEOPLE_QUALIFICATIONS: None,
        }
        for item in CREDIBILITY_ASSESSMENT:
            assert item.achieved_level == published[item.factor], item.factor

    def test_only_ms_management_still_meets_its_threshold(self) -> None:
        met = [item.factor for item in CREDIBILITY_ASSESSMENT if item.meets_threshold]
        assert met == [CredibilityFactor.MS_MANAGEMENT]

    def test_the_shipped_measurement_directory_holds_no_data(self) -> None:
        """A README is not a measurement; anything else here would be one."""
        if not MEASUREMENTS_DIR.is_dir():
            return
        documents = sorted(
            path.name
            for path in MEASUREMENTS_DIR.iterdir()
            if path.suffix in {".yaml", ".yml", ".json"}
        )
        assert documents == []


class TestASyntheticMeasurementMovesTheScore:
    """The apparatus works, rather than merely existing."""

    def test_the_repose_measurement_lifts_validation_off_zero(self) -> None:
        register = MeasurementRegister(records=(synthetic_record(REPOSE),))
        assessment = credibility_assessment(register)
        validation = _find(assessment, CredibilityFactor.VALIDATION)
        assert validation.achieved_level == 1
        assert validation.gap == 2

    def test_it_lifts_that_factor_and_no_other(self) -> None:
        register = MeasurementRegister(records=(synthetic_record(REPOSE),))
        before = {i.factor: i.achieved_level for i in CREDIBILITY_ASSESSMENT}
        moved = {
            item.factor
            for item in credibility_assessment(register)
            if item.achieved_level != before[item.factor]
        }
        assert moved == {CredibilityFactor.VALIDATION}

    def test_one_measurement_never_buys_two_levels(self) -> None:
        """Level 2 needs the divot cast as well; one record cannot reach it."""
        register = MeasurementRegister(records=(synthetic_record(REPOSE),))
        validation = _find(
            credibility_assessment(register), CredibilityFactor.VALIDATION
        )
        assert validation.achieved_level < MAX_CREDIBILITY_LEVEL

    def test_a_partly_satisfied_step_moves_nothing(self) -> None:
        """Input pedigree needs both bench measurements, so one is not enough."""
        register = MeasurementRegister(records=(synthetic_record(BULK_DENSITY),))
        pedigree = _find(
            credibility_assessment(register), CredibilityFactor.INPUT_PEDIGREE
        )
        assert pedigree.achieved_level == 2

    def test_both_bench_measurements_lift_input_pedigree_to_its_threshold(
        self,
    ) -> None:
        register = MeasurementRegister(
            records=(
                synthetic_record(BULK_DENSITY),
                synthetic_record("bunker_sand_drained_friction_angle_deg"),
            )
        )
        pedigree = _find(
            credibility_assessment(register), CredibilityFactor.INPUT_PEDIGREE
        )
        assert pedigree.achieved_level == 3
        assert pedigree.meets_threshold

    def test_a_record_that_misses_the_acceptance_criterion_moves_nothing(self) -> None:
        spec = MEASUREMENT_SPECS[REPOSE]
        thin = MeasurementRecord(
            spec_key=REPOSE,
            basis=MeasurementBasis.SYNTHETIC_FIXTURE,
            source=f"{SYNTHETIC_SOURCE_MARKER} under-sampled",
            instrument="none",
            conditions="none",
            sample_count=spec.acceptance.min_samples - 1,
            relative_expanded_uncertainty=(
                spec.acceptance.max_relative_expanded_uncertainty
            ),
            unit=spec.unit,
            value=None,
        )
        register = MeasurementRegister(records=(thin,))
        assert VALIDATION_LEDGER.satisfied_spec_keys(register) == frozenset()
        validation = _find(
            credibility_assessment(register), CredibilityFactor.VALIDATION
        )
        assert validation.achieved_level == 0

    def test_a_record_in_the_wrong_unit_moves_nothing(self) -> None:
        spec = MEASUREMENT_SPECS[REPOSE]
        wrong = MeasurementRecord(
            spec_key=REPOSE,
            basis=MeasurementBasis.SYNTHETIC_FIXTURE,
            source=f"{SYNTHETIC_SOURCE_MARKER} wrong unit",
            instrument="none",
            conditions="none",
            sample_count=spec.acceptance.min_samples,
            relative_expanded_uncertainty=(
                spec.acceptance.max_relative_expanded_uncertainty
            ),
            unit="rad",
            value=None,
        )
        assert (
            VALIDATION_LEDGER.satisfied_spec_keys(MeasurementRegister(records=(wrong,)))
            == frozenset()
        )

    def test_the_shortfall_says_what_is_wrong_with_the_record(self) -> None:
        spec = MEASUREMENT_SPECS[REPOSE]
        thin = MeasurementRecord(
            spec_key=REPOSE,
            basis=MeasurementBasis.SYNTHETIC_FIXTURE,
            source=f"{SYNTHETIC_SOURCE_MARKER} under-sampled",
            instrument="none",
            conditions="none",
            sample_count=1,
            relative_expanded_uncertainty=1.0,
            unit=spec.unit,
            value=None,
        )
        shortfall = spec.acceptance.shortfall(thin)
        assert "sample" in shortfall
        assert "uncertainty" in shortfall


class TestTheProvenanceFlip:
    """A measurement is only in the model when the model says it is."""

    def test_nothing_is_measured_while_the_register_is_empty(self) -> None:
        assert provenance_updates(MeasurementRegister(records=())) == {}

    def test_the_bench_measurement_flips_its_properties_to_measured(self) -> None:
        register = MeasurementRegister(records=(synthetic_record(BULK_DENSITY),))
        updates = provenance_updates(register)
        assert sorted(updates) == ["moisture", "packing", "particle_density_kg_m3"]
        for name, entry in updates.items():
            assert entry.basis is ProvenanceBasis.MEASURED, name
            assert entry.is_measured, name

    def test_the_flip_carries_the_record_it_came_from(self) -> None:
        register = MeasurementRegister(records=(synthetic_record(BULK_DENSITY),))
        entry = provenance_updates(register)["packing"]
        assert SYNTHETIC_SOURCE_MARKER in entry.source

    def test_a_measurement_that_targets_no_property_flips_nothing(self) -> None:
        """Repose is a system response, not a material constant.

        Reading a friction angle straight off a repose cone is exactly the
        substitution issue #7999 caught once already, in a new costume.
        """
        register = MeasurementRegister(records=(synthetic_record(REPOSE),))
        assert provenance_updates(register) == {}


class TestConventionIsNotAnUpgrade:
    """The distinction PR #9238 established, pinned so it cannot erode."""

    def test_fitting_to_a_simulated_target_buys_no_evidential_rank(self) -> None:
        assert not is_provenance_upgrade(
            ProvenanceBasis.BORROWED_ANALOGUE, ProvenanceBasis.CONVENTION
        )

    def test_nor_does_the_move_back(self) -> None:
        assert not is_provenance_upgrade(
            ProvenanceBasis.CONVENTION, ProvenanceBasis.BORROWED_ANALOGUE
        )

    def test_the_three_not_a_measurement_bases_share_one_rank(self) -> None:
        lateral = (
            ProvenanceBasis.BORROWED_ANALOGUE,
            ProvenanceBasis.CONVENTION,
            ProvenanceBasis.ESTIMATED,
        )
        assert len({EVIDENTIAL_RANK[basis] for basis in lateral}) == 1

    def test_only_a_measurement_outranks_them(self) -> None:
        for basis in (
            ProvenanceBasis.BORROWED_ANALOGUE,
            ProvenanceBasis.CONVENTION,
            ProvenanceBasis.ESTIMATED,
        ):
            assert is_provenance_upgrade(basis, ProvenanceBasis.MEASURED), basis

    def test_a_basis_never_outranks_itself(self) -> None:
        for basis in ProvenanceBasis:
            assert not is_provenance_upgrade(basis, basis), basis


class TestLeverage:
    """Not everything is equally worth measuring, and the code says so."""

    def test_the_ranking_covers_every_spec(self) -> None:
        ranked = VALIDATION_LEDGER.leverage_ranking(MeasurementRegister(records=()))
        assert sorted(item.spec.key for item in ranked) == sorted(MEASUREMENT_SPECS)

    def test_the_ranking_is_ordered_by_leverage(self) -> None:
        ranked = VALIDATION_LEDGER.leverage_ranking(MeasurementRegister(records=()))
        scores = [item.leverage for item in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_the_angle_of_repose_is_the_highest_leverage_measurement(self) -> None:
        """It is the only bench test that touches the factor sitting at zero."""
        ranked = VALIDATION_LEDGER.leverage_ranking(MeasurementRegister(records=()))
        assert ranked[0].spec.key == REPOSE
        assert ranked[0].spec.effort is EffortClass.BENCH_DAY

    def test_the_two_rig_measurements_have_no_immediate_leverage(self) -> None:
        """Not unimportant: unreachable. They buy a level nothing can reach yet.

        Both sit behind validation level 2, and validation is at 0. Spending
        a high-speed rig before the bench work is spending it on a comparison
        that cannot attribute its error to the model.
        """
        ranked = {
            item.spec.key: item
            for item in VALIDATION_LEDGER.leverage_ranking(
                MeasurementRegister(records=())
            )
        }
        for key in ("ejecta_launch_high_speed_video", "clubhead_delivery_shaft_strain"):
            assert ranked[key].leverage == 0.0, key
            assert ranked[key].credit == 0.0, key

    def test_leverage_is_recomputed_against_the_register(self) -> None:
        """Once the bench work lands, the rig work becomes reachable."""
        register = MeasurementRegister(records=(synthetic_record(REPOSE),))
        ranked = {
            item.spec.key: item for item in VALIDATION_LEDGER.leverage_ranking(register)
        }
        assert ranked["splash_shot_divot_cast_volume_m3"].leverage > 0.0

    def test_a_satisfied_spec_drops_out_of_the_ranking_credit(self) -> None:
        register = MeasurementRegister(records=(synthetic_record(REPOSE),))
        ranked = {
            item.spec.key: item for item in VALIDATION_LEDGER.leverage_ranking(register)
        }
        assert ranked[REPOSE].credit == 0.0

    def test_a_level_bought_on_a_wider_gap_is_worth_more(self) -> None:
        """The declared weighting, stated so it can be argued with."""
        assert VALIDATION_LEDGER.step_weight(CredibilityFactor.VALIDATION) == 3
        assert VALIDATION_LEDGER.step_weight(CredibilityFactor.INPUT_PEDIGREE) == 1

    def test_effort_units_are_ordered_by_effort(self) -> None:
        units = [effort.cost_units for effort in EffortClass]
        assert units == sorted(units)
        assert all(unit > 0 for unit in units)


class TestTheIntakePath:
    """A defined way to supply a measurement, and a refusal for everything else."""

    def test_a_well_formed_document_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "synthetic.yaml"
        path.write_text(_synthetic_document(), encoding="utf-8")
        records = load_measurement_document(path)
        assert len(records) == 1
        assert records[0].spec_key == REPOSE
        assert records[0].basis is MeasurementBasis.SYNTHETIC_FIXTURE

    def test_a_loaded_synthetic_document_moves_the_score(self, tmp_path: Path) -> None:
        """End to end: file on disk, through the loader, into the level."""
        path = tmp_path / "synthetic.yaml"
        path.write_text(_synthetic_document(), encoding="utf-8")
        register = MeasurementRegister(records=load_measurement_document(path))
        validation = _find(
            credibility_assessment(register), CredibilityFactor.VALIDATION
        )
        assert validation.achieved_level == 1

    def test_an_unknown_schema_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "future.yaml"
        path.write_text(
            _synthetic_document().replace("schema_version: 1", "schema_version: 99"),
            encoding="utf-8",
        )
        with pytest.raises(MeasurementIntakeError, match="schema_version"):
            load_measurement_document(path)

    def test_a_record_naming_no_spec_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "orphan.yaml"
        path.write_text(
            _synthetic_document().replace(REPOSE, "no_such_measurement"),
            encoding="utf-8",
        )
        with pytest.raises(MeasurementIntakeError, match="no_such_measurement"):
            load_measurement_document(path)

    def test_a_missing_field_is_refused_by_name(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.yaml"
        text = "\n".join(
            line
            for line in _synthetic_document().splitlines()
            if "instrument:" not in line
        )
        path.write_text(text, encoding="utf-8")
        with pytest.raises(MeasurementIntakeError, match="instrument"):
            load_measurement_document(path)

    def test_a_synthetic_record_may_not_carry_a_value(self) -> None:
        with pytest.raises(MeasurementIntakeError, match="carry a value"):
            MeasurementRecord(
                spec_key=REPOSE,
                basis=MeasurementBasis.SYNTHETIC_FIXTURE,
                source=f"{SYNTHETIC_SOURCE_MARKER} x",
                instrument="none",
                conditions="none",
                sample_count=5,
                relative_expanded_uncertainty=0.05,
                unit="deg",
                value=31.0,
            )

    def test_a_synthetic_record_must_say_so_in_its_source(self) -> None:
        with pytest.raises(MeasurementIntakeError, match=SYNTHETIC_SOURCE_MARKER):
            MeasurementRecord(
                spec_key=REPOSE,
                basis=MeasurementBasis.SYNTHETIC_FIXTURE,
                source="a lab that does not exist",
                instrument="none",
                conditions="none",
                sample_count=5,
                relative_expanded_uncertainty=0.05,
                unit="deg",
                value=None,
            )

    def test_an_instrument_record_must_carry_a_value(self) -> None:
        with pytest.raises(MeasurementIntakeError, match="must carry a value"):
            MeasurementRecord(
                spec_key=REPOSE,
                basis=MeasurementBasis.INSTRUMENT,
                source="a laboratory report",
                instrument="a repose apparatus",
                conditions="as played",
                sample_count=5,
                relative_expanded_uncertainty=0.05,
                unit="deg",
                value=None,
            )

    def test_a_non_positive_uncertainty_is_refused(self) -> None:
        with pytest.raises(MeasurementIntakeError, match="uncertainty"):
            MeasurementRecord(
                spec_key=REPOSE,
                basis=MeasurementBasis.SYNTHETIC_FIXTURE,
                source=f"{SYNTHETIC_SOURCE_MARKER} x",
                instrument="none",
                conditions="none",
                sample_count=5,
                relative_expanded_uncertainty=0.0,
                unit="deg",
                value=None,
            )

    def test_the_shipped_register_refuses_a_synthetic_document(
        self, tmp_path: Path
    ) -> None:
        """A fixture must never be able to reach the published score."""
        path = tmp_path / "synthetic.yaml"
        path.write_text(_synthetic_document(), encoding="utf-8")
        with pytest.raises(MeasurementIntakeError, match="synthetic"):
            shipped_register(directory=tmp_path)

    def test_an_absent_directory_yields_an_empty_register(self, tmp_path: Path) -> None:
        assert shipped_register(directory=tmp_path / "nowhere").is_empty


class TestTheRoadmapDocumentIsFresh:
    """CI keeps the published roadmap matching the ledger."""

    def test_the_document_exists(self) -> None:
        assert ROADMAP_DOC.is_file()

    @pytest.mark.parametrize(
        "name",
        ["roadmap-table", "leverage-table", "measurement-specs"],
    )
    def test_a_generated_block_matches_the_module(self, name: str) -> None:
        renderers: dict[str, Callable[[], str]] = {
            "roadmap-table": roadmap_table_markdown,
            "leverage-table": leverage_table_markdown,
            "measurement-specs": measurement_spec_table_markdown,
        }
        text = ROADMAP_DOC.read_text(encoding="utf-8")
        assert _normalised(_generated_block(text, name)) == _normalised(
            renderers[name]()
        ), (
            f"the {name!r} block in validation-roadmap.md is stale. Regenerate "
            "it from bunkershot3d.vandv.roadmap rather than editing by hand."
        )

    def test_the_document_names_every_measurement_spec(self) -> None:
        text = ROADMAP_DOC.read_text(encoding="utf-8")
        for key in MEASUREMENT_SPECS:
            assert key in text, key

    def test_the_document_says_the_roadmap_is_not_evidence(self) -> None:
        text = ROADMAP_DOC.read_text(encoding="utf-8")
        assert "not evidence" in text
        assert "0 of 4" in text


def _synthetic_document() -> str:
    """A measurement document that is obviously not a measurement."""
    return (
        "schema_version: 1\n"
        f"document_id: {SYNTHETIC_SOURCE_MARKER} apparatus test\n"
        "records:\n"
        f"  - spec_key: {REPOSE}\n"
        "    basis: synthetic_fixture\n"
        f"    source: {SYNTHETIC_SOURCE_MARKER} tests/bunkershot3d/vandv\n"
        "    instrument: none\n"
        "    conditions: none\n"
        "    sample_count: 5\n"
        "    relative_expanded_uncertainty: 0.05\n"
        "    unit: deg\n"
        "    value: null\n"
    )


def _generated_block(text: str, name: str) -> str:
    """Extract the block between ``<!-- generated:name -->`` markers."""
    opening = f"<!-- generated:{name} -->"
    closing = f"<!-- end:{name} -->"
    start = text.index(opening) + len(opening)
    return text[start : text.index(closing)].strip()


def _normalised(block: str) -> str:
    """Strip the padding prettier owns, keeping the content the module owns."""
    lines = []
    for raw in block.strip().splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        line = re.sub(r"-{2,}", "--", line)
        if line.startswith("|"):
            line = " | ".join(cell.strip() for cell in line.strip("|").split("|"))
        if line:
            lines.append(line)
    return "\n".join(lines)


def _find(
    assessment: tuple[FactorAssessment, ...], factor: CredibilityFactor
) -> FactorAssessment:
    """Look up one factor in an assessment."""
    for item in assessment:
        if item.factor is factor:
            return item
    raise AssertionError(f"{factor} is not in the assessment")


def _factor(factor: CredibilityFactor) -> FactorAssessment:
    """Look up one factor in the shipped assessment."""
    return _find(CREDIBILITY_ASSESSMENT, factor)
