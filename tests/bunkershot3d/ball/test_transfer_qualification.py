"""The measurement-to-prediction program for the sand-to-ball transfer (#9543).

Nothing in these tests is a measurement. The strokes below are built from
the model itself with a known "true" transfer and a deterministic
perturbation, and are labelled as instrument records only so that the intake
path, the fit, the held-out comparison and the verdict wiring can be
exercised end to end. The shipped state -- no strokes on file, every launch
verdict floored at ``BEYOND_VALIDATION`` -- is pinned alongside.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from bunkershot3d.ball import splash as splash_module
from bunkershot3d.ball.lie import BallLie, BallLieType, BallProperties
from bunkershot3d.ball.qualification import (
    INTENDED_USE_MATRIX,
    MEASUREMENT_PROTOCOL,
    PRACTICAL_TOLERANCES,
    THREE_CAMERA_RIG_CAPABILITY,
    FitOutcome,
    FitStatus,
    IntendedUseMatrix,
    MeasuredStroke,
    MeasurementProtocol,
    ObjectiveDisposition,
    QualificationDataset,
    TransferQualification,
    TransferQualificationError,
    UseRegime,
    objective_disposition,
    rig_capability_markdown,
)
from bunkershot3d.ball.qualification_fit import (
    MIN_CALIBRATION_STROKES,
    fit_transfer,
    predict_launch,
    qualification_report_markdown,
    qualify,
    validate_holdout,
)
from bunkershot3d.ball.splash import (
    DEFAULT_MOMENTUM_TRANSFER,
    MomentumTransfer,
    compute_ball_launch_from_splash,
    launch_verdict,
)
from bunkershot3d.sand import ProvenanceBasis
from bunkershot3d.solvers import EnvelopeStatus
from bunkershot3d.vandv import (
    NoReferenceDataError,
    NumericalUncertainty,
    ValidationComparison,
    VandVError,
)
from bunkershot3d.vandv.measurement import (
    SYNTHETIC_SOURCE_MARKER,
    MeasurementBasis,
    MeasurementRecord,
    MeasurementRegister,
)
from bunkershot3d.vandv.measurement_intake import is_provenance_upgrade

from .test_splash_transfer import GREENSIDE_LOFT_RAD, delivery, solver_verdict

pytestmark = pytest.mark.unit

TRUE_TRANSFER = MomentumTransfer(efficiency=0.35, packing_sensitivity=0.30)
"""The transfer the fixture strokes are generated from. Not a calibration."""

FIXTURE_SOURCE = (
    "unit-test fixture stroke, generated from the model and not a measurement "
    "(tests/bunkershot3d/ball/test_transfer_qualification.py)"
)

TEST_MATRIX = IntendedUseMatrix(
    regimes=(
        UseRegime("standard-firm", BallLieType.STANDARD, (0.6, 1.0), (0.05, 30.0)),
        UseRegime("standard-loose", BallLieType.STANDARD, (0.0, 0.6), (0.05, 30.0)),
        UseRegime("plugged", BallLieType.PLUGGED, (0.0, 1.0), (0.05, 30.0)),
    )
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    spec_key: str,
    value: float | None,
    *,
    unit: str,
    basis: MeasurementBasis = MeasurementBasis.INSTRUMENT,
    uncertainty: float = 0.04,
) -> MeasurementRecord:
    return MeasurementRecord(
        spec_key=spec_key,
        basis=basis,
        source=FIXTURE_SOURCE
        if basis is MeasurementBasis.INSTRUMENT
        else f"{SYNTHETIC_SOURCE_MARKER}: {FIXTURE_SOURCE}",
        instrument="three-camera global-shutter rig, 60 fps, calibrated volume",
        conditions="greenside splash from a standard lie at the played moisture",
        sample_count=1,
        relative_expanded_uncertainty=uncertainty,
        unit=unit,
        value=value,
        measured_on="2026-09-18" if basis is MeasurementBasis.INSTRUMENT else "",
    )


def _sand_register(batch: str, *, synthetic: bool = False) -> MeasurementRegister:
    """A register satisfying both sand specs the protocol requires."""
    basis = (
        MeasurementBasis.SYNTHETIC_FIXTURE if synthetic else MeasurementBasis.INSTRUMENT
    )
    records = []
    for key, unit, value, n, u in (
        ("bunker_sand_bulk_density_kg_m3", "kg/m^3", 1550.0, 3, 0.02),
        ("bunker_sand_drained_friction_angle_deg", "deg", 33.0, 5, 0.05),
    ):
        records.append(
            MeasurementRecord(
                spec_key=key,
                basis=basis,
                source=(
                    f"{SYNTHETIC_SOURCE_MARKER}: {FIXTURE_SOURCE}"
                    if synthetic
                    else f"{FIXTURE_SOURCE}; batch {batch}"
                ),
                instrument="bench apparatus of the class the ledger spec names",
                conditions=f"batch {batch} at the played moisture and compaction",
                sample_count=n,
                relative_expanded_uncertainty=u,
                unit=unit,
                value=None if synthetic else value,
                measured_on="" if synthetic else "2026-09-17",
            )
        )
    return MeasurementRegister(records=tuple(records))


def make_stroke(
    index: int,
    *,
    session: str,
    batch: str = "batch-A",
    relative_density: float = 0.7,
    speed_m_s: float = 25.0,
    depth_m: float = 0.005,
    true_transfer: MomentumTransfer = TRUE_TRANSFER,
    perturbation: float = 0.0,
    with_angle: bool = True,
    ball_speed: MeasurementRecord | None = None,
    digest: str | None = None,
) -> MeasuredStroke:
    """One fixture stroke whose 'measured' launch is the model's own."""
    strike = delivery(speed_m_s=speed_m_s, bed_relative_density=relative_density)
    lie = BallLie(depth_m=depth_m)
    truth = compute_ball_launch_from_splash(
        lie=lie,
        ball=BallProperties(),
        delivery=strike,
        club_loft_rad=GREENSIDE_LOFT_RAD,
        transfer=true_transfer,
    )
    measured_speed = truth.ball_speed_m_s * (1.0 + perturbation)
    stroke_id = f"{session}-{index:02d}"
    return MeasuredStroke(
        stroke_id=stroke_id,
        session_id=session,
        sand_batch_id=batch,
        measured_on="2026-09-18",
        raw_data_digest=digest or _digest(stroke_id),
        delivery=strike,
        lie=lie,
        club_loft_rad=GREENSIDE_LOFT_RAD,
        ball_speed=ball_speed
        or _record("ball_launch_speed_m_s", measured_speed, unit="m/s"),
        numerical=NumericalUncertainty(u_h=0.01),
        launch_angle=(
            _record("ball_launch_angle_rad", truth.launch_angle_rad, unit="rad")
            if with_angle
            else None
        ),
    )


def make_strokes(
    session: str,
    count: int,
    *,
    densities: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8, 0.95),
    batch: str = "batch-A",
    amplitude: float = 0.01,
) -> tuple[MeasuredStroke, ...]:
    """``count`` strokes across the density spread with +/-1 % perturbation."""
    return tuple(
        make_stroke(
            i,
            session=session,
            batch=batch,
            relative_density=densities[i % len(densities)],
            speed_m_s=22.0 + (i % 5),
            perturbation=amplitude * math.sin(1.7 * i + 0.3),
        )
        for i in range(count)
    )


def make_dataset(
    *,
    calibration: tuple[MeasuredStroke, ...] | None = None,
    holdout: tuple[MeasuredStroke, ...] | None = None,
    matrix: IntendedUseMatrix = TEST_MATRIX,
    batches: dict[str, MeasurementRegister] | None = None,
) -> QualificationDataset:
    calibration = calibration or make_strokes("cal-1", 12)
    holdout = holdout or make_strokes("hold-1", 10, batch="batch-B")
    return QualificationDataset(
        protocol=MEASUREMENT_PROTOCOL,
        intended_use=matrix,
        sand_batches=batches
        or {"batch-A": _sand_register("batch-A"), "batch-B": _sand_register("batch-B")},
        strokes=calibration + holdout,
        calibration_session_ids=frozenset(s.session_id for s in calibration),
    )


class TestTheShippedStateIsUnqualified:
    """Software setup only: the register is empty and the floor stands."""

    def test_the_default_launch_is_still_floored(self) -> None:
        result = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert result.provenance.entry("transfer_efficiency").basis is (
            ProvenanceBasis.ESTIMATED
        )

    def test_the_registered_matrix_covers_the_design_band(self) -> None:
        for regime in INTENDED_USE_MATRIX.regimes:
            assert regime.entry_speed_range_m_s == (20.0, 27.0)
        assert {r.lie_type for r in INTENDED_USE_MATRIX.regimes} == {
            BallLieType.STANDARD,
            BallLieType.BURIED,
            BallLieType.PLUGGED,
        }

    def test_the_registered_protocol_needs_characterised_sand(self) -> None:
        assert "bunker_sand_bulk_density_kg_m3" in (
            MEASUREMENT_PROTOCOL.sand_characterisation_spec_keys
        )
        assert "bunker_sand_drained_friction_angle_deg" in (
            MEASUREMENT_PROTOCOL.sand_characterisation_spec_keys
        )

    def test_a_protocol_naming_an_unknown_sand_spec_is_refused(self) -> None:
        with pytest.raises(TransferQualificationError, match="ledger lacks"):
            MeasurementProtocol(
                name="x",
                version="1",
                apparatus="rig",
                calibration_reference="cert",
                sand_characterisation_spec_keys=("no_such_spec",),
            )

    def test_the_tolerances_are_fixed_with_a_rationale(self) -> None:
        assert PRACTICAL_TOLERANCES.max_abs_relative_bias == 0.05
        assert PRACTICAL_TOLERANCES.max_relative_rms_error == 0.10
        assert PRACTICAL_TOLERANCES.min_interval_coverage == 0.80
        assert "window" in PRACTICAL_TOLERANCES.rationale
        assert "8614" in PRACTICAL_TOLERANCES.rationale

    def test_calibrated_is_not_measured(self) -> None:
        assert not is_provenance_upgrade(
            ProvenanceBasis.CALIBRATED, ProvenanceBasis.SPECIFICATION
        )
        assert is_provenance_upgrade(
            ProvenanceBasis.ESTIMATED, ProvenanceBasis.CALIBRATED
        )
        assert is_provenance_upgrade(
            ProvenanceBasis.CALIBRATED, ProvenanceBasis.MEASURED
        )


class TestIntakeRefusals:
    """A stroke is data or it is refused; nothing in between reaches a fit."""

    def test_a_synthetic_launch_record_is_refused(self) -> None:
        fixture = _record(
            "ball_launch_speed_m_s",
            None,
            unit="m/s",
            basis=MeasurementBasis.SYNTHETIC_FIXTURE,
        )
        with pytest.raises(TransferQualificationError, match="synthetic fixture"):
            make_stroke(0, session="s", ball_speed=fixture)

    def test_a_record_in_the_wrong_unit_is_refused(self) -> None:
        wrong = _record("ball_launch_speed_m_s", 40.0, unit="mph")
        with pytest.raises(TransferQualificationError, match="must be in 'm/s'"):
            make_stroke(0, session="s", ball_speed=wrong)

    def test_a_record_against_another_key_is_refused(self) -> None:
        wrong = _record("ball_spin_rate_rad_s", 12.0, unit="m/s")
        with pytest.raises(TransferQualificationError, match="offered against"):
            make_stroke(0, session="s", ball_speed=wrong)

    def test_a_malformed_raw_digest_is_refused(self) -> None:
        with pytest.raises(TransferQualificationError, match="SHA-256"):
            make_stroke(0, session="s", digest="not-a-digest")

    def test_missing_quantities_stay_missing(self) -> None:
        stroke = make_stroke(0, session="s", with_angle=False)
        assert stroke.launch_angle is None
        assert stroke.spin_rate is None

    def test_two_strokes_with_one_raw_capture_are_refused(self) -> None:
        shared = _digest("one capture")
        calibration = make_strokes("cal-1", 12)
        holdout = (
            make_stroke(0, session="hold-1", batch="batch-B", digest=shared),
            make_stroke(1, session="hold-1", batch="batch-B", digest=shared),
        )
        with pytest.raises(TransferQualificationError, match="same raw-data digest"):
            make_dataset(calibration=calibration, holdout=holdout)

    def test_an_uncharacterised_batch_is_refused(self) -> None:
        with pytest.raises(TransferQualificationError, match="no characterisation"):
            make_dataset(batches={"batch-A": _sand_register("batch-A")})

    def test_a_synthetically_characterised_batch_is_refused(self) -> None:
        with pytest.raises(TransferQualificationError, match="synthetic fixture"):
            make_dataset(
                batches={
                    "batch-A": _sand_register("batch-A"),
                    "batch-B": _sand_register("batch-B", synthetic=True),
                }
            )

    def test_a_batch_failing_the_ledger_acceptance_is_refused(self) -> None:
        """The #9286 gates apply: a two-sample density does not characterise."""
        coarse = MeasurementRegister(
            records=(
                MeasurementRecord(
                    spec_key="bunker_sand_bulk_density_kg_m3",
                    basis=MeasurementBasis.INSTRUMENT,
                    source=FIXTURE_SOURCE,
                    instrument="bench apparatus",
                    conditions="batch-B at the played moisture",
                    sample_count=2,
                    relative_expanded_uncertainty=0.02,
                    unit="kg/m^3",
                    value=1550.0,
                    measured_on="2026-09-17",
                ),
            )
        )
        with pytest.raises(TransferQualificationError, match="9286"):
            make_dataset(
                batches={"batch-A": _sand_register("batch-A"), "batch-B": coarse}
            )

    def test_designating_every_session_for_calibration_is_refused(self) -> None:
        strokes = make_strokes("cal-1", 12)
        with pytest.raises(TransferQualificationError, match="nothing held out"):
            QualificationDataset(
                protocol=MEASUREMENT_PROTOCOL,
                intended_use=TEST_MATRIX,
                sand_batches={"batch-A": _sand_register("batch-A")},
                strokes=strokes,
                calibration_session_ids=frozenset({"cal-1"}),
            )

    def test_a_stroke_outside_the_matrix_is_refused(self) -> None:
        narrow = IntendedUseMatrix(
            regimes=(
                UseRegime("plugged", BallLieType.PLUGGED, (0.0, 1.0), (20.0, 27.0)),
            )
        )
        with pytest.raises(TransferQualificationError, match="outside every regime"):
            make_dataset(matrix=narrow)

    def test_the_split_is_by_session_and_disjoint(self) -> None:
        dataset = make_dataset()
        calibration = {s.stroke_id for s in dataset.calibration_strokes}
        holdout = {s.stroke_id for s in dataset.holdout_strokes}
        assert calibration and holdout and not calibration & holdout
        assert dataset.unseen_holdout_batches == ("batch-B",)


class TestTheCalibrationFit:
    """Bounded, identifiable, and preserved when it fails."""

    def test_the_fit_recovers_the_generating_parameters(self) -> None:
        fit = fit_transfer(make_strokes("cal-1", 15))
        assert fit.succeeded and fit.transfer is not None
        assert fit.transfer.efficiency == pytest.approx(
            TRUE_TRANSFER.efficiency, abs=0.02
        )
        assert fit.transfer.packing_sensitivity == pytest.approx(
            TRUE_TRANSFER.packing_sensitivity, abs=0.08
        )
        assert all(se > 0.0 for se in fit.standard_errors.values())
        assert fit.sensitivities["efficiency"] == pytest.approx(1.0, abs=1e-3)
        assert fit.sensitivities["packing_sensitivity"] < 0.0

    def test_too_few_strokes_is_a_preserved_failure(self) -> None:
        fit = fit_transfer(make_strokes("cal-1", MIN_CALIBRATION_STROKES - 1))
        assert fit.status is FitStatus.INSUFFICIENT_DATA
        assert fit.transfer is None
        assert len(fit.calibration_stroke_ids) == MIN_CALIBRATION_STROKES - 1
        assert any(str(MIN_CALIBRATION_STROKES) in r for r in fit.reasons)

    def test_one_packing_cannot_identify_the_packing_sensitivity(self) -> None:
        fit = fit_transfer(make_strokes("cal-1", 12, densities=(0.7,)))
        assert fit.status is FitStatus.UNIDENTIFIABLE
        assert any("packing_sensitivity" in r for r in fit.reasons)

    def test_efficiency_alone_is_identifiable_at_one_packing(self) -> None:
        fit = fit_transfer(
            make_strokes("cal-1", 12, densities=(0.7,)), parameters=("efficiency",)
        )
        assert fit.succeeded

    def test_friction_needs_a_spin_record_on_every_stroke(self) -> None:
        fit = fit_transfer(
            make_strokes("cal-1", 12), parameters=("efficiency", "sand_ball_friction")
        )
        assert fit.status is FitStatus.INSUFFICIENT_DATA
        assert any("spin" in r for r in fit.reasons)

    def test_the_lever_arm_is_never_fittable(self) -> None:
        with pytest.raises(TransferQualificationError, match="distinct names"):
            fit_transfer(
                make_strokes("cal-1", 12), parameters=("spin_lever_arm_fraction",)
            )

    def test_a_failed_fit_cannot_carry_parameters(self) -> None:
        with pytest.raises(TransferQualificationError, match="must not carry"):
            FitOutcome(
                status=FitStatus.INSUFFICIENT_DATA,
                parameters=("efficiency",),
                transfer=DEFAULT_MOMENTUM_TRANSFER,
                standard_errors={},
                sensitivities={},
                covariance=(),
                residual_rms=None,
                calibration_stroke_ids=(),
                reasons=("x",),
            )


class TestHeldOutQualification:
    """Frozen parameters, independent sessions, predeclared tolerances."""

    def test_qualify_is_rerunnable_and_versioned(self) -> None:
        dataset = make_dataset()
        first = qualify(dataset)
        second = qualify(dataset)
        assert first.version == second.version
        assert first.version.startswith("transfer-qualification/1+")
        assert first.evidence_digest == dataset.evidence_digest()
        assert not set(first.calibration_stroke_ids) & set(first.holdout_stroke_ids)

    def test_regimes_with_held_out_evidence_qualify_and_the_rest_are_rejected(
        self,
    ) -> None:
        qualification = qualify(make_dataset())
        by_key = {v.regime.key: v for v in qualification.verdicts}
        assert by_key["standard-firm"].qualified
        assert by_key["standard-firm"].independent_sand_batches
        assert by_key["standard-firm"].holdout_session_ids == ("hold-1",)
        assert not by_key["plugged"].qualified
        assert any("no held-out stroke" in r for r in by_key["plugged"].reasons)
        assert [v.regime.key for v in qualification.rejected_verdicts] == [
            "standard-loose",
            "plugged",
        ]

    def test_a_regime_with_too_few_held_out_strokes_is_rejected(self) -> None:
        qualification = qualify(make_dataset())
        loose = next(
            v for v in qualification.verdicts if v.regime.key == "standard-loose"
        )
        assert not loose.qualified
        assert any("held-out stroke(s) against the 5" in r for r in loose.reasons)
        assert loose.relative_bias is not None  # statistics are still recorded

    def test_a_biased_model_is_rejected_with_the_failure_recorded(self) -> None:
        biased = tuple(
            make_stroke(
                i,
                session="hold-1",
                batch="batch-B",
                relative_density=0.8,
                perturbation=0.25,
            )
            for i in range(6)
        )
        qualification = qualify(make_dataset(holdout=biased))
        firm = next(
            v for v in qualification.verdicts if v.regime.key == "standard-firm"
        )
        assert not firm.qualified
        assert any("relative bias" in r for r in firm.reasons)
        assert firm.model_error_detected_count == 6
        assert firm.dominant_uncertainty in {"measurement", "numerical", "input"}

    def test_a_failed_fit_rejects_every_regime(self) -> None:
        qualification = qualify(
            make_dataset(calibration=make_strokes("cal-1", 12, densities=(0.7,)))
        )
        assert qualification.transfer is None
        assert qualification.qualified_regimes == ()
        assert all("unidentifiable" in v.reasons[0] for v in qualification.verdicts)

    def test_a_qualified_regime_cannot_rest_on_a_failed_fit(self) -> None:
        good = qualify(make_dataset())
        bad_fit = fit_transfer(make_strokes("cal-1", 3))
        with pytest.raises(TransferQualificationError, match="cannot be qualified"):
            TransferQualification(
                version=good.version,
                evidence_digest=good.evidence_digest,
                protocol=good.protocol,
                intended_use=good.intended_use,
                tolerances=good.tolerances,
                fit=bad_fit,
                verdicts=good.verdicts,
                calibration_stroke_ids=good.calibration_stroke_ids,
                holdout_stroke_ids=good.holdout_stroke_ids,
            )

    def test_validate_holdout_uses_only_held_out_strokes(self) -> None:
        dataset = make_dataset()
        fit = fit_transfer(dataset.calibration_strokes)
        verdicts = validate_holdout(dataset, fit)
        used = {sid for v in verdicts for sid in v.holdout_stroke_ids}
        assert used == {s.stroke_id for s in dataset.holdout_strokes}

    def test_each_stroke_distinguishes_noise_from_model_error(self) -> None:
        qualification = qualify(make_dataset())
        firm = next(
            v for v in qualification.verdicts if v.regime.key == "standard-firm"
        )
        assert firm.noise_limited_count + firm.model_error_detected_count == len(
            firm.holdout_stroke_ids
        )

    def test_the_report_names_rejections_uncertainty_and_the_rig(self) -> None:
        report = qualification_report_markdown(qualify(make_dataset()))
        assert "rejected" in report
        assert "Dominant u" in report
        assert "three-camera rig" in report
        assert "ball spin | no" in report
        assert "No camera purchase" in report
        assert PRACTICAL_TOLERANCES.rationale in report


class TestTheVerdictFloorIsLiftedOnlyByEvidence:
    """Step 5 of #9543: versioned evidence for the qualified regime, floor elsewhere."""

    @pytest.fixture
    def qualification(self) -> TransferQualification:
        return qualify(make_dataset())

    def test_inside_a_qualified_regime_the_launch_model_reads_within(
        self, qualification: TransferQualification
    ) -> None:
        within = solver_verdict(speed_m_s=0.1, feature_lengths_m={"clubhead": 0.1})
        strike = delivery(
            speed_m_s=0.1, impulse_n_s=0.02, bed_relative_density=0.8, verdict=within
        )
        verdict = launch_verdict(
            strike, lie=BallLie(depth_m=0.005), qualification=qualification
        )
        assert verdict.status is EnvelopeStatus.WITHIN
        assert any(qualification.version in r for r in verdict.reasons)
        assert not any("uncalibrated" in r for r in verdict.reasons)

    def test_a_carry_never_reads_better_than_the_shot_behind_it(
        self, qualification: TransferQualification
    ) -> None:
        result = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(bed_relative_density=0.8),
            club_loft_rad=GREENSIDE_LOFT_RAD,
            qualification=qualification,
        )
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert any(qualification.version in r for r in result.verdict.reasons)
        assert result.transfer_efficiency == pytest.approx(
            qualification.transfer.efficiency_for(0.8)  # type: ignore[union-attr]
        )

    def test_outside_the_qualified_regime_the_floor_is_preserved(
        self, qualification: TransferQualification
    ) -> None:
        within = solver_verdict(speed_m_s=0.1, feature_lengths_m={"clubhead": 0.1})
        plugged = launch_verdict(
            delivery(speed_m_s=0.1, impulse_n_s=0.02, verdict=within),
            lie=BallLie(depth_m=0.04),
            qualification=qualification,
        )
        assert plugged.status is EnvelopeStatus.BEYOND_VALIDATION
        assert any("rejected regime" in r for r in plugged.reasons)
        assert any("uncalibrated" in r for r in plugged.reasons)

    def test_outside_the_matrix_the_floor_is_preserved(
        self, qualification: TransferQualification
    ) -> None:
        within = solver_verdict(speed_m_s=0.1, feature_lengths_m={"clubhead": 0.1})
        buried = launch_verdict(
            delivery(speed_m_s=0.1, impulse_n_s=0.02, verdict=within),
            lie=BallLie(depth_m=0.015),
            qualification=qualification,
        )
        assert buried.status is EnvelopeStatus.BEYOND_VALIDATION
        assert any("outside every regime" in r for r in buried.reasons)

    def test_a_qualification_without_a_lie_is_refused(
        self, qualification: TransferQualification
    ) -> None:
        with pytest.raises(ValueError, match="pass the lie"):
            launch_verdict(delivery(), qualification=qualification)

    def test_other_parameters_beside_a_qualification_are_refused(
        self, qualification: TransferQualification
    ) -> None:
        with pytest.raises(ValueError, match="differs from the parameters"):
            compute_ball_launch_from_splash(
                lie=BallLie(depth_m=0.005),
                ball=BallProperties(),
                delivery=delivery(),
                club_loft_rad=GREENSIDE_LOFT_RAD,
                transfer=MomentumTransfer(efficiency=0.9),
                qualification=qualification,
            )

    def test_the_fitted_parameters_are_calibrated_and_still_not_measured(
        self, qualification: TransferQualification
    ) -> None:
        result = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(bed_relative_density=0.8),
            club_loft_rad=GREENSIDE_LOFT_RAD,
            qualification=qualification,
        )
        entry = result.provenance.entry("transfer_efficiency")
        assert entry.basis is ProvenanceBasis.CALIBRATED
        assert qualification.version in entry.source
        assert "standard-firm" in entry.note
        assert result.provenance.entry("sand_ball_friction").basis is (
            ProvenanceBasis.ESTIMATED
        )
        assert result.measured_constants() == ()

    def test_a_failed_qualification_changes_nothing(self) -> None:
        failed = qualify(
            make_dataset(calibration=make_strokes("cal-1", 12, densities=(0.7,)))
        )
        result = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(bed_relative_density=0.8),
            club_loft_rad=GREENSIDE_LOFT_RAD,
            qualification=failed,
        )
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert result.transfer_efficiency == pytest.approx(
            DEFAULT_MOMENTUM_TRANSFER.efficiency_for(0.8)
        )
        assert result.provenance.entry("transfer_efficiency").basis is (
            ProvenanceBasis.ESTIMATED
        )


class TestTheObjectiveDispositionOf9239:
    """Unavailable until calibrated; then degenerate or supported per target."""

    def test_unavailable_without_a_qualification(self) -> None:
        assert (
            objective_disposition(
                None,
                delivery(),
                BallLie(depth_m=0.005),
                nominal_carry_m=1.6,
                target_carry_m=12.0,
            )
            is ObjectiveDisposition.UNAVAILABLE_UNCALIBRATED
        )

    def test_unavailable_outside_the_qualified_regime(self) -> None:
        qualification = qualify(make_dataset())
        assert (
            objective_disposition(
                qualification,
                delivery(),
                BallLie(depth_m=0.04),
                nominal_carry_m=12.0,
                target_carry_m=12.0,
            )
            is ObjectiveDisposition.UNAVAILABLE_OUTSIDE_QUALIFIED_REGIME
        )

    def test_a_target_the_qualified_model_cannot_reach_is_degenerate(self) -> None:
        qualification = qualify(make_dataset())
        assert (
            objective_disposition(
                qualification,
                delivery(bed_relative_density=0.8),
                BallLie(depth_m=0.005),
                nominal_carry_m=1.6,
                target_carry_m=12.0,
            )
            is ObjectiveDisposition.DEGENERATE_TARGET
        )

    def test_a_reachable_target_is_supported(self) -> None:
        qualification = qualify(make_dataset())
        assert (
            objective_disposition(
                qualification,
                delivery(bed_relative_density=0.8),
                BallLie(depth_m=0.005),
                nominal_carry_m=11.5,
                target_carry_m=12.0,
            )
            is ObjectiveDisposition.SUPPORTED
        )


class TestAnOnFileMeasurementLiftsTheLiteratureRefusal:
    """``ValidationComparison`` admits a launch quantity only with a real record."""

    def test_the_refusal_stands_without_a_record(self) -> None:
        with pytest.raises(NoReferenceDataError):
            ValidationComparison(
                quantity="ball_speed_m_s",
                unit="m/s",
                simulation_value=10.0,
                experiment_value=9.5,
                numerical=NumericalUncertainty(u_h=0.01),
                u_input=0.0,
                u_exp=0.2,
                reference="x",
            )

    def test_an_instrument_record_carrying_d_is_admitted(self) -> None:
        record = _record("ball_launch_speed_m_s", 9.5, unit="m/s")
        comparison = ValidationComparison(
            quantity="ball_speed_m_s",
            unit="m/s",
            simulation_value=10.0,
            experiment_value=9.5,
            numerical=NumericalUncertainty(u_h=0.01),
            u_input=0.0,
            u_exp=0.2,
            reference=record.source,
            measured_record=record,
        )
        assert comparison.experiment_value == 9.5

    def test_a_synthetic_record_is_refused(self) -> None:
        fixture = _record(
            "ball_launch_speed_m_s",
            None,
            unit="m/s",
            basis=MeasurementBasis.SYNTHETIC_FIXTURE,
        )
        with pytest.raises(VandVError, match="synthetic fixture"):
            ValidationComparison(
                quantity="ball_speed_m_s",
                unit="m/s",
                simulation_value=10.0,
                experiment_value=9.5,
                numerical=NumericalUncertainty(u_h=0.01),
                u_input=0.0,
                u_exp=0.2,
                reference="x",
                measured_record=fixture,
            )

    @pytest.mark.parametrize(
        ("unit", "value", "match"),
        [("mph", 9.5, "on-file record is in"), ("m/s", 9.0, "must be the value")],
    )
    def test_a_record_that_does_not_carry_d_is_refused(
        self, unit: str, value: float, match: str
    ) -> None:
        record = _record("ball_launch_speed_m_s", value, unit=unit)
        with pytest.raises(VandVError, match=match):
            ValidationComparison(
                quantity="ball_speed_m_s",
                unit="m/s",
                simulation_value=10.0,
                experiment_value=9.5,
                numerical=NumericalUncertainty(u_h=0.01),
                u_input=0.0,
                u_exp=0.2,
                reference="x",
                measured_record=record,
            )


class TestTheRigCapabilityRegister:
    """What the three-camera rig can and cannot measure, stated as data."""

    def test_launch_is_measurable_and_spin_ejecta_force_are_not(self) -> None:
        by_quantity = {c.quantity: c.measurable for c in THREE_CAMERA_RIG_CAPABILITY}
        assert by_quantity["ball launch speed and direction"]
        assert not by_quantity["ball spin"]
        assert not by_quantity["ejecta speed and sand motion"]
        assert not by_quantity["force and impulse on the head"]

    def test_the_table_renders_every_row(self) -> None:
        table = rig_capability_markdown()
        assert table.count("\n") == len(THREE_CAMERA_RIG_CAPABILITY) + 1

    def test_predict_launch_is_the_shipped_pipeline(self) -> None:
        stroke = make_stroke(0, session="s")
        assert predict_launch(stroke, TRUE_TRANSFER).ball_speed_m_s == pytest.approx(
            stroke.measured_ball_speed_m_s
        )


def test_the_splash_module_names_the_program() -> None:
    assert splash_module.__doc__ is not None
    assert "9543" in splash_module.__doc__
