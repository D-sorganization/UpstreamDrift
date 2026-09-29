"""Tests for the F1 constitutive calibration record (issue #8733 section 6).

The subject of these tests is not the fit -- that is
``test_f1_shear_cell_8733`` -- but **what the fit is allowed to claim**.
A calibration that quietly relabelled an unfitted parameter, or that let
"fitted to a declared target" drift into "measured", would be the #7999
failure with better documentation, so the boundary is pinned here.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from bunkershot3d.calibration.f1_continuum import (
    F1_CALIBRATION_HONESTY_NOTE,
    F1_UNCALIBRATED_PROPERTIES,
    calibrate_f1_friction_angle,
    calibrated_continuum,
    calibrated_sand,
    f1_calibrated_provenance,
    main,
)
from bunkershot3d.calibration.f1_shear_cell import (
    F1DrainedShearCellExperiment,
    plane_strain_friction_angle_deg,
)
from bunkershot3d.sand.provenance import (
    QUIKRETE_FRICTION_ANGLE_DEG,
    ProvenanceBasis,
)
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.envelope import F1_STANDING_CAVEATS

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def calibration():
    """The closed-form fit; the stochastic search is exercised separately."""
    return calibrate_f1_friction_angle(search=False)


class TestTheFitIsReportedAgainstWhatItReplaced:
    def test_the_borrowed_angle_is_the_quikrete_one(self, calibration) -> None:
        if calibration.borrowed_friction_angle_deg != QUIKRETE_FRICTION_ANGLE_DEG:
            raise AssertionError(calibration.borrowed_friction_angle_deg)

    def test_the_fitted_angle_hits_the_target_midpoint_in_plane_strain(
        self, calibration
    ) -> None:
        midpoint = 0.5 * (
            calibration.target_phi_peak_deg + calibration.target_phi_res_deg
        )
        if abs(calibration.fitted_plane_strain_angle_deg - midpoint) > 1.0e-6:
            raise AssertionError(calibration.fitted_plane_strain_angle_deg)

    def test_the_plane_strain_angle_is_reported_beside_the_input_angle(
        self, calibration
    ) -> None:
        """Quoting only the input angle would overstate the cone by ~2 deg."""
        expected = plane_strain_friction_angle_deg(
            calibration.fitted_friction_angle_deg
        )
        if abs(calibration.fitted_plane_strain_angle_deg - expected) > 1.0e-9:
            raise AssertionError(calibration.fitted_plane_strain_angle_deg)
        if calibration.fitted_plane_strain_angle_deg >= (
            calibration.fitted_friction_angle_deg
        ):
            raise AssertionError("phi* should be the softer of the two here")

    def test_the_fit_improves_on_the_borrowed_value(self, calibration) -> None:
        if calibration.fitted_residual_deg2 > calibration.borrowed_residual_deg2:
            raise AssertionError(
                f"{calibration.fitted_residual_deg2} > "
                f"{calibration.borrowed_residual_deg2}"
            )

    def test_the_fit_reaches_the_structural_floor(self, calibration) -> None:
        """Everything left is the peak-residual gap the model cannot make."""
        if not calibration.fit_is_at_the_structural_floor:
            raise AssertionError(
                f"fitted {calibration.fitted_residual_deg2} vs floor "
                f"{calibration.irreducible_residual_deg2}"
            )

    def test_most_of_the_residual_is_irreducible(self, calibration) -> None:
        """State it, so 12.5 deg^2 is never read as a bad fit."""
        if calibration.removable_residual_deg2 >= (
            calibration.irreducible_residual_deg2
        ):
            raise AssertionError(
                "the removable part is no longer the smaller share; the "
                "reporting in to_mapping needs revisiting"
            )

    def test_the_shift_is_small_and_signed(self, calibration) -> None:
        shift = calibration.friction_angle_shift_deg
        if not math.isfinite(shift) or shift <= 0.0:
            raise AssertionError(shift)
        if shift > 5.0:
            raise AssertionError(
                f"a {shift} deg shift is large enough that the report must say "
                "so prominently; update the record rather than the tolerance"
            )


class TestProvenanceUpgradesOnlyWhatWasFitted:
    def test_the_friction_angle_stops_being_borrowed(self, calibration) -> None:
        cell = F1DrainedShearCellExperiment()
        provenance = f1_calibrated_provenance(cell.sand, calibration)
        entry = provenance.entry("friction_angle_deg")
        if entry.basis is not ProvenanceBasis.CONVENTION:
            raise AssertionError(entry.basis)

    def test_the_friction_angle_is_never_called_measured(self, calibration) -> None:
        cell = F1DrainedShearCellExperiment()
        provenance = f1_calibrated_provenance(cell.sand, calibration)
        if provenance.measured_properties():
            raise AssertionError(
                f"the fit claimed measurements: {provenance.measured_properties()}"
            )

    def test_nothing_else_stops_being_borrowed(self, calibration) -> None:
        """Packing, gradation and grain density were not touched by the fit."""
        cell = F1DrainedShearCellExperiment()
        before = set(cell.sand.provenance.borrowed_properties())
        after = set(
            f1_calibrated_provenance(cell.sand, calibration).borrowed_properties()
        )
        if before - after != {"friction_angle_deg"}:
            raise AssertionError(
                f"the fit changed the basis of {before - after}, but it only "
                "fitted the friction angle"
            )

    def test_the_shear_modulus_keeps_its_estimate(self, calibration) -> None:
        """The headline restraint: an unfitted parameter stays ESTIMATED."""
        cell = F1DrainedShearCellExperiment()
        continuum = calibrated_continuum(cell.sand, calibration)
        entry = continuum.provenance.entry("elastic_shear_modulus_pa")
        if entry.basis is not ProvenanceBasis.ESTIMATED:
            raise AssertionError(
                f"the shear modulus was relabelled {entry.basis}; the "
                "calibration could not identify it and must not claim to have"
            )
        if "Hardin" not in entry.source:
            raise AssertionError(entry.source)

    def test_the_record_names_what_was_not_calibrated(self, calibration) -> None:
        joined = " ".join(F1_UNCALIBRATED_PROPERTIES)
        for name in ("elastic_shear_modulus_pa", "poisson_ratio", "compressive_cap"):
            if name not in joined:
                raise AssertionError(f"{name} missing from the not-calibrated list")

    def test_the_calibrated_sand_carries_the_fitted_angle(self, calibration) -> None:
        cell = F1DrainedShearCellExperiment()
        sand = calibrated_sand(cell.sand, calibration)
        if sand.friction_angle_deg != calibration.fitted_friction_angle_deg:
            raise AssertionError(sand.friction_angle_deg)
        sand.provenance.require_keys()

    def test_the_calibrated_continuum_moves_its_cone(self, calibration) -> None:
        cell = F1DrainedShearCellExperiment()
        before = SandContinuum.from_sand_state(cell.sand).alpha
        after = calibrated_continuum(cell.sand, calibration).alpha
        if not after > before:
            raise AssertionError(f"alpha {before} -> {after}")


class TestTheHonestyBoundary:
    def test_the_note_refuses_the_word_validation(self) -> None:
        lowered = F1_CALIBRATION_HONESTY_NOTE.lower()
        for fragment in (
            "not measured",
            "does not validate",
            "0 of 4",
            "beyond_validation",
            "1.44",
        ):
            if fragment not in lowered:
                raise AssertionError(f"{fragment!r} missing from the honesty note")

    def test_the_calibration_never_claims_a_measurement(self, calibration) -> None:
        if calibration.is_measured_on_bunker_sand:
            raise AssertionError("the calibration claimed to be a measurement")
        record = calibration.to_mapping()
        if record["provenance"]["measured_on_bunker_sand"]:
            raise AssertionError(record["provenance"])

    def test_the_record_pins_validation_at_zero_of_four(self, calibration) -> None:
        record = calibration.to_mapping()["provenance"]
        if record["nasa_std_7009b_validation_levels_met"] != 0:
            raise AssertionError(record)
        if record["nasa_std_7009b_validation_levels_total"] != 4:
            raise AssertionError(record)

    def test_the_f1_standing_caveats_are_untouched(self) -> None:
        """A calibration must not quietly retire a caveat."""
        if not F1_STANDING_CAVEATS:
            raise AssertionError("F1 lost its standing caveats")

    def test_the_max_validated_speed_is_untouched(self) -> None:
        from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S

        if abs(MAX_VALIDATED_SPEED_M_S - 1.44) > 1.0e-12:
            raise AssertionError(MAX_VALIDATED_SPEED_M_S)


class TestTheRecord:
    def test_it_round_trips_through_yaml(self, calibration) -> None:
        text = yaml.dump(calibration.to_mapping(), sort_keys=False)
        loaded = yaml.safe_load(text)
        if loaded["replaced"]["friction_angle_deg"] != (
            calibration.borrowed_friction_angle_deg
        ):
            raise AssertionError(loaded["replaced"])

    def test_it_reports_the_cost(self, calibration) -> None:
        cost = calibration.to_mapping()["cost"]
        if "objective_evaluations" not in cost or "wall_clock_s" not in cost:
            raise AssertionError(cost)

    def test_it_explains_the_irreducible_residual(self, calibration) -> None:
        because = calibration.to_mapping()["residuals_deg2"]["irreducible_because"]
        if "softening" not in because:
            raise AssertionError(because)

    def test_the_cli_writes_a_record(self, tmp_path: Path) -> None:
        destination = tmp_path / "f1_continuum.yaml"
        if main(["--no-search", "--output", str(destination)]) != 0:
            raise AssertionError("the CLI reported failure")
        record = yaml.safe_load(destination.read_text(encoding="utf-8"))
        if record["provenance"]["basis"] != ProvenanceBasis.CONVENTION.value:
            raise AssertionError(record["provenance"])
        if record["provenance"]["calibrated"] != ["friction_angle_deg"]:
            raise AssertionError(record["provenance"])


@pytest.mark.slow
class TestTheStochasticSearchAgrees:
    """The full ``differential_evolution`` run: ~40 solves, tens of seconds."""

    def test_the_search_finds_the_closed_form(self) -> None:
        calibration = calibrate_f1_friction_angle(search=True)
        gap = abs(
            calibration.searched_friction_angle_deg
            - calibration.fitted_friction_angle_deg
        )
        if gap > 0.05:
            raise AssertionError(f"search and closed form differ by {gap} deg")
        if calibration.n_objective_evaluations < 1:
            raise AssertionError("the search reported no objective evaluations")
