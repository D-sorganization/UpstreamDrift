"""Fit F1's constitutive model, and record exactly what the fit earned.

Issue #8733 section 6.  :mod:`bunkershot3d.calibration.f1_shear_cell`
runs the experiment; this module runs it through
:class:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer`,
compares the fitted friction angle against the Quikrete-borrowed one it
replaces, and writes a provenance record that upgrades **only** what was
actually fitted.

The honesty boundary
--------------------

The drained-shear-cell targets are **simulated experiments, not
measurements of real bunker sand**.  Fitting F1 to them makes it
*self-consistent with a stated experiment*: it replaces "borrowed from a
hardware-store analogue" with "fitted to a declared target".  It does
**not** validate the model.

Concretely, and enforced by the test suite:

* NASA-STD-7009B validation for this package stays at **0 of 4**;
* every F1 verdict stays
  :attr:`~bunkershot3d.solvers.envelope.EnvelopeStatus.BEYOND_VALIDATION`;
* ``MAX_VALIDATED_SPEED_M_S`` stays 1.44 m/s;
* the fitted friction angle is recorded as
  :attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.CONVENTION` --
  a modelling convention chosen for reproducibility -- and **never** as
  :attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.MEASURED`.
  ``CONVENTION`` is not an upgrade over ``BORROWED_ANALOGUE``; it is a
  different, and more checkable, kind of not-a-measurement.

What the fit does not earn
--------------------------

The elastic shear modulus keeps its Hardin & Richart (1963)
:attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.ESTIMATED` record,
unchanged, because the drained limit ratio ``q/p`` cancels the elastic
constants exactly and no shear cell can identify them.  The continuum is
therefore built with the modulus *derived* rather than supplied, so
:func:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum.from_sand_state`
keeps emitting the Hardin & Richart provenance entry rather than the
"caller-supplied" one.  Poisson's ratio, the compressive cap and the
cohesive tip are likewise untouched.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

from ..sand import SandState
from ..sand.provenance import (
    PropertyProvenance,
    ProvenanceBasis,
    SandProvenance,
)
from ..solvers.exceptions import CalibrationError
from ..solvers.mpm.constitutive import SandContinuum, drucker_prager_alpha
from .f1_shear_cell import (
    F1_SHEAR_CELL_TARGET_NOTE,
    F1DrainedShearCellExperiment,
    plane_strain_friction_angle_deg,
)
from .optimizer import CalibrationOptimizer

__all__ = [
    "F1_CALIBRATION_HONESTY_NOTE",
    "F1_UNCALIBRATED_PROPERTIES",
    "F1FrictionAngleCalibration",
    "calibrate_f1_friction_angle",
    "f1_calibrated_provenance",
    "main",
]

logger = structlog.get_logger()

F1_CALIBRATION_HONESTY_NOTE = (
    "FITTED TO A SIMULATED EXPERIMENT, NOT MEASURED. The angle-of-repose and "
    "drained-shear-cell targets in bunkershot3d.calibration are declared "
    "numbers, not measurements of real golf bunker sand: issue #8610 records "
    "that no published bunker-sand friction angle was found and issue #8616 "
    "that no published measurement exists for any quantity F1 produces. "
    "Fitting F1's constitutive model to them makes it self-consistent with a "
    "stated experiment -- it replaces 'borrowed from a hardware-store "
    "analogue' with 'fitted to a declared target' -- and does NOT validate "
    "the model. NASA-STD-7009B validation for this package stays at 0 of 4, "
    "every F1 verdict stays BEYOND_VALIDATION, and MAX_VALIDATED_SPEED_M_S "
    "stays 1.44 m/s. Nothing here makes an F1 answer more trustworthy."
)
"""The boundary, stated where the number is produced rather than only in prose."""

F1_UNCALIBRATED_PROPERTIES: tuple[str, ...] = (
    "elastic_shear_modulus_pa (Hardin & Richart 1963 estimate; the drained "
    "limit ratio q/p cancels the elastic constants exactly, so no shear cell "
    "can identify it -- CalibrationOptimizer.check_sensitivity refuses it)",
    "poisson_ratio (textbook convention for sands, 0.25-0.35)",
    "compressive_cap (derived from the packing state, not fitted)",
    "cohesion_pa (from the moisture model; the shear-cell envelope reports an "
    "intercept but no target constrains it)",
    "grain_diameter_m, density_kg_m3 (carried from the sand state)",
)
"""Everything F1 still carries that this calibration did not touch."""

_PROVENANCE_KEY = "friction_angle_deg"
_MODULUS_KEY = "elastic_shear_modulus_pa"
_CALIBRATION_KEY = "f1_constitutive_calibration"

#: How far the stochastic search may sit from the closed-form optimum before
#: the result is refused. ``differential_evolution`` stops on a relative
#: tolerance, so a few hundredths of a degree is expected; a whole degree
#: would mean the search, not the material, chose the answer (issue #7999).
_SEARCH_AGREEMENT_TOLERANCE_DEG = 0.05


class _CountedExperiment:
    """Forwards to a shear cell and counts the solves the search asks for.

    A wrapper rather than a monkey-patch: the cost figure this module
    reports has to be the cost of the run that produced the number beside
    it, and an experiment that had been mutated and restored could not be
    trusted to still be the one that was fitted.
    """

    def __init__(self, experiment: F1DrainedShearCellExperiment) -> None:
        self._experiment = experiment
        self.n_calls = 0
        self.calibrated_parameters = experiment.calibrated_parameters
        self.parameter_bounds = experiment.parameter_bounds
        self.target_phi_peak = experiment.target_phi_peak
        self.target_phi_res = experiment.target_phi_res

    def run_simulation(self, params: dict) -> tuple[float, float]:
        """Forward one solve and count it."""
        self.n_calls += 1
        return self._experiment.run_simulation(params)


@dataclass(frozen=True, slots=True)
class F1FrictionAngleCalibration:
    """What the fit produced, beside what it replaced.

    Every "fitted" field has a "borrowed" twin so the comparison cannot be
    lost between the calculation and the report.

    Attributes:
        sand_name: The bed the calibration was run on.
        borrowed_friction_angle_deg: The Quikrete-analogue value (#7999).
        fitted_friction_angle_deg: The value fitted to the declared
            targets.
        borrowed_alpha: Drucker-Prager cone slope before the fit.
        fitted_alpha: Cone slope after it.
        borrowed_plane_strain_angle_deg: ``phi*`` before the fit -- the
            angle F1 actually enforced.
        fitted_plane_strain_angle_deg: ``phi*`` after it.
        target_phi_peak_deg: Declared peak target.
        target_phi_res_deg: Declared residual target.
        borrowed_phi_peak_deg: Shear-cell response at the borrowed angle.
        borrowed_phi_res_deg: Residual response at the borrowed angle.
        fitted_phi_peak_deg: Shear-cell response at the fitted angle.
        fitted_phi_res_deg: Residual response at the fitted angle.
        borrowed_residual_deg2: Objective at the borrowed angle.
        fitted_residual_deg2: Objective at the fitted angle.
        irreducible_residual_deg2: The part of the objective no parameter
            value can remove, because the model has no softening.
        searched_friction_angle_deg: What the stochastic search returned,
            kept beside the closed form so the two can be compared.
        shear_modulus_pa: The modulus the fitted continuum carries --
            **not** fitted; see :data:`F1_UNCALIBRATED_PROPERTIES`.
        n_objective_evaluations: Shear-cell solves the search consumed.
        wall_clock_s: How long the search took.
    """

    sand_name: str
    borrowed_friction_angle_deg: float
    fitted_friction_angle_deg: float
    borrowed_alpha: float
    fitted_alpha: float
    borrowed_plane_strain_angle_deg: float
    fitted_plane_strain_angle_deg: float
    target_phi_peak_deg: float
    target_phi_res_deg: float
    borrowed_phi_peak_deg: float
    borrowed_phi_res_deg: float
    fitted_phi_peak_deg: float
    fitted_phi_res_deg: float
    borrowed_residual_deg2: float
    fitted_residual_deg2: float
    irreducible_residual_deg2: float
    searched_friction_angle_deg: float
    shear_modulus_pa: float
    n_objective_evaluations: int
    wall_clock_s: float

    @property
    def friction_angle_shift_deg(self) -> float:
        """How far the fit moved the friction angle, signed."""
        return self.fitted_friction_angle_deg - self.borrowed_friction_angle_deg

    @property
    def removable_residual_deg2(self) -> float:
        """Objective the fit actually removed, above the structural floor."""
        return self.borrowed_residual_deg2 - self.fitted_residual_deg2

    @property
    def fit_is_at_the_structural_floor(self) -> bool:
        """True when the fit reached everything the model can reach."""
        return abs(self.fitted_residual_deg2 - self.irreducible_residual_deg2) <= 1.0e-6

    @property
    def is_measured_on_bunker_sand(self) -> bool:
        """Always ``False``. Kept as a property so it can be asserted."""
        return False

    def to_mapping(self) -> dict[str, Any]:
        """Return a YAML-serialisable record of the whole calibration."""
        return {
            "sand_parameters": {
                "friction_angle_deg": float(self.fitted_friction_angle_deg),
                "drucker_prager_alpha": float(self.fitted_alpha),
                "plane_strain_friction_angle_deg": float(
                    self.fitted_plane_strain_angle_deg
                ),
                "shear_modulus_pa": float(self.shear_modulus_pa),
            },
            "replaced": {
                "friction_angle_deg": float(self.borrowed_friction_angle_deg),
                "drucker_prager_alpha": float(self.borrowed_alpha),
                "plane_strain_friction_angle_deg": float(
                    self.borrowed_plane_strain_angle_deg
                ),
                "basis": ProvenanceBasis.BORROWED_ANALOGUE.value,
                "shift_deg": float(self.friction_angle_shift_deg),
            },
            "targets": {
                "phi_peak_deg": float(self.target_phi_peak_deg),
                "phi_res_deg": float(self.target_phi_res_deg),
                "note": F1_SHEAR_CELL_TARGET_NOTE,
            },
            "response": {
                "borrowed_phi_peak_deg": float(self.borrowed_phi_peak_deg),
                "borrowed_phi_res_deg": float(self.borrowed_phi_res_deg),
                "fitted_phi_peak_deg": float(self.fitted_phi_peak_deg),
                "fitted_phi_res_deg": float(self.fitted_phi_res_deg),
            },
            "residuals_deg2": {
                "borrowed": float(self.borrowed_residual_deg2),
                "fitted": float(self.fitted_residual_deg2),
                "irreducible": float(self.irreducible_residual_deg2),
                "removed_by_the_fit": float(self.removable_residual_deg2),
                "irreducible_because": (
                    "rate-independent perfect plasticity has no peak-to-"
                    "residual softening, so phi_peak == phi_res identically "
                    "and the 5 deg gap the targets ask for cannot be produced "
                    "by any parameter value"
                ),
            },
            "cost": {
                "objective_evaluations": int(self.n_objective_evaluations),
                "wall_clock_s": float(self.wall_clock_s),
                "note": (
                    "One objective evaluation is three drained cells on the "
                    "constitutive model, not an MPM solve. F1's MPM "
                    "discretisation is not in this loop; see "
                    "bunkershot3d.calibration.f1_repose for what an MPM "
                    "angle-of-repose target would cost and why it is not "
                    "used as one."
                ),
            },
            "provenance": {
                "basis": ProvenanceBasis.CONVENTION.value,
                "measured_on_bunker_sand": self.is_measured_on_bunker_sand,
                "calibrated": ["friction_angle_deg"],
                "not_calibrated": list(F1_UNCALIBRATED_PROPERTIES),
                "honesty": F1_CALIBRATION_HONESTY_NOTE,
                "nasa_std_7009b_validation_levels_met": 0,
                "nasa_std_7009b_validation_levels_total": 4,
            },
        }


def f1_calibrated_provenance(
    sand: SandState, calibration: F1FrictionAngleCalibration
) -> SandProvenance:
    """Carry the sand's provenance forward, upgrading only what was fitted.

    The friction angle moves from
    :attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.BORROWED_ANALOGUE`
    to :attr:`~bunkershot3d.sand.provenance.ProvenanceBasis.CONVENTION`,
    naming the target it was fitted to and the residual it left behind.
    Nothing else changes: in particular the shear modulus keeps whatever
    record :class:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum`
    gave it, which for a derived modulus is Hardin & Richart and
    ``ESTIMATED``.

    Args:
        sand: The bed whose provenance is being carried forward.
        calibration: The result of :func:`calibrate_f1_friction_angle`.

    Returns:
        The updated provenance record.

    Raises:
        CalibrationError: If ``sand`` is not a
            :class:`~bunkershot3d.sand.state.SandState`.
    """
    if not isinstance(sand, SandState):
        raise CalibrationError(f"expected a SandState, got {type(sand).__name__}")
    entries = dict(sand.provenance.entries)
    entries[_PROVENANCE_KEY] = PropertyProvenance(
        basis=ProvenanceBasis.CONVENTION,
        source=(
            "fitted to the declared drained-shear-cell targets of "
            "bunkershot3d.calibration (peak "
            f"{calibration.target_phi_peak_deg:g} deg, residual "
            f"{calibration.target_phi_res_deg:g} deg) through F1's own "
            "Drucker-Prager return map (issue #8733 section 6)"
        ),
        note=(
            f"{calibration.fitted_friction_angle_deg:.4f} deg, replacing the "
            f"{calibration.borrowed_friction_angle_deg:.4f} deg borrowed from "
            "the Quikrete analogue. In plane strain this cone enforces "
            f"{calibration.fitted_plane_strain_angle_deg:.4f} deg, not the "
            "number above. Residual "
            f"{calibration.fitted_residual_deg2:.4f} deg^2, of which "
            f"{calibration.irreducible_residual_deg2:.4f} deg^2 cannot be "
            "removed by any parameter value because the model has no "
            f"peak-to-residual softening. {F1_CALIBRATION_HONESTY_NOTE}"
        ),
    )
    entries[_CALIBRATION_KEY] = PropertyProvenance(
        basis=ProvenanceBasis.CONVENTION,
        source=("bunkershot3d.calibration.f1_continuum.calibrate_f1_friction_angle"),
        note=(
            "Calibrated: friction_angle_deg only. Not calibrated: "
            + "; ".join(F1_UNCALIBRATED_PROPERTIES)
            + ". "
            + F1_CALIBRATION_HONESTY_NOTE
        ),
    )
    return SandProvenance(entries=entries)


def calibrated_sand(
    sand: SandState, calibration: F1FrictionAngleCalibration
) -> SandState:
    """Return the sand state with the fitted angle and its new provenance.

    Args:
        sand: The bed to update.
        calibration: The fit.

    Returns:
        A new :class:`~bunkershot3d.sand.state.SandState`.

    Raises:
        CalibrationError: If ``sand`` is not a
            :class:`~bunkershot3d.sand.state.SandState`.
    """
    provenance = f1_calibrated_provenance(sand, calibration)
    return dataclasses.replace(
        sand,
        friction_angle_deg=calibration.fitted_friction_angle_deg,
        provenance=provenance,
    )


def calibrated_continuum(
    sand: SandState, calibration: F1FrictionAngleCalibration
) -> SandContinuum:
    """Build the F1 continuum the calibration describes.

    The shear modulus is **derived**, not supplied, so the continuum keeps
    the Hardin & Richart ``ESTIMATED`` provenance entry rather than the
    "caller-supplied" one.  Passing the fitted continuum's own modulus back
    in would relabel an unfitted number as a deliberate choice.

    Args:
        sand: The bed.
        calibration: The fit.

    Returns:
        The continuum.
    """
    return SandContinuum.from_sand_state(calibrated_sand(sand, calibration))


def calibrate_f1_friction_angle(
    experiment: F1DrainedShearCellExperiment | None = None,
    *,
    search: bool = True,
) -> F1FrictionAngleCalibration:
    """Fit F1's friction angle to the declared shear-cell targets.

    Runs the stochastic search
    :meth:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer.optimize`
    performs *and* the closed form
    :meth:`~bunkershot3d.calibration.f1_shear_cell.F1DrainedShearCellExperiment.fit_friction_angle_deg`,
    and refuses to return if they disagree by more than
    :data:`_SEARCH_AGREEMENT_TOLERANCE_DEG`.  A global optimiser that
    disagreed with an available closed form would be reporting its own
    population rather than the material -- the exact #7999 failure -- and
    that has to be a raise rather than a log line.

    Args:
        experiment: The shear cell. Defaults to a fresh
            :class:`~bunkershot3d.calibration.f1_shear_cell.F1DrainedShearCellExperiment`.
        search: Run the stochastic search. Setting this False reports the
            closed form for both, which is a fast path for tests and is
            recorded as zero objective evaluations.

    Returns:
        The calibration record.

    Raises:
        CalibrationError: If the search and the closed form disagree.
    """
    cell = F1DrainedShearCellExperiment() if experiment is None else experiment
    borrowed_angle = float(cell.sand.friction_angle_deg)
    closed_form = cell.fit_friction_angle_deg()

    evaluations = 0
    started = time.perf_counter()
    if search:
        counted = _CountedExperiment(cell)
        searched = float(CalibrationOptimizer(counted).optimize()["friction_angle_deg"])
        evaluations = counted.n_calls
    else:
        searched = closed_form
    elapsed = time.perf_counter() - started

    if abs(searched - closed_form) > _SEARCH_AGREEMENT_TOLERANCE_DEG:
        raise CalibrationError(
            f"the stochastic search returned {searched:.6f} deg but the closed "
            f"form of the same objective is {closed_form:.6f} deg, a "
            f"{abs(searched - closed_form):.6f} deg disagreement. An optimiser "
            "that misses an available closed form is reporting its own "
            "population, not the material (issue #7999); fix the search rather "
            "than writing this number to disk"
        )

    borrowed_peak, borrowed_res = cell.run_simulation(
        {"friction_angle_deg": borrowed_angle}
    )
    fitted_peak, fitted_res = cell.run_simulation({"friction_angle_deg": closed_form})

    def residual(peak: float, res: float) -> float:
        return (peak - cell.target_phi_peak) ** 2 + (res - cell.target_phi_res) ** 2

    calibration = F1FrictionAngleCalibration(
        sand_name=cell.sand.name,
        borrowed_friction_angle_deg=borrowed_angle,
        fitted_friction_angle_deg=closed_form,
        borrowed_alpha=drucker_prager_alpha(borrowed_angle),
        fitted_alpha=drucker_prager_alpha(closed_form),
        borrowed_plane_strain_angle_deg=plane_strain_friction_angle_deg(borrowed_angle),
        fitted_plane_strain_angle_deg=plane_strain_friction_angle_deg(closed_form),
        target_phi_peak_deg=cell.target_phi_peak,
        target_phi_res_deg=cell.target_phi_res,
        borrowed_phi_peak_deg=borrowed_peak,
        borrowed_phi_res_deg=borrowed_res,
        fitted_phi_peak_deg=fitted_peak,
        fitted_phi_res_deg=fitted_res,
        borrowed_residual_deg2=residual(borrowed_peak, borrowed_res),
        fitted_residual_deg2=residual(fitted_peak, fitted_res),
        irreducible_residual_deg2=cell.irreducible_residual_deg2,
        searched_friction_angle_deg=searched,
        shear_modulus_pa=SandContinuum.from_sand_state(cell.sand).shear_modulus_pa,
        n_objective_evaluations=evaluations,
        wall_clock_s=elapsed,
    )
    logger.info(
        "F1 constitutive calibration complete",
        borrowed_friction_angle_deg=calibration.borrowed_friction_angle_deg,
        fitted_friction_angle_deg=calibration.fitted_friction_angle_deg,
        fitted_residual_deg2=calibration.fitted_residual_deg2,
        irreducible_residual_deg2=calibration.irreducible_residual_deg2,
        measured_on_bunker_sand=False,
    )
    return calibration


def _config_path() -> Path:
    """Where the F1 calibration record is written."""
    return Path(__file__).resolve().parent / "configs" / "f1_continuum.yaml"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: fit F1 and write the record.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-search",
        action="store_true",
        help=(
            "Report the closed form without running the stochastic search. "
            "The written record says how many objective evaluations were used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the record; defaults to configs/f1_continuum.yaml.",
    )
    args = parser.parse_args(argv)

    calibration = calibrate_f1_friction_angle(search=not args.no_search)
    record = calibration.to_mapping()
    destination = _config_path() if args.output is None else args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        yaml.dump(record, handle, default_flow_style=False, sort_keys=False)
    logger.info("Saved F1 calibration record", path=str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
