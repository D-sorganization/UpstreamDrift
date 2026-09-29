"""Calibration module for BunkerShot3D.

Two families live here.

The **backend contact-model** experiments (:mod:`.angle_of_repose`,
:mod:`.drained_shear_cell`) fit a DEM ``friction_coefficient``.  Issue
#7999 records what they are and are not: only the mock paths are
implemented, and their output must never be presented as measured data.

The **F1 constitutive** experiments (:mod:`.f1_shear_cell`,
:mod:`.f1_repose`, :mod:`.f1_continuum`) fit
:class:`~bunkershot3d.solvers.mpm.constitutive.SandContinuum` -- the
material F1 and the F2 reference share, which ADR-0033 chose MPM in order
to calibrate once.  Issue #8733 section 6 records that gap and this is the
part of it that could be closed: the drained shear cell identifies the
friction angle; the shear modulus is not identifiable from any declared
target and keeps its Hardin & Richart estimate; the angle of repose does
not settle to the model's own limit angle and refuses to return a number.

**Nothing in this package is a measurement of real golf bunker sand.**
See :data:`~bunkershot3d.calibration.f1_continuum.F1_CALIBRATION_HONESTY_NOTE`.
"""

from .angle_of_repose import AngleOfReposeExperiment
from .drained_shear_cell import DrainedShearCellExperiment
from .f1_continuum import (
    F1_CALIBRATION_HONESTY_NOTE,
    F1_UNCALIBRATED_PROPERTIES,
    F1FrictionAngleCalibration,
    calibrate_f1_friction_angle,
    f1_calibrated_provenance,
)
from .f1_repose import (
    F1_REPOSE_ARREST_NOTE,
    F1AngleOfReposeExperiment,
    SlopeRelaxation,
    SlopeRelaxationSettings,
    relax_slope,
)
from .f1_shear_cell import (
    F1_SHEAR_CELL_TARGET_NOTE,
    F1DrainedShearCellExperiment,
    MohrCoulombEnvelope,
    drained_biaxial_path,
    friction_angle_for_plane_strain_angle_deg,
    plane_strain_friction_angle_deg,
)
from .optimizer import CalibrationOptimizer

__all__: list[str] = [
    "F1_CALIBRATION_HONESTY_NOTE",
    "F1_REPOSE_ARREST_NOTE",
    "F1_SHEAR_CELL_TARGET_NOTE",
    "F1_UNCALIBRATED_PROPERTIES",
    "AngleOfReposeExperiment",
    "CalibrationOptimizer",
    "DrainedShearCellExperiment",
    "F1AngleOfReposeExperiment",
    "F1DrainedShearCellExperiment",
    "F1FrictionAngleCalibration",
    "MohrCoulombEnvelope",
    "SlopeRelaxation",
    "SlopeRelaxationSettings",
    "calibrate_f1_friction_angle",
    "drained_biaxial_path",
    "f1_calibrated_provenance",
    "friction_angle_for_plane_strain_angle_deg",
    "plane_strain_friction_angle_deg",
    "relax_slope",
]
