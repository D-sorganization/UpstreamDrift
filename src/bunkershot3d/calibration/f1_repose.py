"""An angle-of-repose experiment on the F1 **solver**, and why it refuses.

Issue #8733 section 6 asks for the material calibration ADR-0033 promised
would carry between F1 and the F2 reference.
:mod:`bunkershot3d.calibration.f1_shear_cell` supplies that for the
friction angle by driving the *constitutive model*.  This module attempts
the second experiment the harness already declares -- the angle of repose
-- on the *discretisation*, because a repose angle is a property of the
solved boundary-value problem rather than of a single material point.

The protocol
------------

A dynamic column collapse is **not** an angle-of-repose measurement.  The
deposit slope of a collapsed column is set by the initial aspect ratio,
not by the material's stable angle (Lube, Huppert, Sparks & Hallworth,
*J. Fluid Mech.* **508**:175-199, 2005; Lajeunesse, Monnier & Homsy,
*Phys. Fluids* **17**:103302, 2005), and measured here it reported 15 deg
against a model whose own plane-strain limit is 31.94 deg.  What this
module runs instead is a **quasi-static slope relaxation**: a wedge whose
free surface starts a little above the expected stable angle, seeded in
its own geostatic stress, released, and watched until the slope stops
changing.  That is the measurement the target names.

The result: it does not settle to the material's own angle
----------------------------------------------------------

Measured on the primary development machine at ``dx = 4 mm``, 1,048
particles, a 60 mm wedge released from 45 deg onto a rough base.  First,
the history at the 34 deg every preset carries, whose plane-strain limit
is 31.94 deg:

======  ==============
``t``   fitted slope
======  ==============
0.05 s  44.4 deg
0.15 s  39.6 deg
0.25 s  33.7 deg
0.30 s  32.5 deg
0.40 s  28.9 deg
0.45 s  29.5 deg
0.88 s  26.1 deg
======  ==============

The slope passes **through** the model's own limit angle and keeps
falling; the drift over the last stride is still 8 deg/s.  The per-bin
particle counts show the whole wedge shearing, not a thin surface veneer:
the plateau bins lose material steadily while the mid-slope fills, and the
toe advances by under a millimetre over the whole run, so this is not
runout.  The 95th-percentile particle speed is still 24 mm/s at 0.45 s and
the kinetic energy decays roughly like ``1/t`` without reaching zero.

Second, three friction angles marched to 0.88 s:

=========  ==============  ============  ==========
``phi``    ``phi*``        measured      error
=========  ==============  ============  ==========
28 deg     25.37 deg       21.76 deg     -3.61 deg
34 deg     31.94 deg       26.06 deg     -5.88 deg
40 deg     39.03 deg       39.75 deg     +0.72 deg
=========  ==============  ============  ==========

The error is not an offset that could be calibrated out; it tracks **how
far the release angle sat above the limit**.  At 40 deg the wedge starts
only 6 deg above its own limit, collapses barely at all, and stops near
it; at 28 deg it starts 20 deg above, collapses hard, and overshoots by
3.6 deg.  The endpoint is being set by the collapse transient, not by the
material.

**The measured angle is therefore a property of the settle time and the
release angle rather than of the sand**, and an angle-of-repose target
cannot identify a friction angle for F1 as the solver stands.  A plausible
mechanism, stated as a hypothesis and not as a finding: the flow rule is
non-associated and volume preserving by design (Klar et al. 2016
section 4.2), so a surface at yield gains no dilatancy hardening as it
shears, and a rate-independent perfectly plastic slope sitting exactly at
its limit angle is in *neutral* equilibrium -- any numerical perturbation,
including the three-cell redistribution the quadratic B-spline transfers
perform every step, moves it downhill and nothing moves it back.

What this module therefore does
-------------------------------

It runs the experiment, measures the drift, and **raises** rather than
returning a number that the stopping rule chose.
:meth:`F1AngleOfReposeExperiment.run_simulation` is shaped for
:class:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer` so that
the day the solver arrests, the loop closes with no further plumbing;
until then the refusal is the deliverable.  Reporting 26.1 deg as "F1's
angle of repose" would be the #7999 failure mode with a longer runtime.

Cost, so the trade is visible
-----------------------------

At ``dx = 4 mm`` and 1,048 particles the march costs about 5.2 ms per
step and the CFL step is 12.5 us, so one second of settling is roughly
80,000 steps and **7 minutes** of wall clock.  A ``differential_evolution``
search over one parameter takes ~38 objective evaluations against the
constitutive shear cell (23 s in total); the same search against this
experiment would be four to five hours, and that is before the arrest
problem above makes the answer meaningless.  This experiment is a
cross-check, not a loop.

**Nothing here is a measurement of real bunker sand.** The target angle
is a declared number; see
:data:`~bunkershot3d.calibration.f1_continuum.F1_CALIBRATION_HONESTY_NOTE`.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..sand import SandState
from ..sand.presets import PlayingCondition, playing_condition
from ..solvers.envelope import GRAVITY_M_S2
from ..solvers.exceptions import CalibrationError
from ..solvers.mpm.constitutive import SandContinuum
from ..solvers.mpm.grid import PlaneStrainGrid
from ..solvers.mpm.solver import PlaneStrainMPMSolver, cfl_time_step_s
from ..solvers.mpm.state import (
    DomainWalls,
    ParticleState,
    WallCondition,
    surface_profile_m,
)

__all__ = [
    "F1_REPOSE_ARREST_NOTE",
    "F1_REPOSE_MIN_FRICTION_ANGLE_DEG",
    "F1AngleOfReposeExperiment",
    "SlopeRelaxation",
    "SlopeRelaxationSettings",
    "SlopeSample",
    "relax_slope",
    "wedge_bed",
]


F1_REPOSE_ARREST_NOTE = (
    "F1's plane-strain MPM discretisation does not arrest at an angle of "
    "repose. A wedge released from 45 deg relaxes through the model's own "
    "plane-strain limit angle (31.94 deg for the 34 deg every preset carries) "
    "and keeps flattening: 32.5 deg at 0.30 s, 28.9 deg at 0.40 s, 26.1 deg "
    "at 0.88 s, with the whole wedge shearing rather than a surface veneer "
    "running out. Across friction angles the error against the model's own "
    "limit is -3.61 / -5.88 / +0.72 deg at 28 / 34 / 40 deg, tracking how far "
    "the release sat above the limit rather than any property of the sand. "
    "The measured angle is therefore a property of the settle time and the "
    "release angle, not of the material, so this experiment refuses to return "
    "one. Issue #8733 section 6."
)
"""Why this experiment raises instead of returning a number."""

F1_REPOSE_MIN_FRICTION_ANGLE_DEG = 26.5
"""Below this the geostatic seed is already outside the yield surface.

The wedge is seeded with the uniaxial geostatic stress ``sigma_zz =
-rho g h``, ``sigma_xx = 0``, which
:func:`~bunkershot3d.solvers.mpm.state.settled_bed` also refuses when it
falls outside the cone.  For the Poisson ratio F1 uses that state is
admissible only while ``sqrt(2) alpha > 0.4``, i.e. above about 26.35 deg.
The bound is set a little above that so a search cannot walk onto the
edge of it and turn a modelling limit into an optimiser failure."""

_DIMENSION = 2
_MIN_SLOPE_BINS = 3
"""Bins the slope fit needs before it is a line rather than a guess."""

#: Slope drift, in degrees per second of simulated time, below which the
#: relaxation counts as arrested. One degree per second is two orders of
#: magnitude below the 8-20 deg/s the 4 mm run still shows at 0.9 s, so a
#: run that passes this has genuinely stopped rather than merely slowed.
_ARREST_TOLERANCE_DEG_PER_S = 1.0

_SURFACE_FIT_LOWER_FRACTION = 0.15
"""Bins below this fraction of the wedge height are toe, not slope."""

_SURFACE_FIT_UPPER_FRACTION = 0.90
"""Bins above this fraction are the plateau, not slope."""


@dataclass(frozen=True, slots=True)
class SlopeRelaxationSettings:
    """Geometry and resolution of the slope-relaxation experiment.

    Attributes:
        cell_size_m: Grid ``dx``.
        height_m: Wedge height above the base.
        plateau_length_m: Flat crest before the slope starts.
        initial_slope_deg: Slope the wedge is released from. Above the
            expected stable angle, so the surface has somewhere to go.
        settle_time_s: Simulated time the relaxation is watched for.
        n_strides: Slope samples taken along the way. The arrest test
            needs at least two.
        runout_margin_m: Empty bed beyond the toe.
        particles_per_cell_axis: MPM quadrature density.
        effective_width_m: Out-of-plane width the solver is built with.
            No wrench is read here, so it does not enter any result; it is
            required by the solver and is recorded rather than hidden.
    """

    cell_size_m: float = 4.0e-3
    height_m: float = 0.060
    plateau_length_m: float = 0.040
    initial_slope_deg: float = 45.0
    settle_time_s: float = 0.30
    n_strides: int = 6
    runout_margin_m: float = 0.060
    particles_per_cell_axis: int = 2
    effective_width_m: float = 0.05

    def __post_init__(self) -> None:
        positive = {
            "cell_size_m": self.cell_size_m,
            "height_m": self.height_m,
            "plateau_length_m": self.plateau_length_m,
            "settle_time_s": self.settle_time_s,
            "runout_margin_m": self.runout_margin_m,
            "effective_width_m": self.effective_width_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise CalibrationError(f"{name} must be positive, got {value!r}")
        if not math.isfinite(self.initial_slope_deg) or not (
            0.0 < self.initial_slope_deg < 90.0
        ):
            raise CalibrationError(
                "initial_slope_deg must lie strictly in (0, 90), got "
                f"{self.initial_slope_deg!r}"
            )
        if int(self.n_strides) < 2:
            raise CalibrationError(
                "at least two strides are needed: the arrest test compares the "
                f"slope at two times, got {self.n_strides!r}"
            )
        if int(self.particles_per_cell_axis) < 1:
            raise CalibrationError(
                "particles_per_cell_axis must be at least 1, got "
                f"{self.particles_per_cell_axis!r}"
            )

    @property
    def toe_x_m(self) -> float:
        """Where the initial surface meets the base."""
        return self.plateau_length_m + self.height_m / math.tan(
            math.radians(self.initial_slope_deg)
        )

    @property
    def bed_length_m(self) -> float:
        """Horizontal extent of the domain the wedge sits in."""
        return self.toe_x_m + self.runout_margin_m

    def surface_height_m(self, x_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Initial free surface: a plateau, then a straight slope, then zero."""
        slope = math.tan(math.radians(self.initial_slope_deg))
        return np.where(
            x_m <= self.plateau_length_m,
            self.height_m,
            np.maximum(self.height_m - (x_m - self.plateau_length_m) * slope, 0.0),
        )


@dataclass(frozen=True, slots=True)
class SlopeSample:
    """One slope reading along the relaxation.

    Attributes:
        time_s: Simulated time.
        slope_deg: Least-squares slope of the free surface, in degrees.
        n_bins: Bins the fit used.
        kinetic_energy_j: Bed kinetic energy per unit width.
        toe_x_m: Furthest particle in ``x``.
    """

    time_s: float
    slope_deg: float
    n_bins: int
    kinetic_energy_j: float
    toe_x_m: float


@dataclass(frozen=True, slots=True)
class SlopeRelaxation:
    """The whole relaxation, and whether it ever stopped.

    Attributes:
        samples: Slope readings in time order.
        friction_angle_deg: Angle the continuum was built with.
        plane_strain_limit_deg: ``phi*``, the angle the model enforces --
            what an arrested slope would have to match.
        n_particles: Bed size.
        n_steps: Steps marched.
        time_step_s: The CFL step.
        wall_clock_s: How long the march took.
        settings: The configuration used.
    """

    samples: tuple[SlopeSample, ...]
    friction_angle_deg: float
    plane_strain_limit_deg: float
    n_particles: int
    n_steps: int
    time_step_s: float
    wall_clock_s: float
    settings: SlopeRelaxationSettings = field(default_factory=SlopeRelaxationSettings)

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise CalibrationError(
                "a relaxation with fewer than two samples cannot be tested for "
                f"arrest, got {len(self.samples)}"
            )

    @property
    def final_slope_deg(self) -> float:
        """The slope at the end of the watched interval."""
        return self.samples[-1].slope_deg

    @property
    def drift_deg_per_s(self) -> float:
        """How fast the slope was still changing over the second half.

        Measured across the last two samples rather than across the whole
        run, because the early transient is the wedge collapsing and is
        expected to be fast; what matters is whether it has stopped.
        """
        last, previous = self.samples[-1], self.samples[-2]
        span = last.time_s - previous.time_s
        if span <= 0.0:
            raise CalibrationError(
                f"the last two samples are {span!r} s apart; the drift is not defined"
            )
        return abs(last.slope_deg - previous.slope_deg) / span

    @property
    def has_arrested(self) -> bool:
        """True when the slope has stopped moving to within tolerance."""
        return self.drift_deg_per_s <= _ARREST_TOLERANCE_DEG_PER_S

    @property
    def ms_per_step(self) -> float:
        """Measured cost per step, so the trade is reported not guessed."""
        return 1.0e3 * self.wall_clock_s / max(self.n_steps, 1)

    def require_arrested(self) -> float:
        """Return the repose angle, or refuse because there is not one.

        Returns:
            The final slope in degrees, once the relaxation has arrested.

        Raises:
            CalibrationError: If the slope is still moving. The message
                carries the whole history, because a caller who sees only
                "did not converge" will be tempted to lengthen the run
                until it appears to, and the run does not converge at any
                length yet measured.
        """
        if self.has_arrested:
            return self.final_slope_deg
        history = ", ".join(
            f"{s.time_s:.3f} s: {s.slope_deg:.2f} deg" for s in self.samples
        )
        raise CalibrationError(
            "the F1 slope relaxation has not arrested: the surface is still "
            f"flattening at {self.drift_deg_per_s:.2f} deg/s after "
            f"{self.samples[-1].time_s:.3f} s, having passed through this "
            f"material's own plane-strain limit of "
            f"{self.plane_strain_limit_deg:.2f} deg. History: {history}. "
            "Returning the final slope would report the stopping time rather "
            f"than the material. {F1_REPOSE_ARREST_NOTE}"
        )


def wedge_bed(
    material: SandContinuum, settings: SlopeRelaxationSettings
) -> ParticleState:
    """Build a geostatically seeded wedge of sand.

    :func:`~bunkershot3d.solvers.mpm.state.settled_bed` fills a rectangle
    under a flat surface; a repose experiment needs a sloped one, so the
    lattice is filled and then cut under
    :meth:`SlopeRelaxationSettings.surface_height_m`, with each particle's
    vertical stress formed from **its own** depth below that surface
    rather than from a single free-surface height.

    Args:
        material: The continuum, for density and stiffness.
        settings: Geometry and resolution.

    Returns:
        The bed.

    Raises:
        CalibrationError: If the wedge holds no particles, or if the
            geostatic seed is already outside the yield surface -- which
            it is for a friction angle below about 26.35 deg, and which
            must be a refusal rather than a first step of large plastic
            correction.
    """
    spacing = settings.cell_size_m / int(settings.particles_per_cell_axis)
    count_x = max(int(round(settings.bed_length_m / spacing)), 1)
    count_z = max(int(round(settings.height_m / spacing)), 1)
    mesh_x, mesh_z = np.meshgrid(
        (np.arange(count_x) + 0.5) * spacing,
        (np.arange(count_z) + 0.5) * spacing,
        indexing="ij",
    )
    position = np.stack([mesh_x.ravel(), mesh_z.ravel()], axis=1)
    position = position[position[:, 1] < settings.surface_height_m(position[:, 0])]
    if position.shape[0] < 1:
        raise CalibrationError(
            "the wedge holds no particles; the geometry and the cell size are "
            "incompatible"
        )

    volume = np.full(position.shape[0], spacing * spacing)
    depth = np.maximum(settings.surface_height_m(position[:, 0]) - position[:, 1], 0.0)
    vertical_stress = -material.density_kg_m3 * GRAVITY_M_S2 * depth
    vertical_strain = vertical_stress / material.p_wave_modulus_pa
    worst = float(
        material.yield_value(
            np.stack([np.zeros_like(vertical_strain), vertical_strain], axis=1)
        ).max()
    )
    if worst > 0.0:
        raise CalibrationError(
            f"the geostatic wedge seed is outside the yield surface (y = "
            f"{worst:.4g} Pa) at a friction angle of "
            f"{material.friction_angle_deg:.4g} deg: this wedge cannot stand "
            "up under its own weight at zero lateral confinement, so the first "
            "step would be a large plastic correction rather than a relaxation. "
            f"Keep the friction angle above {F1_REPOSE_MIN_FRICTION_ANGLE_DEG} "
            "deg"
        )

    gradient = np.tile(np.eye(_DIMENSION), (position.shape[0], 1, 1))
    gradient[:, 1, 1] = np.exp(vertical_strain)
    return ParticleState(
        position_m=position,
        velocity_m_s=np.zeros_like(position),
        affine=np.zeros((position.shape[0], _DIMENSION, _DIMENSION)),
        deformation_gradient=gradient,
        mass_kg=material.density_kg_m3 * volume,
        initial_volume_m2=volume,
    )


def _fit_slope_deg(
    particles: ParticleState, settings: SlopeRelaxationSettings
) -> tuple[float, int]:
    """Least-squares slope of the free surface, in degrees.

    Bins whose surface sits on the plateau or down in the toe are
    excluded: including them fits a line through two corners and reports
    the wedge's aspect ratio rather than its slope.

    Args:
        particles: The bed.
        settings: Geometry, for the bin range and the height window.

    Returns:
        ``(slope_deg, n_bins_used)``; the slope is ``nan`` when fewer than
        :data:`_MIN_SLOPE_BINS` bins fall inside the window.
    """
    n_bins = max(int(round(settings.bed_length_m / settings.cell_size_m)), 1)
    centres, heights = surface_profile_m(
        particles, x_bounds_m=(0.0, settings.bed_length_m), n_bins=n_bins
    )
    inside = (
        np.isfinite(heights)
        & (heights < _SURFACE_FIT_UPPER_FRACTION * settings.height_m)
        & (heights > _SURFACE_FIT_LOWER_FRACTION * settings.height_m)
    )
    used = int(inside.sum())
    if used < _MIN_SLOPE_BINS:
        return float("nan"), used
    gradient, _ = np.polyfit(centres[inside], heights[inside], 1)
    return math.degrees(math.atan(-float(gradient))), used


def relax_slope(
    material: SandContinuum,
    settings: SlopeRelaxationSettings | None = None,
) -> SlopeRelaxation:
    """Release a wedge and watch its surface slope until told to stop.

    The bed is marched with **no intruder**, so the only thing acting on
    it is gravity and its own stress divergence -- which is what makes an
    arrested slope a statement about the material rather than about a
    boundary.

    Args:
        material: The continuum under test.
        settings: Geometry, resolution and settle time.

    Returns:
        The relaxation, arrested or not. Call
        :meth:`SlopeRelaxation.require_arrested` to get a number out of
        it.

    Raises:
        CalibrationError: If the wedge cannot be built, or if the march
            produces a slope the fit cannot read.
    """
    if not isinstance(material, SandContinuum):
        raise CalibrationError(
            f"expected a SandContinuum, got {type(material).__name__}"
        )
    config = SlopeRelaxationSettings() if settings is None else settings
    particles = wedge_bed(material, config)

    walls = DomainWalls(
        lower_x=WallCondition.SLIP,
        upper_x=WallCondition.SEPARATE,
        lower_z=WallCondition.STICKY,
        upper_z=WallCondition.FREE,
    )
    solver = PlaneStrainMPMSolver(
        material=material,
        cell_size_m=config.cell_size_m,
        effective_width_m=config.effective_width_m,
        bed_depth_m=config.height_m,
        walls=walls,
        max_steps=2_000_000,
    )
    cell = config.cell_size_m
    grid = PlaneStrainGrid(
        (-cell, -cell),
        cell,
        (
            int(round((config.bed_length_m + 2.0 * cell) / cell)) + 4,
            int(round(config.height_m / cell)) + 10,
        ),
    )
    time_step = cfl_time_step_s(
        cell_size_m=cell,
        elastic_wave_speed_m_s=material.elastic_wave_speed_m_s,
        max_material_speed_m_s=math.sqrt(2.0 * GRAVITY_M_S2 * config.height_m),
    )
    stride = max(int(config.settle_time_s / (time_step * config.n_strides)), 1)

    samples: list[SlopeSample] = []
    started = time.perf_counter()
    for index in range(int(config.n_strides)):
        solver.march(
            particles,
            None,
            grid,
            n_steps=stride,
            time_step_s=time_step,
            free_surface_height_m=config.height_m,
            bed_x_bounds_m=(0.0, config.bed_length_m),
        )
        slope, used = _fit_slope_deg(particles, config)
        if not math.isfinite(slope):
            raise CalibrationError(
                f"the free surface gave only {used} usable bins after "
                f"{(index + 1) * stride} steps, so no slope can be fitted; the "
                "wedge has spread past the window the fit reads"
            )
        samples.append(
            SlopeSample(
                time_s=float((index + 1) * stride * time_step),
                slope_deg=slope,
                n_bins=used,
                kinetic_energy_j=float(particles.kinetic_energy_j()),
                toe_x_m=float(particles.position_m[:, 0].max()),
            )
        )
    elapsed = time.perf_counter() - started

    return SlopeRelaxation(
        samples=tuple(samples),
        friction_angle_deg=material.friction_angle_deg,
        plane_strain_limit_deg=math.degrees(math.asin(math.sqrt(2.0) * material.alpha)),
        n_particles=particles.n_particles,
        n_steps=int(config.n_strides) * stride,
        time_step_s=time_step,
        wall_clock_s=elapsed,
        settings=config,
    )


class F1AngleOfReposeExperiment:
    """The harness's angle-of-repose target, pointed at the F1 solver.

    Shaped for
    :class:`~bunkershot3d.calibration.optimizer.CalibrationOptimizer` --
    ``target_angle``, ``run_simulation``, ``calibrated_parameters``,
    ``parameter_bounds`` -- so that when the solver arrests, the loop
    closes with no further plumbing.

    It does **not** arrest today.  :meth:`run_simulation` therefore
    raises; see :data:`F1_REPOSE_ARREST_NOTE` and the module docstring for
    the measurement behind that.  It is also far too expensive to search:
    one evaluation is minutes of wall clock against the 0.6 s the
    constitutive shear cell costs.

    Attributes:
        sand: The bed being calibrated.
        settings: Geometry and resolution of the relaxation.
        target_angle: The declared repose angle, in degrees. **Not a
            measurement of bunker sand.**
        calibrated_parameters: What the optimiser may search.
        parameter_bounds: Search bounds. The lower end of the friction
            angle is set by the geostatic seed, not by taste; see
            :data:`F1_REPOSE_MIN_FRICTION_ANGLE_DEG`.
    """

    calibrated_parameters: tuple[str, ...] = ("friction_angle_deg",)

    parameter_bounds: dict[str, tuple[float, float]] = {
        "friction_angle_deg": (F1_REPOSE_MIN_FRICTION_ANGLE_DEG, 55.0),
    }

    #: This experiment simulates a *declared* target. It is not data.
    is_measured_on_bunker_sand: bool = False

    def __init__(
        self,
        sand: SandState | None = None,
        *,
        settings: SlopeRelaxationSettings | None = None,
        target_angle: float = 32.0,
    ) -> None:
        """Initialise the experiment.

        Args:
            sand: Bed to calibrate. Defaults to the fluffy USGA preset,
                the only cohesionless one.
            settings: Geometry and resolution.
            target_angle: Declared repose angle in degrees.

        Raises:
            CalibrationError: If the sand or the target is unusable.
        """
        self.sand = playing_condition(PlayingCondition.FLUFFY) if sand is None else sand
        if not isinstance(self.sand, SandState):
            raise CalibrationError(
                f"sand must be a SandState, got {type(self.sand).__name__}"
            )
        if not math.isfinite(target_angle) or not 0.0 < target_angle < 90.0:
            raise CalibrationError(
                f"target_angle must be an angle in (0, 90) deg, got {target_angle!r}"
            )
        self.settings = SlopeRelaxationSettings() if settings is None else settings
        self.target_angle = float(target_angle)

    def relax(self, params: dict) -> SlopeRelaxation:
        """Run the relaxation without asking it for an angle.

        This is the honest entry point while the solver does not arrest:
        it returns the whole history, drift included, so a caller can see
        what the solver did instead of being handed a number.

        Args:
            params: May carry ``friction_angle_deg``.

        Returns:
            The relaxation.

        Raises:
            CalibrationError: If the friction angle is unusable or the
                wedge cannot be seeded.
        """
        angle = float(params.get("friction_angle_deg", self.sand.friction_angle_deg))
        if not math.isfinite(angle) or not 0.0 < angle < 90.0:
            raise CalibrationError(
                f"friction_angle_deg must lie in (0, 90), got {angle!r}"
            )
        material = SandContinuum.from_sand_state(
            dataclasses.replace(self.sand, friction_angle_deg=angle)
        )
        return relax_slope(material, self.settings)

    def run_simulation(self, params: dict) -> float:
        """Return the measured repose angle, or refuse.

        Args:
            params: May carry ``friction_angle_deg``.

        Returns:
            The arrested slope in degrees.

        Raises:
            CalibrationError: While the relaxation does not arrest, which
                is the state measured in issue #8733 section 6.
        """
        return self.relax(params).require_arrested()
