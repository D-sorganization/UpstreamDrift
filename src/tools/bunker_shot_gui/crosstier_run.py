"""Running the cross-tier check from the workbench's own inputs (#8713).

A thin assembly layer, deliberately outside
:class:`~src.tools.bunker_shot_gui.model.WorkbenchModel`: this is not on the
per-shot path and must never be put there. F1 has no shot history yet
(issue #8733), so each probe is a separate march to one recorded pose, and
the whole check costs minutes where a design evaluation costs milliseconds.
Keeping it a separate call is what stops it being run by accident.

The defaults below are the interactive ones, and every one of them is
recorded on the result rather than assumed by the reader: the grid F1 was
solved on is drawn in the figure, and the declared effective width travels
with every magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence

from bunkershot3d.solvers import DRFTSolver, MaterialResponse, RefusalPolicy
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver

from .bridge import compare_tiers, entry_kinematics, validity_band
from .crosstier import CrossTierComparison
from .design import SandCondition, SwingSetup, WedgeDesign, WorkbenchInputError
from .model import WorkbenchModel
from .shot3d import shot_scene

__all__ = [
    "F1_BED_DEPTH_M",
    "F1_CELL_SIZE_M",
    "F1_MAX_STEPS",
    "F1_PROBE_COUNT",
    "F1_SWEEP_SPEEDS_M_S",
    "cross_tier_check",
]

F1_CELL_SIZE_M = 0.003
"""Grid ``dx`` the interactive cross-tier check runs F1 at.

Coarser than the 1-2 mm bulk resolution ADR-0033 specifies, and stated as
such: the cost of a probe goes roughly as ``dx**-3`` -- particle count in
two dimensions and step count in one -- so the specified resolution turns a
minutes-long check into a tens-of-minutes one. Refining does not make the
tier quotable for club force either way; that stays F0's at every
resolution. The value used is recorded on the comparison and drawn in the
figure, so nobody has to ask which one produced a picture."""

F1_BED_DEPTH_M = 0.080
"""Sand depth below the free surface for F1's bed.

Deep enough that a 12 mm divot does not reach the fixed floor, whose
sticky condition would otherwise show up as a stiffer bed."""

F1_PROBE_COUNT = 3
"""Engaged samples of the F0 record handed to F1, by default."""

F1_SWEEP_SPEEDS_M_S: tuple[float, ...] = (5.0, 12.0, 25.0)
"""ADR-0033's own sweep, which is what brackets the inertial-share crossing.

A greenside shot enters at 25 m/s and leaves above 17, so it never
decelerates through the crossover; the sweep is a separate, declared
experiment at one recorded pose and the view labels it as one."""

F1_MAX_STEPS = 200000
"""Cap on one F1 march. Generous, because the approach length depends on
the queried pose; the CFL check inside the solver is the real guard."""


def cross_tier_check(
    model: WorkbenchModel,
    design: WedgeDesign,
    sand: SandCondition,
    swing: SwingSetup,
    *,
    cell_size_m: float = F1_CELL_SIZE_M,
    bed_depth_m: float = F1_BED_DEPTH_M,
    n_probes: int = F1_PROBE_COUNT,
    sweep_speeds_m_s: Sequence[float] = F1_SWEEP_SPEEDS_M_S,
) -> CrossTierComparison:
    """Put the F1 continuum beside F0 on one design (issue #8713).

    **Not** on the per-shot path, and it must not be put there: F1 has
    no shot history yet (issue #8733), so every probe is a separate
    march to one recorded pose and the whole check costs minutes where
    a shot costs milliseconds.

    Both solvers are built with a permissive refusal policy. A greenside
    bunker shot is far outside either tier's stated envelope, and a
    strict policy would turn a refusal into an exception before there
    was anything to compare -- which is a correct refusal and a useless
    comparison. The verdict itself is unchanged: it travels on the band
    and on the licence statement, and nothing here improves it.

    Args:
        model: The workbench model whose settings the F0 record is run
            under. Passed rather than owned so the check uses exactly the
            discretisation the design panel is showing.
        design: The designer's inputs.
        sand: The playing condition.
        swing: The delivery.
        cell_size_m: F1 grid ``dx``. The default is coarser than the
            1-2 mm bulk resolution ADR-0033 specifies, because the
            check is interactive and the cost goes as ``dx**-3``;
            the resolution actually used is recorded on the result and
            drawn in the figure rather than left to whoever ran it.
        bed_depth_m: Sand depth below the free surface for F1's bed.
        n_probes: How many engaged samples of the F0 record to probe.
        sweep_speeds_m_s: Speeds for the declared sweep that brackets
            the inertial-share crossover. A greenside shot does not
            decelerate through the crossing, so without these the view
            can only report which side of it the whole record sits on.

    Returns:
        The comparison.

    Raises:
        WorkbenchInputError: If the design is not a constructible sole.
        OutOfEnvelopeError: If the F0 march itself refuses, in which
            case there is no record to compare against.
    """
    geometry = design.geometry()
    state = sand.sand_state()
    build = model.head_build(geometry)
    kinematics = entry_kinematics(build, swing)
    result = model.shot_result(geometry, state, swing)
    f0 = DRFTSolver(
        material=MaterialResponse.from_sand_state(state),
        dynamic_terms_active=bool(swing.dynamic_terms_active),
        refusal_policy=RefusalPolicy.REPORT,
    )
    band = validity_band(f0, build, result, kinematics.orientation)
    scene = shot_scene(build, result)
    if band is None or scene is None:
        raise WorkbenchInputError(
            "the shot recorded fewer than 2 samples, which is too short to "
            "carry a validity band or a swept divot section, and therefore too "
            "short to cross-check"
        )
    return compare_tiers(
        f0_solver=f0,
        f1_solver=PlaneStrainMPMSolver(
            material=SandContinuum.from_sand_state(state),
            cell_size_m=cell_size_m,
            effective_width_m=geometry.sole_width_m,
            bed_depth_m=bed_depth_m,
            refusal_policy=RefusalPolicy.REPORT,
            max_steps=F1_MAX_STEPS,
        ),
        build=build,
        result=result,
        kinematics=kinematics,
        f0_divot_section_area_m2=scene.divot.section_area_m2,
        band=band,
        head_mass_kg=geometry.head_mass_kg,
        bulk_density_kg_m3=state.bulk_density_kg_m3,
        n_probes=n_probes,
        sweep_speeds_m_s=sweep_speeds_m_s,
    )
