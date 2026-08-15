"""Designer metrics for BunkerShot3D (issue #8614, W7 of epic #8607).

Baseline finding B24: nothing in the package computed the quantities a wedge
designer actually needs. This package does, and it computes them from the
**result artifact** rather than from solver internals, so the same metric means
the same thing at every fidelity tier of ADR-0032 -- F0 DRFT, F1 continuum, F2
MPM, F3 DEM. See :mod:`bunkershot3d.metrics.trace` for the input contract.

What is measured, and in what units
-----------------------------------

===================================== ===================== ==============================
Metric                                Unit                  Module
===================================== ===================== ==============================
Sole depth vs along-track travel      m vs m                :mod:`.divot`
Entry point / distance behind ball    m                     :mod:`.divot`
Maximum divot depth                   m                     :mod:`.divot`
Exit point, divot length              m                     :mod:`.divot`
Divot section area / volume / mass    m^2 / m^3 / kg        :mod:`.divot`
Dig-vs-skid slope ratio and verdict   dimensionless         :mod:`.divot`
Vertical impulse balance              N.s                   :mod:`.divot`
Club KE loss, work on sand, ball      J, and fractions      :mod:`.energy`
Peak / mean head deceleration         m/s^2 (also g)        :mod:`.loads`
Peak resultant force and moment       N, N.m                :mod:`.loads`
Head twist about the shaft axis       N.m, N.s.m, rad       :mod:`.loads`
Playability window area               axis unit product     :mod:`.playability`
Bounce utilisation of the sole        m^2 and fraction      :mod:`.bounce_map`
Forgiveness sensitivities             r, m/unit, fraction   :mod:`.forgiveness`
===================================== ===================== ==============================

Two of these are the headline numbers for the epic:

* :func:`~bunkershot3d.metrics.playability.playability_objective` -- the area in
  (entry distance x attack angle) or (entry distance x sand firmness) space over
  which carry lands within +/-10 % of target. **The primary scalar objective.**
* :func:`~bunkershot3d.metrics.bounce_map.bounce_utilisation` -- the fraction of
  sole area that actually carried load, resolved spatially. It says *where to
  grind*, which makes the tool prescriptive rather than merely evaluative.

And one is new work rather than a reproduction:
:func:`~bunkershot3d.metrics.loads.head_twist_metrics` quantifies head rotation
under sand load -- the moment about the shaft axis and about the CG. The
literature search for this epic found it published nowhere.
"""

from __future__ import annotations

from .bounce_map import (
    DEFAULT_LOAD_THRESHOLD_FRACTION,
    BounceUtilisation,
    LoadProfile,
    SoleLoadTrace,
    bounce_utilisation,
)
from .divot import (
    DEFAULT_DIG_SLOPE_RATIO,
    DEFAULT_ENTRY_WINDOW_M,
    DEFAULT_SKID_SLOPE_RATIO,
    DigSkidResult,
    DivotMetrics,
    SoleDepthProfile,
    StrikeInterval,
    dig_vs_skid,
    divot_metrics,
    sole_depth_profile,
    submerged_interval,
)
from .energy import BallLaunch, EnergyPartition, energy_partition, head_kinetic_energy_J
from .enums import DigSkidVerdict, WrenchReference
from .forgiveness import (
    SWEEP_RANGES,
    WIVOU_2016_CARRY_CORRELATION,
    FactorSensitivity,
    ForgivenessReport,
    SweepRange,
    forgiveness_report,
    forgiveness_sensitivity,
)
from .loads import (
    HeadLoadMetrics,
    HeadTwistMetrics,
    head_load_metrics,
    head_twist_metrics,
    shaft_travel_loft_axes,
)
from .playability import (
    DEFAULT_CARRY_TOLERANCE_FRACTION,
    PlayabilityAxis,
    PlayabilityWindow,
    playability_objective,
    playability_window,
)
from .trace import (
    STANDARD_GRAVITY_MPS2,
    WORLD_UP,
    HeadModel,
    StrikeScene,
    StrikeTrace,
    angular_velocity_world_radps,
    centre_of_mass_moment_Nm,
    rotate_body_to_world,
    rotate_world_to_body,
)

__all__ = [
    "DEFAULT_CARRY_TOLERANCE_FRACTION",
    "DEFAULT_DIG_SLOPE_RATIO",
    "DEFAULT_ENTRY_WINDOW_M",
    "DEFAULT_LOAD_THRESHOLD_FRACTION",
    "DEFAULT_SKID_SLOPE_RATIO",
    "STANDARD_GRAVITY_MPS2",
    "SWEEP_RANGES",
    "WIVOU_2016_CARRY_CORRELATION",
    "WORLD_UP",
    "BallLaunch",
    "BounceUtilisation",
    "DigSkidResult",
    "DigSkidVerdict",
    "DivotMetrics",
    "EnergyPartition",
    "FactorSensitivity",
    "ForgivenessReport",
    "HeadLoadMetrics",
    "HeadModel",
    "HeadTwistMetrics",
    "LoadProfile",
    "PlayabilityAxis",
    "PlayabilityWindow",
    "SoleDepthProfile",
    "SoleLoadTrace",
    "StrikeInterval",
    "StrikeScene",
    "StrikeTrace",
    "SweepRange",
    "WrenchReference",
    "angular_velocity_world_radps",
    "bounce_utilisation",
    "centre_of_mass_moment_Nm",
    "dig_vs_skid",
    "divot_metrics",
    "energy_partition",
    "forgiveness_report",
    "forgiveness_sensitivity",
    "head_kinetic_energy_J",
    "head_load_metrics",
    "head_twist_metrics",
    "playability_objective",
    "playability_window",
    "rotate_body_to_world",
    "rotate_world_to_body",
    "shaft_travel_loft_axes",
    "sole_depth_profile",
    "submerged_interval",
]
