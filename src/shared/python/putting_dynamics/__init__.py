"""Surface-aware putting dynamics (epic #8345, phases P2 + P3).

Advanced putt physics for the 3-D putt simulation: heterogeneous green
surfaces (heightmap slope + spatial friction + seeded stochastic
bumpiness), advanced friction laws (velocity-dependent rolling
resistance, static rolling hold with steep-slope restart, optional
grain anisotropy), a two-body putter-ball collision with finite putter
effective mass, dynamic-loft sweeps, hosel-position face twist, and a
mode-machine RK4 solver (airborne rise/settle -> slide -> roll ->
rest) with hole capture.

Package layout mirrors ``src.shared.python.physics.impact_model``
(types / surfaces / friction / collision / solver / utils behind this
façade); consume the package through this module — the submodule
internals are not a public surface (Law of Demeter).

Reused shared infrastructure (AGENTS.md section A; also credited in
each submodule):

* ``src.shared.python.contracts`` — DbC (require/ensure).
* ``src.shared.python.core.physics_constants`` — USGA ball constants
  and standard gravity (single source).
* Tools ``swing_sim.putting`` (branch ``feat/putting-vertical``,
  Tools#4125 H3) — skid/roll closed forms, stimp derivation, capture
  bound, lofted-impact decomposition: restated with credit because the
  vendored ``vendor/ud-tools`` snapshot predates that branch (UD
  vendors Tools; never the reverse).
* Tools ``swing_sim.variation.engine`` — keyed per-field RNG stream
  discipline for the bumpy fields.
* ``rust_core/upstream-physics/src/contact.rs`` — green restitution
  (0.78) and sliding friction (0.4) defaults.

Quick start::

    from src.shared.python.putting_dynamics import (
        PutterState, SurfaceSpec, simulate_strike,
    )

    surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
    putter = PutterState(
        head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=3.0, speed_mps=1.8
    )
    result = simulate_strike(putter, surface, hole_x_m=3.0, hole_y_m=0.0)

All quantities are SI; convert for display via :mod:`.utils` only.
"""

from __future__ import annotations

from .collision import (
    GROUND_RESTITUTION,
    LOFT_SWEEP_MAX_DEG,
    LOFT_SWEEP_MIN_DEG,
    effective_head_mass,
    strike,
    sweep_dynamic_loft,
)
from .friction import (
    DEFAULT_SLIDING_MU,
    DEFAULT_STATIC_MU,
    STIMP_RELEASE_SPEED_MPS,
    FrictionParams,
    grain_factor,
    is_static_hold,
    rolling_mu,
    rolling_mu_to_stimp,
    sliding_mu,
    stimp_to_rolling_mu,
)
from .solver import (
    DT_S,
    HOLE_RADIUS_M,
    capture_speed_mps,
    simulate_ball,
    simulate_strike,
)
from .surfaces import (
    FrictionField,
    HeightField,
    SurfaceSpec,
    bumpy_friction_field,
    bumpy_height_field,
)
from .types import (
    BallState,
    CollisionReport,
    Mode,
    PuttResult,
    PutterState,
    TrajectorySample,
)
from .utils import (
    ball_kinetic_energy_j,
    energy_balance_error_j,
    m_to_feet,
    m_to_yards,
    mps_to_mph,
)

__all__ = [
    "DEFAULT_SLIDING_MU",
    "DEFAULT_STATIC_MU",
    "DT_S",
    "GROUND_RESTITUTION",
    "HOLE_RADIUS_M",
    "LOFT_SWEEP_MAX_DEG",
    "LOFT_SWEEP_MIN_DEG",
    "STIMP_RELEASE_SPEED_MPS",
    "BallState",
    "CollisionReport",
    "FrictionField",
    "FrictionParams",
    "HeightField",
    "Mode",
    "PuttResult",
    "PutterState",
    "SurfaceSpec",
    "TrajectorySample",
    "ball_kinetic_energy_j",
    "bumpy_friction_field",
    "bumpy_height_field",
    "capture_speed_mps",
    "effective_head_mass",
    "energy_balance_error_j",
    "grain_factor",
    "is_static_hold",
    "m_to_feet",
    "m_to_yards",
    "mps_to_mph",
    "rolling_mu",
    "rolling_mu_to_stimp",
    "simulate_ball",
    "simulate_strike",
    "sliding_mu",
    "stimp_to_rolling_mu",
    "strike",
    "sweep_dynamic_loft",
]
