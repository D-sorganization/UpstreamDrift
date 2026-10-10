"""Computed-torque reference tracking controller (GCV-20 ball-impact split).

Split out of ``full_body_forward_dynamics`` (Issue #10330 module-size
exception) when GCV-20 (#11767) added the impact-split reference-rate path.
``full_body_forward_dynamics`` imports ``tracking_controller`` back at the
bottom of the module so ``full_body_forward_dynamics.tracking_controller``
keeps working unchanged for existing callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.motion_matching.impact_force import (
    reference_rates,
    sample_rate_table,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        Array,
        ComputedTorqueGains,
        Controller,
        FullBodySimulator,
    )


def _tracking_controller_from_gains(
    simulator: FullBodySimulator,
    time_ref: Sequence[float] | Array,
    q_ref: Array,
    gains: ComputedTorqueGains,
    *,
    acceleration_feedforward: float = 1.0,
    split_time_s: float | None = None,
) -> Controller:
    """Computed-torque tracking with pre-built gains.

    ``split_time_s`` (ball impact, GCV-20) differentiates the reference on
    each side of impact separately so the feedforward has no spike there.
    """
    # Deferred import: full_body_forward_dynamics imports this module back
    # at its own bottom, so importing it at module level here would cycle.
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        _computed_torque,
    )

    if not 0.0 <= acceleration_feedforward <= 1.0:
        raise ValueError("acceleration_feedforward must lie in [0, 1]")
    times = np.asarray(time_ref, dtype=float)
    reference = np.asarray(q_ref, dtype=float)
    if (
        times.ndim != 1
        or np.any(np.diff(times) <= 0)
        or reference.shape != (times.size, simulator.nv)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Reference times must increase with one finite q row each")
    velocity, acceleration, last = reference_rates(times, reference, split_time_s)
    acceleration = acceleration_feedforward * acceleration
    split = None if last is None else (float(split_time_s or 0.0), last)

    def sample(table: Array, t: float) -> Array:
        return sample_rate_table(times, table, t, split)

    def controller(t: float, q: Array, v: Array) -> Array:
        q_t, v_t, a_t = (
            sample_rate_table(times, reference, t, None),
            sample(velocity, t),
            sample(acceleration, t),
        )
        com_ref = (
            simulator.centre_of_mass(q_t)[0] if gains.balance is not None else None
        )
        return _computed_torque(simulator, q, v, q_t, v_t, a_t, gains, com_ref)

    return controller


def tracking_controller(
    simulator: FullBodySimulator,
    time_ref: Sequence[float] | Array,
    q_ref: Array,
    *,
    omega_rad_s: float | Array,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    root_regulation: tuple[float, float] | None = None,
    **impact_split: float | None,
) -> Controller:
    """Computed-torque tracking of a reference trajectory (linear interpolation).

    ``split_time_s`` splits the reference rates at the ball impact (GCV-20).

    ``acceleration_feedforward`` and ``split_time_s`` are accepted through
    ``**impact_split`` rather than as named parameters so this function stays
    within the repository's parameter-count budget
    (``scripts/ci/check_architecture_budget.py``); any other keyword raises
    ``TypeError``.
    """
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        _tracking_gains,
    )

    raw_feedforward = impact_split.pop("acceleration_feedforward", 1.0)
    acceleration_feedforward = (
        1.0 if raw_feedforward is None else float(raw_feedforward)
    )
    split_time_s = impact_split.pop("split_time_s", None)
    if impact_split:
        raise TypeError(
            "tracking_controller() got unexpected keyword arguments: "
            f"{sorted(impact_split)}"
        )

    return _tracking_controller_from_gains(
        simulator,
        time_ref,
        q_ref,
        _tracking_gains(omega_rad_s, zeta, balance, root_regulation),
        acceleration_feedforward=acceleration_feedforward,
        split_time_s=split_time_s,
    )
