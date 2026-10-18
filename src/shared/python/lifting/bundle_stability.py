"""RK4 stability of a lift bundle's tracking run (#13041).

:func:`bundle_rk4_stability` records the RK4 real-axis margin of a bundle's
reference run; :func:`stable_tracking_bundle` regenerates a run at the
substep count its own record needs, so no committed bundle is integrated
outside the RK4 real stability region.  Split from :mod:`.bundles` to keep
that module inside its size budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from src.shared.python.motion_matching.same_input import InputBundle, VectorPlant
from src.shared.python.motion_matching.same_input.integrator import StepPolicy
from src.shared.python.motion_matching.same_input.stiffness import (
    rk4_stability_record,
    stable_substeps,
)

#: Reference steps between the states sampled for ``rk4_stability``.
STABILITY_SAMPLE_EVERY = 5
#: Bundle runs tried while raising the substeps to the recorded stiffest rate
#: (:func:`~src.shared.python.motion_matching.same_input.stiffness.\
#: stable_substeps`); each retry at least doubles them.
STABILITY_ATTEMPTS = 4


def bundle_rk4_stability(
    bundle: InputBundle, plant: Any, every: int = STABILITY_SAMPLE_EVERY
) -> dict[str, float | int]:
    """RK4 margin of ``bundle``'s reference run, sampled every ``every`` steps.

    The stiffest open-loop rate of ``plant`` (the bundle's MuJoCo plant) at
    each sampled reference state under its recorded effort, times the
    bundle's RK4 substep (see
    :func:`~src.shared.python.motion_matching.same_input.stiffness.\
rk4_stability_record`).
    """
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError("every must be a positive integer")
    rows = np.arange(0, bundle.steps, every)
    return rk4_stability_record(
        plant,
        bundle.reference_q[rows],
        bundle.reference_v[rows],
        bundle.efforts[rows],
        bundle.dt_s / bundle.substeps,
    )


def stable_tracking_bundle(
    generate: Callable[[StepPolicy], InputBundle],
    policy: StepPolicy,
    *,
    label: str,
) -> tuple[InputBundle, dict[str, float | int], Any]:
    """Run ``generate`` at the fewest substeps that keep RK4 stable.

    Starts at ``policy`` and, while the run's ``rk4_stability`` record needs
    more substeps (:func:`stable_substeps`), regenerates it with them.

    Returns:
        The stable bundle, its stability record and its MuJoCo plant.  The
        record's ``max_abs_z`` is inside the RK4 real stability limit.

    Raises:
        ValueError: If no run is stable within :data:`STABILITY_ATTEMPTS`.
    """
    plant: Any = None
    for _ in range(STABILITY_ATTEMPTS):
        bundle = generate(policy)
        if plant is None:
            plant = VectorPlant("mujoco", bundle.spec_bytes)
        stability = bundle_rk4_stability(bundle, plant)
        substeps = stable_substeps(stability, bundle.dt_s, policy.substeps)
        if substeps == policy.substeps:
            return bundle, stability, plant
        policy = replace(policy, substeps=substeps)
    raise ValueError(f"no stable RK4 substep count for {label}: {stability}")
