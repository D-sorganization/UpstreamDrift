"""Pre-allocate DatasetGenerator simulation recording buffers.

NM-02 (#10617): always-on kinematics/control buffers plus optional
dynamics channels gated by GeneratorConfig flags.
"""

from __future__ import annotations

import numpy as np

from .config import GeneratorConfig
from .labels import ModelDoFLayout


def allocate_sim_buffers(
    config: GeneratorConfig,
    n_steps: int,
    layout: ModelDoFLayout,
) -> tuple[dict[str, np.ndarray | None], dict[str, bool]]:
    """Pre-allocate recording arrays and per-channel failure trackers."""
    n_q, n_v, n_u = layout.n_q, layout.n_v, layout.n_u
    buffers: dict[str, np.ndarray | None] = {
        "times": np.zeros(n_steps),
        "positions": np.zeros((n_steps, n_q)),
        "velocities": np.zeros((n_steps, n_v)),
        "interval_accelerations": np.zeros((n_steps, n_v)),
        "native_accelerations": np.zeros((n_steps, n_v)),
        "requested_controls": np.zeros((n_steps, n_u)),
        "applied_controls": np.zeros((n_steps, n_u)),
        "mass_matrices": (
            np.zeros((n_steps, n_v, n_v)) if config.record_mass_matrix else None
        ),
        "bias_forces": (
            np.zeros((n_steps, n_v)) if config.record_bias_forces else None
        ),
        "gravity": np.zeros((n_steps, n_v)) if config.record_gravity else None,
        "contact": (np.zeros((n_steps, 3)) if config.record_contact_forces else None),
        "drift": (np.zeros((n_steps, n_v)) if config.record_drift_control else None),
        "control_accel": (
            np.zeros((n_steps, n_v)) if config.record_drift_control else None
        ),
        "kinetic_energy": np.zeros(n_steps),
        "potential_energy": np.zeros(n_steps),
    }
    trackers = {
        "mass_matrices": False,
        "bias_forces": False,
        "gravity": False,
        "contact": False,
        "drift": False,
        "control_accel": False,
        "native_accelerations": False,
        "kinetic_energy": False,
        "potential_energy": False,
        "interval_accelerations": False,
        "applied_controls": False,
    }
    return buffers, trackers


__all__ = ["allocate_sim_buffers"]
