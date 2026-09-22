"""Finalize DatasetGenerator channel buffers into evidence-labeled outputs.

NM-02 (#10617): drop failed optional buffers and attach ChannelEvidence so
unavailable dynamics never appear as zero measurements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import GeneratorConfig
from .labels import ChannelAvailability, ChannelEvidence, ModelDoFLayout


def finalize_channels(
    config: GeneratorConfig,
    buffers: dict[str, np.ndarray | None],
    trackers: dict[str, bool],
    layout: ModelDoFLayout,
    n_steps: int,
) -> tuple[dict[str, ChannelEvidence], dict[str, Any]]:
    """Drop failed optional buffers and build channel evidence."""
    finalized = assemble_finalized_buffers(config, buffers, trackers)
    evidence = build_channel_evidence(config, trackers, finalized, layout, n_steps)
    return evidence, finalized


def assemble_finalized_buffers(
    config: GeneratorConfig,
    buffers: dict[str, np.ndarray | None],
    trackers: dict[str, bool],
) -> dict[str, Any]:
    """Keep always-on channels and drop failed optional buffers."""
    return {
        "times": buffers["times"],
        "positions": buffers["positions"],
        "velocities": buffers["velocities"],
        "requested_controls": buffers["requested_controls"],
        "applied_controls": buffers["applied_controls"],
        "interval_accelerations": (
            None
            if trackers["interval_accelerations"]
            else buffers["interval_accelerations"]
        ),
        "native_accelerations": (
            None
            if trackers["native_accelerations"]
            else buffers["native_accelerations"]
        ),
        "mass_matrices": drop_if_failed(
            config.record_mass_matrix,
            trackers["mass_matrices"],
            buffers["mass_matrices"],
        ),
        "bias_forces": drop_if_failed(
            config.record_bias_forces,
            trackers["bias_forces"],
            buffers["bias_forces"],
        ),
        "gravity": drop_if_failed(
            config.record_gravity, trackers["gravity"], buffers["gravity"]
        ),
        "contact": drop_if_failed(
            config.record_contact_forces,
            trackers["contact"],
            buffers["contact"],
        ),
        "drift": drop_if_failed(
            config.record_drift_control, trackers["drift"], buffers["drift"]
        ),
        "control_accel": drop_if_failed(
            config.record_drift_control,
            trackers["control_accel"],
            buffers["control_accel"],
        ),
        "kinetic_energy": (
            None if trackers["kinetic_energy"] else buffers["kinetic_energy"]
        ),
        "potential_energy": (
            None if trackers["potential_energy"] else buffers["potential_energy"]
        ),
    }


def build_channel_evidence(
    config: GeneratorConfig,
    trackers: dict[str, bool],
    finalized: dict[str, Any],
    layout: ModelDoFLayout,
    n_steps: int,
) -> dict[str, ChannelEvidence]:
    """Build per-channel availability evidence for the finalized buffers."""
    return {
        "positions": ChannelEvidence(
            name="positions",
            availability=ChannelAvailability.AVAILABLE,
            semantic="configuration coordinates q",
            units="rad_or_m",
            shape=(n_steps, layout.n_q),
        ),
        "velocities": ChannelEvidence(
            name="velocities",
            availability=ChannelAvailability.AVAILABLE,
            semantic="tangent velocities v",
            units="rad_s_or_m_s",
            shape=(n_steps, layout.n_v),
        ),
        "requested_controls": ChannelEvidence(
            name="requested_controls",
            availability=ChannelAvailability.AVAILABLE,
            semantic="commanded actuator inputs before saturation",
            units="N_m_or_N",
            shape=(n_steps, layout.n_u),
        ),
        "applied_controls": ChannelEvidence(
            name="applied_controls",
            availability=ChannelAvailability.AVAILABLE,
            semantic="applied actuator inputs after saturation",
            units="N_m_or_N",
            shape=(n_steps, layout.n_u),
        ),
        "interval_accelerations": evidence_for(
            name="interval_accelerations",
            requested=True,
            failed=trackers["interval_accelerations"],
            values=finalized["interval_accelerations"],
            semantic="interval finite-difference acceleration from post-step v",
            units="rad_s2_or_m_s2",
            shape=(n_steps, layout.n_v),
        ),
        "native_accelerations": evidence_for(
            name="native_accelerations",
            requested=True,
            failed=trackers["native_accelerations"],
            values=finalized["native_accelerations"],
            semantic="instantaneous native acceleration from engine dynamics",
            units="rad_s2_or_m_s2",
            shape=(n_steps, layout.n_v),
        ),
        "mass_matrices": evidence_for(
            name="mass_matrices",
            requested=config.record_mass_matrix,
            failed=trackers["mass_matrices"],
            values=finalized["mass_matrices"],
            semantic="mass matrix M(q)",
            units="kg_m2",
            shape=(n_steps, layout.n_v, layout.n_v),
            fail_note="engine raised while computing mass matrix",
        ),
        "bias_forces": evidence_for(
            name="bias_forces",
            requested=config.record_bias_forces,
            failed=trackers["bias_forces"],
            values=finalized["bias_forces"],
            semantic="bias forces h(q,v)",
            units="N_m_or_N",
            shape=(n_steps, layout.n_v),
            fail_note="engine raised while computing bias forces",
        ),
        "gravity_forces": evidence_for(
            name="gravity_forces",
            requested=config.record_gravity,
            failed=trackers["gravity"],
            values=finalized["gravity"],
            semantic="gravity generalized forces",
            units="N_m_or_N",
            shape=(n_steps, layout.n_v),
            fail_note="engine raised while computing gravity forces",
        ),
        "contact_forces": evidence_for(
            name="contact_forces",
            requested=config.record_contact_forces,
            failed=trackers["contact"],
            values=finalized["contact"],
            semantic="contact force components",
            units="N",
            shape=(n_steps, 3),
            fail_note="engine raised or lacks contact channels",
        ),
    }


def drop_if_failed(
    requested: bool, failed: bool, buffer: np.ndarray | None
) -> np.ndarray | None:
    if not requested:
        return None
    if failed:
        return None
    return buffer


def evidence_for(
    *,
    name: str,
    requested: bool,
    failed: bool,
    values: np.ndarray | None,
    semantic: str,
    units: str,
    shape: tuple[int, ...],
    fail_note: str = "channel unavailable",
) -> ChannelEvidence:
    if not requested:
        return ChannelEvidence(
            name=name,
            availability=ChannelAvailability.NOT_REQUESTED,
            semantic=semantic,
            units=units,
            shape=None,
            notes="not requested by GeneratorConfig",
        )
    if failed or values is None:
        return ChannelEvidence(
            name=name,
            availability=ChannelAvailability.UNAVAILABLE,
            semantic=semantic,
            units=units,
            shape=None,
            notes=fail_note,
        )
    return ChannelEvidence(
        name=name,
        availability=ChannelAvailability.AVAILABLE,
        semantic=semantic,
        units=units,
        shape=shape,
    )
