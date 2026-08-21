"""Generate exact branch payload contracts for structural atlas checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_structural_checkpoint import (
    ArrayContract,
    structural_checkpoint_array_contract,
)
from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    StructuralExecutionIdentity,
    scientific_configuration_sha256,
)


def _full_shape(configuration: Any) -> tuple[int, ...]:
    return (
        1,
        1,
        2,
        len(configuration.forward.time_steps_s),
        2,
        len(configuration.horizons_s),
    )


def _shaft_contract(
    configuration: ArticulatedShaftAtlasConfig, nq: int
) -> ArrayContract:
    shape = _full_shape(configuration)
    trace_shape = shape[:-1]
    parity_shape = shape[:4] + shape[5:]
    arrays = {
        name: np.empty(shape)
        for name in (
            "peak_force",
            "peak_couple",
            "open_fraction",
            "maximum_virtual_power",
            "maximum_positive_dissipation",
            "maximum_shaft_power_residual",
            "normalized_energy_residual",
            "maximum_bending",
            "maximum_twist",
            "peak_shaft_energy",
            "terminal_dissipated_work",
            "final_speed",
        )
    }
    arrays.update(
        {
            "transition_count": np.empty(shape, dtype=int),
            "initial_energy": np.empty(trace_shape),
            "final_state": np.empty((*shape, nq)),
            "final_elastic": np.empty((*shape, 3)),
            "trajectory_parity": np.empty(parity_shape),
            "force_parity": np.empty(parity_shape),
            "active_set_parity": np.empty(parity_shape, dtype=bool),
        }
    )
    return structural_checkpoint_array_contract(arrays)


def _ground_contract(
    configuration: ArticulatedGroundAtlasConfig, nq: int
) -> ArrayContract:
    shape = _full_shape(configuration)
    trace_shape = shape[:-1]
    parity_shape = shape[:4] + shape[5:]
    arrays = {
        name: np.empty(shape)
        for name in (
            "peak_grip_force",
            "peak_grip_couple",
            "open_fraction",
            "peak_ground_force",
            "peak_intrinsic_moment",
            "peak_transported_moment",
            "peak_ground_energy",
            "terminal_ground_work",
            "terminal_total_work",
            "normalized_energy_residual",
            "maximum_virtual_power",
            "maximum_shaft_power_residual",
            "maximum_ground_power_residual",
            "maximum_positive_dissipation",
            "maximum_bending",
            "maximum_twist",
            "maximum_base_translation",
            "maximum_base_pitch",
            "final_speed",
        )
    }
    arrays.update(
        {
            "transition_count": np.empty(shape, dtype=int),
            "initial_energy": np.empty(trace_shape),
            "final_q": np.empty((*shape, nq)),
            "final_base_full": np.empty((*shape, 3)),
            "trajectory_parity": np.empty(parity_shape),
            "force_parity": np.empty(parity_shape),
            "ground_force_parity": np.empty(parity_shape),
            "active_set_parity": np.empty(parity_shape, dtype=bool),
        }
    )
    return structural_checkpoint_array_contract(arrays)


def structural_branch_contracts(
    identity: StructuralExecutionIdentity,
    configuration: ArticulatedShaftAtlasConfig | ArticulatedGroundAtlasConfig,
    *,
    nq: int,
) -> dict[tuple[str, int], ArrayContract]:
    """Bind every branch to an independently generated exact array contract."""

    if not isinstance(identity, StructuralExecutionIdentity):
        raise TypeError("identity must be a StructuralExecutionIdentity")
    if identity.pathway not in {"shaft", "ground"}:
        raise ValueError("identity pathway must be shaft or ground")
    if type(nq) is not int or nq <= 0:
        raise ValueError("nq must be a positive integer")
    if identity.pathway == "shaft":
        if not isinstance(configuration, ArticulatedShaftAtlasConfig):
            raise TypeError("shaft identity requires ArticulatedShaftAtlasConfig")
        expected_branches = tuple(
            ("activation", slot) for slot in range(len(configuration.activations))
        )
        contract = _shaft_contract(configuration, nq)
    else:
        if not isinstance(configuration, ArticulatedGroundAtlasConfig):
            raise TypeError("ground identity requires ArticulatedGroundAtlasConfig")
        expected_branches = tuple(
            [("primary", slot) for slot in range(len(configuration.ground_activations))]
            + [("control", slot) for slot in range(len(configuration.control_names))]
        )
        contract = _ground_contract(configuration, nq)
    if (
        scientific_configuration_sha256(configuration)
        != identity.scientific_configuration_sha256
    ):
        raise RuntimeError("configuration digest does not reproduce the identity")
    expected_states = tuple(
        (case, sample)
        for case in configuration.case_indices
        for sample in configuration.sample_indices
    )
    if identity.registered_states != expected_states:
        raise RuntimeError("identity states do not reproduce the configuration")
    if identity.registered_branches != expected_branches:
        raise RuntimeError("identity branches do not reproduce the configuration")
    return {branch: dict(contract) for branch in expected_branches}


__all__ = ["structural_branch_contracts"]
