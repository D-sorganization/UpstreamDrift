"""Tests for independently generated structural branch payload contracts."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
    _buffers as ground_buffers,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
    _buffers as shaft_buffers,
)
from scripts.research.proximal_distal_energy.articulated_structural_branch_contract import (
    structural_branch_contracts,
)
from scripts.research.proximal_distal_energy.articulated_structural_checkpoint import (
    structural_checkpoint_array_contract,
)
from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    resolve_structural_execution_identity,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _authority() -> ArticulatedAtlasAuthority:
    return ArticulatedAtlasAuthority.from_scaled(
        load_scaled_authority(
            DATA / "articulated_structural_authority_nominal.json",
            DATA / "articulated_structural_authority_nominal.npz",
        )
    )


def _actual_contract(buffer) -> dict:
    return structural_checkpoint_array_contract(
        {field.name: getattr(buffer, field.name) for field in fields(buffer)}
    )


def test_shaft_branch_contract_matches_actual_local_buffer() -> None:
    configuration = ArticulatedShaftAtlasConfig()
    identity = resolve_structural_execution_identity(
        _authority(),
        corner_id="nominal",
        pathway="shaft",
        configuration=configuration,
    )
    contracts = structural_branch_contracts(identity, configuration, nq=20)
    shape = (
        1,
        1,
        2,
        len(configuration.forward.time_steps_s),
        2,
        len(configuration.horizons_s),
    )

    assert tuple(contracts) == identity.registered_branches
    assert len({id(contract) for contract in contracts.values()}) == len(contracts)
    assert all(
        contract == _actual_contract(shaft_buffers(shape, 20))
        for contract in contracts.values()
    )


def test_ground_branch_contract_matches_actual_local_buffer() -> None:
    configuration = ArticulatedGroundAtlasConfig()
    identity = resolve_structural_execution_identity(
        _authority(),
        corner_id="nominal",
        pathway="ground",
        configuration=configuration,
    )
    contracts = structural_branch_contracts(identity, configuration, nq=20)
    shape = (
        1,
        1,
        2,
        len(configuration.forward.time_steps_s),
        2,
        len(configuration.horizons_s),
    )

    assert tuple(contracts) == identity.registered_branches
    assert all(
        contract == _actual_contract(ground_buffers(shape, 20))
        for contract in contracts.values()
    )


def test_branch_contract_rejects_pathway_configuration_and_identity_drift() -> None:
    shaft = ArticulatedShaftAtlasConfig()
    identity = resolve_structural_execution_identity(
        _authority(),
        corner_id="nominal",
        pathway="shaft",
        configuration=shaft,
    )

    with pytest.raises(TypeError, match="shaft identity"):
        structural_branch_contracts(identity, ArticulatedGroundAtlasConfig(), nq=20)
    with pytest.raises(ValueError, match="identity pathway"):
        structural_branch_contracts(
            replace(identity, pathway="unknown"),  # type: ignore[arg-type]
            shaft,
            nq=20,
        )
    with pytest.raises(ValueError, match="positive integer"):
        structural_branch_contracts(identity, shaft, nq=True)
    with pytest.raises(RuntimeError, match="configuration digest"):
        structural_branch_contracts(
            identity,
            replace(shaft, total_stiffness_n_m=shaft.total_stiffness_n_m + 1.0),
            nq=20,
        )
    assert structural_branch_contracts(
        identity,
        replace(shaft, worker_count=1),
        nq=20,
    )
    with pytest.raises(RuntimeError, match="states"):
        structural_branch_contracts(
            replace(identity, registered_states=identity.registered_states[:-1]),
            shaft,
            nq=20,
        )
    with pytest.raises(RuntimeError, match="branches"):
        structural_branch_contracts(
            replace(identity, registered_branches=identity.registered_branches[:-1]),
            shaft,
            nq=20,
        )
