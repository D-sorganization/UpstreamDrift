"""Contracts for digest-bound structural atlas execution identities."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    resolve_structural_execution_identity,
    scientific_configuration_sha256,
    structural_checkpoint_metadata,
    validate_structural_checkpoint_metadata,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


def _load(corner_id: str) -> ArticulatedAtlasAuthority:
    return ArticulatedAtlasAuthority.from_scaled(
        load_scaled_authority(
            DATA / f"articulated_structural_authority_{corner_id}.json",
            DATA / f"articulated_structural_authority_{corner_id}.npz",
        )
    )


@pytest.mark.parametrize(
    ("pathway", "configuration"),
    [
        ("shaft", ArticulatedShaftAtlasConfig()),
        ("ground", ArticulatedGroundAtlasConfig()),
    ],
)
def test_identity_resolves_every_registered_checkpoint_prefix_field(
    pathway: str,
    configuration: ArticulatedShaftAtlasConfig | ArticulatedGroundAtlasConfig,
) -> None:
    authority = _load("nominal")

    identity = resolve_structural_execution_identity(
        authority,
        corner_id="nominal",
        pathway=pathway,
        configuration=configuration,
    )
    prefix = identity.checkpoint_prefix()

    assert prefix["corner_id"] == "nominal"
    assert prefix["pathway"] == pathway
    assert prefix["authority_sha256"] == authority.authority_sha256
    assert prefix["scales"] == {
        "height": 1.0,
        "body_mass": 1.0,
        "joint_limit": 1.0,
    }
    assert set(prefix["model_sha256"]) == {"0", "8", "9", "17"}
    assert len(prefix["planned_states"]) == 12
    assert prefix["retained_failures"] == []
    for key in (
        "atlas_source_sha256",
        "scientific_configuration_sha256",
        "plan_design_sha256",
        "plan_contract_sha256",
    ):
        assert len(prefix[key]) == 64


def test_worker_count_is_operational_not_scientific_identity() -> None:
    authority = _load("nominal")
    first = resolve_structural_execution_identity(
        authority,
        corner_id="nominal",
        pathway="shaft",
        configuration=ArticulatedShaftAtlasConfig(worker_count=1),
    )
    second = resolve_structural_execution_identity(
        authority,
        corner_id="nominal",
        pathway="shaft",
        configuration=ArticulatedShaftAtlasConfig(worker_count=12),
    )

    assert first == second
    assert scientific_configuration_sha256(
        ArticulatedShaftAtlasConfig(worker_count=1)
    ) == scientific_configuration_sha256(ArticulatedShaftAtlasConfig(worker_count=7))


def test_identity_checkpoints_only_feasible_states_and_retains_plan_denominator() -> (
    None
):
    authority = _load("height_scale_low")

    identity = resolve_structural_execution_identity(
        authority,
        corner_id="height_scale-low",
        pathway="shaft",
        configuration=ArticulatedShaftAtlasConfig(),
    )

    assert len(identity.registered_states) == 11
    assert len(identity.planned_states) == 12
    assert (0, 12) not in identity.registered_states
    assert identity.retained_failures == ((0, 12, "ik_nonconvergence"),)
    assert identity.registered_states == authority.feasible_states(
        (0, 8, 9, 17),
        (0, 6, 12),
    )


def test_identity_rejects_configuration_drift() -> None:
    authority = _load("nominal")
    changed = replace(ArticulatedShaftAtlasConfig(), total_stiffness_n_m=1800.1)

    with pytest.raises(RuntimeError, match="scientific configuration"):
        resolve_structural_execution_identity(
            authority,
            corner_id="nominal",
            pathway="shaft",
            configuration=changed,
        )


def test_identity_rejects_wrong_corner_authority() -> None:
    authority = _load("height_scale_high")

    with pytest.raises(RuntimeError, match="corner authority"):
        resolve_structural_execution_identity(
            authority,
            corner_id="nominal",
            pathway="ground",
            configuration=ArticulatedGroundAtlasConfig(),
        )


def test_identity_rejects_pathway_and_configuration_type_mismatch() -> None:
    authority = _load("nominal")

    with pytest.raises(TypeError, match="ArticulatedShaftAtlasConfig"):
        resolve_structural_execution_identity(
            authority,
            corner_id="nominal",
            pathway="shaft",
            configuration=ArticulatedGroundAtlasConfig(),
        )
    with pytest.raises(ValueError, match="pathway"):
        resolve_structural_execution_identity(
            authority,
            corner_id="nominal",
            pathway="unknown",
            configuration=ArticulatedGroundAtlasConfig(),
        )


def test_checkpoint_metadata_is_exact_and_json_round_trip_safe() -> None:
    identity = resolve_structural_execution_identity(
        _load("nominal"),
        corner_id="nominal",
        pathway="ground",
        configuration=ArticulatedGroundAtlasConfig(),
    )

    metadata = structural_checkpoint_metadata(
        identity,
        state_slot=4,
        state=(8, 6),
        branch_kind="primary",
        branch_slot=2,
    )
    round_trip = json.loads(json.dumps(metadata, sort_keys=True))

    assert round_trip == metadata
    assert metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert metadata["state"] == [8, 6]
    assert (
        validate_structural_checkpoint_metadata(
            round_trip,
            identity,
            state_slot=4,
            state=(8, 6),
            branch_kind="primary",
            branch_slot=2,
        )
        == metadata
    )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "corner_id",
        "authority_sha256",
        "scales",
        "model_sha256",
        "atlas_source_sha256",
        "scientific_configuration_sha256",
        "planned_states",
        "retained_failures",
        "plan_design_sha256",
        "plan_contract_sha256",
        "state_slot",
        "state",
        "pathway",
        "branch_kind",
        "branch_slot",
    ],
)
def test_checkpoint_metadata_rejects_every_tampered_identity_field(field: str) -> None:
    identity = resolve_structural_execution_identity(
        _load("nominal"),
        corner_id="nominal",
        pathway="shaft",
        configuration=ArticulatedShaftAtlasConfig(),
    )
    metadata = structural_checkpoint_metadata(
        identity,
        state_slot=0,
        state=(0, 0),
        branch_kind="activation",
        branch_slot=0,
    )
    tampered = deepcopy(metadata)
    tampered[field] = "tampered"

    with pytest.raises(RuntimeError, match=field):
        validate_structural_checkpoint_metadata(
            tampered,
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="activation",
            branch_slot=0,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_checkpoint_metadata_rejects_missing_or_extra_fields(mutation: str) -> None:
    identity = resolve_structural_execution_identity(
        _load("nominal"),
        corner_id="nominal",
        pathway="shaft",
        configuration=ArticulatedShaftAtlasConfig(),
    )
    metadata = structural_checkpoint_metadata(
        identity,
        state_slot=0,
        state=(0, 0),
        branch_kind="activation",
        branch_slot=0,
    )
    if mutation == "missing":
        metadata.pop("authority_sha256")
    else:
        metadata["unregistered"] = True

    with pytest.raises(RuntimeError, match="structural checkpoint identity"):
        validate_structural_checkpoint_metadata(
            metadata,
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="activation",
            branch_slot=0,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"state_slot": -1, "state": (0, 0), "branch_kind": "state", "branch_slot": 0},
        {
            "state_slot": 0,
            "state": (0, 6),
            "branch_kind": "activation",
            "branch_slot": 0,
        },
        {
            "state_slot": True,
            "state": (0, 0),
            "branch_kind": "activation",
            "branch_slot": 0,
        },
        {
            "state_slot": 0,
            "state": (False, 0),
            "branch_kind": "activation",
            "branch_slot": 0,
        },
        {"state_slot": 0, "state": (0, 0), "branch_kind": "primary", "branch_slot": 0},
        {
            "state_slot": 0,
            "state": (0, 0),
            "branch_kind": "activation",
            "branch_slot": 4,
        },
        {
            "state_slot": 0,
            "state": (0, 0),
            "branch_kind": "activation",
            "branch_slot": True,
        },
    ],
)
def test_checkpoint_metadata_rejects_invalid_local_identity(arguments) -> None:
    identity = resolve_structural_execution_identity(
        _load("nominal"),
        corner_id="nominal",
        pathway="shaft",
        configuration=ArticulatedShaftAtlasConfig(),
    )

    with pytest.raises(ValueError):
        structural_checkpoint_metadata(identity, **arguments)
