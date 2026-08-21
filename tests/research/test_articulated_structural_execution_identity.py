"""Contracts for digest-bound structural atlas execution identities."""

from __future__ import annotations

from dataclasses import replace
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
    resolve_structural_execution_identity,
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
