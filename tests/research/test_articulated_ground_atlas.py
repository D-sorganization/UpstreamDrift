from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    CONTROL_NAMES,
    GROUND_ACTIVATIONS,
    ArticulatedGroundAtlasConfig,
    _resolve_states,
)
from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def test_ground_atlas_contract_is_complete_and_fails_closed() -> None:
    config = ArticulatedGroundAtlasConfig()
    assert config.ground_activations == GROUND_ACTIVATIONS
    assert config.control_names == CONTROL_NAMES
    assert len(config.case_indices) * len(config.sample_indices) == 12
    assert config.horizons_s == (0.004, 0.01, 0.025, 0.05)
    assert (
        len(config.case_indices)
        * len(config.sample_indices)
        * len(config.ground_activations)
        * 2
        * len(config.forward.time_steps_s)
        * 2
        == 384
    )
    with pytest.raises(ValueError, match="ground_activations"):
        ArticulatedGroundAtlasConfig(ground_activations=("fixed", "coupled"))
    with pytest.raises(ValueError, match="control_names"):
        ArticulatedGroundAtlasConfig(control_names=("rigid_shaft",))
    with pytest.raises(ValueError, match="worker_count"):
        ArticulatedGroundAtlasConfig(worker_count=0)
    with pytest.raises(ValueError, match="horizons_s"):
        ArticulatedGroundAtlasConfig(horizons_s=(0.004, 0.049))
    with pytest.raises(ValueError, match="shaft_damping_ratio"):
        ArticulatedGroundAtlasConfig(shaft_damping_ratio=1.0)
    with pytest.raises(ValueError, match="ground_translation_stiffness_scale"):
        ArticulatedGroundAtlasConfig(ground_translation_stiffness_scale=0.0)
    with pytest.raises(ValueError, match="ground_free_moment_damping_scale"):
        ArticulatedGroundAtlasConfig(ground_free_moment_damping_scale=-1.0)


def test_ground_atlas_retains_infeasible_state_and_executes_feasible_subset() -> None:
    scaled = load_scaled_authority(
        DATA / "articulated_structural_authority_height_scale_low.json",
        DATA / "articulated_structural_authority_height_scale_low.npz",
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)

    selection = _resolve_states(authority, ArticulatedGroundAtlasConfig())

    assert len(selection.planned_states) == 12
    assert len(selection.feasible_states) == 11
    assert selection.retained_failures == (
        {"case_index": 0, "phase_index": 12, "failure_class": "ik_nonconvergence"},
    )
