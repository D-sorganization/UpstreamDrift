from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    CONTROL_NAMES,
    GROUND_ACTIVATIONS,
    ArticulatedGroundAtlasConfig,
)

pytestmark = pytest.mark.scientific


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
