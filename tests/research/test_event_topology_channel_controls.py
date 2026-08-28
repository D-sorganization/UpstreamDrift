from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from scripts.research.proximal_distal_energy.event_robustness_noise import (
    CommonRandomPerturbations,
)
from scripts.research.proximal_distal_energy.event_topology_channel_controls import (
    ChannelMask,
    apply_channel_mask,
    event_metric_records,
    mask_common_random_perturbations,
    registered_channel_masks,
)
from scripts.research.proximal_distal_energy.event_topology_robustness import (
    CrossingDirection,
    EventTopologyStatus,
    GlobalEventTopology,
    GlobalGuardEvent,
)
from src.shared.python.simulation_backends import GolfModelParams


def test_registered_channel_masks_are_complete_and_coordinate_explicit() -> None:
    masks = registered_channel_masks()

    assert {mask.name: mask.values for mask in masks} == {
        "both": (1.0, 1.0),
        "shoulder_only": (1.0, 0.0),
        "wrist_only": (0.0, 1.0),
        "zero": (0.0, 0.0),
    }


def test_channel_mask_rejects_fractional_or_unidentified_authority() -> None:
    with pytest.raises(ValueError, match="binary"):
        ChannelMask("partial", (1.0, 0.5))
    with pytest.raises(ValueError, match="nonempty"):
        ChannelMask("", (1.0, 1.0))


def test_mask_applies_to_command_and_command_noise_without_mutation() -> None:
    mask = ChannelMask("shoulder_only", (1.0, 0.0))
    controls = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    perturbations = CommonRandomPerturbations(
        initial_state_delta=np.zeros((2, 4)),
        command_delta_nm=np.asarray(
            [[[5.0, 6.0], [7.0, 8.0]], [[-5.0, -6.0], [-7.0, -8.0]]]
        ),
        guard_offset_delta=np.asarray([0.1, -0.1]),
    )

    masked_controls = apply_channel_mask(controls, mask)
    masked_noise = mask_common_random_perturbations(perturbations, mask)

    np.testing.assert_array_equal(masked_controls, [[1.0, 0.0], [3.0, 0.0]])
    np.testing.assert_array_equal(
        masked_noise.command_delta_nm,
        [[[5.0, 0.0], [7.0, 0.0]], [[-5.0, 0.0], [-7.0, 0.0]]],
    )
    np.testing.assert_array_equal(controls, [[1.0, 2.0], [3.0, 4.0]])
    assert not masked_controls.flags.writeable
    assert not masked_noise.command_delta_nm.flags.writeable


def test_zero_mask_cannot_gain_command_authority_from_noise() -> None:
    mask = ChannelMask("zero", (0.0, 0.0))
    perturbations = CommonRandomPerturbations(
        initial_state_delta=np.ones((2, 4)),
        command_delta_nm=np.ones((2, 3, 2)),
        guard_offset_delta=np.asarray([0.1, -0.1]),
    )

    masked = mask_common_random_perturbations(perturbations, mask)

    assert np.count_nonzero(masked.command_delta_nm) == 0
    np.testing.assert_array_equal(
        masked.initial_state_delta, perturbations.initial_state_delta
    )
    np.testing.assert_array_equal(
        masked.guard_offset_delta, perturbations.guard_offset_delta
    )


def test_event_metrics_keep_speed_separate_from_topology_identity() -> None:
    event = GlobalGuardEvent(
        direction=CrossingDirection.POSITIVE,
        sample_index=3,
        time_s=0.3,
        state=np.asarray([0.2, -0.2, 4.0, 8.0]),
        guard_residual=0.0,
        transversality_per_s=12.0,
        near_grazing=False,
    )
    topology = GlobalEventTopology(
        EventTopologyStatus.UNIQUE_TRANSVERSE,
        (event,),
    )

    records = event_metric_records(topology, GolfModelParams.default())

    assert len(records) == 1
    assert records[0]["direction"] == "negative_to_nonnegative"
    assert records[0]["event_time_s"] == pytest.approx(0.3)
    assert records[0]["clubhead_speed_m_s"] > 0.0
    assert records[0]["event_state"] == pytest.approx([0.2, -0.2, 4.0, 8.0])


def test_absent_topology_does_not_fabricate_event_metrics() -> None:
    topology = GlobalEventTopology(EventTopologyStatus.ABSENT, ())

    assert event_metric_records(topology, GolfModelParams.default()) == []
