"""Independent common-reference wrench checks for epic #15494."""

import json

import numpy as np
import pytest

from src.shared.python.loop_nullspace.allocation_comparison import (
    compare_hand_wrench_histories,
)
from src.shared.python.loop_nullspace.evidence import GripFrame
from src.shared.python.loop_nullspace.grip_split import grasp_map, hand_wrenches

pytestmark = pytest.mark.unit


def _inputs() -> dict:
    """Two-sample fixture: rotating grip frames carrying equal net wrenches."""
    # Moving and rotating geometry; wrench components remain in grip axes.
    rotations = (np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    frames = tuple(
        GripFrame(rotation, np.array([k, 2 * k, 0.0]))
        for k, rotation in enumerate(rotations)
    )
    lead = np.array([f.origin + f.rotation @ [-0.1, 0, 0] for f in frames])
    trail = np.array([f.origin + f.rotation @ [0.1, 0, 0] for f in frames])
    baseline, candidate = [], []
    for k, frame in enumerate(frames):
        g = grasp_map(frame, lead[k], trail[k])
        net = np.array([2, 4, 6, 0.1, 0.2, 0.3]) * (k + 1)
        baseline.append(np.concatenate(hand_wrenches(net, np.zeros(6), g)))
        candidate.append(np.concatenate(hand_wrenches(net, np.ones(6), g)))
    return {
        "times_s": [0.0, 0.02],
        "grip_frames": frames,
        "lead_points_world_m": lead,
        "trail_points_world_m": trail,
        "baseline": baseline,
        "candidate": candidate,
        "force_atol_n": 1e-9,
        "moment_atol_nm": 1e-9,
    }


def test_changed_hand_split_preserves_net_wrench() -> None:
    """A different hand split under equal net wrenches is equivalent, not replay-proof."""
    inputs = _inputs()
    assert not np.allclose(inputs["baseline"], inputs["candidate"])
    report = compare_hand_wrench_histories(**inputs)
    assert report["equivalent"] is True
    assert report["failed_sample_indices"] == []
    assert report["max_force_error_n"] < 1e-12
    assert report["max_moment_error_nm"] < 1e-12
    assert report["evidence_scope"] == "sampled_net_wrench_equivalence"
    assert report["forward_replay_verified"] is False
    assert report["feasibility_verified"] is False
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_equal_opposite_forces_can_change_net_moment() -> None:
    """Equal opposite force deltas net to zero force yet still shift the moment."""
    inputs = _inputs()
    candidate = np.array(inputs["baseline"])
    candidate[1, 1] += 3.0
    candidate[1, 7] -= 3.0
    inputs["candidate"] = candidate
    report = compare_hand_wrench_histories(**inputs)
    # Independent cross-product oracle: (-.1 ex) x (3 ey)
    # + (.1 ex) x (-3 ey) = -.6 ez N m, despite zero net force.
    assert report["force_error_n"] == pytest.approx([0, 0])
    assert report["moment_error_nm"] == pytest.approx([0, 0.6])
    assert report["max_moment_error_nm"] == pytest.approx(0.6)
    assert report["equivalent"] is False
    assert report["failed_sample_indices"] == [1]


def test_force_and_moment_use_separate_dimensional_tolerances() -> None:
    """Force and moment tolerances gate their own components independently."""
    inputs = _inputs()
    candidate = np.array(inputs["baseline"])
    candidate[0, 0] += 0.5  # Axial force: no extra moment.
    candidate[1, 3] += 0.125  # Exactly representable boundary; pure free couple.
    inputs.update(candidate=candidate, force_atol_n=0.5, moment_atol_nm=0.125)
    assert compare_hand_wrench_histories(**inputs)["equivalent"] is True
    inputs["force_atol_n"] = 0.49
    assert compare_hand_wrench_histories(**inputs)["failed_sample_indices"] == [0]
    inputs.update(force_atol_n=0.5, moment_atol_nm=0.124)
    assert compare_hand_wrench_histories(**inputs)["failed_sample_indices"] == [1]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("times_s", []),
        ("times_s", [0, 0]),
        ("times_s", [0.02, 0]),
        ("times_s", [0, np.nan]),
        ("baseline", np.zeros((1, 12))),
        ("candidate", np.full((2, 12), np.inf)),
        ("candidate", np.zeros((2, 6))),
        ("lead_points_world_m", np.zeros((1, 3))),
        ("grip_frames", ()),
        ("grip_frames", (None, None)),
        ("force_atol_n", -1),
        ("moment_atol_nm", np.nan),
        ("force_atol_n", np.inf),
    ],
)
def test_invalid_or_missing_inputs_are_rejected(key: str, value: object) -> None:
    """Malformed times, histories, geometry or tolerances raise rather than pass."""
    inputs = _inputs()
    inputs[key] = value
    with pytest.raises((ValueError, TypeError)):
        compare_hand_wrench_histories(**inputs)


def test_inconsistent_reference_geometry_is_rejected() -> None:
    """Lead and trail points that no longer bracket a midpoint are rejected."""
    inputs = _inputs()
    inputs["trail_points_world_m"][1] += [0, 0, 0.1]
    with pytest.raises(ValueError, match="midpoint"):
        compare_hand_wrench_histories(**inputs)


def test_identical_histories_accept_zero_tolerances_without_mutating_inputs() -> None:
    """Zero tolerances still accept identical histories and leave inputs unmutated."""
    inputs = _inputs()
    baseline = np.array(inputs["baseline"])
    inputs.update(candidate=baseline.copy(), force_atol_n=0, moment_atol_nm=0)
    before = baseline.copy()
    assert compare_hand_wrench_histories(**inputs)["equivalent"] is True
    np.testing.assert_array_equal(inputs["candidate"], before)


def test_vector_norm_exceedance_is_not_hidden_by_componentwise_comparison() -> None:
    """Within-atol components still fail once their vector norm exceeds atol."""
    inputs = _inputs()
    candidate = np.array(inputs["baseline"])
    candidate[0, :2] += 0.4
    inputs.update(candidate=candidate, force_atol_n=0.5, moment_atol_nm=1)
    report = compare_hand_wrench_histories(**inputs)
    assert report["force_error_n"][0] == pytest.approx(np.sqrt(0.32))
    assert report["failed_sample_indices"] == [0]
