"""Versioned native motion batches preserve actuator branches and power."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
)
from src.shared.python.pose_interchange.native_motion_sequence import (
    export_native_motion,
    restore_native_motion,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def adapter() -> NativeJointStateAdapter:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    return NativeJointStateAdapter(json.loads(path.read_text()))


@pytest.fixture
def samples(adapter: NativeJointStateAdapter) -> tuple[np.ndarray, ...]:
    q = np.full((3, len(adapter.coordinate_order)), 0.2)
    for group in adapter.groups:
        q[:, adapter.coordinate_order.index(group.coordinates[1])] = [-2.0, -2.1, -2.2]
        q[:, adapter.coordinate_order.index(group.coordinates[0])] = [6.4, 6.5, 6.6]
    return q, np.full_like(q, 0.7), np.full_like(q, -0.3), np.full_like(q, 2.0)


def test_real_native_roundtrip_and_power(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    assert motion.coordinate_order == adapter.coordinate_order
    assert motion.rotation_groups == adapter.groups
    assert motion.times_s == (0, 0.1, 0.2)
    assert motion.convention_tag == "native-motion-sequence-v1"
    restored = restore_native_motion(adapter, motion)
    for actual, expected in zip(restored, samples, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    for index, state in enumerate(motion.states):
        power = sum(
            np.dot(g.omega_parent_rad_s, g.moment_parent_nm)
            for g in state.rotations.values()
        )
        power += sum(s[1] * s[3] for s in state.scalars.values())
        assert power == pytest.approx(samples[1][index] @ samples[3][index])


def test_deep_immutable_snapshot(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    samples[0][0, 0] = 99
    assert motion.native_reference[0][0] == 0.2
    with pytest.raises(TypeError):
        motion.states[0].rotations["bad"] = None
    with pytest.raises(TypeError):
        motion.states[0].scalars["bad"] = (1, 2, 3, 4)


@pytest.mark.parametrize("times", [[0, 0, 1], [0, 2, 1], [0, np.nan, 1], [], [0, 1]])
def test_invalid_times(
    adapter: NativeJointStateAdapter,
    samples: tuple[np.ndarray, ...],
    times: list[float],
) -> None:
    with pytest.raises(ValueError):
        export_native_motion(adapter, times, *samples)


@pytest.mark.parametrize("field", range(4))
def test_invalid_sample_shape_and_values(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...], field: int
) -> None:
    values = list(samples)
    values[field] = values[field][:, :-1]
    with pytest.raises(ValueError):
        export_native_motion(adapter, [0, 0.1, 0.2], *values)
    values = list(samples)
    values[field][1, 0] = np.inf
    with pytest.raises(ValueError):
        export_native_motion(adapter, [0, 0.1, 0.2], *values)


def test_identity_version_inventory_guards(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    for kwargs in [
        {"convention_tag": "future"},
        {"coordinate_order": ("duplicate",) * 27},
        {"specification_sha256": "wrong"},
    ]:
        with pytest.raises(ValueError):
            restore_native_motion(adapter, replace(motion, **kwargs))


def test_singular_sample_has_context(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    middle = adapter.coordinate_order.index(adapter.groups[0].coordinates[1])
    samples[0][1, middle] = np.pi / 2
    with pytest.raises(ValueError, match=r"sample 1.*0.1"):
        export_native_motion(adapter, [0, 0.1, 0.2], *samples)


def test_restore_rejects_singular_reference_with_context(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    reference = np.array(motion.native_reference)
    reference[1, adapter.coordinate_order.index(adapter.groups[0].coordinates[1])] = (
        np.pi / 2
    )
    with pytest.raises(ValueError, match=r"sample 1.*0.1"):
        restore_native_motion(adapter, replace(motion, native_reference=reference))


def test_nested_input_state_is_copied(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    original = motion.states[0]
    mutable_scalars = dict(original.scalars)
    replacement = replace(original, scalars=mutable_scalars)
    copied = replace(motion, states=(replacement, *motion.states[1:]))
    mutable_scalars.clear()
    assert len(copied.states[0].scalars) == 18


def test_invalid_quaternion_and_state_clock_count(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    with pytest.raises(ValueError, match="count"):
        replace(motion, states=motion.states[:-1])
    state = motion.states[1]
    rotations = dict(state.rotations)
    name = next(iter(rotations))
    rotations[name] = replace(rotations[name], quaternion_wxyz=(2.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"sample 1.*0.1.*unit quaternion"):
        replace(
            motion,
            states=(
                motion.states[0],
                replace(state, rotations=rotations),
                motion.states[2],
            ),
        )


def test_valid_but_different_spec_identity_rejected(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    other = "0" * 64
    states = tuple(
        replace(state, specification_sha256=other) for state in motion.states
    )
    alternate = replace(motion, specification_sha256=other, states=states)
    with pytest.raises(ValueError, match="identity"):
        restore_native_motion(adapter, alternate)


def test_quaternion_sign_flip_preserves_entire_native_motion(
    adapter: NativeJointStateAdapter, samples: tuple[np.ndarray, ...]
) -> None:
    motion = export_native_motion(adapter, [0, 0.1, 0.2], *samples)
    states = tuple(
        replace(
            state,
            rotations={
                name: replace(
                    rotation,
                    quaternion_wxyz=tuple(-x for x in rotation.quaternion_wxyz),
                )
                for name, rotation in state.rotations.items()
            },
        )
        for state in motion.states
    )
    flipped = replace(motion, states=states)
    for actual, expected in zip(
        restore_native_motion(adapter, flipped), samples, strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
