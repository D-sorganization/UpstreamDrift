"""Native state interchange preserves mechanics, inventory, and frame identity."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def adapter() -> NativeJointStateAdapter:
    root = Path(__file__).resolve().parents[3]
    spec = json.loads(
        (
            root
            / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
        ).read_text()
    )
    return NativeJointStateAdapter(spec)


def test_round_trip_real_native_inventory_and_power(
    adapter: NativeJointStateAdapter,
) -> None:
    q = dict.fromkeys(adapter.coordinate_order, 0.2)
    v = dict.fromkeys(q, 0.3)
    a = dict.fromkeys(q, -0.4)
    tau = dict.fromkeys(q, 2.0)
    state = adapter.export(q, v, a, tau)
    restored = adapter.restore(state, q)
    for actual, expected in zip(restored, (q, v, a, tau), strict=True):
        np.testing.assert_allclose(
            list(actual.values()), list(expected.values()), atol=1e-10
        )
    assert len(state.rotations) > 0
    for name, values in state.scalars.items():
        assert values == (q[name], v[name], a[name], tau[name])
    for group in adapter.groups:
        value = state.rotations[group.name]
        assert np.dot(
            value.moment_parent_nm, value.omega_parent_rad_s
        ) == pytest.approx(sum(tau[n] * v[n] for n in group.coordinates))


def test_inventory_mismatch_rejected(adapter: NativeJointStateAdapter) -> None:
    q = dict.fromkeys(adapter.coordinate_order, 0.0)
    with pytest.raises(ValueError, match="inventory"):
        adapter.export({}, q, q, q)


def test_wrong_spec_identity_rejected(adapter: NativeJointStateAdapter) -> None:
    from dataclasses import replace

    q = dict.fromkeys(adapter.coordinate_order, 0.1)
    state = adapter.export(q, q, q, q)
    with pytest.raises(ValueError, match="identity"):
        adapter.restore(replace(state, specification_sha256="other"), q)
    with pytest.raises(ValueError, match="convention"):
        adapter.restore(replace(state, convention_tag="other"), q)


def test_never_groups_rotation_axes_across_body_edges() -> None:
    joints = []
    for index, axis in enumerate("xyz"):
        joints.append(
            {
                "name": axis,
                "parent": str(index),
                "child": str(index + 1),
                "parent_to_base": np.eye(4).tolist(),
                "child_to_follower": np.eye(4).tolist(),
                "primitives": [{"coordinate": axis, "primitive": "R" + axis}],
            }
        )
    adapter = NativeJointStateAdapter(
        {"schema_version": 1, "coordinate_order": list("xyz"), "joints": joints}
    )
    assert not adapter.groups
    assert adapter.scalar_coordinates == tuple("xyz")


def test_rotation_inventory_matches_native_primitives(
    adapter: NativeJointStateAdapter,
) -> None:
    grouped = [name for group in adapter.groups for name in group.coordinates]
    assert len(adapter.groups) == 3
    assert len(grouped) == len(set(grouped)) == 9
    assert set(grouped) | set(adapter.scalar_coordinates) == set(
        adapter.coordinate_order
    )
    for group in adapter.groups:
        assert group.axes == "XYZ"
        assert [adapter.primitive_types[n] for n in group.coordinates] == [
            "Rx",
            "Ry",
            "Rz",
        ]
    root = Path(__file__).resolve().parents[3]
    spec = json.loads(
        (
            root
            / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
        ).read_text()
    )
    joints = {j["name"]: j for j in spec["joints"]}
    for group in adapter.groups:
        assert group.parent_body == joints[group.name]["parent"]
        assert group.child_body == joints[group.name]["child"]
        np.testing.assert_array_equal(
            group.parent_to_base, joints[group.name]["parent_to_base"]
        )
        np.testing.assert_array_equal(
            group.child_to_follower, joints[group.name]["child_to_follower"]
        )


def test_singular_effort_inverse_explicitly_rejected(
    adapter: NativeJointStateAdapter,
) -> None:
    from src.shared.python.pose_interchange.joint_chart import SingularChartError

    q = dict.fromkeys(adapter.coordinate_order, 0.0)
    q[adapter.groups[0].coordinates[1]] = np.pi / 2
    zeros = dict.fromkeys(q, 0.0)
    with pytest.raises(SingularChartError):
        adapter.export(q, zeros, zeros, zeros)


@pytest.mark.live_simulation
@pytest.mark.requires_mujoco
def test_native_mujoco_frames_survive_representation_round_trip(
    adapter: NativeJointStateAdapter,
) -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.native_model import NativeMujocoModel

    root = Path(__file__).resolve().parents[3]
    model = NativeMujocoModel(
        (
            root
            / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
        ).read_bytes()
    )
    q = dict.fromkeys(adapter.coordinate_order, 0.2)
    zeros = dict.fromkeys(q, 0.0)
    expected = model.frame_poses(q)
    restored, _, _, _ = adapter.restore(adapter.export(q, zeros, zeros, zeros), q)
    actual = model.frame_poses(restored)
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-12)
