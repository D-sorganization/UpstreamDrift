"""Wave 6 coverage: src.learning.imitation.dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.learning.imitation.dataset import Demonstration, DemonstrationDataset


def _make_demo(
    n_frames: int = 10,
    n_joints: int = 3,
    *,
    with_actions: bool = True,
    with_ee: bool = False,
    with_contact: bool = False,
    task_id: str | None = "t",
    success: bool = True,
) -> Demonstration:
    return Demonstration(
        timestamps=np.arange(n_frames, dtype=float) * 0.01,
        joint_positions=np.arange(n_frames * n_joints, dtype=float).reshape(
            n_frames, n_joints
        ),
        joint_velocities=np.ones((n_frames, n_joints)),
        actions=np.full((n_frames, n_joints), 0.5) if with_actions else None,
        end_effector_poses=np.zeros((n_frames, 7)) if with_ee else None,
        contact_states=(
            [[{"link": "f"}] for _ in range(n_frames)] if with_contact else None
        ),
        task_id=task_id,
        success=success,
    )


class TestDemonstration:
    def test_props(self) -> None:
        d = _make_demo(20, 4)
        assert d.n_frames == 20
        assert d.n_joints == 4
        assert d.duration == pytest.approx(0.19, rel=1e-6)

    def test_validation_position_mismatch(self) -> None:
        with pytest.raises(ValueError, match="joint_positions"):
            Demonstration(
                timestamps=np.zeros(5),
                joint_positions=np.zeros((4, 3)),
                joint_velocities=np.zeros((5, 3)),
            )

    def test_validation_velocity_mismatch(self) -> None:
        with pytest.raises(ValueError, match="joint_velocities"):
            Demonstration(
                timestamps=np.zeros(5),
                joint_positions=np.zeros((5, 3)),
                joint_velocities=np.zeros((4, 3)),
            )

    def test_get_frame_basic(self) -> None:
        d = _make_demo(5, 2)
        frame = d.get_frame(2)
        assert frame["timestamp"] == pytest.approx(0.02)
        assert "action" in frame
        assert "ee_pose" not in frame

    def test_get_frame_with_ee(self) -> None:
        d = _make_demo(5, 2, with_ee=True)
        frame = d.get_frame(0)
        assert "ee_pose" in frame

    def test_subsample(self) -> None:
        d = _make_demo(10, 2, with_ee=True, with_contact=True)
        sub = d.subsample(2)
        assert sub.n_frames == 5
        assert sub.end_effector_poses is not None
        assert sub.contact_states is not None
        assert len(sub.contact_states) == 5

    def test_subsample_no_optional(self) -> None:
        d = _make_demo(10, 2, with_actions=False)
        sub = d.subsample(3)
        assert sub.actions is None
        assert sub.end_effector_poses is None

    def test_roundtrip_dict(self) -> None:
        d = _make_demo(4, 2, with_ee=True, with_contact=True)
        data = d.to_dict()
        d2 = Demonstration.from_dict(data)
        np.testing.assert_array_equal(d.joint_positions, d2.joint_positions)
        assert d2.actions is not None
        assert d2.end_effector_poses is not None
        assert d2.contact_states == d.contact_states

    def test_from_dict_minimal(self) -> None:
        data = {
            "timestamps": [0.0, 0.1],
            "joint_positions": [[0.0], [1.0]],
            "joint_velocities": [[0.0], [0.0]],
        }
        d = Demonstration.from_dict(data)
        assert d.actions is None
        assert d.success is True
        assert d.source == "unknown"


class TestDataset:
    def test_basics(self) -> None:
        ds = DemonstrationDataset()
        assert len(ds) == 0
        ds.add(_make_demo(5, 2))
        ds.extend([_make_demo(3, 2), _make_demo(4, 2, success=False)])
        assert len(ds) == 3
        assert list(iter(ds))[0].n_frames == 5
        assert ds[0].n_frames == 5

    def test_totals(self) -> None:
        ds = DemonstrationDataset([_make_demo(5, 2), _make_demo(3, 2)])
        assert ds.total_frames == 8
        assert ds.total_transitions == (5 - 1) + (3 - 1)

    def test_filter_successful(self) -> None:
        ds = DemonstrationDataset(
            [_make_demo(3, 2, success=True), _make_demo(3, 2, success=False)]
        )
        ok = ds.filter_successful()
        assert len(ok) == 1

    def test_filter_by_task(self) -> None:
        ds = DemonstrationDataset(
            [_make_demo(3, 2, task_id="a"), _make_demo(3, 2, task_id="b")]
        )
        out = ds.filter_by_task("a")
        assert len(out) == 1

    def test_to_transitions(self) -> None:
        ds = DemonstrationDataset([_make_demo(5, 2)])
        s, a, ns = ds.to_transitions()
        assert s.shape == (4, 4)
        assert a.shape == (4, 2)
        assert ns.shape == (4, 4)

    def test_to_transitions_skips_no_actions(self) -> None:
        ds = DemonstrationDataset([_make_demo(5, 2, with_actions=False)])
        s, a, ns = ds.to_transitions()
        assert s.size == 0

    def test_state_action_pairs(self) -> None:
        ds = DemonstrationDataset([_make_demo(4, 2)])
        s, a = ds.to_state_action_pairs()
        assert s.shape == (3, 4)
        assert a.shape == (3, 2)

    def test_augment(self) -> None:
        ds = DemonstrationDataset([_make_demo(3, 2, with_ee=True)])
        aug = ds.augment(
            noise_std=0.01, num_augmentations=2, rng=np.random.default_rng(0)
        )
        # original + 2 augmented
        assert len(aug) == 3
        assert aug[1].metadata.get("augmented") is True
        assert aug[1].source.endswith("_augmented")

    def test_augment_default_rng(self) -> None:
        ds = DemonstrationDataset([_make_demo(3, 2)])
        aug = ds.augment(noise_std=0.0, num_augmentations=1)
        assert len(aug) == 2

    def test_save_load(self, tmp_path: Path) -> None:
        ds = DemonstrationDataset([_make_demo(4, 2, with_ee=True)])
        out = tmp_path / "ds.json"
        ds.save(out)
        loaded = DemonstrationDataset.load(out)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0].joint_positions, ds[0].joint_positions)

    def test_save_json_structure(self, tmp_path: Path) -> None:
        ds = DemonstrationDataset([_make_demo(3, 1)])
        out = tmp_path / "x.json"
        ds.save(out)
        data = json.loads(out.read_text())
        assert data["version"] == "1.0"
        assert data["n_demonstrations"] == 1

    def test_sample(self) -> None:
        ds = DemonstrationDataset([_make_demo(3, 2) for _ in range(5)])
        sub = ds.sample(3, rng=np.random.default_rng(0))
        assert len(sub) == 3
        # n > len truncated
        sub2 = ds.sample(99)
        assert len(sub2) == 5

    def test_statistics_empty(self) -> None:
        assert DemonstrationDataset().get_statistics() == {"n_demonstrations": 0}

    def test_statistics_full(self) -> None:
        ds = DemonstrationDataset(
            [_make_demo(4, 2, success=True), _make_demo(3, 2, success=False)]
        )
        stats = ds.get_statistics()
        assert stats["n_demonstrations"] == 2
        assert stats["total_frames"] == 7
        assert stats["success_rate"] == pytest.approx(0.5)
        assert len(stats["position_mean"]) == 2
