"""Unit tests for SimulationDataStore and SwingDataSequence.

Covers construction (with/without ``storage_path``), validation,
add/get/list semantics, and HDF5 round-trip persistence.

Closes #5477 (test side).
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.data_store import SimulationDataStore, SwingDataSequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sequence(
    sequence_id: str = "seq-001",
    n_steps: int = 16,
    num_joints: int = 7,
    seed: int = 0,
) -> SwingDataSequence:
    """Build a small but realistically-shaped SwingDataSequence for tests."""
    rng = np.random.default_rng(seed)
    return SwingDataSequence(
        sequence_id=sequence_id,
        timestamps=np.linspace(0.0, 1.0, n_steps),
        joint_angles=rng.standard_normal((n_steps, num_joints)),
        joint_velocities=rng.standard_normal((n_steps, num_joints)),
        club_pose=rng.standard_normal((n_steps, 7)),
        applied_torques=rng.standard_normal((n_steps, num_joints)),
        metadata={"engine": "pinocchio", "subject": "test"},
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_construct_without_storage_path(self):
        store = SimulationDataStore()
        assert store.storage_path is None
        assert store.list_sequences() == []

    def test_construct_with_storage_path(self, tmp_path):
        path = tmp_path / "store.h5"
        store = SimulationDataStore(storage_path=str(path))
        assert store.storage_path == str(path)
        assert store.list_sequences() == []

    def test_construct_rejects_nonexistent_parent_dir(self, tmp_path):
        bad = tmp_path / "does" / "not" / "exist" / "store.h5"
        with pytest.raises(ValueError, match="parent"):
            SimulationDataStore(storage_path=str(bad))


# ---------------------------------------------------------------------------
# Sequence validation
# ---------------------------------------------------------------------------


class TestSequenceValidation:
    def test_mismatched_joint_angles_rejected(self):
        with pytest.raises(ValueError, match="joint_angles"):
            SwingDataSequence(
                sequence_id="bad",
                timestamps=np.zeros(10),
                joint_angles=np.zeros((9, 7)),  # mismatched
                joint_velocities=np.zeros((10, 7)),
                club_pose=np.zeros((10, 7)),
                applied_torques=np.zeros((10, 7)),
            )

    def test_mismatched_club_pose_rejected(self):
        with pytest.raises(ValueError, match="club_pose"):
            SwingDataSequence(
                sequence_id="bad",
                timestamps=np.zeros(10),
                joint_angles=np.zeros((10, 7)),
                joint_velocities=np.zeros((10, 7)),
                club_pose=np.zeros((8, 7)),  # mismatched
                applied_torques=np.zeros((10, 7)),
            )


# ---------------------------------------------------------------------------
# Add / get / list
# ---------------------------------------------------------------------------


class TestAddGetList:
    def test_add_and_get_round_trip(self):
        store = SimulationDataStore()
        seq = _make_sequence()
        store.add_sequence(seq)
        retrieved = store.get_sequence("seq-001")
        np.testing.assert_array_equal(retrieved.joint_angles, seq.joint_angles)

    def test_duplicate_sequence_id_rejected(self):
        store = SimulationDataStore()
        store.add_sequence(_make_sequence("dup"))
        with pytest.raises(KeyError, match="dup"):
            store.add_sequence(_make_sequence("dup"))

    def test_missing_sequence_raises_key_error_not_notimplemented(self):
        store = SimulationDataStore()
        with pytest.raises(KeyError):
            store.get_sequence("missing")

    def test_list_sequences_returns_all_ids(self):
        store = SimulationDataStore()
        store.add_sequence(_make_sequence("a", seed=1))
        store.add_sequence(_make_sequence("b", seed=2))
        assert set(store.list_sequences()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Disk persistence (HDF5)
# ---------------------------------------------------------------------------


class TestDiskPersistence:
    def test_flush_without_storage_path_is_noop(self):
        store = SimulationDataStore()
        store.add_sequence(_make_sequence())
        # Should not raise even though no path is set.
        store.flush_to_disk()

    def test_flush_then_load_round_trip(self, tmp_path):
        path = tmp_path / "store.h5"
        store = SimulationDataStore(storage_path=str(path))
        seq_a = _make_sequence("a", n_steps=12, seed=11)
        seq_b = _make_sequence("b", n_steps=20, seed=22)
        store.add_sequence(seq_a)
        store.add_sequence(seq_b)
        store.flush_to_disk()

        assert path.exists()

        reloaded = SimulationDataStore.from_disk(str(path))
        assert set(reloaded.list_sequences()) == {"a", "b"}

        for original in (seq_a, seq_b):
            got = reloaded.get_sequence(original.sequence_id)
            np.testing.assert_array_equal(got.timestamps, original.timestamps)
            np.testing.assert_array_equal(got.joint_angles, original.joint_angles)
            np.testing.assert_array_equal(
                got.joint_velocities, original.joint_velocities
            )
            np.testing.assert_array_equal(got.club_pose, original.club_pose)
            np.testing.assert_array_equal(got.applied_torques, original.applied_torques)
            assert got.metadata == original.metadata

    def test_from_disk_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SimulationDataStore.from_disk(str(tmp_path / "nope.h5"))

    def test_flush_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "store.h5"
        first = SimulationDataStore(storage_path=str(path))
        first.add_sequence(_make_sequence("only", seed=7))
        first.flush_to_disk()

        second = SimulationDataStore(storage_path=str(path))
        second.add_sequence(_make_sequence("other", seed=8))
        second.flush_to_disk()

        reloaded = SimulationDataStore.from_disk(str(path))
        assert reloaded.list_sequences() == ["other"]
