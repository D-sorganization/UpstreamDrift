"""Tests for src/shared/data_store/store.py (HDF5-backed simulation store)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.data_store.store import (
    SimulationDataStore,
    SwingDataSequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seq(
    seq_id: str = "swing-001",
    n_steps: int = 8,
    n_joints: int = 4,
    metadata: dict[str, str] | None = None,
) -> SwingDataSequence:
    rng = np.random.default_rng(seed=hash(seq_id) & 0xFFFF)
    if metadata is None:
        metadata = {"subject": "S1", "club": "driver"}
    return SwingDataSequence(
        sequence_id=seq_id,
        timestamps=np.linspace(0.0, 1.0, n_steps),
        joint_angles=rng.normal(size=(n_steps, n_joints)),
        joint_velocities=rng.normal(size=(n_steps, n_joints)),
        club_pose=rng.normal(size=(n_steps, 7)),
        applied_torques=rng.normal(size=(n_steps, n_joints)),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SwingDataSequence dimension validation
# ---------------------------------------------------------------------------


class TestSwingDataSequenceValidation:
    def test_valid_sequence_constructs(self):
        seq = _make_seq()
        assert seq.sequence_id == "swing-001"
        assert seq.timestamps.shape == (8,)
        assert seq.metadata["subject"] == "S1"

    def test_default_metadata_is_empty_dict(self):
        seq = SwingDataSequence(
            sequence_id="s",
            timestamps=np.zeros(3),
            joint_angles=np.zeros((3, 2)),
            joint_velocities=np.zeros((3, 2)),
            club_pose=np.zeros((3, 7)),
            applied_torques=np.zeros((3, 2)),
        )
        assert seq.metadata == {}

    @pytest.mark.parametrize(
        "field_name",
        ["joint_angles", "joint_velocities", "club_pose", "applied_torques"],
    )
    def test_mismatched_field_length_raises(self, field_name: str):
        kwargs = {
            "sequence_id": "bad",
            "timestamps": np.zeros(5),
            "joint_angles": np.zeros((5, 2)),
            "joint_velocities": np.zeros((5, 2)),
            "club_pose": np.zeros((5, 7)),
            "applied_torques": np.zeros((5, 2)),
        }
        # Replace one field with a wrong leading dim.
        bad_shape = list(kwargs[field_name].shape)
        bad_shape[0] = 4
        kwargs[field_name] = np.zeros(tuple(bad_shape))
        with pytest.raises(ValueError, match=field_name):
            SwingDataSequence(**kwargs)


# ---------------------------------------------------------------------------
# SimulationDataStore in-memory behavior
# ---------------------------------------------------------------------------


class TestSimulationDataStoreInMemory:
    def test_init_no_storage_path(self):
        store = SimulationDataStore()
        assert store.storage_path is None
        assert store.list_sequences() == []

    def test_init_with_valid_storage_path(self, tmp_path):
        path = tmp_path / "store.h5"
        store = SimulationDataStore(storage_path=str(path))
        assert store.storage_path == str(path)

    def test_init_with_missing_parent_dir_raises(self, tmp_path):
        bad = tmp_path / "no_such_dir" / "store.h5"
        with pytest.raises(ValueError, match="parent directory does not exist"):
            SimulationDataStore(storage_path=str(bad))

    def test_add_and_get_sequence(self):
        store = SimulationDataStore()
        seq = _make_seq("a")
        store.add_sequence(seq)
        assert store.get_sequence("a") is seq

    def test_add_duplicate_raises(self):
        store = SimulationDataStore()
        store.add_sequence(_make_seq("dup"))
        with pytest.raises(KeyError, match="already exists"):
            store.add_sequence(_make_seq("dup"))

    def test_get_unknown_raises(self):
        store = SimulationDataStore()
        with pytest.raises(KeyError):
            store.get_sequence("nope")

    def test_list_sequences_returns_all_ids(self):
        store = SimulationDataStore()
        for sid in ("a", "b", "c"):
            store.add_sequence(_make_seq(sid))
        assert sorted(store.list_sequences()) == ["a", "b", "c"]

    def test_list_sequences_returns_new_list(self):
        store = SimulationDataStore()
        store.add_sequence(_make_seq("a"))
        result = store.list_sequences()
        result.append("mutated")
        assert "mutated" not in store.list_sequences()


# ---------------------------------------------------------------------------
# Disk persistence (HDF5)
# ---------------------------------------------------------------------------


class TestDiskPersistence:
    def test_flush_no_storage_path_is_noop(self):
        store = SimulationDataStore()
        store.add_sequence(_make_seq("x"))
        store.flush_to_disk()  # must not raise

    def test_flush_creates_file(self, tmp_path):
        path = tmp_path / "out.h5"
        store = SimulationDataStore(storage_path=str(path))
        store.add_sequence(_make_seq("only"))
        store.flush_to_disk()
        assert path.is_file()

    def test_flush_empty_store_writes_valid_file(self, tmp_path):
        path = tmp_path / "empty.h5"
        SimulationDataStore(storage_path=str(path)).flush_to_disk()
        restored = SimulationDataStore.from_disk(str(path))
        assert restored.list_sequences() == []

    def test_roundtrip_preserves_data(self, tmp_path):
        path = tmp_path / "rt.h5"
        store = SimulationDataStore(storage_path=str(path))
        seq = _make_seq("rt", n_steps=12, n_joints=5)
        store.add_sequence(seq)
        store.flush_to_disk()

        restored = SimulationDataStore.from_disk(str(path))
        out = restored.get_sequence("rt")
        np.testing.assert_array_equal(out.timestamps, seq.timestamps)
        np.testing.assert_array_equal(out.joint_angles, seq.joint_angles)
        np.testing.assert_array_equal(out.joint_velocities, seq.joint_velocities)
        np.testing.assert_array_equal(out.club_pose, seq.club_pose)
        np.testing.assert_array_equal(out.applied_torques, seq.applied_torques)
        assert out.metadata == seq.metadata
        assert out.sequence_id == "rt"

    def test_roundtrip_multiple_sequences(self, tmp_path):
        path = tmp_path / "multi.h5"
        store = SimulationDataStore(storage_path=str(path))
        store.add_sequence(_make_seq("a", n_steps=4, n_joints=2))
        store.add_sequence(_make_seq("b", n_steps=6, n_joints=3))
        store.add_sequence(_make_seq("c", n_steps=10, n_joints=2))
        store.flush_to_disk()

        restored = SimulationDataStore.from_disk(str(path))
        assert sorted(restored.list_sequences()) == ["a", "b", "c"]
        assert restored.get_sequence("b").joint_angles.shape == (6, 3)

    def test_flush_overwrites_existing(self, tmp_path):
        path = tmp_path / "ow.h5"
        s1 = SimulationDataStore(storage_path=str(path))
        s1.add_sequence(_make_seq("first"))
        s1.flush_to_disk()

        s2 = SimulationDataStore(storage_path=str(path))
        s2.add_sequence(_make_seq("second"))
        s2.flush_to_disk()

        restored = SimulationDataStore.from_disk(str(path))
        assert restored.list_sequences() == ["second"]

    def test_from_disk_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SimulationDataStore.from_disk(str(tmp_path / "nope.h5"))

    def test_from_disk_bad_format_version_raises(self, tmp_path):
        import h5py

        path = tmp_path / "bad.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["format_version"] = 999
            h5.create_group("sequences")
        with pytest.raises(ValueError, match="Unsupported store format_version"):
            SimulationDataStore.from_disk(str(path))

    def test_from_disk_missing_format_version_raises(self, tmp_path):
        import h5py

        path = tmp_path / "noversion.h5"
        with h5py.File(path, "w") as h5:
            h5.create_group("sequences")
        with pytest.raises(ValueError, match="format_version"):
            SimulationDataStore.from_disk(str(path))

    def test_from_disk_sets_storage_path(self, tmp_path):
        path = tmp_path / "sp.h5"
        SimulationDataStore(storage_path=str(path)).flush_to_disk()
        restored = SimulationDataStore.from_disk(str(path))
        assert restored.storage_path == str(path)

    def test_roundtrip_empty_metadata(self, tmp_path):
        path = tmp_path / "nometa.h5"
        store = SimulationDataStore(storage_path=str(path))
        seq = _make_seq("m", metadata={})
        store.add_sequence(seq)
        store.flush_to_disk()
        restored = SimulationDataStore.from_disk(str(path))
        assert restored.get_sequence("m").metadata == {}

    def test_roundtrip_bytes_metadata_decoded(self, tmp_path):
        """from_disk should decode bytes attrs to str (h5py may return bytes)."""
        import h5py

        from shared.data_store.store import _FORMAT_VERSION

        path = tmp_path / "bytesmeta.h5"
        # Hand-craft a file with a bytes-valued attr.
        with h5py.File(path, "w") as h5:
            h5.attrs["format_version"] = _FORMAT_VERSION
            sequences = h5.create_group("sequences")
            grp = sequences.create_group("s")
            n = 3
            grp.create_dataset("timestamps", data=np.zeros(n))
            grp.create_dataset("joint_angles", data=np.zeros((n, 2)))
            grp.create_dataset("joint_velocities", data=np.zeros((n, 2)))
            grp.create_dataset("club_pose", data=np.zeros((n, 7)))
            grp.create_dataset("applied_torques", data=np.zeros((n, 2)))
            grp.attrs.create("note", np.bytes_(b"hello"))

        restored = SimulationDataStore.from_disk(str(path))
        assert restored.get_sequence("s").metadata["note"] == "hello"
