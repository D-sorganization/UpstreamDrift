"""Tests for src.shared.python.engine_core.checkpoint (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.engine_core.checkpoint import StateCheckpoint

# ---------------------------------------------------------------------------
# StateCheckpoint.create
# ---------------------------------------------------------------------------


class TestStateCheckpointCreate:
    def _make(self, **kwargs) -> StateCheckpoint:
        defaults = {
            "engine_type": "mujoco",
            "engine_state": {"step": 0},
            "q": np.array([1.0, 2.0, 3.0]),
            "v": np.array([0.1, 0.2, 0.3]),
            "timestamp": 0.5,
        }
        defaults.update(kwargs)
        return StateCheckpoint.create(**defaults)

    def test_returns_state_checkpoint(self) -> None:
        cp = self._make()
        assert isinstance(cp, StateCheckpoint)

    def test_id_starts_with_cp(self) -> None:
        cp = self._make()
        assert cp.id.startswith("cp_")

    def test_engine_type_stored(self) -> None:
        cp = self._make(engine_type="pinocchio")
        assert cp.engine_type == "pinocchio"

    def test_timestamp_stored(self) -> None:
        cp = self._make(timestamp=1.23)
        assert cp.timestamp == 1.23

    def test_step_count_stored(self) -> None:
        cp = self._make(step_count=42)
        assert cp.step_count == 42

    def test_step_count_defaults_zero(self) -> None:
        cp = self._make()
        assert cp.step_count == 0

    def test_q_stored_as_tuple(self) -> None:
        q = np.array([1.0, 2.0])
        cp = self._make(q=q)
        assert isinstance(cp.q, tuple)

    def test_v_stored_as_tuple(self) -> None:
        v = np.array([0.5, 0.6])
        cp = self._make(v=v)
        assert isinstance(cp.v, tuple)

    def test_metadata_stored(self) -> None:
        cp = self._make(metadata={"label": "test"})
        assert cp.metadata["label"] == "test"

    def test_metadata_defaults_empty(self) -> None:
        cp = self._make()
        assert cp.metadata == {}

    def test_checksum_non_empty(self) -> None:
        cp = self._make()
        assert len(cp.checksum) > 0

    def test_engine_state_copied(self) -> None:
        state = {"x": 1}
        cp = self._make(engine_state=state)
        state["x"] = 99
        assert cp.engine_state["x"] == 1  # immutable copy


# ---------------------------------------------------------------------------
# StateCheckpoint.get_q / get_v
# ---------------------------------------------------------------------------


class TestStateCheckpointArrayAccessors:
    def test_get_q_returns_array(self) -> None:
        q = np.array([1.0, 2.0, 3.0])
        cp = StateCheckpoint.create("eng", {}, q, np.zeros(3), 0.0)
        result = cp.get_q()
        assert isinstance(result, np.ndarray)

    def test_get_q_values_match(self) -> None:
        q = np.array([1.0, 2.0, 3.0])
        cp = StateCheckpoint.create("eng", {}, q, np.zeros(3), 0.0)
        np.testing.assert_array_almost_equal(cp.get_q(), q)

    def test_get_v_values_match(self) -> None:
        v = np.array([0.1, 0.2])
        cp = StateCheckpoint.create("eng", {}, np.zeros(2), v, 0.0)
        np.testing.assert_array_almost_equal(cp.get_v(), v)


# ---------------------------------------------------------------------------
# StateCheckpoint.verify_checksum
# ---------------------------------------------------------------------------


class TestStateCheckpointVerifyChecksum:
    def test_fresh_checkpoint_valid(self) -> None:
        cp = StateCheckpoint.create("mujoco", {}, np.ones(3), np.ones(3), 0.0)
        assert cp.verify_checksum() is True

    def test_tampered_checksum_invalid_via_replace(self) -> None:
        import dataclasses

        cp = StateCheckpoint.create("mujoco", {}, np.ones(3), np.ones(3), 0.0)
        bad_cp = dataclasses.replace(cp, checksum="deadbeef00000000")
        assert bad_cp.verify_checksum() is False


# ---------------------------------------------------------------------------
# StateCheckpoint.to_dict
# ---------------------------------------------------------------------------


class TestStateCheckpointToDict:
    def test_checkpoint_returns_dict(self) -> None:
        cp = StateCheckpoint.create("mujoco", {}, np.ones(2), np.zeros(2), 0.0)
        assert isinstance(cp.to_dict(), dict)

    def test_id_in_dict(self) -> None:
        cp = StateCheckpoint.create("mujoco", {}, np.ones(2), np.zeros(2), 0.0)
        assert "id" in cp.to_dict()

    def test_q_is_list_in_dict(self) -> None:
        cp = StateCheckpoint.create(
            "mujoco", {}, np.array([1.0, 2.0]), np.zeros(2), 0.0
        )
        result = cp.to_dict()
        assert isinstance(result["q"], list)

    def test_all_fields_present(self) -> None:
        cp = StateCheckpoint.create("mujoco", {}, np.ones(2), np.zeros(2), 1.0)
        d = cp.to_dict()
        for key in (
            "id",
            "timestamp",
            "wall_time",
            "engine_type",
            "engine_state",
            "q",
            "v",
            "step_count",
            "metadata",
            "checksum",
        ):
            assert key in d


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------


class TestStateCheckpointFrozen:
    def test_mutation_raises(self) -> None:
        cp = StateCheckpoint.create("mujoco", {}, np.ones(2), np.zeros(2), 0.0)
        with pytest.raises((AttributeError, TypeError)):
            cp.engine_type = "changed"  # type: ignore[misc]
