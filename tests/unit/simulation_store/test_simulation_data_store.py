"""Tests for SimulationDataStore — Epic #5396.

Covers:
- CRUD operations (save, load, list, delete)
- Persistence across independent instances
- Missing / unknown run IDs
- Invalid run_id values (DbC preconditions)
- Overwrite semantics
- Empty store
- Large payloads
- Nested / complex data structures
- run_exists helper
- list_runs ordering
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.simulation_store import SimulationDataStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> SimulationDataStore:
    """Return a fresh, isolated store backed by a temp directory."""
    return SimulationDataStore(base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    def test_save_and_load_simple_dict(self, store: SimulationDataStore) -> None:
        data = {"engine": "drake", "score": 0.95}
        store.save_run("run_001", data)
        assert store.load_run("run_001") == data

    def test_save_creates_backing_file(
        self, store: SimulationDataStore, tmp_path: Path
    ) -> None:
        store.save_run("run_001", {"x": 1})
        assert (tmp_path / "run_001.json").exists()

    def test_load_returns_dict(self, store: SimulationDataStore) -> None:
        store.save_run("abc", {"k": "v"})
        result = store.load_run("abc")
        assert isinstance(result, dict)

    def test_overwrite_replaces_data(self, store: SimulationDataStore) -> None:
        store.save_run("run_001", {"v": 1})
        store.save_run("run_001", {"v": 2})
        assert store.load_run("run_001") == {"v": 2}

    def test_save_empty_dict(self, store: SimulationDataStore) -> None:
        store.save_run("empty_run", {})
        assert store.load_run("empty_run") == {}

    def test_save_nested_dict(self, store: SimulationDataStore) -> None:
        data = {"params": {"alpha": 1.0, "beta": [1, 2, 3]}, "meta": {"ok": True}}
        store.save_run("nested", data)
        assert store.load_run("nested") == data

    def test_save_list_values(self, store: SimulationDataStore) -> None:
        data = {"trajectory": [0.1, 0.2, 0.3], "tags": ["swing", "iron"]}
        store.save_run("traj_run", data)
        assert store.load_run("traj_run") == data

    def test_save_preserves_none_values(self, store: SimulationDataStore) -> None:
        data = {"result": None, "error": None}
        store.save_run("none_run", data)
        assert store.load_run("none_run") == data

    def test_large_payload(self, store: SimulationDataStore) -> None:
        data = {"frames": list(range(10_000))}
        store.save_run("large", data)
        loaded = store.load_run("large")
        assert loaded["frames"] == list(range(10_000))


# ---------------------------------------------------------------------------
# List runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_store_returns_empty_list(self, store: SimulationDataStore) -> None:
        assert store.list_runs() == []

    def test_list_after_single_save(self, store: SimulationDataStore) -> None:
        store.save_run("run_a", {})
        assert store.list_runs() == ["run_a"]

    def test_list_multiple_sorted(self, store: SimulationDataStore) -> None:
        store.save_run("run_b", {})
        store.save_run("run_a", {})
        store.save_run("run_c", {})
        assert store.list_runs() == ["run_a", "run_b", "run_c"]

    def test_list_returns_list_type(self, store: SimulationDataStore) -> None:
        store.save_run("r1", {})
        result = store.list_runs()
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_list_after_delete_excludes_deleted(
        self, store: SimulationDataStore
    ) -> None:
        store.save_run("keep", {})
        store.save_run("remove", {})
        store.delete_run("remove")
        assert store.list_runs() == ["keep"]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDeleteRun:
    def test_delete_removes_run(self, store: SimulationDataStore) -> None:
        store.save_run("run_del", {"x": 1})
        store.delete_run("run_del")
        assert "run_del" not in store.list_runs()

    def test_delete_removes_backing_file(
        self, store: SimulationDataStore, tmp_path: Path
    ) -> None:
        store.save_run("run_del", {})
        store.delete_run("run_del")
        assert not (tmp_path / "run_del.json").exists()

    def test_delete_unknown_run_raises_key_error(
        self, store: SimulationDataStore
    ) -> None:
        with pytest.raises(KeyError):
            store.delete_run("nonexistent")

    def test_delete_does_not_affect_other_runs(
        self, store: SimulationDataStore
    ) -> None:
        store.save_run("keep", {"k": "v"})
        store.save_run("gone", {})
        store.delete_run("gone")
        assert store.load_run("keep") == {"k": "v"}


# ---------------------------------------------------------------------------
# Missing run
# ---------------------------------------------------------------------------


class TestLoadMissingRun:
    def test_load_missing_raises_key_error(self, store: SimulationDataStore) -> None:
        with pytest.raises(KeyError):
            store.load_run("does_not_exist")

    def test_load_after_delete_raises_key_error(
        self, store: SimulationDataStore
    ) -> None:
        store.save_run("ephemeral", {})
        store.delete_run("ephemeral")
        with pytest.raises(KeyError):
            store.load_run("ephemeral")


# ---------------------------------------------------------------------------
# Invalid run_id (DbC preconditions)
# ---------------------------------------------------------------------------


class TestInvalidRunId:
    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "   ",
            "has spaces",
            "has/slash",
            "has\\backslash",
            "has.dot",
            "a" * 257,  # too long (exceeds 256 char limit)
        ],
    )
    def test_save_invalid_id_raises(
        self, store: SimulationDataStore, bad_id: str
    ) -> None:
        with pytest.raises((ValueError, Exception)):
            store.save_run(bad_id, {})

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "has spaces",
            "has/slash",
        ],
    )
    def test_load_invalid_id_raises(
        self, store: SimulationDataStore, bad_id: str
    ) -> None:
        with pytest.raises((ValueError, Exception)):
            store.load_run(bad_id)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "has spaces",
        ],
    )
    def test_delete_invalid_id_raises(
        self, store: SimulationDataStore, bad_id: str
    ) -> None:
        with pytest.raises((ValueError, Exception)):
            store.delete_run(bad_id)

    def test_save_requires_dict_not_list(self, store: SimulationDataStore) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            store.save_run("run_ok", [1, 2, 3])  # type: ignore[arg-type]

    def test_save_requires_dict_not_string(self, store: SimulationDataStore) -> None:
        with pytest.raises((ValueError, TypeError, Exception)):
            store.save_run("run_ok", "not_a_dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Valid edge-case IDs
# ---------------------------------------------------------------------------


class TestValidRunIds:
    @pytest.mark.parametrize(
        "valid_id",
        [
            "run_001",
            "abc",
            "A",
            "a1b2c3",
            "run-2024-01-01",
            "UPPERCASE",
            "mixed_CASE-123",
            "a" * 64,  # long but filesystem-safe
        ],
    )
    def test_valid_id_accepted(self, store: SimulationDataStore, valid_id: str) -> None:
        store.save_run(valid_id, {"ok": True})
        assert store.load_run(valid_id) == {"ok": True}


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------


class TestPersistenceAcrossInstances:
    def test_second_instance_reads_data_from_first(self, tmp_path: Path) -> None:
        store_a = SimulationDataStore(base_dir=tmp_path)
        store_a.save_run("persistent", {"value": 42})

        store_b = SimulationDataStore(base_dir=tmp_path)
        assert store_b.load_run("persistent") == {"value": 42}

    def test_second_instance_lists_runs_from_first(self, tmp_path: Path) -> None:
        store_a = SimulationDataStore(base_dir=tmp_path)
        store_a.save_run("r1", {})
        store_a.save_run("r2", {})

        store_b = SimulationDataStore(base_dir=tmp_path)
        assert store_b.list_runs() == ["r1", "r2"]

    def test_second_instance_can_delete_first_run(self, tmp_path: Path) -> None:
        store_a = SimulationDataStore(base_dir=tmp_path)
        store_a.save_run("shared_run", {})

        store_b = SimulationDataStore(base_dir=tmp_path)
        store_b.delete_run("shared_run")
        assert store_a.list_runs() == []


# ---------------------------------------------------------------------------
# run_exists helper
# ---------------------------------------------------------------------------


class TestRunExists:
    def test_exists_returns_false_when_absent(self, store: SimulationDataStore) -> None:
        assert store.run_exists("missing") is False

    def test_exists_returns_true_after_save(self, store: SimulationDataStore) -> None:
        store.save_run("present", {})
        assert store.run_exists("present") is True

    def test_exists_returns_false_after_delete(
        self, store: SimulationDataStore
    ) -> None:
        store.save_run("transient", {})
        store.delete_run("transient")
        assert store.run_exists("transient") is False


# ---------------------------------------------------------------------------
# JSON file format integrity
# ---------------------------------------------------------------------------


class TestJsonFileFormat:
    def test_backing_file_is_valid_json(
        self, store: SimulationDataStore, tmp_path: Path
    ) -> None:
        store.save_run("json_check", {"key": "val"})
        raw = (tmp_path / "json_check.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed == {"key": "val"}

    def test_multiple_runs_are_separate_files(
        self, store: SimulationDataStore, tmp_path: Path
    ) -> None:
        store.save_run("r1", {"a": 1})
        store.save_run("r2", {"b": 2})
        files = {p.stem for p in tmp_path.glob("*.json")}
        assert files == {"r1", "r2"}
