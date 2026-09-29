"""Tests for sidekick.utils.state_manager (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

from sidekick.utils.state_manager import (
    StateManager,
    safe_read_json,
    safe_write_json,
)

# ---------------------------------------------------------------------------
# safe_read_json
# ---------------------------------------------------------------------------


class TestSafeReadJson:
    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        result = safe_read_json(tmp_path / "nonexistent.json", default=42)
        assert result == 42

    def test_missing_file_default_none(self, tmp_path: Path) -> None:
        result = safe_read_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_reads_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text('{"x": 1}', encoding="utf-8")
        result = safe_read_json(p)
        assert result == {"x": 1}

    def test_corrupt_json_returns_default(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not valid json!!!", encoding="utf-8")
        result = safe_read_json(p, default="fallback")
        assert result == "fallback"


# ---------------------------------------------------------------------------
# safe_write_json
# ---------------------------------------------------------------------------


class TestSafeWriteJson:
    def test_writes_and_reads_back(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        data = {"key": "value", "num": 3.14}
        assert safe_write_json(p, data) is True
        assert p.exists()
        result = safe_read_json(p)
        assert result == data

    def test_state_manager_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c.json"
        assert safe_write_json(p, {"x": 1}) is True
        assert p.exists()

    def test_returns_false_on_unserializable(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        result = safe_write_json(p, object())  # object() is not JSON serializable
        assert result is False


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------


class TestStateManagerInit:
    def test_init_with_tmp_dir(self, tmp_path: Path) -> None:
        mgr = StateManager(base_directory=str(tmp_path / "states"))
        assert mgr is not None

    def test_list_states_empty_initially(self, tmp_path: Path) -> None:
        mgr = StateManager(base_directory=str(tmp_path / "states"))
        states = mgr.list_states()
        assert isinstance(states, list)
        assert len(states) == 0


class TestStateManagerSaveLoad:
    def _mgr(self, tmp_path: Path) -> StateManager:
        return StateManager(base_directory=str(tmp_path / "states"))

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        data = {"temperature": 500.0, "pressure": 101.325}
        mgr.save_state("test_run", data)
        loaded = mgr.load_state("test_run")
        assert loaded is not None
        assert loaded["temperature"] == 500.0

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        result = mgr.load_state("does_not_exist")
        assert result is None

    def test_save_appears_in_list(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        mgr.save_state("my_state", {"val": 1})
        states = mgr.list_states()
        names = [s["name"] for s in states]
        assert "my_state" in names

    def test_delete_state(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        mgr.save_state("temp_state", {"x": 1})
        deleted = mgr.delete_state("temp_state")
        assert deleted is True
        assert mgr.load_state("temp_state") is None

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        result = mgr.delete_state("ghost")
        assert result is False
