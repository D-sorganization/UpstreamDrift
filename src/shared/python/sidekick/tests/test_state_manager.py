import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sidekick.utils.state_manager import (
    StateManager,
    _StateManagerHolder,
    get_state_manager,
)


@pytest.fixture
def manager(tmp_path) -> Any:
    # Use tmp_path to isolate states for testing
    mgr = StateManager(base_directory=str(tmp_path))
    return mgr


def test_initialization(manager, tmp_path) -> Any:
    assert manager.base_directory == tmp_path
    assert manager.states_dir.exists()
    assert manager.sessions_dir.exists()
    assert manager.backups_dir.exists()
    assert manager.exports_dir.exists()
    assert manager.protected_states == set()


def test_protect_unprotect_state(manager) -> Any:
    # Protect a state that doesn't exist yet
    manager.protect_state("test_state")
    assert "test_state" in manager.protected_states

    manager.unprotect_state("test_state")
    assert "test_state" not in manager.protected_states


def test_save_load_state(manager) -> Any:
    state_data = {"key": "value", "num": 42}
    success = manager.save_state("test_save", state_data, description="test desc")
    assert success is True

    loaded = manager.load_state("test_save")
    assert loaded == state_data


def test_save_state_with_protection(manager) -> Any:
    state_data = {"foo": "bar"}
    manager.save_state("protected_state", state_data, protected=True)
    assert "protected_state" in manager.protected_states

    # Verify we can't delete it without force
    deleted = manager.delete_state("protected_state")
    assert deleted is False

    # Force delete works
    deleted = manager.delete_state("protected_state", force=True)
    assert deleted is True


def test_delete_state(manager) -> Any:
    manager.save_state("to_delete", {"a": 1})
    assert manager.delete_state("to_delete") is True
    assert manager.load_state("to_delete") is None


def test_list_states(manager) -> Any:
    manager.save_state("state1", {"1": 1})
    manager.save_state("state2", {"2": 2})

    states = manager.list_states()
    assert len(states) == 2
    names = [s["name"] for s in states]
    assert "state1" in names
    assert "state2" in names


def test_export_import_state(manager, tmp_path) -> Any:
    manager.save_state("export_me", {"data": "yes"})

    export_path = tmp_path / "export.json"
    actual_path = manager.export_state("export_me", str(export_path))

    assert actual_path == str(export_path)
    assert Path(export_path).exists()

    # Import it under a new name
    success = manager.import_state(str(export_path), "imported_state")
    assert success is True

    loaded = manager.load_state("imported_state")
    assert loaded == {"data": "yes"}

    # Attempt to import over existing state
    success = manager.import_state(str(export_path), "imported_state")
    assert success is False  # already exists


def test_save_load_session(manager) -> Any:
    session_data = {"current_tab": 2}
    manager.save_session(session_data)

    loaded = manager.load_session()
    assert loaded == session_data


def test_cleanup_backups(manager, tmp_path) -> Any:
    manager.save_state("old_state", {"old": 1})
    # Save it again to trigger a backup
    manager.save_state("old_state", {"new": 2})

    backups = list(manager.backups_dir.glob("*.backup"))
    assert len(backups) == 1

    # Modify mtime to be old
    old_time = datetime.now(timezone.utc).timestamp() - (35 * 24 * 3600)  # noqa: UP017
    import os

    os.utime(backups[0], (old_time, old_time))

    manager.cleanup_old_backups(max_age_days=30)
    assert len(list(manager.backups_dir.glob("*.backup"))) == 0


def test_json_serializer(manager) -> Any:
    from pathlib import Path

    dt = datetime(2023, 1, 1, 12, 0, 0)
    res = manager._json_serializer(dt)
    assert res == "2023-01-01T12:00:00"

    p = Path("/tmp/foo")  # nosec B108
    res = manager._json_serializer(p)
    assert res == str(p)

    class Dummy:
        pass

    d = Dummy()
    d.attr = "val"  # type: ignore[attr-defined]
    res = manager._json_serializer(d)
    assert res == {"attr": "val"}


def test_load_state_invalid_format(manager, tmp_path) -> Any:
    # Create invalid state file
    file_path = manager.states_dir / "invalid.json"
    with open(file_path, "w") as f:
        f.write("not valid json")

    with pytest.raises(json.JSONDecodeError):
        manager.load_state("invalid")

    # Valid JSON but missing schema struct
    with open(file_path, "w") as f:
        json.dump({"bad": "state"}, f)

    loaded = manager.load_state("invalid")
    assert loaded is None


def test_permission_error_save_load(manager) -> Any:
    # Mock open to raise PermissionError
    with patch("builtins.open", side_effect=PermissionError):
        # Save
        assert manager.save_state("err_state", {"data": 1}) is False

        # Load (requires file to exist first, so we mock exists)
        with patch("pathlib.Path.exists", return_value=True):
            assert manager.load_state("err_state") is None

        # Export
        with patch.object(manager, "load_state", return_value={"data": 1}):
            assert manager.export_state("err_state") is None

        # Import
        with patch("pathlib.Path.exists", return_value=True):
            assert manager.import_state("some_path.json") is False

        # Save session
        assert manager.save_session({"s": 1}) is False

        # Load session
        with patch("pathlib.Path.exists", return_value=True):
            assert manager.load_session() is None


def test_singleton_get_state_manager(tmp_path) -> Any:
    _StateManagerHolder.instance = None
    mgr1 = get_state_manager(str(tmp_path))
    mgr2 = get_state_manager(str(tmp_path))
    assert mgr1 is mgr2
    # Ensure cleanup
    _StateManagerHolder.instance = None


def test_delete_state_not_found(manager) -> Any:
    assert manager.delete_state("does_not_exist") is False


def test_protect_state_existing(manager) -> Any:
    manager.save_state("exists", {"a": 1})

    # Actually protect it (writes to metadata)
    assert manager.protect_state("exists") is True
    loaded = manager.load_state("exists")
    assert loaded is not None
    assert "exists" in manager.protected_states

    # Now unprotect it
    assert manager.unprotect_state("exists") is True
    assert "exists" not in manager.protected_states


def test_list_states_errors_and_cache(manager, tmp_path) -> Any:
    manager.save_state("s1", {"b": 2})

    # Add a non-json file to the dir
    (manager.states_dir / "not_a_state.txt").write_text("hello")

    states = manager.list_states()
    assert len(states) == 1
    assert states[0]["name"] == "s1"

    # Run list again to hit cache logic
    states2 = manager.list_states()
    assert states == states2


def test_unprotect_permission_error(manager) -> Any:
    manager.save_state("some", {"a": 1})
    with patch("builtins.open", side_effect=PermissionError):
        assert manager.protect_state("some") is False
        assert manager.unprotect_state("some") is False


def test_delete_state_permission_error(manager) -> Any:
    manager.save_state("del_err", {"a": 1})
    with patch("pathlib.Path.unlink", side_effect=PermissionError):
        assert manager.delete_state("del_err") is False
