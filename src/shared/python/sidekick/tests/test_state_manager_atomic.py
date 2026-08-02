"""Atomicity tests for StateManager JSON persistence (#3276).

A crash / disk-full mid-write must never leave a previously-saved state file
truncated or unparseable: the on-disk state must be either the old contents or
the new contents, never a half-written file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sidekick.utils.state_manager import StateManager

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(tmp_path: Path) -> StateManager:
    return StateManager(base_directory=str(tmp_path))


def _state_file(manager: StateManager, name: str) -> Path:
    safe = manager._sanitize_filename(name)
    return Path(manager.states_dir) / f"{safe}.json"


def test_save_state_is_atomic_on_write_failure(manager: StateManager) -> None:
    """A failure mid-write leaves the prior saved file intact and loadable."""
    assert manager.save_state("doc", {"version": 1, "value": "original"}) is True
    state_file = _state_file(manager, "doc")
    original_bytes = state_file.read_bytes()

    # Simulate the destination write (os.replace) failing partway. Because the
    # write goes to a temp file first, the destination must be untouched.
    with patch(
        "utils.file_utils.os.replace",
        side_effect=OSError("simulated disk-full during replace"),
    ):
        result = manager.save_state("doc", {"version": 2, "value": "new"})

    assert result is False
    # The previously-saved file is never truncated — it still loads as-is.
    assert state_file.read_bytes() == original_bytes
    loaded = manager.load_state("doc")
    assert loaded is not None
    assert loaded["value"] == "original"

    # No stray temp file left behind in the states directory.
    leftover = list(manager.states_dir.glob(".*.tmp"))
    assert leftover == []


def test_atomic_write_text_leaves_prior_file_intact_on_failure(
    tmp_path: Path,
) -> None:
    """The atomic helper never truncates the destination on a mid-write error."""
    from utils.file_utils import atomic_write_text

    target = tmp_path / "data.json"
    target.write_text("OLD", encoding="utf-8")

    with patch("utils.file_utils.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError):
            atomic_write_text(target, "NEW")

    # Destination still holds the old contents; no temp file remains.
    assert target.read_text(encoding="utf-8") == "OLD"
    assert list(tmp_path.glob(".*.tmp")) == []
