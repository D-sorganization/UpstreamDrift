"""Tests that tile launch failures surface searched paths and the submodule remedy.

Refs #13023 (follow-up to #12993 / PR #13022).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def test_format_source_root_not_found_includes_paths_and_remedy(tmp_path: Path) -> None:
    from src.launchers.launch_failure_message import format_source_root_not_found
    from src.shared.python.config.model_source_providers import (
        ModelSourcePathPolicy,
        SourceRootNotFoundError,
    )

    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    policy = ModelSourcePathPolicy(repo)
    with pytest.raises(SourceRootNotFoundError) as exc_info:
        policy.require_source_root("Missing_Sibling_Repo")

    formatted = format_source_root_not_found(exc_info.value)

    assert isinstance(formatted, str)
    for candidate in policy.source_root_candidates(Path("Missing_Sibling_Repo")):
        assert str(candidate) in formatted
    assert "git submodule update --init vendor/ud-tools" in formatted


def test_special_app_handler_surfaces_source_root_not_found_in_status_message(
    tmp_path: Path,
) -> None:
    from src.launchers.launcher_model_handlers import SpecialAppHandler
    from src.shared.python.config.model_source_providers import ModelSourcePathPolicy

    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    policy = ModelSourcePathPolicy(repo)

    model = SimpleNamespace(
        id="test_missing_app",
        name="Test Missing App",
        type="special_app",
        path="src/tools/test/__main__.py",
        source_root="Missing_Sibling_Repo",
    )

    handler = SpecialAppHandler()
    process_manager = MagicMock()

    success = handler.launch(model, repo, process_manager)
    assert not success
    process_manager.launch_script.assert_not_called()
    process_manager.launch_module.assert_not_called()

    status_msg = handler.status_message(model)
    assert status_msg is not None
    for candidate in policy.source_root_candidates(Path("Missing_Sibling_Repo")):
        assert str(candidate) in status_msg
    assert "git submodule update --init vendor/ud-tools" in status_msg


def test_tile_launch_shows_toast_with_searched_paths_and_remedy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qapp,
) -> None:
    from src.launchers.launcher_model_handlers import ModelHandlerRegistry
    from src.launchers.launcher_simulation import SimulationManager
    from src.shared.python.config.model_source_providers import ModelSourcePathPolicy

    repo = tmp_path / "UpstreamDrift"
    repo.mkdir()
    monkeypatch.setattr("src.launchers.launcher_simulation.REPOS_ROOT", repo)
    policy = ModelSourcePathPolicy(repo)

    model = SimpleNamespace(
        id="test_missing_app",
        name="Test Missing App",
        type="special_app",
        path="src/tools/test/__main__.py",
        source_root="Missing_Sibling_Repo",
    )

    mock_launcher = MagicMock()
    mock_launcher.show_toast = MagicMock()
    mock_launcher.lbl_status = MagicMock()
    mock_launcher.model_handler_registry = ModelHandlerRegistry()
    mock_launcher.process_manager = MagicMock()
    mock_launcher._get_model = MagicMock(return_value=model)
    mock_launcher.selected_model = "test_missing_app"

    sim_manager = SimulationManager(mock_launcher)
    sim_manager._execute_local_launch(model)

    mock_launcher.show_toast.assert_called_once()
    toast_args = mock_launcher.show_toast.call_args[0]
    toast_text, toast_type = toast_args[0], toast_args[1]
    assert toast_type == "error"
    for candidate in policy.source_root_candidates(Path("Missing_Sibling_Repo")):
        assert str(candidate) in toast_text
    assert "git submodule update --init vendor/ud-tools" in toast_text


def test_launcher_model_handlers_module_size_budget() -> None:
    handlers_file = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "launchers"
        / "launcher_model_handlers.py"
    )
    line_count = len(handlers_file.read_text(encoding="utf-8").splitlines())
    assert line_count <= 1200, (
        f"launcher_model_handlers.py exceeds 1200-line budget: {line_count} lines"
    )
