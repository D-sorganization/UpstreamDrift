"""Tests for the LauncherService.

Provides isolated unit tests for the LauncherService class, mocking out
the underlying ProcessManager and ModelHandlerRegistry to ensure tests
run quickly and deterministically without launching actual processes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.launcher_service import LauncherService


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Fixture providing a temporary repository root directory."""
    return tmp_path


@pytest.fixture
def launcher_service(repo_root: Path) -> LauncherService:
    """Fixture providing a LauncherService instance."""
    return LauncherService(repo_root=repo_root)


class TestLauncherServiceInitialization:
    """Tests for LauncherService initialization."""

    def test_init_success(self, repo_root: Path) -> None:
        """Test successful initialization with a valid directory."""
        service = LauncherService(repo_root=repo_root)
        assert service._repo_root == repo_root
        assert service._process_manager is None
        assert service._handler_registry is None

    def test_init_none_root(self) -> None:
        """Test initialization with None raises ValueError."""
        with pytest.raises(ValueError, match="repo_root must not be None"):
            LauncherService(repo_root=None)  # type: ignore

    def test_init_missing_root(self, tmp_path: Path) -> None:
        """Test initialization with non-existent directory raises FileNotFoundError."""
        missing_dir = tmp_path / "non_existent_dir"
        with pytest.raises(FileNotFoundError, match="repo_root does not exist"):
            LauncherService(repo_root=missing_dir)


class TestLauncherServiceProperties:
    """Tests for lazy-loaded properties."""

    def test_process_manager_lazy_load(
        self, launcher_service: LauncherService, repo_root: Path
    ) -> None:
        """Test that process_manager is lazily loaded."""
        mock_process_manager_cls = MagicMock()
        mock_instance = MagicMock()
        mock_process_manager_cls.return_value = mock_instance

        # Create a mock module for src.launchers.launcher_process_manager
        mock_module = MagicMock()
        mock_module.ProcessManager = mock_process_manager_cls

        with patch.dict(
            sys.modules, {"src.launchers.launcher_process_manager": mock_module}
        ):
            # First access should initialize it
            pm = launcher_service.process_manager
            assert pm is mock_instance
            mock_process_manager_cls.assert_called_once_with(repo_root=repo_root)

            # Second access should return the cached instance
            pm2 = launcher_service.process_manager
            assert pm2 is mock_instance
            assert mock_process_manager_cls.call_count == 1

    def test_handler_registry_lazy_load(
        self, launcher_service: LauncherService
    ) -> None:
        """Test that handler_registry is lazily loaded."""
        mock_handler_registry_cls = MagicMock()
        mock_instance = MagicMock()
        mock_handler_registry_cls.return_value = mock_instance

        # Create a mock module for src.launchers.launcher_model_handlers
        mock_module = MagicMock()
        mock_module.ModelHandlerRegistry = mock_handler_registry_cls

        with patch.dict(
            sys.modules, {"src.launchers.launcher_model_handlers": mock_module}
        ):
            # First access should initialize it
            registry = launcher_service.handler_registry
            assert registry is mock_instance
            mock_handler_registry_cls.assert_called_once_with()

            # Second access should return the cached instance
            registry2 = launcher_service.handler_registry
            assert registry2 is mock_instance
            assert mock_handler_registry_cls.call_count == 1


class TestLauncherServiceMethods:
    """Tests for LauncherService methods."""

    def test_get_handler_success(self, launcher_service: LauncherService) -> None:
        """Test get_handler with a valid model type."""
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_registry.get_handler.return_value = mock_handler
        launcher_service._handler_registry = mock_registry

        handler = launcher_service.get_handler("test_model")

        assert handler is mock_handler
        mock_registry.get_handler.assert_called_once_with("test_model")

    def test_get_handler_empty_string(self, launcher_service: LauncherService) -> None:
        """Test get_handler with an empty string raises ValueError."""
        with pytest.raises(ValueError, match="Model type must be a non-empty string"):
            launcher_service.get_handler("")

    def test_get_handler_none(self, launcher_service: LauncherService) -> None:
        """Test get_handler with None raises ValueError."""
        with pytest.raises(ValueError, match="Model type must be a non-empty string"):
            launcher_service.get_handler(None)  # type: ignore

    def test_get_running_processes(self, launcher_service: LauncherService) -> None:
        """Test get_running_processes returns correct info."""
        mock_pm = MagicMock()

        mock_proc1 = MagicMock()
        mock_proc1.pid = 1234
        mock_proc1.poll.return_value = None  # Running

        mock_proc2 = MagicMock()
        mock_proc2.pid = 5678
        mock_proc2.poll.return_value = 0  # Finished successfully

        mock_pm.running_processes = {
            "proc1": mock_proc1,
            "proc2": mock_proc2,
        }
        launcher_service._process_manager = mock_pm

        processes = launcher_service.get_running_processes()

        assert len(processes) == 2
        assert processes["proc1"] == {"pid": 1234, "running": True, "exit_code": None}
        assert processes["proc2"] == {"pid": 5678, "running": False, "exit_code": 0}
        mock_proc1.poll.assert_called_once()
        mock_proc2.poll.assert_called_once()

    @patch("src.api.services.launcher_service.kill_process_tree", create=True)
    def test_stop_process_success(
        self, mock_kill_tree: MagicMock, launcher_service: LauncherService
    ) -> None:
        """Test stop_process successfully stops and removes a process."""
        mock_pm = MagicMock()
        mock_proc = MagicMock()
        mock_proc.pid = 1234

        running_processes = {"test_proc": mock_proc}
        mock_pm.running_processes = running_processes
        launcher_service._process_manager = mock_pm

        # Mock the import inside the method
        mock_module = MagicMock()
        mock_module.kill_process_tree = mock_kill_tree

        with patch.dict(
            sys.modules, {"src.shared.python.security.subprocess_utils": mock_module}
        ):
            result = launcher_service.stop_process("test_proc")

            assert result is True
            mock_module.kill_process_tree.assert_called_once_with(1234)
            assert "test_proc" not in running_processes

    def test_stop_process_not_found(self, launcher_service: LauncherService) -> None:
        """Test stop_process returns False when process not found."""
        mock_pm = MagicMock()
        mock_pm.running_processes = {}
        launcher_service._process_manager = mock_pm

        result = launcher_service.stop_process("non_existent_proc")

        assert result is False

    def test_stop_process_empty_name(self, launcher_service: LauncherService) -> None:
        """Test stop_process with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Process name must be a non-empty string"):
            launcher_service.stop_process("")

    def test_stop_process_none_name(self, launcher_service: LauncherService) -> None:
        """Test stop_process with None raises ValueError."""
        with pytest.raises(ValueError, match="Process name must be a non-empty string"):
            launcher_service.stop_process(None)  # type: ignore
