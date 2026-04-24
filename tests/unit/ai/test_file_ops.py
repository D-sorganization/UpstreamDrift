"""Tests for src.shared.python.ai.tools.file_ops (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.tools.file_ops import register_file_tools


def _make_registry() -> ToolRegistry:
    """Create registry and register file tools."""
    registry = ToolRegistry()
    register_file_tools(registry)
    return registry


class TestRegisterFileTools:
    def test_registration_succeeds(self) -> None:
        registry = _make_registry()
        assert registry is not None

    def test_read_file_registered(self) -> None:
        registry = _make_registry()
        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names

    def test_list_directory_registered(self) -> None:
        registry = _make_registry()
        tools = registry.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_directory" in tool_names


class TestReadFileTool:
    def _call_read_file(self, registry: ToolRegistry, file_path: str) -> str:
        result = registry.execute("read_file", {"file_path": file_path})
        return str(result.result or result.error or "")

    def test_read_existing_file(self, tmp_path: Path) -> None:
        registry = _make_registry()
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello, world!")
        result = self._call_read_file(registry, str(test_file))
        assert "Hello, world!" in result

    def test_read_nonexistent_file(self) -> None:
        registry = _make_registry()
        result = self._call_read_file(registry, "/nonexistent/path/file.txt")
        assert "Error" in result

    def test_read_directory_returns_error(self, tmp_path: Path) -> None:
        registry = _make_registry()
        result = self._call_read_file(registry, str(tmp_path))
        assert "Error" in result


class TestListDirectoryTool:
    def _call_list_dir(self, registry: ToolRegistry, dir_path: str) -> str:
        result = registry.execute("list_directory", {"directory_path": dir_path})
        return str(result.result or result.error or "")

    def test_list_existing_directory(self, tmp_path: Path) -> None:
        registry = _make_registry()
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "subdir").mkdir()
        result = self._call_list_dir(registry, str(tmp_path))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_shows_files_and_dirs(self, tmp_path: Path) -> None:
        registry = _make_registry()
        (tmp_path / "myfile.txt").write_text("hi")
        (tmp_path / "mydir").mkdir()
        result = self._call_list_dir(registry, str(tmp_path))
        assert "myfile.txt" in result
        assert "mydir" in result

    def test_list_nonexistent_directory_returns_error(self) -> None:
        registry = _make_registry()
        result = self._call_list_dir(registry, "/nonexistent/path/dir")
        assert "Error" in result

    def test_list_file_as_directory_returns_error(self, tmp_path: Path) -> None:
        registry = _make_registry()
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")
        result = self._call_list_dir(registry, str(test_file))
        assert "Error" in result
