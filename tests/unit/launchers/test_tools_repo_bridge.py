"""Tests for the Tools-repo bridge registry (#5334).

Verifies that UpstreamDrift's launcher can discover and represent tools
from the sibling Tools repository via tools.json, without requiring the
Tools repo to be installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.gui_launcher.tools_repo_bridge import (
    ExternalTool,
    ToolsRepoBridge,
    load_tools_from_repo,
)


# ── Fixtures ─────────────────────────────────────────────────────────


SAMPLE_TOOLS_JSON: dict = {
    "Signal Processing": [
        {
            "name": "Function Generator (PyQt6)",
            "path": "src/function_generator/launch_pyqt6.py",
            "type": "python",
            "desc": "Generate waveforms",
        },
        {
            "name": "Signal Processing Studio",
            "path": "src/signal_processing_studio/launch_pyqt6.py",
            "type": "python",
            "desc": "Unified signal processing",
        },
    ],
    "Robotics": [
        {
            "name": "Humanoid Character Builder",
            "path": "src/humanoid_builder_gui/launch_pyqt6.py",
            "type": "python",
            "desc": "Build parametric humanoids",
        },
    ],
}


@pytest.fixture()
def tools_json_file(tmp_path: Path) -> Path:
    """Write a sample tools.json to a temp directory."""
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps(SAMPLE_TOOLS_JSON), encoding="utf-8")
    return tools_file


@pytest.fixture()
def mock_tools_repo(tmp_path: Path) -> Path:
    """Create a minimal mock Tools repo directory with tools.json."""
    repo_root = tmp_path / "Tools"
    repo_root.mkdir()
    (repo_root / "tools.json").write_text(
        json.dumps(SAMPLE_TOOLS_JSON), encoding="utf-8"
    )
    return repo_root


# ── ExternalTool unit tests ───────────────────────────────────────────


class TestExternalTool:
    def test_has_required_fields(self) -> None:
        tool = ExternalTool(
            name="Test Tool",
            category="Signal Processing",
            description="A test tool",
            launch_path="src/test_tool/launch_pyqt6.py",
            repo_root=Path("/some/repo"),
            repo_name="Tools",
        )
        assert tool.name == "Test Tool"
        assert tool.category == "Signal Processing"
        assert tool.description == "A test tool"
        assert tool.launch_path == "src/test_tool/launch_pyqt6.py"
        assert tool.repo_name == "Tools"

    def test_absolute_script_path(self, tmp_path: Path) -> None:
        tool = ExternalTool(
            name="Test",
            category="Cat",
            description="Desc",
            launch_path="src/tool/launch.py",
            repo_root=tmp_path,
            repo_name="Tools",
        )
        expected = tmp_path / "src" / "tool" / "launch.py"
        assert tool.absolute_script_path == expected

    def test_is_available_false_when_no_script(self, tmp_path: Path) -> None:
        tool = ExternalTool(
            name="Ghost",
            category="X",
            description="Y",
            launch_path="src/ghost/launch.py",
            repo_root=tmp_path,
            repo_name="Tools",
        )
        assert tool.is_available is False

    def test_is_available_true_when_script_exists(self, tmp_path: Path) -> None:
        script = tmp_path / "src" / "tool" / "launch.py"
        script.parent.mkdir(parents=True)
        script.write_text("# stub", encoding="utf-8")
        tool = ExternalTool(
            name="Real",
            category="X",
            description="Y",
            launch_path="src/tool/launch.py",
            repo_root=tmp_path,
            repo_name="Tools",
        )
        assert tool.is_available is True

    def test_requires_non_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ExternalTool(
                name="",
                category="X",
                description="Y",
                launch_path="src/t/l.py",
                repo_root=Path("/"),
                repo_name="Tools",
            )

    def test_requires_non_empty_category(self) -> None:
        with pytest.raises(ValueError, match="category"):
            ExternalTool(
                name="T",
                category="",
                description="Y",
                launch_path="src/t/l.py",
                repo_root=Path("/"),
                repo_name="Tools",
            )


# ── ToolsRepoBridge unit tests ────────────────────────────────────────


class TestToolsRepoBridge:
    def test_load_from_file(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="Tools")
        tools = bridge.load()
        assert len(tools) == 3

    def test_tools_have_correct_categories(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="Tools")
        tools = bridge.load()
        categories = {t.category for t in tools}
        assert "Signal Processing" in categories
        assert "Robotics" in categories

    def test_tools_have_non_empty_descriptions(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="Tools")
        for tool in bridge.load():
            assert tool.description, f"Tool '{tool.name}' missing description"

    def test_filter_by_category(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="Tools")
        tools = bridge.load(category_filter="Robotics")
        assert len(tools) == 1
        assert tools[0].name == "Humanoid Character Builder"

    def test_load_gracefully_handles_missing_file(self, tmp_path: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tmp_path, repo_name="Tools")
        tools = bridge.load()
        assert tools == []

    def test_load_gracefully_handles_malformed_json(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "tools.json"
        bad_json.write_text("{not valid json}", encoding="utf-8")
        bridge = ToolsRepoBridge(repo_root=tmp_path, repo_name="Tools")
        tools = bridge.load()
        assert tools == []

    def test_list_categories(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="Tools")
        cats = bridge.list_categories()
        assert sorted(cats) == sorted(["Signal Processing", "Robotics"])

    def test_repo_name_stored(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="MyRepo")
        tools = bridge.load()
        for tool in tools:
            assert tool.repo_name == "MyRepo"

    def test_load_idempotent(self, tools_json_file: Path) -> None:
        bridge = ToolsRepoBridge(repo_root=tools_json_file.parent, repo_name="Tools")
        first = bridge.load()
        second = bridge.load()
        assert [t.name for t in first] == [t.name for t in second]

    def test_requires_valid_repo_root(self) -> None:
        with pytest.raises(TypeError):
            ToolsRepoBridge(repo_root="not-a-path", repo_name="T")  # type: ignore[arg-type]


# ── load_tools_from_repo convenience function ─────────────────────────


class TestLoadToolsFromRepo:
    def test_loads_from_sibling_repo(self, mock_tools_repo: Path) -> None:
        tools = load_tools_from_repo(repo_root=mock_tools_repo)
        assert len(tools) == 3

    def test_returns_empty_for_missing_repo(self, tmp_path: Path) -> None:
        tools = load_tools_from_repo(repo_root=tmp_path / "NonExistent")
        assert tools == []

    def test_auto_discovers_sibling_tools_repo(self, tmp_path: Path) -> None:
        """load_tools_from_repo can auto-discover sibling Tools repo."""
        sibling_tools = tmp_path / "Tools"
        sibling_tools.mkdir()
        (sibling_tools / "tools.json").write_text(
            json.dumps(
                {
                    "Robotics": [
                        {"name": "T", "path": "p.py", "type": "python", "desc": "D"}
                    ]
                }
            ),
            encoding="utf-8",
        )
        # Pretend UpstreamDrift is at tmp_path / "UpstreamDrift"
        ud_root = tmp_path / "UpstreamDrift"
        ud_root.mkdir()
        tools = load_tools_from_repo(ud_root=ud_root)
        assert len(tools) == 1
        assert tools[0].name == "T"
