from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from src.shared.python.ai.sample_tools import register_golf_suite_tools
from src.shared.python.ai.tool_registry import ToolRegistry


def test_register_golf_suite_tools():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)
    assert len(reg) > 0
    assert "list_sample_files" in reg
    assert "load_c3d" in reg
    assert "get_marker_info" in reg
    assert "run_inverse_dynamics" in reg
    assert "interpret_torques" in reg
    assert "explain_concept" in reg
    assert "list_glossary_terms" in reg
    assert "search_glossary" in reg
    assert "validate_cross_engine" in reg
    assert "check_energy_conservation" in reg
    assert "list_physics_engines" in reg


def test_list_sample_files(tmp_path: Path):
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    with patch("src.shared.python.ai.sample_tools.Path") as MockPath:
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        MockPath.return_value = mock_path_obj

        res = reg.execute("list_sample_files", {})
        assert res.success is True
        assert len(res.result["files"]) == 0
        assert "No sample data" in res.result["message"]

        # Test with files
        mock_file = Mock()
        mock_file.stem = "test_file"
        mock_stat = Mock()
        mock_stat.st_size = 2048
        mock_file.stat.return_value = mock_stat
        mock_path_obj.exists.return_value = True
        mock_path_obj.glob.return_value = [mock_file]

        res2 = reg.execute("list_sample_files", {})
        assert res2.success is True
        assert len(res2.result["files"]) == 1
        assert res2.result["files"][0]["name"] == "test_file"
        assert res2.result["files"][0]["size_kb"] == 2


def test_load_c3d():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    # Missing file
    res = reg.execute("load_c3d", {"file_path": "missing.c3d"})
    assert res.success is True  # It returns a structured dict
    assert res.result["success"] is False
    assert "File not found" in res.result["error"]

    with patch("src.shared.python.ai.sample_tools.Path") as MockPath:
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.suffix = ".txt"
        MockPath.return_value = mock_path_obj

        res_suffix = reg.execute("load_c3d", {"file_path": "test.txt"})
        assert res_suffix.result["success"] is False
        assert "must be a .c3d file" in res_suffix.result["error"]


def test_get_marker_info():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    with patch("src.shared.python.ai.sample_tools.Path") as MockPath:
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        MockPath.return_value = mock_path_obj

        res = reg.execute("get_marker_info", {"file_path": "missing.c3d"})
        assert res.success is True
        assert res.result["success"] is False


def test_run_inverse_dynamics():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute(
        "run_inverse_dynamics", {"file_path": "test.c3d", "engine": "mujoco"}
    )
    assert res.success is True
    assert res.result["success"] is True
    assert res.result["engine"] == "mujoco"

    res_bad = reg.execute(
        "run_inverse_dynamics", {"file_path": "test.c3d", "engine": "bad_engine"}
    )
    assert res_bad.result["success"] is False


def test_interpret_torques():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute(
        "interpret_torques",
        {"shoulder_torque": 20, "hip_torque": 100, "wrist_torque": 100},
    )
    assert res.success is True
    assert "Below typical" in res.result["shoulder"]["classification"]
    assert "Within typical range" in res.result["hip"]["classification"]
    assert "Above typical" in res.result["wrist"]["classification"]


def test_explain_concept():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute(
        "explain_concept", {"term": "inverse_dynamics", "expertise_level": 3}
    )
    assert res.success is True
    assert res.result["term"] == "inverse_dynamics"
    assert "M(q)q" in res.result["explanation"]


def test_list_glossary_terms():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute("list_glossary_terms", {"category": "golf"})
    assert res.success is True
    assert "kinetic_chain" in res.result["terms"]


def test_search_glossary():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute("search_glossary", {"query": "pinocchio"})
    assert res.success is True
    assert any(r["term"] == "Pinocchio" for r in res.result["results"])


def test_validate_cross_engine():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute("validate_cross_engine", {"file_path": "test.c3d"})
    assert res.success is True
    assert res.result["status"] == "validation_pending"


def test_check_energy_conservation():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    res = reg.execute("check_energy_conservation", {})
    assert res.success is True
    assert res.result["status"] == "check_pending"


def test_list_physics_engines():
    reg = ToolRegistry()
    register_golf_suite_tools(reg)

    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: True if name == "mujoco" else None

        res = reg.execute("list_physics_engines", {})
        assert res.success is True
        assert res.result["available_count"] == 1
        engines = {e["name"]: e["status"] for e in res.result["engines"]}
        assert engines["MuJoCo"] == "available"
        assert engines["Drake"] == "not installed"
