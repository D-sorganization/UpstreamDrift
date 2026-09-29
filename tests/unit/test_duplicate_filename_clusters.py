"""Regression coverage for intentional duplicate filename clusters."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


def test_code_quality_check_wrappers_delegate_to_shared_implementation() -> None:
    wrapper_paths = [
        "src/engines/physics_engines/drake/tools/code_quality_check.py",
        "src/engines/physics_engines/mujoco/tools/code_quality_check.py",
        "src/engines/physics_engines/pinocchio/tools/code_quality_check.py",
        "src/engines/Simscape_Multibody_Models/2D_Golf_Model/tools/code_quality_check.py",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/tools/code_quality_check.py",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/scripts/quality-check.py",
    ]

    for relative_path in wrapper_paths:
        content = _read_text(relative_path)
        assert "from src.tools.code_quality_check import main" in content
        assert 'if __name__ == "__main__":' in content
        assert "BANNED_PATTERNS" not in content
        assert "PASS_PATTERNS" not in content
        assert "MAGIC_NUMBERS" not in content
        assert "check_banned_patterns" not in content
        assert "check_magic_numbers" not in content
        assert "check_ast_issues" not in content
        assert content.count("def ") == 0
        assert content.count("main()") == 1


def test_matlab_codeissuesgui_copies_stay_identical() -> None:
    gui_paths = [
        "src/engines/pendulum_models/tools/matlab_code_analyzer_gui/codeIssuesGUI.m",
        "src/engines/physics_engines/drake/tools/matlab_code_analyzer_gui/codeIssuesGUI.m",
        "src/engines/physics_engines/pinocchio/tools/matlab_code_analyzer_gui/codeIssuesGUI.m",
        "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/code_analysis_gui/codeIssuesGUI.m",
    ]

    digests = {
        relative_path: sha256(_read_text(relative_path).encode("utf-8")).hexdigest()
        for relative_path in gui_paths
    }

    assert len(set(digests.values())) == 1, digests
