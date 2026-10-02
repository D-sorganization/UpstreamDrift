"""Clean worker launch environments preserve settings without parent mutation."""

import os
import pytest

from src.shared.python.core import repo_python_environment

pytestmark = pytest.mark.unit


def test_repo_environment_prioritizes_src_without_mutating_base(tmp_path):
    src = str(tmp_path / "src")
    base = {"PYTHONPATH": os.pathsep.join(["external", src, src]), "MUJOCO_GL": "egl"}
    original = dict(base)
    environment = repo_python_environment(tmp_path, base)
    assert environment["PYTHONPATH"].split(os.pathsep) == [src, "external"]
    assert environment["MUJOCO_GL"] == "egl"
    assert "MUJOCO_PLUGIN_PATH" not in environment
    assert repo_python_environment(tmp_path, environment) == environment
    assert base == original


def test_repo_environment_works_without_pythonpath(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert repo_python_environment(tmp_path)["PYTHONPATH"] == str(tmp_path / "src")
