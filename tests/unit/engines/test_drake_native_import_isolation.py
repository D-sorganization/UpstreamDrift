"""Ordinary imports of the native Drake adapter stay SDK-lazy."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_native_replay_import_does_not_load_optional_engine_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import importlib
import sys
from importlib.abc import MetaPathFinder

class ForbiddenOptionalDependency(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "pydrake" or fullname.startswith("pydrake."):
            raise AssertionError("native replay import attempted to load pydrake")
        if fullname == "bunkershot3d" or fullname.startswith("bunkershot3d."):
            raise AssertionError("native replay import attempted to load bunkershot3d")
        return None

sys.meta_path.insert(0, ForbiddenOptionalDependency())
module = importlib.import_module(
    "src.engines.physics_engines.drake.python.native_torque_replay"
)
drake = importlib.import_module("src.engines.physics_engines.drake")
python = importlib.import_module("src.engines.physics_engines.drake.python")
assert module.__name__.endswith("native_torque_replay")
assert "Engine" in drake.__all__
assert "simulate_with_coefficients" in python.__all__
assert not any(name == "pydrake" or name.startswith("pydrake.") for name in sys.modules)
assert not any(
    name == "bunkershot3d" or name.startswith("bunkershot3d.")
    for name in sys.modules
)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
