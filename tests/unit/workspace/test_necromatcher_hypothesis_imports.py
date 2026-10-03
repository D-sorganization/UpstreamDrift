"""The authored hypothesis facade remains available without optional native SDKs."""

from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_public_hypothesis_import_does_not_load_native_or_gui_sdk() -> None:
    from src.shared.python.core import repo_python_environment

    root = Path(__file__).resolve().parents[3]
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'PyQt6', 'PySide6', 'pinocchio', 'pydrake'}:
            raise ImportError('native/GUI SDK blocked: ' + fullname)
sys.meta_path.insert(0, Block())
from src.shared.python.workspace import (
    NativeHypothesisRequest, NativeModelBinding, bind_native_hypothesis,
    author_native_hypothesis, load_native_model_binding,
)
assert NativeHypothesisRequest and NativeModelBinding
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=repo_python_environment(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
