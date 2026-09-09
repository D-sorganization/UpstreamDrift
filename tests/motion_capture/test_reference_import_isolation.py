"""Native reference startup must not pull unrelated optional format backends."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_body_json_import_does_not_import_legacy_c3d_loader() -> None:
    code = """
import importlib.abc, sys
class BlockLegacyC3D(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.shared.python.motion_matching.loaders.c3d":
            raise ImportError("unrelated optional C3D dependency was imported")
sys.meta_path.insert(0, BlockLegacyC3D())
from src.shared.python.motion_matching.loaders.body_json import load_body_target_json
assert callable(load_body_target_json)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
