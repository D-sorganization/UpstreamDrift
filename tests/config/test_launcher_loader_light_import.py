"""``src.config`` must import without pydantic-settings (#15150, G13).

The capability atlas CI job imports ``src.config`` in a light environment
that does not install pydantic-settings; the settings module is loaded lazily.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = """
import importlib.abc, sys

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pydantic_settings" or name.startswith("pydantic_settings."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _Block())
import src.config  # noqa: F401
print("ok")
"""


def test_src_config_imports_without_pydantic_settings() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
