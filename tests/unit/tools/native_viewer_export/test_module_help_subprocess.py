"""Subprocess smoke: ``python -m src.tools.native_viewer_export --help`` (#14748)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_module_help_exits_zero_without_pytest_path_hacks() -> None:
    """CLI help must work from a clean interpreter (no tests/conftest path setup)."""
    pytest.importorskip("defusedxml")
    repo_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.pop("TOOLS_REPO_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-m", "src.tools.native_viewer_export", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "usage:" in out.lower()
