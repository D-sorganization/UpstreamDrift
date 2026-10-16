"""Launcher startup must not import ezc3d (#12883).

The Python ``ezc3d`` extension and the native OpenSim SDK ship ABI-incompatible
``ezc3d.dll`` builds on Windows; importing ezc3d at startup makes OpenSim fail
to load for the whole session.  A stub ``ezc3d`` package that records its import
makes this deterministic whether or not the real wheel is installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE = """
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, {repo!r})
from src.launchers import embedded_tool_bootstrap as b
b.bootstrap_embeddable_tools()
from src.shared.python.motion_pipeline.sources.c3d_adapter import C3DAdapter
from src.shared.python.motion_matching.loaders import c3d
print("EZC3D_LOADED=" + str("ezc3d" in sys.modules))
"""


@pytest.mark.unit
def test_bootstrap_does_not_import_ezc3d(tmp_path: Path) -> None:
    stub = tmp_path / "stubs" / "ezc3d"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("c3d = None\n", encoding="utf-8")
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(tmp_path / "stubs"),
        "QT_QPA_PLATFORM": "offscreen",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO_ROOT))],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert "EZC3D_LOADED=False" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-2000:]
    )
