"""Issue #12901: Startup import profiling and deferred heavy imports.

The built-in API server took ~90 s to become ready (and ~27 s on cold import)
due to eager heavy dependencies (bunkershot3d, trimesh/character builder,
and loop_nullspace facade) imported during route discovery and config loading.
These unit tests assert that those heavy packages remain deferred until their
respective endpoints or factories are actually exercised.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_python_code(
    code: str, *, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "QT_QPA_PLATFORM": "offscreen",
        "MPLBACKEND": "Agg",
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.unit
def test_config_import_defers_bunkershot3d() -> None:
    """Importing api.config must not eagerly load bunkershot3d."""
    code = (
        "import sys\n"
        "import src.api.config\n"
        "assert 'bunkershot3d' not in sys.modules, "
        "f'bunkershot3d was eagerly imported by api.config'\n"
    )
    proc = _run_python_code(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.unit
def test_character_appearance_route_defers_trimesh() -> None:
    """Importing character_appearance route must not eagerly import trimesh."""
    code = (
        "import sys\n"
        "import src.api.routes.character_appearance\n"
        "assert 'trimesh' not in sys.modules, "
        "f'trimesh was eagerly imported by character_appearance route'\n"
    )
    proc = _run_python_code(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.unit
def test_loop_nullspace_route_defers_facade() -> None:
    """Importing analysis_loop_nullspace route must not eagerly import loop_nullspace.facade."""
    code = (
        "import sys\n"
        "import src.api.routes.analysis_loop_nullspace\n"
        "assert 'src.shared.python.loop_nullspace.facade' not in sys.modules, "
        "f'loop_nullspace.facade was eagerly imported by analysis_loop_nullspace route'\n"
    )
    proc = _run_python_code(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.unit
def test_server_startup_time_under_budget() -> None:
    """Server import and lifespan startup must complete in under 15 seconds (issue #12901)."""
    code = (
        "import time, asyncio, sys\n"
        "t0 = time.perf_counter()\n"
        "import src.api.server as s\n"
        "async def main():\n"
        "    async with s.lifespan(s.app):\n"
        "        pass\n"
        "asyncio.run(main())\n"
        "elapsed = time.perf_counter() - t0\n"
        "print(f'READY_TIME={elapsed:.2f}')\n"
        "assert elapsed < 15.0, f'Server startup exceeded 15 s budget: {elapsed:.2f} s'\n"
    )
    proc = _run_python_code(code, timeout=45.0)
    assert proc.returncode == 0, proc.stderr or proc.stdout
