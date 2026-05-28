from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_conftest_import(module_name: str) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        [
            "import importlib",
            "import pathlib",
            "import sys",
            f"repo = pathlib.Path(r'{REPO_ROOT}')",
            "repo_str = str(repo)",
            "if repo_str not in sys.path:",
            "    sys.path.insert(0, repo_str)",
            f"importlib.import_module('{module_name}')",
            "import src.shared.python._contracts_primitives",
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_simscape_three_d_gui_conftest_keeps_src_shared_importable() -> None:
    result = _run_conftest_import("tests.unit.engines.simscape.three_d_gui.conftest")
    assert result.returncode == 0, result.stderr or result.stdout


def test_c3d_viewer_ui_conftest_keeps_src_shared_importable() -> None:
    result = _run_conftest_import("tests.unit.c3d_viewer.ui.conftest")
    assert result.returncode == 0, result.stderr or result.stdout
