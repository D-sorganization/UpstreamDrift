"""NSX-43: the adapter stays lazy and the pure modules stay Qt-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "src" / "tools" / "grip_wrench_plots"


def test_adapter_import_does_not_load_pyqt() -> None:
    code = (
        "import sys; "
        "import src.tools.grip_wrench_plots._embed_adapter; "
        "bad = [m for m in sys.modules if m.startswith('PyQt6')]; "
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.parametrize("name", ["nsx_view.py", "nsx_job.py", "nsx_draw.py"])
def test_pure_modules_do_not_import_pyqt(name: str) -> None:
    text = (PACKAGE / name).read_text(encoding="utf-8")
    assert "PyQt6" not in text.replace("PyQt, matplotlib", "")
    assert "import PyQt" not in text
