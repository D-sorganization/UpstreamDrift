"""Every studio script parses as a classic browser script (#14141 step 8).

The studio opens from ``file://``, so each ``js/*.js`` file must be a plain
script, not a module. ``node --check`` catches a syntax error in any slice
the moment it lands; the headless smoke run in step 8l loads them together.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_JS = Path(__file__).resolve().parents[3] / "src/apps/putting_studio/js"
_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
@pytest.mark.parametrize("script", sorted(_JS.glob("*.js")), ids=lambda p: p.name)
def test_script_parses_without_module_syntax(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert "\nimport " not in source and "\nexport " not in source
    run = subprocess.run(
        [str(_NODE), "--check", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert run.returncode == 0, run.stderr
