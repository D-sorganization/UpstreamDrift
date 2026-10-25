from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_pinned_tools_exposes_public_fts_sanitizer() -> None:
    repository = Path(__file__).resolve().parents[3]
    tools_source = repository / "vendor" / "ud-tools" / "src" / "shared" / "python"
    script = """
from pathlib import Path
import sys
import ai.knowledge.pack as pack
root = Path(sys.argv[1]).resolve()
module = Path(pack.__file__).resolve()
assert root in module.parents, (module, root)
assert pack.sanitize_fts_query('quasar OR (diagnostic)') == '\"quasar\" OR \"OR\" OR \"diagnostic\"'
"""
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(tools_source) + (
        os.pathsep + inherited if inherited else ""
    )
    subprocess.run(
        [sys.executable, "-c", script, str(tools_source)],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
