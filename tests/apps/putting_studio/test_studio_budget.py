"""Offline size and startup budget of the standalone studio (#14141 step 8l).

The budgets were set from measurement on 2026-11-01: an 835 KB bundle and a
163 ms headless script load. They leave room for growth and slow CI runners;
SUNSET.md records them as the bar a generated or bridged core must meet.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_STUDIO = Path(__file__).resolve().parents[3] / "src/apps/putting_studio"
_SMOKE = Path(__file__).with_name("studio_smoke.js")
_NODE = shutil.which("node")
BUNDLE_BUDGET_BYTES = 1_000_000
LOAD_BUDGET_MS = 2_000.0
_NETWORK = re.compile(r"https?://|fetch\(|XMLHttpRequest|WebSocket|import\(")


def _bundle() -> list[Path]:
    globs = ("index.html", "css/*.css", "js/*.js", "vendor/*.js", "vendor/*.css")
    return [path for pattern in globs for path in sorted(_STUDIO.glob(pattern))]


def test_offline_bundle_fits_its_size_budget() -> None:
    total = sum(path.stat().st_size for path in _bundle())
    assert 0 < total <= BUNDLE_BUDGET_BYTES, f"studio bundle is {total} bytes"


def test_studio_code_makes_no_network_access() -> None:
    own = [p for p in _bundle() if p.parent.name != "vendor"]
    for path in own:
        assert not _NETWORK.search(path.read_text(encoding="utf-8")), path.name


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_every_script_loads_headless_within_the_startup_budget() -> None:
    run = subprocess.run(
        [str(_NODE), str(_SMOKE), str(_STUDIO)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    load = re.search(r"SMOKE_LOAD_MS=([0-9.]+)", run.stdout)
    assert load is not None
    assert float(load.group(1)) <= LOAD_BUDGET_MS
