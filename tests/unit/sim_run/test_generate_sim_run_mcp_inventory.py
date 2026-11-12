"""Freshness checks for generated ``ud-sim-mcp`` capability inventory (#15020)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def test_sim_run_mcp_capability_inventory_is_current() -> None:
    """The committed inventory must match the live service probe."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.generate_sim_run_mcp_inventory", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout or proc.stderr
