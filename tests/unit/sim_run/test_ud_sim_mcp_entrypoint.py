"""Packaging checks for the ``ud-sim-mcp`` console script (#15020)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


@pytest.mark.unit
def test_ud_sim_mcp_console_script_declared(pyproject: dict) -> None:
    scripts = pyproject.get("project", {}).get("scripts", {})
    assert scripts.get("ud-sim-mcp") == "src.sim_run.mcp:main"


@pytest.mark.unit
def test_ud_sim_mcp_main_callable() -> None:
    from src.sim_run.mcp import main

    assert callable(main)
