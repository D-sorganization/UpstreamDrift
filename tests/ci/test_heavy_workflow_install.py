"""Heavy-lane install contract (#13017).

The 2026-10-18 heavy run failed every lane for two environmental reasons:

* the workflow ran ``pip install pinocchio pink``, which are unrelated PyPI
  projects (0.4.3 / 0.8.1) -- the robotics libraries are ``pin`` / ``pin-pink``
  -- so ``pinocchio.Model`` / ``buildModelFromUrdf`` did not exist;
* lanes installed into the runner's shared toolcache Python concurrently, so
  differing ``platformdirs`` versions corrupted each other's entry points
  (``No module named 'platformdirs.pytest_plugin'``).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "heavy-tests-opt-in.yml"
DOCKERFILE = ROOT / "Dockerfile.heavy_test"

sys.path.insert(0, str(ROOT / "scripts"))
import check_heavy_dep_parity as parity  # noqa: E402


def _pip_packages(path: Path) -> set[str]:
    return parity._extract_pip_packages(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", [WORKFLOW, DOCKERFILE], ids=lambda p: p.name)
def test_robotics_packages_use_real_pypi_names(path: Path) -> None:
    pkgs = _pip_packages(path)
    assert "pin" in pkgs
    assert "pin-pink" in pkgs
    assert "pinocchio" not in pkgs, "PyPI 'pinocchio' is not the robotics library"
    assert "pink" not in pkgs, "PyPI 'pink' is not pin-pink"


def test_parity_script_canonical_names_are_real() -> None:
    assert {"pin", "pin-pink"} <= parity.CANONICAL_HEAVY_DEPS
    assert not {"pinocchio", "pink"} & parity.CANONICAL_HEAVY_DEPS


def test_workflow_installs_into_isolated_venv_before_pip() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    venv = re.search(r"python -m venv", text)
    first_pip = re.search(r"pip install --upgrade pip", text)
    assert venv and first_pip and venv.start() < first_pip.start()
    assert "GITHUB_PATH" in text
