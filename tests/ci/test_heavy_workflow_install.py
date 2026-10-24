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

import importlib.util
from pathlib import Path
import re
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "heavy-tests-opt-in.yml"
DOCKERFILE = ROOT / "Dockerfile.heavy_test"


def _load_parity() -> ModuleType:
    """Load the script by file path.

    Never put ``scripts/`` on ``sys.path``: it contains ``opensim`` and
    ``motion_capture`` packages that would shadow the real ones for every
    later test in the same process.
    """
    spec = importlib.util.spec_from_file_location(
        "_check_heavy_dep_parity", ROOT / "scripts" / "check_heavy_dep_parity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = _load_parity()


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


def test_heavy_lane_propagates_pytest_exit_status() -> None:
    run_step = next(
        step
        for step in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
            "run-heavy-tests"
        ]["steps"]
        if "Run heavy" in step.get("name", "")
    )
    assert "set -o pipefail" in run_step["run"]


def test_public_repo_routes_heavy_lanes_to_hosted_runner() -> None:
    step = next(
        step
        for step in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
            "pick-runner"
        ]["steps"]
        if step.get("id") == "check"
    )
    assert "ubuntu-latest" in step["run"]
