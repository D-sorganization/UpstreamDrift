"""The same-input facade must not load unrelated optional engine code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.unit
def test_package_and_closure_import_do_not_load_optional_or_unrelated_modules() -> None:
    script = r"""
import importlib.abc
import sys

blocked = {
    "cv2",
    "src.shared.python.motion_matching.historical_fit",
    "src.shared.python.motion_matching.pipeline",
    "src.shared.python.motion_matching.visualization",
    "src.shared.python.motion_matching.same_input.reference",
    "src.shared.python.motion_matching.same_input.integrator",
}

class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(f"unexpected eager import: {fullname}")
        return None

sys.meta_path.insert(0, BlockImports())
import src.shared.python.motion_matching.same_input as facade
assert "project_to_closure" in dir(facade)
from src.shared.python.motion_matching.same_input.closure import project_to_closure
assert "reference" not in sys.modules
assert "src.shared.python.motion_matching.same_input.integrator" not in sys.modules
assert project_to_closure is facade.project_to_closure
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.unit
def test_lazy_facade_preserves_all_public_object_identities_and_unknown_attribute() -> (
    None
):
    script = r"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
import src.shared.python.motion_matching.same_input as facade

providers = {
    "SCHEMA": "bundle", "InputBundle": "bundle",
    "ClosureProjection": "closure", "project_to_closure": "closure",
    "ROOT_COORDINATES": "integrator", "Rollout": "integrator",
    "StepPolicy": "integrator", "integrate": "integrator",
    "open_loop": "integrator", "zoh_rk4_step": "integrator",
    "ALL_ENGINES": "plant", "OPTIONAL_ENGINES": "plant",
    "PARITY_ENGINES": "plant", "VectorPlant": "plant",
    "closed_loop": "reference", "generate_reference_bundle": "reference",
    "ReplayScore": "scoring", "growth_rate": "scoring",
    "score_replay": "scoring", "segmented_replay": "scoring",
}
assert set(facade.__all__) == set(providers)
assert set(facade.__all__).issubset(dir(facade))
for name, module_name in providers.items():
    provider = importlib.import_module(
        f"src.shared.python.motion_matching.same_input.{module_name}"
    )
    assert getattr(facade, name) is getattr(provider, name)
try:
    facade.unknown_public_name
except AttributeError:
    pass
else:
    raise AssertionError("unknown package attributes must raise AttributeError")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
