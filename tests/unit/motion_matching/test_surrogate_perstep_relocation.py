"""Verify issue #4044 relocation of the per-step surrogate.

The implementation moved from
``MachineLearning/{train_dynamics_surrogate,optimize_torque_sequence_for_club,
extract_dynamics_dataset}.py`` to
``src.shared.python.motion_matching.surrogate.perstep.{train,optimize,
extract_dataset}``. Backwards-compatible shims at the old paths re-export the
public symbols and emit a ``DeprecationWarning`` on import.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import warnings
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
)


@pytest.fixture(autouse=True)
def _ensure_machinelearning_on_syspath() -> None:
    """The shims live in MachineLearning/ — that directory must be importable."""
    path_str = str(ML_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    # Ensure a fresh import each test so the warning fires reliably.
    for name in (
        "train_dynamics_surrogate",
        "optimize_torque_sequence_for_club",
        "extract_dynamics_dataset",
    ):
        sys.modules.pop(name, None)


def _import_with_warning(name: str) -> tuple[object, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    return module, list(caught)


@pytest.mark.parametrize(
    "old_name",
    [
        "train_dynamics_surrogate",
        "optimize_torque_sequence_for_club",
        "extract_dynamics_dataset",
    ],
)
def test_old_paths_still_importable_with_warning(old_name: str) -> None:
    """The old ``MachineLearning/`` paths still import and emit a deprecation."""
    pytest.importorskip("torch") if old_name != "extract_dynamics_dataset" else None
    module, caught = _import_with_warning(old_name)
    assert module is not None
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, (
        f"Expected DeprecationWarning when importing {old_name}; got {caught!r}"
    )
    text = " ".join(str(w.message) for w in deprecations)
    assert "perstep" in text and "#4044" in text


def test_new_path_imports() -> None:
    """The relocated package and its submodules import cleanly from the new path."""
    pytest.importorskip("torch")
    pkg = importlib.import_module("src.shared.python.motion_matching.surrogate.perstep")
    assert hasattr(pkg, "DynamicsMLP")
    assert hasattr(pkg, "TrainConfig")
    assert hasattr(pkg, "train_dynamics_surrogate")
    assert hasattr(pkg, "optimize_torque_sequence")
    assert hasattr(pkg, "extract_dataset")

    train = importlib.import_module(
        "src.shared.python.motion_matching.surrogate.perstep.train"
    )
    assert hasattr(train, "DynamicsMLP")
    assert hasattr(train, "TrainConfig")
    assert hasattr(train, "main")


def test_new_path_apis_match_old() -> None:
    """The shim re-exports must expose identical objects to the new package."""
    pytest.importorskip("torch")
    new_train = importlib.import_module(
        "src.shared.python.motion_matching.surrogate.perstep.train"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_train = importlib.import_module("train_dynamics_surrogate")

    # The DynamicsMLP class object must be the same across both import paths.
    assert old_train.DynamicsMLP is new_train.DynamicsMLP
    assert old_train.TrainConfig is new_train.TrainConfig
    # The CLI entry-point must exist on both and have the same signature.
    assert inspect.signature(old_train.main) == inspect.signature(new_train.main)
