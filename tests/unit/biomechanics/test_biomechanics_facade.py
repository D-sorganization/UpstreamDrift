"""Tests for the lazy biomechanics facade (Issue #12397).

Verifies that importing the top-level biomechanics package or individual core
muscle kernels isolates imports, preventing eager loading of GUI/visualization
libraries, analysis pipelines, or unrelated optional modules. Also asserts
export identity, attribute error semantics, dir() introspection, __all__ parity,
sentinel-based optional groups, and exception propagation.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run Python snippet in a clean subprocess with repository on PYTHONPATH."""
    pythonpath = os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT)])
    env = {**os.environ, "PYTHONPATH": pythonpath}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )


# ============================================================================
# Fresh-process isolation tests (clean interpreter)
# ============================================================================


def test_top_level_package_import_isolates_unrelated_modules() -> None:
    """Importing src.shared.python.biomechanics must not load unrelated modules."""
    code = """
import sys
import src.shared.python.biomechanics

loaded = set(sys.modules.keys())
forbidden = [
    "matplotlib",
    "PyQt6",
    "src.shared.python.analysis",
    "src.shared.python.simulation_backends",
    "bunkershot3d",
    "src.shared.python.biomechanics.dynamic_com",
    "src.shared.python.biomechanics.grf_visualization",
    "src.shared.python.biomechanics.drift_control_transfer",
    "src.shared.python.biomechanics.kinematic_sequence",
]
found = [pkg for pkg in forbidden if any(m == pkg or m.startswith(pkg + ".") for m in loaded)]
if found:
    print("LEAKED:", found)
    sys.exit(1)
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (
        f"Unrelated modules leaked on import:\n{result.stdout}\n{result.stderr}"
    )


def test_kernel_import_and_instantiation_isolates_unrelated_modules() -> None:
    """Importing and using core Hill/activation kernels must not load unrelated modules."""
    code = """
import sys
from src.shared.python.biomechanics import ActivationDynamics, HillMuscleModel, MuscleParameters

params = MuscleParameters(F_max=100.0, l_opt=0.1, l_slack=0.2)
model = HillMuscleModel(params)
act = ActivationDynamics()

loaded = set(sys.modules.keys())
forbidden = [
    "matplotlib",
    "PyQt6",
    "src.shared.python.analysis",
    "src.shared.python.simulation_backends",
    "bunkershot3d",
    "src.shared.python.biomechanics.dynamic_com",
    "src.shared.python.biomechanics.grf_visualization",
    "src.shared.python.biomechanics.drift_control_transfer",
    "src.shared.python.biomechanics.kinematic_sequence",
]
found = [pkg for pkg in forbidden if any(m == pkg or m.startswith(pkg + ".") for m in loaded)]
if found:
    print("LEAKED:", found)
    sys.exit(1)
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (
        f"Unrelated modules leaked on kernel use:\n{result.stdout}\n{result.stderr}"
    )


def test_real_submodule_import_isolates_unrelated_modules() -> None:
    """Importing real submodules directly must not trigger eager package exports."""
    code = """
import sys
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel
from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics

loaded = set(sys.modules.keys())
forbidden = [
    "matplotlib",
    "PyQt6",
    "src.shared.python.analysis",
    "src.shared.python.simulation_backends",
    "bunkershot3d",
    "src.shared.python.biomechanics.dynamic_com",
    "src.shared.python.biomechanics.grf_visualization",
    "src.shared.python.biomechanics.drift_control_transfer",
    "src.shared.python.biomechanics.kinematic_sequence",
]
found = [pkg for pkg in forbidden if any(m == pkg or m.startswith(pkg + ".") for m in loaded)]
if found:
    print("LEAKED:", found)
    sys.exit(1)
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (
        f"Unrelated modules leaked on direct submodule import:\n{result.stdout}\n{result.stderr}"
    )


# ============================================================================
# Facade export identity and semantics
# ============================================================================


def test_unconditional_exports_identity_matches_source_submodules() -> None:
    """Every unconditional symbol exported by the facade must match its canonical submodule object."""
    import src.shared.python.biomechanics as bm

    unconditional = bm._UNCONDITIONAL_EXPORTS
    assert len(unconditional) >= 42

    for sym, mod_relpath in unconditional.items():
        submod = importlib.import_module(mod_relpath, "src.shared.python.biomechanics")
        expected_obj = getattr(submod, sym)
        facade_obj = getattr(bm, sym)
        assert facade_obj is expected_obj, (
            f"Symbol '{sym}' from '{mod_relpath}' does not match identity in facade"
        )


def test_repeated_access_returns_cached_identity() -> None:
    """Accessing an export multiple times returns the identical object."""
    import src.shared.python.biomechanics as bm

    first = bm.HillMuscleModel
    second = bm.HillMuscleModel
    assert first is second


def test_submodule_import_and_facade_interop_preserves_identity() -> None:
    """Direct submodule import followed by facade access preserves object identity."""
    from src.shared.python.biomechanics.muscle_equilibrium import (
        EquilibriumSolver as DirectSolver,
    )
    import src.shared.python.biomechanics as bm

    assert bm.EquilibriumSolver is DirectSolver


def test_unknown_attribute_raises_attribute_error() -> None:
    """Accessing an unknown attribute raises AttributeError with informative text."""
    import src.shared.python.biomechanics as bm

    with pytest.raises(
        AttributeError, match="has no attribute 'non_existent_symbol_123'"
    ):
        _ = bm.non_existent_symbol_123


def test_dir_exposes_documented_names_without_loading_optional_modules() -> None:
    """dir() on the package exposes all documented names without triggering optional imports."""
    code = """
import sys
import src.shared.python.biomechanics as bm

names = dir(bm)
assert "HillMuscleModel" in names
assert "BiomechanicalModel" in names
assert "plot_grf_and_com_3d" in names
assert "f_l" in names

# Verify optional modules were not loaded
loaded = set(sys.modules.keys())
forbidden = [
    "src.shared.python.biomechanics.dynamic_com",
    "src.shared.python.biomechanics.grf_visualization",
]
found = [pkg for pkg in forbidden if pkg in loaded]
if found:
    print("OPTIONAL MODULE LOADED BY DIR():", found)
    sys.exit(1)
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (
        f"dir() triggered optional module load:\n{result.stdout}\n{result.stderr}"
    )


def test_all_property_returns_ordered_mutable_list_and_matches_legacy_order() -> None:
    """__all__ returns a mutable list matching base exports followed by active optional symbols."""
    import src.shared.python.biomechanics as bm

    all_list = bm.__all__
    assert isinstance(all_list, list)

    # Base symbols appear first in sequence
    base_symbols = list(bm._BASE_ALL)
    assert all_list[: len(base_symbols)] == base_symbols

    # All items in __all__ are accessible via getattr
    for name in all_list:
        assert hasattr(bm, name)
        assert getattr(bm, name) is not None

    # Mutability: list can be modified by caller without breaking internal structure
    all_list.append("test_symbol")
    assert "test_symbol" in bm.__all__


def test_import_star_populates_all_symbols() -> None:
    """from src.shared.python.biomechanics import * imports all symbols in __all__."""
    code = """
import sys
test_globals = {}
exec("from src.shared.python.biomechanics import *", test_globals)

import src.shared.python.biomechanics as bm
for sym in bm.__all__:
    assert sym in test_globals, f"Missing {sym} from import *"
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, f"import * failed:\n{result.stdout}\n{result.stderr}"


def test_optional_import_error_sets_paired_symbols_to_none() -> None:
    """When an optional module raises ImportError, all paired symbols in the group resolve to None."""
    code = """
import sys
from unittest.mock import patch
import importlib

orig_import = importlib.import_module

def fake_import(name, package=None):
    if name == ".dynamic_com":
        raise ImportError("Mocked missing dynamic_com")
    return orig_import(name, package)

with patch("importlib.import_module", side_effect=fake_import):
    import src.shared.python.biomechanics as bm
    assert bm.BiomechanicalModel is None
    assert bm.SegmentDefinition is None
    assert "BiomechanicalModel" not in bm.__all__
    assert "SegmentDefinition" not in bm.__all__
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (
        f"Optional ImportError handling failed:\n{result.stdout}\n{result.stderr}"
    )


def test_non_import_error_propagates_without_swallowing() -> None:
    """Exceptions other than ImportError during optional import must propagate."""
    code = """
import sys
from unittest.mock import patch
import importlib

orig_import = importlib.import_module

def fake_import(name, package=None):
    if name == ".rust_muscle":
        raise RuntimeError("Non-ImportError defect in rust_muscle")
    return orig_import(name, package)

with patch("importlib.import_module", side_effect=fake_import):
    import src.shared.python.biomechanics as bm
    try:
        _ = bm.f_l
        sys.exit(2)
    except RuntimeError as e:
        if "Non-ImportError defect" in str(e):
            sys.exit(0)
        sys.exit(3)
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (
        f"Non-ImportError defect was not propagated properly:\n{result.stdout}\n{result.stderr}"
    )
