import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.physics.rust_kernel import is_rust_available

# Import paths configured at test runner level via pyproject.toml/conftest.py
project_root = get_repo_root()


# Fix import names since they start with numbers
# Actually direct import of 01... is invalid syntax.
# We need importlib or renaming.
# Let's verify importability via importlib.


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Could not load spec for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is None:
        raise ImportError(f"No loader found for {name}")
    spec.loader.exec_module(module)
    return module


example01_path = project_root / "examples" / "01_basic_simulation.py"
example02_path = project_root / "examples" / "02_parameter_sweeps.py"
example03_path = project_root / "examples" / "03_injury_risk_tutorial.py"
aerodynamics_path = project_root / "examples" / "aerodynamics_demo.py"
flight_path = project_root / "examples" / "basic_flight_simulation.py"
topography_path = project_root / "examples" / "topography_demo.py"
motion_training_path = project_root / "examples" / "motion_training_demo.py"


def test_example_01_runs() -> None:
    """Test Example 01 runs without error (mocked)."""
    # Mock engine manager to simulate missing engine and return False
    with patch(
        "src.shared.python.engine_core.engine_manager.EngineManager"
    ) as MockManager:
        instance = MockManager.return_value
        instance.switch_engine.return_value = False

        # Load and run
        mod = load_module("ex01", example01_path)
        mod.main()

        # Verify it handled missing engine gracefully
        assert instance.switch_engine.called


def test_example_02_runs() -> None:
    """Test Example 02 runs without error."""
    with patch("src.shared.python.data_io.output_manager.OutputManager") as MockOutput:
        mod = load_module("ex02", example02_path)
        mod.main()

        assert MockOutput.return_value.create_output_structure.called
        assert MockOutput.return_value.save_simulation_results.called


def test_example_03_runs() -> None:
    """Test Example 03 (injury risk tutorial) runs without error."""
    mod = load_module("ex03", example03_path)
    mod.run_tutorial()


def test_aerodynamics_demo_runs() -> None:
    """Test aerodynamics_demo.py runs without error."""
    mod = load_module("aerodynamics_demo", aerodynamics_path)
    mod.main()


@pytest.mark.skipif(
    not is_rust_available(),
    reason="upstream-physics Rust kernel not available — basic_flight_simulation requires it",
)
def test_basic_flight_simulation_runs() -> None:
    """Test basic_flight_simulation.py runs without error (requires Rust kernel)."""
    mod = load_module("basic_flight_simulation", flight_path)
    mod.main()


def test_topography_demo_runs() -> None:
    """Test topography_demo.py runs without error."""
    mod = load_module("topography_demo", topography_path)
    mod.main()


def test_motion_training_demo_importable() -> None:
    """Test motion_training_demo.py can be imported (lazy imports inside functions).

    The script inserts the pinocchio/python directory into sys.path at module
    level.  We save/restore sys.path so that worker-level state is not polluted,
    which would cause other tests (e.g. dtack backend tests) to change behaviour.
    """
    orig_path = sys.path.copy()
    try:
        mod = load_module("motion_training_demo", motion_training_path)
        # Verify key entry points are present
        assert callable(mod.main)
        assert callable(mod.run_ik_demo)
        assert callable(mod.run_trajectory_analysis)
    finally:
        sys.path[:] = orig_path
