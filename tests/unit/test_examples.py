import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.shared.python.path_utils import get_repo_root

# Add project root to path
project_root = get_repo_root()
sys.path.append(str(project_root))


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


def test_example_01_runs() -> None:
    """Test Example 01 runs without error (mocked)."""
    # Mock engine manager to simulate missing engine and return False
    with patch("shared.python.engine_manager.EngineManager") as MockManager:
        instance = MockManager.return_value
        instance.switch_engine.return_value = False

        # Load and run
        mod = load_module("ex01", example01_path)
        mod.main()

        # Verify it handled missing engine gracefully
        assert instance.switch_engine.called


def test_example_02_runs() -> None:
    """Test Example 02 runs without error."""
    # This example requires registry constants
    # Mock output manager to avoid disk writes in main logic?
    # Actually it writes to project_root/output by default.
    # We should mock OutputManager to prevent clutter or use temp dir.

    with patch("shared.python.output_manager.OutputManager") as MockOutput:
        mod = load_module("ex02", example02_path)
        mod.main()

        assert MockOutput.return_value.create_output_structure.called
        assert MockOutput.return_value.save_simulation_results.called
