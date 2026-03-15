from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def _assert_from_tools(path: Path) -> None:
    normalized = _normalized_path(str(path))
    assert any(
        marker in normalized
        for marker in (
            "/_tools_dep/",
            "/vendor/ud-tools/",
            "/repositories/tools/",
            "/tools/",
        )
    ), f"Expected Tools-backed provider path, got: {path}"


def test_signal_toolkit_imports_resolve_from_tools_provider() -> None:
    module = importlib.import_module("signal_toolkit")
    _assert_from_tools(Path(module.__file__).resolve())

    signal = module.SignalGenerator.sinusoid(
        np.linspace(0.0, 1.0, 16), amplitude=1.0, frequency=2.0
    )
    assert len(signal.values) == 16


def test_humanoid_character_builder_imports_resolve_from_tools_provider() -> None:
    module = importlib.import_module("humanoid_character_builder")
    _assert_from_tools(Path(module.__file__).resolve())

    params = module.BodyParameters(height_m=1.75, mass_kg=72.0)
    assert params.height_m == 1.75
    assert params.mass_kg == 72.0


def test_model_generation_imports_resolve_from_tools_provider() -> None:
    module = importlib.import_module("model_generation")
    _assert_from_tools(Path(module.__file__).resolve())

    assert callable(module.quick_urdf)
    assert module.DEFAULT_HEIGHT_M > 0


def test_upstream_drift_tools_imports_resolve_from_tools_provider() -> None:
    module = importlib.import_module("upstream_drift_tools")
    _assert_from_tools(Path(module.__file__).resolve())

    state_manager = importlib.import_module("upstream_drift_tools.utils.state_manager")
    _assert_from_tools(Path(state_manager.__file__).resolve())
    assert hasattr(state_manager, "StateManager")
