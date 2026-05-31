"""
Python-specific shared utilities and libraries.
This package contains reusable Python logic for tools.

Available packages:
    - chat: Portable AI chat dock widget and Pydantic models
    - notes: Project-backed notes workspace with recycle-bin semantics
    - theme: Fleet-wide color theme management for PyQt6 applications
    - sidekick: Process engineering calculators
    - signal_toolkit: Signal processing and analysis
    - humanoid_character_builder: anthropometric domain layer for humanoid generation
    - model_generation: canonical generic URDF/MJCF toolkit and conversion surface
    - estimation: canonical-core MAP estimation helpers

Preferred imports (direct from package, since src/shared/python is on sys.path):
    from shared.python.theme import ThemeManager, get_theme_manager  # theme: keep prefix
    from humanoid_character_builder import CharacterBuilder, BodyParameters
    from model_generation import quick_urdf, ManualBuilder, FrankensteinEditor
    from signal_toolkit import Signal, SignalGenerator, FunctionFitter
    from sidekick.process_calculators import FlareCalculator
    from gui_launcher import GUIType, LaunchConfig, register_gui
    from plot_engine.specs import PlotSpec, SeriesData
    from plot_theme import apply_plot_theme
"""

import importlib
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# Suite root — the repository root (3 levels up from src/shared/python)
SUITE_ROOT: Path = Path(__file__).parent.parent.parent.parent

# Default output root — can be patched in tests
OUTPUT_ROOT: Path = SUITE_ROOT / "output"

__all__ = [
    "SUITE_ROOT",
    "OUTPUT_ROOT",
    "biomechanics",
    "calc_backend",
    "chat",
    "config",
    "estimation",
    "humanoid_character_builder",
    "model_generation",
    "notes",
    "pose_estimation",
    "signal_toolkit",
    "theme",
    "sidekick",
]

_LAZY_SUBPACKAGES = {
    "biomechanics",
    "calc_backend",
    "chat",
    "config",
    "estimation",
    "humanoid_character_builder",
    "model_generation",
    "notes",
    "pose_estimation",
    "signal_toolkit",
    "theme",
    "sidekick",
}


def __getattr__(name: str):
    """Expose shared.python subpackages for import-cache-sensitive patching."""
    if name in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
