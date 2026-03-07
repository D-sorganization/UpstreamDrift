"""
Python-specific shared utilities and libraries.
This package contains reusable Python logic for tools.

Available packages:
    - chat: Portable AI chat dock widget and Pydantic models
    - notes: Project-backed notes workspace with recycle-bin semantics
    - theme: Fleet-wide color theme management for PyQt6 applications
    - upstream_drift_tools: Process engineering calculators
    - signal_toolkit: Signal processing and analysis
    - humanoid_character_builder: URDF humanoid model generation
    - model_generation: URDF/MJCF model building and conversion

Preferred imports (direct from package, since src/shared/python is on sys.path):
    from shared.python.theme import ThemeManager, get_theme_manager  # theme: keep prefix
    from humanoid_character_builder import CharacterBuilder, BodyParameters
    from model_generation import quick_urdf, ManualBuilder, FrankensteinEditor
    from signal_toolkit import Signal, SignalGenerator, FunctionFitter
    from upstream_drift_tools.process_calculators import FlareCalculator
    from gui_launcher import GUIType, LaunchConfig, register_gui
    from plot_engine.specs import PlotSpec, SeriesData
    from plot_theme import apply_plot_theme
"""

from pathlib import Path

# Suite root — the repository root (3 levels up from src/shared/python)
SUITE_ROOT: Path = Path(__file__).parent.parent.parent.parent

# Default output root — can be patched in tests
OUTPUT_ROOT: Path = SUITE_ROOT / "output"

__all__ = [
    "SUITE_ROOT",
    "OUTPUT_ROOT",
    "chat",
    "humanoid_character_builder",
    "model_generation",
    "notes",
    "signal_toolkit",
    "theme",
    "upstream_drift_tools",
]
