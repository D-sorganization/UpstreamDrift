# ruff: noqa: E501
"""
Python-specific shared utilities and libraries.
This package contains reusable Python logic for tools.

Available packages:
    - ai: Agent-agnostic assistant integration helpers and GUI support
    - chat: Portable AI chat dock widget and Pydantic models
    - notes: Project-backed notes workspace with recycle-bin semantics
    - theme: Fleet-wide color theme management for PyQt6 applications
    - upstream_drift_tools: Process engineering calculators
    - signal_toolkit: Signal processing and analysis
    - humanoid_character_builder: URDF humanoid model generation
    - model_generation: URDF/MJCF model building and conversion

Preferred imports:
    from shared.python.theme import ThemeManager, get_theme_manager  # theme: keep prefix
    from shared.python import ai  # package-level assistant helpers
    from shared.python.humanoid_character_builder import CharacterBuilder, BodyParameters
    from shared.python.model_generation import quick_urdf, ManualBuilder, FrankensteinEditor
    from shared.python.signal_toolkit import Signal, SignalGenerator, FunctionFitter
    from shared.python.sidekick.process_calculators import FlareCalculator
    from shared.python.gui_launcher import GUIType, LaunchConfig, register_gui
    from shared.python.plot_engine.specs import PlotSpec, SeriesData
    from shared.python.plot_theme import apply_plot_theme
"""

from pathlib import Path

from . import _seam_redirect as _seam_redirect

# Retired Tools-owned packages (UD #9406) resolve to the pinned Tools tree under
# every spelling; this must run before any `src.shared.python.<root>` import.
_seam_redirect.install()
# Retired roots exist only in the pinned Tools tree; keep it as the trailing
# search location so `shared.python.<root>` finds them whichever copy of this
# package Python bound to the name (tests/conftest.py puts src/ first).
__path__ = _seam_redirect.extend_shared_python_path(__path__)

SUITE_ROOT = Path(__file__).resolve().parents[3]

from . import ai as ai
from . import cli_utils as cli_utils

__all__ = [
    "SUITE_ROOT",
    "apply_plot_theme",
    "get_logger",
    "ai",
    "cli_utils",
    "chat",
    "humanoid_character_builder",
    "model_generation",
    "notes",
    "signal_toolkit",
    "theme",
    "upstream_drift_tools",
]
