"""How a clip's motion was produced, labelled on every clip (#15150, D3).

Three honest modes, decided from the bundle and never guessed:

* kinematic replay: every effort is zero, the motion is recorded, not simulated;
* feedback-tracked dynamics: non-zero efforts written by the closed-loop tracker
  (the bundle's provenance records its ``controller``);
* open-loop dynamics from t0: efforts replayed from an initial state, declared by
  the bundle's ``replay_mode`` provenance field.

Anything else is ``unknown_replay_mode``. Labels and keys live in
``replay_mode_settings.json``.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from src.tools.native_viewer_export.kinematic_replay import is_kinematic_replay

SETTINGS_PATH = Path(__file__).with_name("replay_mode_settings.json")


class ReplayMode(str, Enum):
    """The replay mode of a clip; the value is the manifest string."""

    KINEMATIC = "kinematic_replay"
    FEEDBACK_TRACKED = "feedback_tracked_dynamics"
    OPEN_LOOP = "open_loop_dynamics"
    UNKNOWN = "unknown_replay_mode"


def _read(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))


def load_labels(path: Path | None = None) -> dict[str, str]:
    """Display text per mode value; ``ValueError`` if a mode has no label."""
    labels = _read(path)["labels"]
    missing = [m.value for m in ReplayMode if not str(labels.get(m.value, "")).strip()]
    if missing:
        raise ValueError(f"replay mode label missing for {missing}")
    return {m.value: str(labels[m.value]) for m in ReplayMode}


def display_text(mode: ReplayMode) -> str:
    """The label text of ``mode``; ``TypeError`` for anything else."""
    if not isinstance(mode, ReplayMode):
        raise TypeError(f"mode must be a ReplayMode, got {type(mode).__name__}")
    return load_labels()[mode.value]


def hud_line(mode: ReplayMode) -> str:
    """The HUD line stating ``mode``."""
    return f"{_read()['hud_prefix']}{display_text(mode)}"


def replay_mode_of(bundle: Any) -> ReplayMode:
    """Mode of ``bundle``: zero efforts decide kinematic; else a declared
    dynamics label, else a recorded tracker controller; else unknown.

    Raises ``ValueError`` for an empty effort table.
    """
    if is_kinematic_replay(bundle):
        return ReplayMode.KINEMATIC
    settings = _read()
    provenance = getattr(bundle, "provenance", None) or {}
    declared = provenance.get(settings["provenance_key"])
    if declared in (ReplayMode.FEEDBACK_TRACKED.value, ReplayMode.OPEN_LOOP.value):
        return ReplayMode(declared)
    if declared is None and provenance.get(settings["controller_key"]):
        return ReplayMode.FEEDBACK_TRACKED
    return ReplayMode.UNKNOWN


def hud_line_of(bundle: Any) -> str:
    """The HUD line for the replay mode recorded in ``bundle``."""
    return hud_line(replay_mode_of(bundle))
