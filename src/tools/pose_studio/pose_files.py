"""Engine-native pose save/load for Pose Studio (issue #8882, EPIC #4895).

Every path here delegates to
:mod:`src.shared.python.pose_interchange.pose_io`, which already owns the
five engine-native on-disk shapes documented in
``docs/user_guide/pose_studio/save_formats.md`` and holds round-trip
parity to 1e-9. This module adds only the two things the GUI needs and
``pose_io`` deliberately does not know about: the per-engine Qt file
filter and the default suffix to pre-fill in the save dialog.

Keeping it separate from ``gui.py`` means the file-format knowledge is
unit-testable without a ``QApplication``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.pose_io import (
    load_initial_state,
    save_initial_state,
)

__all__ = [
    "ENGINE_FORMATS",
    "EngineFormat",
    "engine_format",
    "load_pose",
    "save_pose",
]


class EngineFormat:
    """The Qt-facing description of one engine's on-disk pose format."""

    __slots__ = ("description", "engine", "suffix")

    def __init__(self, engine: str, description: str, suffix: str) -> None:
        self.engine = engine
        self.description = description
        self.suffix = suffix

    @property
    def name_filter(self) -> str:
        """``QFileDialog`` name filter for this engine, plus an all-files escape."""
        return f"{self.description} (*{self.suffix});;All Files (*)"

    def default_name(self, stem: str = "pose") -> str:
        """Suggested file name for the save dialog."""
        return f"{stem}{self.suffix}"


#: Mirrors ``pose_io``'s five engine writers. ``pose_io.SUPPORTED_ENGINES``
#: stays the source of truth for *which* engines exist; this table only
#: describes how each one looks in a file dialog.
ENGINE_FORMATS: Final[dict[str, EngineFormat]] = {
    "drake": EngineFormat("drake", "Drake initial state (q, v, metadata)", ".json"),
    "mujoco": EngineFormat("mujoco", "MuJoCo initial state (qpos, qvel)", ".json"),
    "pinocchio": EngineFormat("pinocchio", "Pinocchio initial state archive", ".npz"),
    "opensim": EngineFormat("opensim", "OpenSim storage", ".sto"),
    "simscape": EngineFormat("simscape", "Simscape starting pose", ".json"),
}


def engine_format(engine: str) -> EngineFormat:
    """Return the :class:`EngineFormat` for ``engine``.

    Raises:
        ValueError: if ``engine`` has no on-disk pose format.
    """
    try:
        return ENGINE_FORMATS[engine]
    except KeyError:
        raise ValueError(
            f"no pose file format for engine {engine!r}; "
            f"known engines: {sorted(ENGINE_FORMATS)}"
        ) from None


def save_pose(pose: CanonicalPose, engine: str, path: Path | str) -> Path:
    """Write ``pose`` as an engine-native initial-state file.

    Returns the path actually written. NumPy appends ``.npz`` for the
    Pinocchio archive, so the returned path is what the caller should
    report to the user rather than the path they asked for.
    """
    fmt = engine_format(engine)
    target = Path(path)
    save_initial_state(pose, engine, target)
    if not target.exists() and fmt.suffix == ".npz":
        appended = target.with_suffix(target.suffix + ".npz")
        if appended.exists():
            return appended
    return target


def load_pose(engine: str, path: Path | str) -> CanonicalPose:
    """Read an engine-native initial-state file back as a canonical pose."""
    engine_format(engine)
    return load_initial_state(engine, Path(path))
