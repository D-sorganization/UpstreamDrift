"""ToolShell UI migration registry (UD #14260)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ToolShellSpec:
    """ToolShell layout configuration for migrated tool pages."""

    tool_id: str
    tool_name: str
    workspace_id: str
    migrated: bool = True
    next_step_href: str = ""


MIGRATED_TOOL_SHELLS: Final[dict[str, ToolShellSpec]] = {
    "video_analyzer": ToolShellSpec(
        "video_analyzer",
        "Video Analyzer",
        "capture_analyze",
        next_step_href="/tools/swing-objective-lab",
    ),
    "swing_objective_lab": ToolShellSpec(
        "swing_objective_lab",
        "Swing Objective Lab",
        "optimize_train",
        next_step_href="/tools/golf-simulator",
    ),
    "golf_simulator": ToolShellSpec(
        "golf_simulator",
        "Golf Simulator",
        "shot_course_lab",
        next_step_href="/tools/ball-flight",
    ),
}


def is_tool_migrated(tool_id: str) -> bool:
    """Return True if tool has migrated to ToolShell."""
    return tool_id in MIGRATED_TOOL_SHELLS
