"""Tension/compression segment shading for native exports (issue #13146).

The shading itself is one ``ForceColorScale`` (validated once, shared with every
other viewer); this module only legends it and moves the per-frame axial loads
to a render worker as JSON.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale

__all__ = [
    "frames_from_json",
    "frames_to_json",
    "shading_legend",
]


def shading_legend(scale: ForceColorScale) -> str:
    """One HUD line naming both colours and the saturation load.

    Preconditions: ``scale`` is a ``ForceColorScale`` (``TypeError`` otherwise).
    """
    if not isinstance(scale, ForceColorScale):
        raise TypeError("scale must be a ForceColorScale")
    if scale.tension_limit_n == scale.compression_limit_n:
        sat = f"{scale.tension_limit_n:g} N"
    else:
        sat = f"{scale.tension_limit_n:g} N / {scale.compression_limit_n:g} N"
    return (
        f"Tension {scale.tension_color} | Compression {scale.compression_color}"
        f" | Saturation {sat} | Opacity {scale.opacity:g} ({scale.blend_mode})"
    )


def frames_to_json(frames: Sequence[AxialLoadFrame | None], path: Path) -> None:
    """Write per-position axial frames (``None`` stays ``null``) as a JSON list."""
    docs: list[dict[str, Any] | None] = []
    for frame in frames:
        if frame is not None and not isinstance(frame, AxialLoadFrame):
            raise TypeError("frames must be AxialLoadFrame or None")
        docs.append(None if frame is None else frame.to_dict())
    path.write_text(json.dumps(docs), encoding="utf-8")


def frames_from_json(path: Path) -> list[AxialLoadFrame | None]:
    """Inverse of :func:`frames_to_json`; unavailable values stay ``None``."""
    docs = json.loads(path.read_text(encoding="utf-8"))
    return [
        None
        if d is None
        else AxialLoadFrame(float(d["time_s"]), d["values_n"], str(d["source"]))
        for d in docs
    ]
