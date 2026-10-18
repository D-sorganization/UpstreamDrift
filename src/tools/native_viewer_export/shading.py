"""Tension/compression segment shading for native exports (issue #13146).

The shading itself is one ``ForceColorScale`` (validated once, shared with every
other viewer); this module only legends it and moves the per-frame axial loads
to a render worker as JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import json
import logging
from pathlib import Path
from typing import Any

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.meshcat_force_colors import MeshcatForceColors

logger = logging.getLogger(__name__)

__all__ = [
    "MeshcatShading",
    "build_meshcat_shading",
    "frames_from_json",
    "frames_to_json",
    "log_coverage",
    "mujoco_scene_coverage",
    "pinocchio_color_bindings",
    "shading_legend",
]

Bindings = dict[str, dict[str, tuple[float, ...]]]


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


def log_coverage(engine: str, shaded: int, total: int) -> str:
    """Log (debug) and return the one-line shaded/total segment coverage summary."""
    if not 0 <= shaded <= total:
        raise ValueError(f"coverage needs 0 <= shaded <= total, got {shaded}/{total}")
    line = f"{engine}: segment shading covers {shaded}/{total} segments"
    logger.debug(line)
    return line


def mujoco_scene_coverage(
    model: Any, frames: Iterable[AxialLoadFrame | None]
) -> tuple[int, int]:
    """(shaded, total) bodies of a MuJoCo ``model`` that own geometry.

    A body counts as shaded when any frame carries a load for its name (axial
    loads are keyed by the MuJoCo body name, which is the specification name).
    """
    import mujoco

    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b))
        for b in set(model.geom_bodyid.tolist())
        if int(b) > 0
    }
    loaded = {
        n
        for f in frames
        if f is not None
        for n, v in f.values_n.items()
        if v is not None
    }
    return len(names & loaded), len(names)


def pinocchio_color_bindings(items: Iterable[Any], root: str = "golfer") -> Bindings:
    """Segment (spec body) to MeshCat leaf path to base RGBA for Pinocchio visuals.

    ``items`` are the exporter's ``VisualItem`` records; Pinocchio's
    ``MeshcatVisualizer`` publishes each geometry object at
    ``<root>/visuals/<name>``. Items on the world body (the floor) are skipped
    so unmapped geometry stays unshaded.
    """
    out: Bindings = {}
    for item in items:
        if not item.body:
            continue
        path = f"{root}/visuals/{item.name}/<object>"
        out.setdefault(item.body, {})[path] = tuple(item.rgba)
    return out


class MeshcatShading:
    """Per-frame segment shading for a MeshCat export (Drake, Pinocchio).

    Wraps the shared ``MeshcatForceColors`` (blending is ``ForceColorScale.
    shade_rgb`` over each object's base colour; alpha is preserved) and counts
    the segments that actually received a load for the coverage summary.
    """

    def __init__(
        self,
        engine: str,
        set_property: Callable[[str, str, list[float]], object],
        bindings: Mapping[str, Mapping[str, Sequence[float]]],
        scale: ForceColorScale,
    ) -> None:
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be a ForceColorScale")
        self._engine = engine
        self._scale = scale
        self._segments = frozenset(bindings)
        self._seen: set[str] = set()
        self._colors = MeshcatForceColors(set_property, bindings)

    def update(self, frame: AxialLoadFrame | None) -> None:
        """Shade (or restore) the bound objects for this frame's loads."""
        self._colors.apply(frame, self._scale)
        if frame is not None:
            self._seen |= {
                s for s in self._segments if frame.values_n.get(s) is not None
            }

    def coverage(self) -> tuple[int, int]:
        """(segments that received a load so far, bound segments)."""
        return len(self._seen), len(self._segments)

    def log_summary(self) -> str:
        return log_coverage(self._engine, *self.coverage())


def build_meshcat_shading(
    engine: str,
    enabled_scale: ForceColorScale | None,
    set_property: Callable[[str, str, list[float]], object],
    bindings: Mapping[str, Mapping[str, Sequence[float]]],
) -> MeshcatShading | None:
    """A :class:`MeshcatShading`, or ``None`` when shading is off or nothing binds."""
    if enabled_scale is None or not enabled_scale.enabled or not bindings:
        return None
    return MeshcatShading(engine, set_property, bindings, enabled_scale)
