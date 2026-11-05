"""Force and torque vectors drawn on the model at swing events of a run (Issue #14439).

:func:`render_event_overlays` reads a MuJoCo run written by
:func:`src.sim_run.mujoco_run.run_mujoco`, checks the model text and trajectory
against the run manifest, restores the recorded state at each event row, samples
the world-frame wrenches through ``MujocoForceTorqueSource`` and draws them with
the shared ADR-0052 glyph builder and Matplotlib renderer. Rendering uses the Agg
canvas directly, so it needs no display, and the PNG carries no timestamp or
software metadata, so the same run renders to the same bytes. Event rows come
from the caller; ``dataclasses.asdict`` of a
:class:`src.shared.python.analysis.swing_events.SwingEventFrames` is one source.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    GlyphSet,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (
    draw_glyphs_3d,
    draw_legend,
    equalize_3d_axes,
    legend_lines,
)
from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import model_revision, sha256_file
from src.sim_run.trajectory_run import TRAJECTORY_FILENAME

_SNAPSHOT_DECIMALS = 6
_PNG_METADATA: Mapping[str, None] = {"Software": None}


@dataclass(frozen=True)
class EventRender:
    """One rendered event: where it sits in the run and what was drawn."""

    event: str
    row: int
    time_s: float
    filename: str
    sha256: str
    arrows: int
    torque_arcs: int
    legend: tuple[str, ...]
    glyphs: dict[str, Any]


def _load_trajectory(manifest: RunManifest, run_dir: Path, model_text: str):
    if manifest.model.revision != model_revision(model_text):
        raise ValueError(
            f"model text does not match recorded revision {manifest.model.revision}"
        )
    path = run_dir / TRAJECTORY_FILENAME
    recorded = manifest.outputs.get(TRAJECTORY_FILENAME)
    if recorded is None or not path.is_file() or sha256_file(path) != recorded:
        raise ValueError(f"{TRAJECTORY_FILENAME} does not match its manifest digest")
    return np.load(path)


def _check_events(events: Mapping[str, int], rows: int) -> None:
    if not events:
        raise ValueError("at least one event row is required")
    for name, row in events.items():
        if not 0 <= row < rows:
            raise ValueError(f"event {name!r} row {row} is outside [0, {rows})")


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, _SNAPSHOT_DECIMALS) + 0.0
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _scene_points(data: mujoco.MjData, glyphs: GlyphSet) -> np.ndarray:
    points = [np.asarray(data.xpos[1:])]
    points.extend(np.array([a.tail_m, a.tip_m]) for a in glyphs.arrows)
    return np.vstack(points)


def _draw(
    path: Path, model: mujoco.MjModel, data: mujoco.MjData, title: str, glyphs: GlyphSet
) -> None:
    fig = Figure(figsize=(10.0, 7.5), dpi=100)
    FigureCanvasAgg(fig)
    ax: Any = fig.add_subplot(projection="3d")
    for body in range(1, model.nbody):
        parent = model.body_parentid[body]
        if parent > 0:
            segment = np.array([data.xpos[parent], data.xpos[body]])
            ax.plot(*segment.T, color="#808080", linewidth=4.0)
    ax.scatter(*np.asarray(data.xpos[1:]).T, color="#404040", s=12)
    draw_glyphs_3d(ax, glyphs)
    equalize_3d_axes(ax, _scene_points(data, glyphs))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    draw_legend(ax, glyphs.legend)
    fig.savefig(path, format="png", metadata=_PNG_METADATA)


def render_event_overlays(
    manifest: RunManifest,
    run_dir: Path,
    model_text: str,
    *,
    events: Mapping[str, int],
    out_dir: Path,
    style: ForceGlyphStyle | None = None,
) -> tuple[EventRender, ...]:
    """Draw the run's force and torque vectors at each event row into ``out_dir``.

    Preconditions: ``model_text`` matches the manifest's model revision, the
    trajectory matches its recorded digest, and every event row indexes the
    trajectory. Raises ``ValueError`` otherwise.

    Postconditions: ``out_dir/<event>.png`` exists for each event, in the order
    given, and each receipt's ``sha256`` is that file's digest. ``glyphs`` is the
    drawn glyph set rounded to six decimals, a renderer-independent snapshot.
    """
    trajectory = _load_trajectory(manifest, Path(run_dir), model_text)
    _check_events(events, trajectory.shape[0])
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_torque_source import (  # noqa: E501
        MujocoForceTorqueSource,
    )

    model = mujoco.MjModel.from_xml_string(model_text)
    data = mujoco.MjData(model)
    source = MujocoForceTorqueSource(model)
    style = style or ForceGlyphStyle()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    renders: list[EventRender] = []
    for name, row in events.items():
        state = trajectory[row]
        data.time = float(state[0])
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq : 1 + model.nq + model.nv]
        mujoco.mj_forward(model, data)
        glyphs = build_glyphs(source.sample(data, include_gravity=True), style)
        path = out / f"{name}.png"
        _draw(path, model, data, f"{name} (t = {data.time:.3f} s)", glyphs)
        renders.append(
            EventRender(
                event=name,
                row=row,
                time_s=float(data.time),
                filename=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arrows=len(glyphs.arrows),
                torque_arcs=len(glyphs.torque_arcs),
                legend=legend_lines(glyphs.legend),
                glyphs=_rounded(json.loads(json.dumps(glyphs.to_dict()))),
            )
        )
    return tuple(renders)
