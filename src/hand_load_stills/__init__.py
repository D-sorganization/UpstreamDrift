"""Tip-to-tail stills: ZTCF + ZVCF = Total at the seven swing phases.

Copies supplied midhands vectors. It does not solve the grip null space.
A non-finite component is UNAVAILABLE and is never drawn as zero. The drawn
tip of ZTCF then ZVCF must meet Total within the relative tolerance (owner
default 1e-4); a miss is labelled CLOSURE FAILED. Face-on uses the shared
face-on camera. Colours are Okabe–Ito. The footer carries the caller's git
SHA and data hash.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from golf_view_presets.presets import get_view_preset
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "PHASE_IDS",
    "PHASE_LABELS",
    "ComponentVectors",
    "PhaseLoads",
    "SimulationOutputLike",
    "StillRecord",
    "adapt_simulation_output",
    "closure_status",
    "render_tip_to_tail",
]

RELATIVE_TOLERANCE = 1e-4
_VIEWS = ("2d", "3d")
_VIEW_LABEL = {
    "3d": "Three-Dimensional Tip-to-Tail",
    "2d": "Two-Dimensional Face-On Tip-to-Tail",
}
_ZTCF, _ZVCF, _TOTAL, _BADGE = "#0072B2", "#D55E00", "#000000", "#F0E442"
_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_HASH = re.compile(r"^[0-9a-fA-F]{8,128}$")
_EVIDENCE = frozenset({"allocation-only", "forward-replayed"})
_BANNED = ("wasted energy", "biological optimality")
_PARTS = ("ztcf", "zvcf", "total")
_SETTINGS = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "swing_report_package"
    / "phase_settings.json"
)


def _catalog(path: Path = _SETTINGS) -> tuple[tuple[str, str], ...]:
    """Seven ``(id, label)`` rows from the swing-report phase settings."""
    phases = json.loads(path.read_text(encoding="utf-8")).get("phases")
    if not isinstance(phases, list):
        raise ValueError("phase settings must list phases")
    rows = []
    for item in phases:
        if not isinstance(item, dict) or not item.get("id") or not item.get("label"):
            raise ValueError(f"phase {item!r} needs an id and a label")
        rows.append((str(item["id"]), str(item["label"])))
    if len(rows) != 7 or len({row[0] for row in rows}) != 7:
        raise ValueError("phase settings must name seven distinct phases")
    return tuple(rows)


def _fold(name: object) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split()).casefold()


PHASE_CATALOG = _catalog()
PHASE_IDS = tuple(row[0] for row in PHASE_CATALOG)
PHASE_LABELS = tuple(row[1] for row in PHASE_CATALOG)
_LABEL = dict(PHASE_CATALOG)
_PHASE_INDEX: dict[str, str] = {}
for _pid, _label in PHASE_CATALOG:
    for _key in (_pid, _label):
        _folded = _fold(_key)
        if _folded in _PHASE_INDEX and _PHASE_INDEX[_folded] != _pid:
            raise ValueError(f"phase alias {_key!r} is ambiguous")
        _PHASE_INDEX[_folded] = _pid
_FACE = get_view_preset("face_on")
_RIGHT, _UP = _FACE.image_right(), _FACE.image_up()


class SimulationOutputLike(Protocol):
    """Fields another lane's SimulationOutput must expose to this adapter."""

    git_sha: str
    data_hash: str
    objective: str
    evidence: str
    phases: Sequence[Mapping[str, Any]]


def _canonical(name: object) -> str:
    found = _PHASE_INDEX.get(_fold(name))
    if found is None:
        raise ValueError(f"unknown phase {name!r}")
    return found


def _as_vec(value: object, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    return out


@dataclass(frozen=True, eq=False)
class ComponentVectors:
    """ZTCF, ZVCF and Total. NaN means unavailable, not a physical zero."""

    ztcf: np.ndarray
    zvcf: np.ndarray
    total: np.ndarray

    def __post_init__(self) -> None:
        for name in _PARTS:
            object.__setattr__(self, name, _as_vec(getattr(self, name), name))


@dataclass(frozen=True, eq=False)
class PhaseLoads:
    """One phase. ``name`` is stored as the phase-settings id."""

    name: str
    force_n: ComponentVectors
    moment_nm: ComponentVectors

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _canonical(self.name))
        for attr in ("force_n", "moment_nm"):
            if not isinstance(getattr(self, attr), ComponentVectors):
                raise TypeError(f"{attr} must be ComponentVectors")


def _hex_token(value: object, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be a hex string")
    return value


def _objective(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("objective must be a non-empty string")
    if any(phrase in value.casefold() for phrase in _BANNED):
        raise ValueError("wasted energy and biological optimality are rejected")
    return value.strip()


@dataclass(frozen=True, eq=False)
class StillRecord:
    """Seven phase loads and the provenance stamped on every still.

    Preconditions: hex provenance, evidence ``allocation-only`` or
    ``forward-replayed``, the seven phase ids in swing order, and a
    ``relative_tolerance`` in ``(0, 1]``.
    """

    git_sha: str
    data_hash: str
    objective: str
    evidence: str
    phases: tuple[PhaseLoads, ...]
    relative_tolerance: float = RELATIVE_TOLERANCE

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        names = tuple(phase.name for phase in phases)
        if names != PHASE_IDS or any(not isinstance(p, PhaseLoads) for p in phases):
            raise ValueError(f"phases must be {PHASE_IDS} in that order")
        tolerance = float(self.relative_tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= 1.0:
            raise ValueError("relative_tolerance must lie in (0, 1]")
        if self.evidence not in _EVIDENCE:
            raise ValueError(f"evidence must be one of {sorted(_EVIDENCE)}")
        object.__setattr__(self, "git_sha", _hex_token(self.git_sha, _SHA, "git_sha"))
        object.__setattr__(
            self, "data_hash", _hex_token(self.data_hash, _HASH, "data_hash")
        )
        object.__setattr__(self, "objective", _objective(self.objective))
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "relative_tolerance", tolerance)


def _drawn(value: object, name: str) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.ndim != 1 or arr.shape[0] not in (2, 3):
        raise ValueError(f"{name} must be a length-2 or length-3 vector")
    if not np.isfinite(arr).all():
        return None
    return arr


def closure_status(
    ztcf: object, zvcf: object, total: object, tolerance: float = RELATIVE_TOLERANCE
) -> str:
    """``closed``, ``open`` or ``unavailable`` for one drawn triple.

    Unavailable when any component is non-finite. Open when
    ``||(ztcf + zvcf) - total|| / scale`` exceeds ``tolerance``. ``scale`` is
    the larger norm, and at least ``1e-12``.
    """
    drawn = [
        _drawn(item, name)
        for item, name in ((ztcf, "ztcf"), (zvcf, "zvcf"), (total, "total"))
    ]
    if any(item is None for item in drawn):
        return "unavailable"
    if not math.isfinite(tolerance) or not 0.0 < tolerance <= 1.0:
        raise ValueError("tolerance must lie in (0, 1]")
    ztcf_v, zvcf_v, total_v = (np.asarray(item) for item in drawn)
    gap = float(np.linalg.norm(ztcf_v + zvcf_v - total_v))
    scale = max(
        float(np.linalg.norm(total_v)), float(np.linalg.norm(ztcf_v + zvcf_v)), 1e-12
    )
    if gap / scale <= tolerance:
        return "closed"
    return "open"


def _read(item: object, key: str) -> object:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _need(source: object, key: str) -> object:
    if isinstance(source, Mapping):
        if key not in source:
            raise ValueError(f"simulation output is missing {key}")
        return source[key]
    if not hasattr(source, key):
        raise ValueError(f"simulation output is missing {key}")
    return getattr(source, key)


def _vectors(item: object | None, key: str) -> ComponentVectors:
    raw = {} if item is None else _read(item, key)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{key} must be a mapping")
    missing = np.full(3, np.nan)
    return ComponentVectors(
        *(missing if raw.get(part) is None else raw[part] for part in _PARTS)
    )


def adapt_simulation_output(source: SimulationOutputLike) -> StillRecord:
    """Copy phase loads. Missing channels and phases are NaN, never zero.

    Total is not invented from ZTCF + ZVCF. Postcondition: seven phases in order.
    """
    raw = _need(source, "phases")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("phases must be a sequence of phase loads")
    found: dict[str, object] = {}
    for item in raw:
        name = _read(item, "name")
        if name is None:
            raise ValueError("each phase needs a name")
        pid = _canonical(name)
        if pid in found:
            raise ValueError(f"duplicate phase {pid}")
        found[pid] = item
    phases = tuple(
        PhaseLoads(
            pid,
            _vectors(found.get(pid), "force_n"),
            _vectors(found.get(pid), "moment_nm"),
        )
        for pid in PHASE_IDS
    )
    return StillRecord(
        git_sha=str(_need(source, "git_sha")),
        data_hash=str(_need(source, "data_hash")),
        objective=str(_need(source, "objective")),
        evidence=str(_need(source, "evidence")),
        phases=phases,
    )


def _project(vec: np.ndarray, view: str) -> np.ndarray:
    if view == "3d":
        return vec
    return np.array([float(np.dot(vec, _RIGHT)), float(np.dot(vec, _UP))])


def _triple(
    triple: ComponentVectors, view: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _project(triple.ztcf, view),
        _project(triple.zvcf, view),
        _project(triple.total, view),
    )


def _plot(
    ax: Axes, tail: np.ndarray, tip: np.ndarray, color: str, style: str, label: str
) -> None:
    series = [[float(tail[i]), float(tip[i])] for i in range(int(tail.shape[0]))]
    ax.plot(
        *series,
        color=color,
        linestyle=style,
        linewidth=0.0 if style == "none" else 2.0,
        marker="x" if style == "none" else None,
        label=label,
    )


def _note(ax: Axes, message: str, gid: str, y: float) -> None:
    kwargs: dict[str, Any] = {
        "transform": ax.transAxes,
        "ha": "center",
        "va": "center",
        "fontsize": 8,
        "color": "#000000",
    }
    axis = cast(Any, ax)
    artist = (
        axis.text2D(0.5, y, message, **kwargs)
        if ax.name == "3d"
        else axis.text(0.5, y, message, **kwargs)
    )
    artist.set_gid(gid)


def _panel(
    ax: Axes, triple: ComponentVectors, view: str, tolerance: float, span: float
) -> None:
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    if view == "3d":
        axis3d = cast(Any, ax)
        axis3d.set_zlim(-span, span)
        axis3d.set_xlabel("X")
        axis3d.set_ylabel("Y")
        axis3d.set_zlabel("Z")
    else:
        ax.set_xlabel("Target")
        ax.set_ylabel("Up")
        ax.set_aspect("equal", adjustable="box")
    ztcf, zvcf, total = _triple(triple, view)
    status = closure_status(ztcf, zvcf, total, tolerance)
    if status == "unavailable":
        _note(ax, "UNAVAILABLE", "unavailable", 0.55)
        return
    origin = np.zeros_like(total)
    specs = (
        (origin, total, _TOTAL, "--", "Total"),
        (origin, ztcf, _ZTCF, "-", "ZTCF"),
        (ztcf, ztcf + zvcf, _ZVCF, "-", "ZVCF"),
    )
    for tail, tip, color, style, label in specs:
        _plot(ax, tail, tip, color, style, label)
    if status == "open":
        _plot(ax, ztcf + zvcf, ztcf + zvcf, _ZVCF, "none", "_sum_tip")
        _note(ax, "CLOSURE FAILED", "closure-failure", 0.9)


def _span(record: StillRecord, view: str) -> float:
    peak = 1.0
    for phase in record.phases:
        for attr in ("force_n", "moment_nm"):
            drawn = _triple(getattr(phase, attr), view)
            if closure_status(*drawn, record.relative_tolerance) == "unavailable":
                continue
            points = (*drawn, drawn[0] + drawn[1])
            peak = max(peak, max(float(np.max(np.abs(point))) for point in points))
    return peak * 1.25


def render_tip_to_tail(record: StillRecord, view: str = "3d") -> Figure:
    """Return a 2×7 still of force and moment at every swing phase.

    Preconditions: ``record`` is a StillRecord and ``view`` is ``3d`` or ``2d``.
    Postcondition: fourteen panels, an objective title, an evidence badge and
    a provenance footer. Closure failures and unavailable panels stay visible.
    The figure is not written to disk.
    """
    if not isinstance(record, StillRecord):
        raise TypeError(f"record must be StillRecord, got {type(record).__name__}")
    if view not in _VIEWS:
        raise ValueError(f"view must be one of {_VIEWS}, got {view!r}")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        len(PHASE_IDS),
        figsize=(16.8, 5.6),
        subplot_kw={"projection": "3d"} if view == "3d" else {},
    )
    fig.suptitle(f"Objective: {record.objective}")
    label = fig.text(0.5, 0.93, _VIEW_LABEL[view], ha="center", va="top", fontsize=10)
    label.set_gid("view-label")
    badge = fig.text(
        0.99,
        0.99,
        record.evidence,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": _BADGE, "ec": "#000000"},
    )
    badge.set_gid("evidence-badge")
    footer = fig.text(
        0.01,
        0.01,
        f"Provenance: git {record.git_sha} | data {record.data_hash}",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    footer.set_gid("provenance-footer")
    span = _span(record, view)
    grid = np.atleast_2d(axes)
    units = (("force_n", "Force (N)"), ("moment_nm", "Moment (N·m)"))
    for row, (attr, unit) in enumerate(units):
        for phase in record.phases:
            axis = grid[row, PHASE_IDS.index(phase.name)]
            axis.set_title(f"{_LABEL[phase.name]} {unit}")
            _panel(axis, getattr(phase, attr), view, record.relative_tolerance, span)
    handles, labels = grid[0, 0].get_legend_handles_labels()
    kept = [
        (h, lab)
        for h, lab in zip(handles, labels, strict=True)
        if not str(lab).startswith("_")
    ]
    if kept:
        chosen_h, chosen_l = zip(*kept, strict=True)
        fig.legend(
            chosen_h, chosen_l, loc="upper left", frameon=False, fontsize=8
        ).set_gid("part-legend")
    fig.subplots_adjust(
        top=0.78, bottom=0.14, left=0.04, right=0.99, wspace=0.55, hspace=0.6
    )
    return fig
