"""Tip-to-tail stills of supplied midhands ZTCF + ZVCF = Total.

Draws the 3D sheet and the face-on 2D sheet. The face-on camera is the shared
preset (azimuth 0, elevation -6): target toward image right, up along the
image. Vectors are copied. The grip null space is not solved here; its
minimum-norm member is alpha = 0 in f = F/2 ± (u × m')/d ± alpha u.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from matplotlib.figure import Figure
from src.shared.python.golf_view_presets import get_view_preset

RELATIVE_TOLERANCE = 1e-4
_ZTCF, _ZVCF, _TOTAL, _BADGE = "#0072B2", "#D55E00", "#000000", "#F0E442"
_VIEW_LABEL = {
    "3d": "Three-Dimensional Tip-to-Tail",
    "2d": "Two-Dimensional Face-On Tip-to-Tail",
}
_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_HASH = re.compile(r"^[0-9a-fA-F]{8,128}$")
_EVIDENCE = frozenset({"allocation-only", "forward-replayed"})
_BANNED = ("wasted energy", "biological optimality")
_PARTS = ("ztcf", "zvcf", "total")
_SETTINGS = (
    Path(__file__).resolve().parents[1]
    / "tools/swing_report_package/phase_settings.json"
)


def _catalog(path: Path = _SETTINGS) -> tuple[tuple[str, str], ...]:
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


class SimulationOutputLike(Protocol):
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


def _components(value: object, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.ndim != 1 or arr.shape[0] not in (2, 3):
        raise ValueError(f"{name} must be a length-2 or length-3 vector")
    return arr


def _vec3(value: object, name: str) -> np.ndarray:
    arr = _components(value, name)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr.copy()


_BASIS: tuple[np.ndarray, np.ndarray] | None = None


def _face_basis() -> tuple[np.ndarray, np.ndarray]:
    """Image right and image up of the shared face-on camera.

    Computed on the first face-on sheet. A preset mismatch raises there and
    leaves the 3D sheet importable.
    """
    global _BASIS
    if _BASIS is None:
        preset = get_view_preset("face_on")
        if (preset.azimuth_deg, preset.elevation_deg) != (0.0, -6.0):
            raise ValueError("face-on preset must be azimuth 0 and elevation -6")
        _BASIS = (
            np.asarray(preset.image_right(), dtype=np.float64),
            np.asarray(preset.image_up(), dtype=np.float64),
        )
    return _BASIS


@dataclass(frozen=True, eq=False)
class ComponentVectors:
    ztcf: np.ndarray
    zvcf: np.ndarray
    total: np.ndarray

    def __post_init__(self) -> None:
        for name in _PARTS:
            arr = _vec3(getattr(self, name), name)
            arr[~np.isfinite(arr)] = np.nan
            object.__setattr__(self, name, arr)


@dataclass(frozen=True, eq=False)
class PhaseLoads:
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


def closure_status(
    ztcf: object, zvcf: object, total: object, tolerance: float = RELATIVE_TOLERANCE
) -> str:
    drawn = [
        _components(value, name)
        for name, value in zip(_PARTS, (ztcf, zvcf, total), strict=True)
    ]
    if any(part.shape != drawn[0].shape for part in drawn):
        raise ValueError("ztcf, zvcf and total must share a length")
    if any(not np.isfinite(part).all() for part in drawn):
        return "unavailable"
    if not math.isfinite(tolerance) or not 0.0 < tolerance <= 1.0:
        raise ValueError("tolerance must lie in (0, 1]")
    ztcf_v, zvcf_v, total_v = drawn
    gap = float(np.linalg.norm(ztcf_v + zvcf_v - total_v))
    scale = max(
        float(np.linalg.norm(total_v)), float(np.linalg.norm(ztcf_v + zvcf_v)), 1e-12
    )
    if gap / scale <= tolerance:
        return "closed"
    return "open"


def _get(item: object, key: str, *, required: bool) -> object:
    if isinstance(item, Mapping):
        if required and key not in item:
            raise ValueError(f"simulation output is missing {key}")
        return item.get(key)
    if required and not hasattr(item, key):
        raise ValueError(f"simulation output is missing {key}")
    return getattr(item, key, None)


def _vectors(item: object | None, key: str) -> ComponentVectors:
    raw = {} if item is None else _get(item, key, required=False)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{key} must be a mapping")
    missing = np.full(3, np.nan)
    return ComponentVectors(
        *(missing if raw.get(part) is None else raw[part] for part in _PARTS)
    )


def adapt_simulation_output(source: SimulationOutputLike) -> StillRecord:
    raw = _get(source, "phases", required=True)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("phases must be a sequence of phase loads")
    found: dict[str, object] = {}
    for item in raw:
        name = _get(item, "name", required=False)
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
        git_sha=str(_get(source, "git_sha", required=True)),
        data_hash=str(_get(source, "data_hash", required=True)),
        objective=str(_get(source, "objective", required=True)),
        evidence=str(_get(source, "evidence", required=True)),
        phases=phases,
    )


def _project(vec: np.ndarray, view: str) -> np.ndarray:
    if view == "3d":
        return vec
    right, up = _face_basis()
    return np.array([float(np.dot(vec, right)), float(np.dot(vec, up))])


def _gid(ax: Any, text: str, gid: str, y: float) -> None:
    kwargs = {"transform": ax.transAxes, "ha": "center", "fontsize": 8}
    if ax.name == "3d":
        artist = ax.text2D(0.5, y, text, **kwargs)
    else:
        artist = ax.text(0.5, y, text, **kwargs)
    artist.set_gid(gid)


def _segment(
    ax: Any, tail: np.ndarray, tip: np.ndarray, color: str, label: str
) -> None:
    style = "--" if label == "Total" else "-"
    coords = [[float(tail[i]), float(tip[i])] for i in range(int(tail.shape[0]))]
    ax.plot(*coords, color=color, lw=2, ls=style, label=label)


def _mark_open(ax: Any, tip: np.ndarray) -> None:
    coords = [[float(tip[i])] for i in range(int(tip.shape[0]))]
    ax.plot(*coords, color=_ZVCF, marker="x", ls="none", label="_sum_tip")
    _gid(ax, "CLOSURE FAILED", "closure-failure", 0.9)


def _axes_labels(ax: Any, view: str) -> None:
    if view == "3d":
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return
    ax.set_xlabel("Target")
    ax.set_ylabel("Up")
    ax.set_aspect("equal", adjustable="box")


def _draw_panel(ax: Any, triple: ComponentVectors, view: str, tolerance: float) -> None:
    """Draw one projected triple, or label it UNAVAILABLE."""
    _axes_labels(ax, view)
    ztcf = _project(triple.ztcf, view)
    zvcf = _project(triple.zvcf, view)
    total = _project(triple.total, view)
    status = closure_status(ztcf, zvcf, total, tolerance)
    if status == "unavailable":
        _gid(ax, "UNAVAILABLE", "unavailable", 0.55)
        return
    origin = np.zeros_like(total)
    _segment(ax, origin, total, _TOTAL, "Total")
    _segment(ax, origin, ztcf, _ZTCF, "ZTCF")
    _segment(ax, ztcf, ztcf + zvcf, _ZVCF, "ZVCF")
    if status == "open":
        _mark_open(ax, ztcf + zvcf)


def render_tip_to_tail(record: StillRecord, view: str = "3d") -> Figure:
    """Return the 2×7 force and moment sheet for ``view``.

    Preconditions: ``record`` is a StillRecord and ``view`` is ``3d`` or ``2d``.
    Postcondition: fourteen panels, the objective, an evidence badge
    (``allocation-only`` or ``forward-replayed``) and a provenance footer.
    A non-finite panel reads UNAVAILABLE and is not drawn as zero. A closure
    miss is drawn and labelled. The face-on sheet checks closure on the
    projected plane, so a gap along the view direction stays closed there and
    fails on the 3D sheet. The figure is not written to disk.
    """
    if not isinstance(record, StillRecord):
        raise TypeError(f"record must be StillRecord, got {type(record).__name__}")
    if view not in _VIEW_LABEL:
        raise ValueError(f"view must be one of {tuple(_VIEW_LABEL)}, got {view!r}")
    import matplotlib.pyplot as plt

    options: dict[str, Any] = {"figsize": (16.8, 5.6)}
    if view == "3d":
        options["subplot_kw"] = {"projection": "3d"}
    fig, axes = plt.subplots(2, len(PHASE_IDS), **options)
    fig.suptitle(f"Objective: {record.objective}")
    label = fig.text(0.5, 0.93, _VIEW_LABEL[view], ha="center", fontsize=10)
    label.set_gid("view-label")
    badge = fig.text(
        0.99,
        0.99,
        record.evidence,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "fc": _BADGE, "ec": "#000000"},
    )
    badge.set_gid("evidence-badge")
    footer = fig.text(
        0.01, 0.01, f"Provenance: git {record.git_sha} | data {record.data_hash}"
    )
    footer.set_gid("provenance-footer")
    grid = np.atleast_2d(axes)
    rows = (("force_n", "Force (N)"), ("moment_nm", "Moment (N·m)"))
    for row, (attr, unit) in enumerate(rows):
        for phase in record.phases:
            ax = grid[row, PHASE_IDS.index(phase.name)]
            ax.set_title(f"{_LABEL[phase.name]} {unit}")
            _draw_panel(ax, getattr(phase, attr), view, record.relative_tolerance)
    return fig
