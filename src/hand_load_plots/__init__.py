"""Hand-load plots for a fixed-motion allocation comparison.

``adapt_simulation_output`` is the seam to the SimulationOutput contract owned
by another lane. This package draws supplied series. It does not solve the
grip null space. ``C(t) = ||f_L|| + ||f_R|| - ||F||`` is opposing-force
cancellation, never wasted energy or biological optimality. Non-finite samples
stay missing; an all-missing series is labelled UNAVAILABLE and is not drawn
as zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "COMPONENTS",
    "HandLoadPlotInput",
    "PhaseMarker",
    "SimulationOutputLike",
    "StrategyLoads",
    "adapt_simulation_output",
    "opposing_force_cancellation",
    "plot_cancellation",
    "plot_invariant_dashboard",
    "series_magnitude",
]

COMPONENTS = ("total", "ztcf", "zvcf")
_IDS = ("baseline", "min_squared_force")
_EVIDENCE = frozenset({"allocation-only", "forward-replayed"})
_LABEL = {"baseline": "baseline", "min_squared_force": "min squared force"}
_STYLE = {"baseline": "-", "min_squared_force": "--"}
_PART = {"total": "Total", "ztcf": "ZTCF", "zvcf": "ZVCF"}
_PART_COLOR = {"total": "#000000", "ztcf": "#0072B2", "zvcf": "#D55E00"}
_HAND_COLOR = {
    ("LH", "baseline"): "#E69F00",
    ("LH", "min_squared_force"): "#56B4E9",
    ("RH", "baseline"): "#CC79A7",
    ("RH", "min_squared_force"): "#009E73",
}
_ID_COLOR = {"baseline": "#E69F00", "min_squared_force": "#0072B2"}
_PHASES = {
    "address": "Address",
    "takeaway": "Takeaway",
    "backswing": "Backswing",
    "top": "Top",
    "downswing": "Downswing",
    "impact": "Impact",
    "follow through": "Follow-Through",
    "finish": "Finish",
}
_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_HASH = re.compile(r"^[0-9a-fA-F]{8,128}$")


class SimulationOutputLike(Protocol):
    """Fields the other lane's SimulationOutput must expose to this adapter."""

    time_s: np.ndarray
    git_sha: str
    data_hash: str
    objective: str
    phases: Sequence[tuple[str, float]]
    strategies: Sequence[Mapping[str, Any]]


@dataclass(frozen=True, eq=False)
class PhaseMarker:
    """Named event on the swing time base."""

    name: str
    time_s: float


@dataclass(frozen=True, eq=False)
class StrategyLoads:
    """One allocation. NaN is a missing sample, not a physical zero.

    Preconditions: id is baseline or min_squared_force; evidence is
    allocation-only or forward-replayed; forces are ``(n, 3)``; alpha is
    ``(n,)``; every midhands component is present.
    """

    strategy_id: str
    evidence: str
    force_left_n: np.ndarray
    force_right_n: np.ndarray
    alpha_n: np.ndarray
    midhands_force_n: Mapping[str, np.ndarray]
    midhands_moment_nm: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.strategy_id not in _IDS:
            raise ValueError(f"unknown strategy_id {self.strategy_id!r}")
        if self.evidence not in _EVIDENCE:
            raise ValueError(f"evidence must be one of {sorted(_EVIDENCE)}")
        left = _shaped(self.force_left_n, None, "force_left_n")
        n = int(left.shape[0])
        object.__setattr__(self, "force_left_n", _shaped(left, (n, 3), "force_left_n"))
        object.__setattr__(
            self, "force_right_n", _shaped(self.force_right_n, (n, 3), "force_right_n")
        )
        object.__setattr__(self, "alpha_n", _shaped(self.alpha_n, (n,), "alpha_n"))
        object.__setattr__(
            self, "midhands_force_n", _channels(self.midhands_force_n, n, force=True)
        )
        object.__setattr__(
            self,
            "midhands_moment_nm",
            _channels(self.midhands_moment_nm, n, force=False),
        )


@dataclass(frozen=True, eq=False)
class HandLoadPlotInput:
    """Local plot record. Midhands are stored on each strategy.

    Preconditions: finite strictly increasing time; hex provenance; an
    objective that is not wasted energy or biological optimality; Address,
    Impact, and Follow-Through in that order; both strategies.
    """

    time_s: np.ndarray
    git_sha: str
    data_hash: str
    objective: str
    phases: tuple[PhaseMarker, ...]
    strategies: tuple[StrategyLoads, ...]

    def __post_init__(self) -> None:
        time = _floats(self.time_s, "time_s")
        if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all():
            raise ValueError("time_s must be a finite vector of length >= 2")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be strictly increasing")
        phases = tuple(_phase(item) for item in self.phases)
        _check_phases(phases, float(time[0]), float(time[-1]))
        _check_strategies(self.strategies, int(time.size))
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "git_sha", _token(self.git_sha, _SHA, "git_sha"))
        object.__setattr__(
            self, "data_hash", _token(self.data_hash, _HASH, "data_hash")
        )
        object.__setattr__(self, "objective", _objective(self.objective))
        object.__setattr__(self, "phases", phases)


def series_magnitude(values: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norm. Non-finite rows stay NaN and are never zeroed."""
    arr = _floats(values, "values")
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got ndim {arr.ndim}")
    out = np.full(arr.shape[0], np.nan)
    finite = np.isfinite(arr).all(axis=1)
    if np.any(finite):
        out[finite] = np.linalg.norm(arr[finite], axis=1)
    return out


def opposing_force_cancellation(
    force_left_n: np.ndarray, force_right_n: np.ndarray, net_force_n: np.ndarray
) -> np.ndarray:
    """Return ``C(t) = ||f_L|| + ||f_R|| - ||F||``. Non-finite samples stay NaN."""
    parts = [
        series_magnitude(item) for item in (force_left_n, force_right_n, net_force_n)
    ]
    if len({part.shape for part in parts}) != 1:
        raise ValueError("force series must share one time length")
    return parts[0] + parts[1] - parts[2]


def adapt_simulation_output(source: SimulationOutputLike) -> HandLoadPlotInput:
    """Copy a SimulationOutput-shaped record. Missing channels become NaN.

    Does not solve for alpha or the hand split, and does not zero-fill.
    """
    time = _floats(source.time_s, "time_s")
    if time.ndim != 1:
        raise ValueError("time_s must be one-dimensional")
    n = int(time.shape[0])
    return HandLoadPlotInput(
        time_s=time,
        git_sha=source.git_sha,
        data_hash=source.data_hash,
        objective=source.objective,
        phases=tuple(_phase(item) for item in source.phases),
        strategies=tuple(_from_mapping(item, n) for item in source.strategies),
    )


def plot_invariant_dashboard(record: HandLoadPlotInput) -> Figure:
    """Unchanged Total, ZTCF, and ZVCF wrench beside the shrinking hand forces.

    Postcondition: objective, units, evidence badge, and provenance footer.
    """
    fig, axes = _figure(record, 2, 2)
    spec = (
        ("Invariant Midhands Force", "Force (N)"),
        ("Invariant Midhands Moment", "Moment (N·m)"),
        ("Left-Hand Force", "Force (N)"),
        ("Right-Hand Force", "Force (N)"),
    )
    for axis, (title, ylabel) in zip(axes, spec, strict=True):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    _invariants(axes[0], record, "force")
    _invariants(axes[1], record, "moment")
    _hands(axes[2], record, "LH")
    _hands(axes[3], record, "RH")
    for axis in axes:
        _legend(axis)
    return fig


def plot_cancellation(record: HandLoadPlotInput) -> Figure:
    """Opposing-force cancellation and supplied axial squeeze, with phases.

    Postcondition: same chrome as the dashboard. ``C`` is not an energy.
    """
    fig, axes = _figure(record, 2, 1)
    axes[0].set_title("Opposing-Force Cancellation")
    axes[0].set_ylabel("Opposing-Force Cancellation (N)")
    axes[1].set_title("Axial Squeeze")
    axes[1].set_ylabel("Axial Squeeze (N)")
    for strategy in _ordered(record):
        label, color = _LABEL[strategy.strategy_id], _ID_COLOR[strategy.strategy_id]
        curve = opposing_force_cancellation(
            strategy.force_left_n,
            strategy.force_right_n,
            strategy.midhands_force_n["total"],
        )
        _draw(axes[0], record.time_s, curve, label, color, "-")
        _draw(axes[1], record.time_s, strategy.alpha_n, label, color, "-")
    for axis in axes:
        _mark_phases(axis, record.phases)
        _legend(axis)
    return fig


def _figure(
    record: HandLoadPlotInput, rows: int, cols: int
) -> tuple[Figure, list[Axes]]:
    if not isinstance(record, HandLoadPlotInput):
        raise TypeError(
            f"record must be HandLoadPlotInput, got {type(record).__name__}"
        )
    fig, raw = plt.subplots(rows, cols, figsize=(9.2, 6.4), sharex=True)
    axes = list(np.atleast_1d(raw).ravel())
    fig.suptitle(f"Objective: {record.objective}")
    seen: list[str] = []
    for strategy in _ordered(record):
        if strategy.evidence not in seen:
            seen.append(strategy.evidence)
    badge = fig.text(
        0.99,
        0.985,
        " ".join(seen),
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "#F0E442", "ec": "#000000"},
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
    for axis in axes:
        axis.set_xlabel("Time (s)")
    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.48, wspace=0.32)
    return fig, axes


def _ordered(record: HandLoadPlotInput) -> tuple[StrategyLoads, ...]:
    by_id = {item.strategy_id: item for item in record.strategies}
    return tuple(by_id[name] for name in _IDS)


def _draw(
    axis: Axes,
    time_s: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    style: str,
) -> None:
    drawn = series_magnitude(values)
    if not np.isfinite(drawn).any():
        axis.text(
            0.02,
            0.95,
            f"UNAVAILABLE: {label}",
            transform=axis.transAxes,
            color="#000000",
            fontsize=8,
            va="top",
        )
        return
    axis.plot(time_s, drawn, label=label, color=color, linestyle=style, linewidth=1.6)


def _invariants(axis: Axes, record: HandLoadPlotInput, kind: str) -> None:
    attr = "midhands_force_n" if kind == "force" else "midhands_moment_nm"
    for part, color in _PART_COLOR.items():
        for strategy in _ordered(record):
            label = f"{_PART[part]} {kind} {_LABEL[strategy.strategy_id]}"
            _draw(
                axis,
                record.time_s,
                getattr(strategy, attr)[part],
                label,
                color,
                _STYLE[strategy.strategy_id],
            )


def _hands(axis: Axes, record: HandLoadPlotInput, side: str) -> None:
    attr = "force_left_n" if side == "LH" else "force_right_n"
    for strategy in _ordered(record):
        _draw(
            axis,
            record.time_s,
            getattr(strategy, attr),
            f"{side} force {_LABEL[strategy.strategy_id]}",
            _HAND_COLOR[(side, strategy.strategy_id)],
            "-",
        )


def _mark_phases(axis: Axes, phases: tuple[PhaseMarker, ...]) -> None:
    for phase in phases:
        axis.axvline(
            phase.time_s,
            color="#000000",
            linestyle=":",
            linewidth=0.8,
            label=f"_{phase.name}",
        )
        axis.text(
            phase.time_s,
            1.01,
            phase.name,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
            color="#000000",
            clip_on=False,
        )


def _legend(axis: Axes) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, fontsize=7, frameon=False)


def _floats(values: object, name: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _shaped(values: object, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    arr = _floats(values, name)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr.copy()


def _channels(values: object, n: int, *, force: bool) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError("midhands channels must be a mapping")
    allowed = {(n, 3)} if force else {(n,), (n, 3)}
    copied: dict[str, np.ndarray] = {}
    for part in COMPONENTS:
        if part not in values:
            raise ValueError(f"midhands channel missing {part}")
        arr = _floats(values[part], part)
        if arr.shape not in allowed:
            raise ValueError(f"{part} has shape {arr.shape}")
        copied[part] = arr.copy()
    return MappingProxyType(copied)


def _token(value: object, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be a hex string")
    return value


def _objective(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("objective must be a non-empty string")
    if any(
        phrase in value.casefold()
        for phrase in ("wasted energy", "biological optimality")
    ):
        raise ValueError("wasted energy and biological optimality are rejected")
    return value.strip()


def _phase(item: PhaseMarker | tuple[str, float]) -> PhaseMarker:
    name, time_s = (item.name, item.time_s) if isinstance(item, PhaseMarker) else item
    key = " ".join(str(name).replace("_", " ").replace("-", " ").split()).lower()
    canonical = _PHASES.get(key)
    if canonical is None:
        raise ValueError(f"unknown phase {name!r}")
    try:
        instant = float(time_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"phase {name!r} time must be a float") from exc
    if not math.isfinite(instant):
        raise ValueError(f"phase {name!r} time must be finite")
    return PhaseMarker(canonical, instant)


def _check_phases(phases: tuple[PhaseMarker, ...], start: float, end: float) -> None:
    names = [phase.name for phase in phases]
    times = [phase.time_s for phase in phases]
    if len(names) != len(set(names)):
        raise ValueError("phase names must be unique")
    if any(later <= earlier for earlier, later in zip(times, times[1:], strict=False)):
        raise ValueError("phase times must be strictly increasing")
    if any(instant < start or instant > end for instant in times):
        raise ValueError("phase times must lie on time_s")
    found = set(names)
    for earlier, later in (("Address", "Impact"), ("Impact", "Follow-Through")):
        if earlier not in found or later not in found:
            raise ValueError(
                "phases must run from Address through Impact to Follow-Through"
            )
        if times[names.index(earlier)] >= times[names.index(later)]:
            raise ValueError(
                "Address, Impact, and Follow-Through must be in that order"
            )


def _check_strategies(strategies: tuple[StrategyLoads, ...], n: int) -> None:
    if not all(isinstance(item, StrategyLoads) for item in strategies):
        raise TypeError("strategies must contain StrategyLoads")
    if sorted(item.strategy_id for item in strategies) != sorted(_IDS):
        raise ValueError("strategies must be baseline and min_squared_force")
    if any(item.force_left_n.shape[0] != n for item in strategies):
        raise ValueError("strategy length must match time_s")


def _from_mapping(item: Mapping[str, Any], n: int) -> StrategyLoads:
    if not isinstance(item, Mapping):
        raise TypeError("each strategy must be a mapping")
    forces = item.get("midhands_force_n") or {}
    moments = item.get("midhands_moment_nm") or {}
    if not isinstance(forces, Mapping) or not isinstance(moments, Mapping):
        raise TypeError("midhands channels must be mappings")
    return StrategyLoads(
        strategy_id=str(item["strategy_id"]),
        evidence=str(item["evidence"]),
        force_left_n=_gap(item, "force_left_n", (n, 3)),
        force_right_n=_gap(item, "force_right_n", (n, 3)),
        alpha_n=_gap(item, "alpha_n", (n,)),
        midhands_force_n={part: _gap(forces, part, (n, 3)) for part in COMPONENTS},
        midhands_moment_nm={part: _gap(moments, part, (n,)) for part in COMPONENTS},
    )


def _gap(mapping: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = mapping.get(key)
    if raw is None:
        return np.full(shape, np.nan)
    return _shaped(raw, shape, key)
