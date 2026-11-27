"""Invariant hand-load dashboard. Missing series stay UNAVAILABLE, never zero. Caller closes the figure."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COMPONENTS = ("total", "ztcf", "zvcf")
_IDS = ("baseline", "min_squared_force")
_EVIDENCE = frozenset({"allocation-only", "forward-replayed"})
_PART = {"total": "Total", "ztcf": "ZTCF", "zvcf": "ZVCF"}
_PART_COLOR = {"total": "#000000", "ztcf": "#0072B2", "zvcf": "#D55E00"}
_HAND_COLOR = {
    ("LH", "baseline"): "#E69F00",
    ("LH", "min_squared_force"): "#56B4E9",
    ("RH", "baseline"): "#CC79A7",
    ("RH", "min_squared_force"): "#009E73",
}
_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_HASH = re.compile(r"^[0-9a-fA-F]{8,128}$")


class SimulationOutputLike(Protocol):
    """Fields the other lane's SimulationOutput exposes to this adapter."""

    time_s: np.ndarray
    git_sha: str
    data_hash: str
    objective: str
    strategies: Sequence[Mapping[str, Any]]


@dataclass(frozen=True, eq=False)
class StrategyLoads:
    """One allocation. NaN is missing data, never a physical zero."""

    strategy_id: str
    evidence: str
    force_left_n: np.ndarray
    force_right_n: np.ndarray
    midhands_force_n: Mapping[str, np.ndarray]
    midhands_moment_nm: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.strategy_id not in _IDS:
            raise ValueError(f"unknown strategy_id {self.strategy_id!r}")
        if self.evidence not in _EVIDENCE:
            raise ValueError(f"evidence must be one of {sorted(_EVIDENCE)}")
        left = _floats(self.force_left_n, "force_left_n")
        n = int(left.shape[0])
        object.__setattr__(self, "force_left_n", _shaped(left, (n, 3), "force_left_n"))
        object.__setattr__(
            self, "force_right_n", _shaped(self.force_right_n, (n, 3), "force_right_n")
        )
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
    """Both strategies, hex provenance, and an objective that is not a banned claim."""

    time_s: np.ndarray
    git_sha: str
    data_hash: str
    objective: str
    strategies: tuple[StrategyLoads, ...]

    def __post_init__(self) -> None:
        time = _floats(self.time_s, "time_s")
        if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all():
            raise ValueError("time_s must be a finite vector of length >= 2")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be strictly increasing")
        _check_strategies(self.strategies, int(time.size))
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "git_sha", _token(self.git_sha, _SHA, "git_sha"))
        object.__setattr__(
            self, "data_hash", _token(self.data_hash, _HASH, "data_hash")
        )
        object.__setattr__(self, "objective", _objective(self.objective))


def adapt_simulation_output(source: SimulationOutputLike) -> HandLoadPlotInput:
    """Copy the record. Missing channels become NaN and are not zero-filled."""
    time = _floats(source.time_s, "time_s")
    if time.ndim != 1:
        raise ValueError("time_s must be one-dimensional")
    n = int(time.shape[0])
    return HandLoadPlotInput(
        time_s=time,
        git_sha=source.git_sha,
        data_hash=source.data_hash,
        objective=source.objective,
        strategies=tuple(_from_mapping(item, n) for item in source.strategies),
    )


def plot_invariant_dashboard(record: HandLoadPlotInput) -> Figure:
    """Total, ZTCF, and ZVCF beside the hand splits, with badge and provenance."""
    fig, raw = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    axes = list(np.atleast_1d(raw).ravel())
    _stamp(fig, record)
    spec = (
        ("Invariant Midhands Force", "Force (N)"),
        ("Invariant Midhands Moment", "Moment (N·m)"),
        ("Left-Hand Force", "Force (N)"),
        ("Right-Hand Force", "Force (N)"),
    )
    for axis, (title, ylabel) in zip(axes, spec, strict=True):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Time (s)")
    _invariants(axes[0], record, "force")
    _invariants(axes[1], record, "moment")
    _hands(axes[2], record, "LH")
    _hands(axes[3], record, "RH")
    return fig


def _ordered(record: HandLoadPlotInput) -> tuple[StrategyLoads, ...]:
    by_id = {item.strategy_id: item for item in record.strategies}
    return tuple(by_id[name] for name in _IDS)


def _stamp(fig: Figure, record: HandLoadPlotInput) -> None:
    fig.suptitle(f"Objective: {record.objective}")
    badge = " ".join(dict.fromkeys(item.evidence for item in _ordered(record)))
    footer = f"Provenance: git {record.git_sha} | data {record.data_hash}"
    marks = (
        ("evidence-badge", badge, 0.99, 0.985, "right", "top"),
        ("provenance-footer", footer, 0.01, 0.01, "left", "bottom"),
    )
    for gid, text, xpos, ypos, ha, va in marks:
        fig.text(xpos, ypos, text, ha=ha, va=va, fontsize=8).set_gid(gid)


def _magnitude(values: np.ndarray) -> np.ndarray:
    """Return a copied norm. Non-finite rows stay NaN."""
    arr = _floats(values, "values")
    if arr.ndim == 1:
        return arr.copy()
    out = np.full(arr.shape[0], np.nan)
    finite = np.isfinite(arr).all(axis=1)
    if np.any(finite):
        out[finite] = np.linalg.norm(arr[finite], axis=1)
    return out


def _draw(
    axis: Axes,
    time_s: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    style: str,
) -> None:
    drawn = _magnitude(values)
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
            name = strategy.strategy_id.replace("_", " ")
            style = "-" if strategy.strategy_id == "baseline" else "--"
            _draw(
                axis,
                record.time_s,
                getattr(strategy, attr)[part],
                f"{_PART[part]} {kind} {name}",
                color,
                style,
            )


def _hands(axis: Axes, record: HandLoadPlotInput, side: str) -> None:
    attr = "force_left_n" if side == "LH" else "force_right_n"
    for strategy in _ordered(record):
        _draw(
            axis,
            record.time_s,
            getattr(strategy, attr),
            f"{side} force {strategy.strategy_id.replace('_', ' ')}",
            _HAND_COLOR[(side, strategy.strategy_id)],
            "-",
        )


def _floats(values: object, name: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _shaped(values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = _floats(values, name)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr.copy()


def _channels(values: object, n: int, *, force: bool) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError("midhands channels must be a mapping")
    allowed: set[tuple[int, ...]] = {(n, 3)} if force else {(n,), (n, 3)}
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
    banned = ("wasted energy", "biological optimality")
    if any(phrase in value.casefold() for phrase in banned):
        raise ValueError("wasted energy and biological optimality are rejected")
    return value.strip()


def _check_strategies(strategies: tuple[StrategyLoads, ...], n: int) -> None:
    if sorted(item.strategy_id for item in strategies) != sorted(_IDS):
        raise ValueError("strategies must be baseline and min_squared_force")
    if any(item.force_left_n.shape[0] != n for item in strategies):
        raise ValueError("strategy length must match time_s")


def _from_mapping(item: Mapping[str, Any], n: int) -> StrategyLoads:
    forces = item.get("midhands_force_n") or {}
    moments = item.get("midhands_moment_nm") or {}
    return StrategyLoads(
        strategy_id=str(item["strategy_id"]),
        evidence=str(item["evidence"]),
        force_left_n=_gap(item, "force_left_n", (n, 3)),
        force_right_n=_gap(item, "force_right_n", (n, 3)),
        midhands_force_n={part: _gap(forces, part, (n, 3)) for part in COMPONENTS},
        midhands_moment_nm={part: _gap(moments, part, (n,)) for part in COMPONENTS},
    )


def _gap(mapping: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = mapping.get(key)
    if raw is None:
        return np.full(shape, np.nan)
    return _shaped(raw, shape, key)
