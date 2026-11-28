"""Synchronized signed component carpets and cumulative waterfalls (#15691)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.physics.energy_ledger_contract import (
    UNAVAILABLE,
    EnergyLedgerSeries,
    EnergyLedgerStep,
    LedgerScalar,
    UnavailableType,
)

_AXES: tuple[tuple[str, int], ...] = (("x", 0), ("y", 1), ("z", 2))
_DEFAULT_COMPS = (
    "kinetic_j",
    "potential_j",
    "actuator_work_j",
    "external_work_j",
    "dissipation_j",
)
_SPECS: dict[str, tuple[str, str, int | None]] = {
    "mechanical_energy_j": ("J", "", None),
    **{
        k: ("J", attr, None)
        for k, attr in (
            ("kinetic_j", "kinetic_j"),
            ("potential_j", "potential_j"),
            ("actuator_work_j", "actuator_work_j"),
            ("w_act", "actuator_work_j"),
            ("external_work_j", "external_work_j"),
            ("w_ext", "external_work_j"),
            ("dissipation_j", "dissipation_j"),
            ("w_diss", "dissipation_j"),
        )
    },
    **{
        f"{p}_{ax}_{s}": (u, attr, idx)
        for p, s, attr, u in (
            ("linear_momentum", "ns", "linear_momentum_ns", "N*s"),
            ("angular_momentum", "nms", "angular_momentum_nms", "N*m*s"),
            ("net_external_force", "n", "net_external_force_n", "N"),
            ("net_external_torque", "nm", "net_external_torque_nm", "N*m"),
        )
        for ax, idx in _AXES
    },
}


def _is_finite(v: object) -> bool:
    """Return True if v is a finite real number (excluding bool)."""
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _req_str(val: object, name: str) -> None:
    """Validate that val is a non-empty string."""
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _freeze_shape(
    arrays: Sequence[NDArray[np.float64] | NDArray[np.bool_]], shape: tuple[int, ...]
) -> None:
    """Verify shape of each array and mark it read-only."""
    for arr in arrays:
        if arr.shape != shape:
            raise ValueError(f"expected shape {shape}, got {arr.shape}")
        arr.setflags(write=False)


def _validate_times(times_s: tuple[float, ...]) -> None:
    """Ensure times_s is non-empty, finite, and strictly increasing."""
    if not times_s:
        raise ValueError("times_s must be non-empty")
    prev: float | None = None
    for idx, t in enumerate(times_s):
        if not _is_finite(t) or (prev is not None and float(t) <= prev):
            raise ValueError(f"invalid or non-increasing times_s at index {idx}")
        prev = float(t)


def _extract_spec(step: EnergyLedgerStep, attr: str, axis: int | None) -> LedgerScalar:
    """Extract scalar or vector-axis value from step, preserving UNAVAILABLE."""
    if not attr:
        ke, pe = step.kinetic_j, step.potential_j
        if isinstance(ke, UnavailableType) or isinstance(pe, UnavailableType):
            return UNAVAILABLE
        return float(ke) + float(pe)
    val = getattr(step, attr)
    if isinstance(val, UnavailableType):
        return UNAVAILABLE
    return float(val[axis]) if axis is not None else float(val)


@dataclass(frozen=True)
class SignedComponentCarpet:
    """Synchronized signed component carpet matrix with zero-centered bounds."""

    schema_version: str
    engine: str
    times_s: tuple[float, ...]
    component_labels: tuple[str, ...]
    values: NDArray[np.float64]
    unavailable_mask: NDArray[np.bool_]
    vmin: float
    vmax: float
    units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate matrix shape, UNAVAILABLE/NaN alignment, and symmetric bounds."""
        _validate_times(self.times_s)
        labels = self.component_labels
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("component_labels must be non-empty and unique")
        for s in (self.schema_version, self.engine, *labels):
            _req_str(s, "carpet label")
        if self.units and len(self.units) != len(labels):
            raise ValueError("units length must match component_labels length")
        _freeze_shape(
            (self.values, self.unavailable_mask), (len(labels), len(self.times_s))
        )
        if self.unavailable_mask.dtype != np.bool_ or not np.array_equal(
            np.isnan(self.values), self.unavailable_mask
        ):
            raise ValueError("unavailable_mask must be bool and align with NaN cells")
        avail = self.values[~self.unavailable_mask]
        if avail.size and not np.all(np.isfinite(avail)):
            raise ValueError("all available carpet cells must be finite")
        if not _is_finite(self.vmax) or self.vmax <= 0.0 or self.vmin != -self.vmax:
            raise ValueError("scale bounds must satisfy vmax > 0 and vmin == -vmax")

    def component_row(self, label: str) -> NDArray[np.float64]:
        """Return the synchronized 1D series for a component label."""
        return self.values[self.component_labels.index(label), :]


@dataclass(frozen=True)
class WaterfallTerm:
    """Single synchronized cumulative term in a work or impulse waterfall."""

    label: str
    unit: str
    sign: int
    available: bool
    running_total: NDArray[np.float64]
    final_value: LedgerScalar

    def __post_init__(self) -> None:
        """Validate term identity, availability consistency, and immutability."""
        _req_str(self.label, "label")
        _req_str(self.unit, "unit")
        if self.sign not in (-1, 1) or not isinstance(self.available, bool):
            raise ValueError("sign must be +/-1 and available must be bool")
        if self.running_total.ndim != 1 or self.running_total.size == 0:
            raise ValueError("running_total must be a non-empty 1D array")
        if not self.available:
            if (
                not np.all(np.isnan(self.running_total))
                or self.final_value is not UNAVAILABLE
            ):
                raise ValueError("unavailable WaterfallTerm requires NaN/UNAVAILABLE")
        elif not np.all(np.isfinite(self.running_total)) or not _is_finite(
            self.final_value
        ):
            raise ValueError("available WaterfallTerm must contain only finite values")
        self.running_total.setflags(write=False)


@dataclass(frozen=True)
class CumulativeWaterfall:
    """Synchronized cumulative work and impulse totals with closure residuals."""

    schema_version: str
    engine: str
    times_s: tuple[float, ...]
    w_act_j: NDArray[np.float64]
    w_ext_j: NDArray[np.float64]
    w_diss_j: NDArray[np.float64]
    de_total_j: NDArray[np.float64]
    energy_residual_j: NDArray[np.float64]
    closure_residual_j: LedgerScalar
    j_ext_ns: NDArray[np.float64]
    dp_ns: NDArray[np.float64]
    impulse_residual_ns: NDArray[np.float64]
    impulse_closure_residual_ns: LedgerScalar
    work_terms: tuple[WaterfallTerm, ...]
    impulse_terms: tuple[WaterfallTerm, ...]

    def __post_init__(self) -> None:
        """Validate array shapes against times_s and freeze array buffers."""
        _validate_times(self.times_s)
        n = len(self.times_s)
        _freeze_shape((self.w_act_j, self.w_ext_j, self.w_diss_j), (n,))
        _freeze_shape(
            (self.de_total_j, self.energy_residual_j, self.impulse_residual_ns), (n,)
        )
        _freeze_shape((self.j_ext_ns, self.dp_ns), (n, 3))

    def work_term(self, label: str) -> WaterfallTerm:
        """Return the WaterfallTerm in work_terms matching label."""
        return next(t for t in self.work_terms if t.label == label)

    def impulse_term(self, label: str) -> WaterfallTerm:
        """Return the WaterfallTerm in impulse_terms matching label."""
        return next(t for t in self.impulse_terms if t.label == label)


def build_signed_component_carpet(
    series: EnergyLedgerSeries,
    *,
    components: Sequence[str] | None = None,
    extra_components: Mapping[str, Sequence[LedgerScalar]] | None = None,
    min_scale: float = 1e-12,
) -> SignedComponentCarpet:
    """Build a synchronized signed component carpet from an EnergyLedgerSeries."""
    if not isinstance(series, EnergyLedgerSeries):
        raise TypeError("series must be an EnergyLedgerSeries")
    if not _is_finite(min_scale) or min_scale <= 0.0:
        raise ValueError("min_scale must be a positive finite float")
    steps = series.steps
    extras = dict(extra_components or {})
    for k, seq in extras.items():
        _req_str(k, "extra label")
        if len(seq) != len(steps) or any(
            v is not UNAVAILABLE and not _is_finite(v) for v in seq
        ):
            raise ValueError(f"extra component {k!r} has invalid length or values")
    raw = (
        list(_DEFAULT_COMPS) + [k for k in extras if k not in _DEFAULT_COMPS]
        if components is None
        else list(components)
    )
    if not raw or len(set(raw)) != len(raw):
        raise ValueError("carpet component labels must be non-empty and unique")
    labels = tuple(raw)
    vals = np.full((len(labels), len(steps)), np.nan, dtype=np.float64)
    mask = np.ones((len(labels), len(steps)), dtype=np.bool_)
    units: list[str] = []
    for r, lbl in enumerate(labels):
        if lbl in extras:
            units.append("custom")
            row: Sequence[LedgerScalar] = extras[lbl]
        elif lbl in _SPECS:
            unit, attr, ax = _SPECS[lbl]
            units.append(unit)
            row = [_extract_spec(s, attr, ax) for s in steps]
        else:
            raise KeyError(f"unknown carpet component: {lbl!r}")
        for c, cell in enumerate(row):
            if not isinstance(cell, UnavailableType):
                vals[r, c], mask[r, c] = float(cell), False
    avail = vals[~mask]
    vmax = max(float(np.max(np.abs(avail))) if avail.size else 0.0, float(min_scale))
    times = tuple(s.time_s for s in steps)
    sch, eng = series.schema_version, series.engine
    return SignedComponentCarpet(
        sch, eng, times, labels, vals, mask, -vmax, vmax, tuple(units)
    )


def _scalar_delta(
    steps: tuple[EnergyLedgerStep, ...], avail: bool, attr: str
) -> NDArray[np.float64]:
    """Compute cumulative delta from step 0, or NaN array if unavailable."""
    if not avail:
        return np.full(len(steps), np.nan, dtype=np.float64)
    vals: list[float] = []
    for step in steps:
        v = _extract_spec(step, attr, None)
        if isinstance(v, UnavailableType):
            return np.full(len(steps), np.nan, dtype=np.float64)
        vals.append(float(v))
    raw = np.array(vals, dtype=np.float64)
    return raw - raw[0]


def _make_term(
    label: str, unit: str, sign: int, avail: bool, arr: NDArray[np.float64]
) -> WaterfallTerm:
    """Construct a WaterfallTerm preserving UNAVAILABLE when not available."""
    return WaterfallTerm(
        label, unit, sign, avail, arr.copy(), float(arr[-1]) if avail else UNAVAILABLE
    )


def _impulse_totals(
    steps: tuple[EnergyLedgerStep, ...], times: tuple[float, ...], avail: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], LedgerScalar]:
    """Compute cumulative impulse J_ext, dP, residual norm, and max residual."""
    n = len(steps)
    if not avail:
        nan3 = np.full((n, 3), np.nan, dtype=np.float64)
        return nan3, nan3.copy(), np.full(n, np.nan, dtype=np.float64), UNAVAILABLE
    p = np.array([s.linear_momentum_ns for s in steps], dtype=np.float64)
    f = np.array([s.net_external_force_n for s in steps], dtype=np.float64)
    dp, j_ext = p - p[0:1, :], np.zeros_like(p)
    if n >= 2:
        dt = np.diff(np.asarray(times, dtype=np.float64))[:, None]
        j_ext[1:, :] = np.cumsum(0.5 * dt * (f[:-1, :] + f[1:, :]), axis=0)
    res = np.linalg.norm(dp - j_ext, axis=1)
    return j_ext, dp, res, float(np.max(res))


def build_cumulative_waterfall(series: EnergyLedgerSeries) -> CumulativeWaterfall:
    """Compute synchronized cumulative work and impulse waterfalls from a ledger."""
    if not isinstance(series, EnergyLedgerSeries):
        raise TypeError("series must be an EnergyLedgerSeries")
    steps, av = series.steps, series.availability
    times = tuple(s.time_s for s in steps)
    w_act = _scalar_delta(steps, av.actuator_work, "actuator_work_j")
    w_ext = _scalar_delta(steps, av.external_work, "external_work_j")
    w_diss = _scalar_delta(steps, av.dissipation, "dissipation_j")
    e_ok = av.kinetic and av.potential
    de_tot = _scalar_delta(steps, e_ok, "")
    if e_ok and av.actuator_work and av.external_work and av.dissipation:
        e_res = de_tot - (w_act + w_ext - w_diss)
        closure_j: LedgerScalar = float(np.max(np.abs(e_res)))
    else:
        e_res, closure_j = np.full(len(steps), np.nan, dtype=np.float64), UNAVAILABLE
    imp_ok = av.linear_momentum
    j_ext, dp, imp_res, imp_close = _impulse_totals(steps, times, imp_ok)
    w_specs = (
        ("W_act", 1, av.actuator_work, w_act),
        ("W_ext", 1, av.external_work, w_ext),
        ("W_diss", -1, av.dissipation, w_diss),
        ("dE_total", 1, e_ok, de_tot),
    )
    w_terms = tuple(_make_term(lbl, "J", sgn, ok, arr) for lbl, sgn, ok, arr in w_specs)
    j_ax = tuple(
        _make_term(f"J_ext_{a}", "N*s", 1, imp_ok, j_ext[:, i]) for a, i in _AXES
    )
    p_ax = tuple(_make_term(f"dP_{a}", "N*s", 1, imp_ok, dp[:, i]) for a, i in _AXES)
    sch, eng, i_terms = series.schema_version, series.engine, j_ax + p_ax
    return CumulativeWaterfall(
        sch,
        eng,
        times,
        w_act,
        w_ext,
        w_diss,
        de_tot,
        e_res,
        closure_j,
        j_ext,
        dp,
        imp_res,
        imp_close,
        w_terms,
        i_terms,
    )
