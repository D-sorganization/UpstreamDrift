"""Sampled feasible component ranges and their classification (#15494, Slice B).

Given several already-admitted allocations of one fixed motion, report the
sampled min/max/width of each component and classify it as a model-implied
invariant, a prior-constrained narrow interval, a variable, a quantity whose
sampled spread is merely small, or one that did not move in the samples. A
small sampled spread is never promoted to "prior-constrained narrow" on its
own: that label needs a caller-declared feasible interval, established
independently of the samples, that is itself narrow. This module does not solve an allocation,
certify feasibility or replay dynamics; invariance is only ever *declared* by
the caller and then checked against the samples, never inferred from them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import readonly_float

__all__ = ["classify_component_ranges"]

_MODEL_INVARIANT = "MODEL_INVARIANT"
_INVARIANT_VIOLATED = "INVARIANT_VIOLATED"
_UNMOVED = "UNMOVED_IN_SAMPLES"
_NARROW = "PRIOR_CONSTRAINED_NARROW"
_SMALL_SPREAD = "SAMPLED_SMALL_SPREAD"
_VARIABLE = "VARIABLE"


def _classify_one(
    max_width: float,
    atol: float,
    declared: bool,
    prior: tuple[float, float] | None,
    feasible: tuple[float, float] | None,
    narrow_fraction: float,
) -> str:
    if max_width <= atol:
        return _MODEL_INVARIANT if declared else _UNMOVED
    if declared:
        return _INVARIANT_VIOLATED
    if prior is None:
        return _VARIABLE
    limit = narrow_fraction * (prior[1] - prior[0])
    if feasible is not None and feasible[1] - feasible[0] <= limit:
        return _NARROW
    return _SMALL_SPREAD if max_width <= limit else _VARIABLE


def _interval(
    label: str, name: str, bounds: tuple[float, float]
) -> tuple[float, float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"{label} for {name!r} must be finite with lo < hi")
    return lo, hi


def _check_names(label: str, given: Iterable[str], known: Sequence[str]) -> None:
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"{label} names unknown components: {unknown}")


def classify_component_ranges(
    histories: ArrayLike,
    *,
    names: Sequence[str],
    units: Sequence[str],
    atols: Sequence[float],
    declared_invariant: Sequence[str] = (),
    prior_bounds: Mapping[str, tuple[float, float]] | None = None,
    feasible_bounds: Mapping[str, tuple[float, float]] | None = None,
    narrow_fraction: float = 0.1,
) -> dict[str, Any]:
    """Classify component spreads over aligned candidate histories.

    Preconditions: ``histories`` is ``(K, T, C)`` with ``K >= 2`` candidates
    on one shared timebase, ``C == len(names) == len(units)``, all finite, in
    the units given. ``atols`` are positive finite absolute tolerances; the
    first is primary and each is classified separately so tolerance dependence
    is visible. ``prior_bounds`` maps a component to a declared ``(lo, hi)``
    prior; samples outside it are rejected, never clipped. ``feasible_bounds`` maps a
    component to a ``(lo, hi)`` feasible interval established *independently of
    the samples* (e.g. solver-certified extrema); it must lie inside the prior
    and contain every sample. Only a feasible interval no wider than
    ``narrow_fraction`` of the prior earns ``PRIOR_CONSTRAINED_NARROW``; a
    small sampled spread alone is ``SAMPLED_SMALL_SPREAD``.

    Postconditions: each component reports per-sample min/max, the largest
    width, its class at every tolerance and ``tolerance_dependent``.
    ``ok`` is false iff a declared invariant moved. Bounds are pointwise per
    sample and not claimed to be jointly reachable over the whole swing.

    Raises:
        ValueError: for invalid shapes, non-finite values, bad tolerances,
            unknown names, malformed priors or samples outside a prior.
    """
    data = readonly_float(histories, "histories")
    if data.ndim != 3 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("histories must be (K>=2 candidates, T>=1, C)")
    names = tuple(names)
    units = tuple(units)
    if data.shape[2] != len(names) or len(units) != len(names):
        raise ValueError("names and units must match the component axis")
    if len(set(names)) != len(names):
        raise ValueError("component names must be unique")
    tols = tuple(float(a) for a in atols)
    if not tols or not all(math.isfinite(a) and a > 0.0 for a in tols):
        raise ValueError("atols must be non-empty, positive and finite")
    if not (math.isfinite(narrow_fraction) and 0.0 < narrow_fraction <= 1.0):
        raise ValueError("narrow_fraction must be in (0, 1]")
    declared = frozenset(declared_invariant)
    priors = dict(prior_bounds or {})
    feasibles = dict(feasible_bounds or {})
    _check_names("declared_invariant", declared, names)
    _check_names("prior_bounds", priors, names)
    _check_names("feasible_bounds", feasibles, names)

    components = []
    for idx, name in enumerate(names):
        series = data[:, :, idx]
        prior = priors.get(name)
        if prior is not None:
            prior = _interval("prior", name, prior)
            if series.min() < prior[0] or series.max() > prior[1]:
                raise ValueError(f"samples for {name!r} fall outside its prior")
        feasible = feasibles.get(name)
        if feasible is not None:
            feasible = _interval("feasible bounds", name, feasible)
            if prior is None:
                raise ValueError(f"feasible bounds for {name!r} need a prior")
            if feasible[0] < prior[0] or feasible[1] > prior[1]:
                raise ValueError(f"feasible bounds for {name!r} exceed its prior")
            if series.min() < feasible[0] or series.max() > feasible[1]:
                raise ValueError(
                    f"samples for {name!r} fall outside its feasible bounds"
                )
        lower, upper = series.min(axis=0), series.max(axis=0)
        max_width = float((upper - lower).max())
        classes = [
            _classify_one(
                max_width, a, name in declared, prior, feasible, narrow_fraction
            )
            for a in tols
        ]
        components.append(
            {
                "name": name,
                "units": units[idx],
                "min": lower.tolist(),
                "max": upper.tolist(),
                "max_width": max_width,
                "classification": classes[0],
                "classification_by_atol": classes,
                "tolerance_dependent": len(set(classes)) > 1,
            }
        )
    return {
        "candidate_count": int(data.shape[0]),
        "sample_count": int(data.shape[1]),
        "atols": list(tols),
        "components": components,
        "ok": all(c["classification"] != _INVARIANT_VIOLATED for c in components)
        and all(
            _INVARIANT_VIOLATED not in c["classification_by_atol"] for c in components
        ),
        "pointwise_bounds_jointly_reachable": False,
    }
