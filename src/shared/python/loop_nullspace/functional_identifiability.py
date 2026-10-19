"""Functional identifiability and per-source attribution quantities (NSE-UD-1).

For one frame the closed-loop allocation is ``x = x_p + N z`` (see
:mod:`decomposition`).  A *linear functional* ``y = s . x`` of the allocation
(a joint torque, a hand-split coordinate, an induced acceleration) either takes
one value on the whole fibre or it varies with ``z``.

Exact theorem versus numerical label
------------------------------------
* **Exact (theorem).**  ``s . x`` is constant over the affine fibre
  ``{x_p + N z}`` iff ``N^T s = 0``, i.e. iff ``s`` lies in ``range(A^T)``.
  This is a statement about exact arithmetic and the exact null space.
* **Numerical (this module).**  :func:`classify_functional` evaluates the
  *normalised annihilation residual*

      rho = ||N^T s||_2 / ||s_x * s||_2        (``s_x`` = ``variable_scale``)

  and labels the functional ``NUMERICALLY_INVARIANT`` when ``rho <= tol`` and
  ``FAMILY_DEPENDENT`` otherwise.  ``N`` already carries ``variable_scale``
  (``N = s_x * V`` with ``V`` orthonormal), so ``rho`` lies in ``[0, 1]`` with
  **no second scaling**.  ``rho`` is a classification quantity only: it is
  unchanged when ``s`` is rescaled and says nothing about the size of the
  variation in physical units.  The label is relative to the declared
  tolerance and the characteristic scales (``C-SCALE-FORCE-MOMENT``); it is
  not a proof of invariance.
* **Unbounded fibre.**  On an unbounded fibre any nonzero residual permits
  unbounded variation of ``s . x``; a small ``rho`` then bounds nothing.
* **Bounded domain (optional).**  When a bounded :class:`FeasibleSet` and a
  certified *enclosing* radius ``R`` (``||z - z0||_2 <= R`` for every admissible
  ``z``, in reduced coordinates, absolute units) are supplied, then

      |s . (x - x0)| <= ||N^T s||_2 * R,   x0 = x_p + N z0,

  and the full envelope width is at most ``2 ||N^T s||_2 R``.  ``R`` is an
  enclosing radius, not the Chebyshev inscribed radius, and the bound is in
  the units of ``s . x`` (it is *not* ``rho * R``; it scales linearly with
  ``s``).

Envelope certificates
---------------------
:func:`attribution_quantities` builds quantities for
:func:`~.envelopes.static_envelopes`, which certifies LP extrema over the
**declared polygonal approximation** of any friction cone.  An *outer*
approximation therefore yields certified outer bounds on the exact cone set;
an *inner* approximation yields inner bounds only.  No exact second-order-cone
envelope is claimed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.decomposition import (
    SCALE_CONVENTION,
    Decomposition,
)
from src.shared.python.loop_nullspace.envelopes import (
    BoundStatus,
    EnvelopeQuantity,
    QuantityFactory,
    static_envelopes,
    x_functional_quantities,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    readonly_float,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet

__all__ = [
    "NUMERIC_INVARIANCE_CONVENTION",
    "FunctionalIdentifiability",
    "FunctionalStatus",
    "VariationBound",
    "attribution_quantities",
    "classify_functional",
    "functional_record",
]

NUMERIC_INVARIANCE_CONVENTION = (
    "C-NUMERIC-INVARIANCE: rho = ||N^T s|| / ||s_x * s|| <= tol; a numerical "
    "label at the declared tolerance, not the exact theorem N^T s = 0"
)
_PARTITION_ATOL = 1e-9
_RADIUS_SLACK = 1e-9


class FunctionalStatus(Enum):
    """Numerical identifiability label of a linear functional."""

    NUMERICALLY_INVARIANT = "numerically_invariant"
    FAMILY_DEPENDENT = "family_dependent"


@dataclass(frozen=True)
class VariationBound:
    """Absolute bounded-domain bound on ``s . (x - x0)``.

    ``deviation_bound = ||N^T s|| * radius`` bounds ``|s . (x - x0)|`` and
    ``width_bound = 2 * deviation_bound`` bounds the envelope width, both in the
    units of ``s . x``.  ``radius`` is the caller's certified enclosing radius
    about ``center`` in reduced coordinates.
    """

    radius: float
    center: Array
    deviation_bound: float
    width_bound: float


@dataclass(frozen=True)
class FunctionalIdentifiability:
    """Result of :func:`classify_functional`.

    ``residual`` is ``rho`` (dimensionless, classification only);
    ``annihilation_norm`` is ``||N^T s||_2`` (units of ``s . x``);
    ``scaled_norm`` is ``||s_x * s||_2``; ``bound`` is present only when a
    bounded feasible set and an enclosing radius were supplied.
    """

    status: FunctionalStatus
    residual: float
    tol: float
    annihilation_norm: float
    scaled_norm: float
    bound: VariationBound | None = None


def _check_tol(tol: float) -> float:
    if isinstance(tol, bool) or not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be positive and finite, got {tol!r}")
    return float(tol)


def _check_selector(dec: Decomposition, selector: Any) -> Array:
    sel = np.asarray(selector, dtype=float)
    if sel.shape != (dec.dim_x,):
        raise ValueError(f"selector must have shape ({dec.dim_x},), got {sel.shape}")
    if not np.all(np.isfinite(sel)):
        raise ValueError("selector must be finite")
    return sel


def _coordinate_box(fs: FeasibleSet) -> tuple[Array, Array]:
    """Per-coordinate ``[lo, hi]`` of the set; ``ValueError`` unless bounded."""
    eye = np.eye(fs.n_z)
    quantities = [
        EnvelopeQuantity(f"z{i}", eye[i], 0.0, "1", 1.0) for i in range(fs.n_z)
    ]
    series = static_envelopes([fs], quantities)
    statuses = series.status_lower[0] + series.status_upper[0]
    if any(s is BoundStatus.INFEASIBLE for s in statuses):
        raise ValueError("feasible_set is infeasible; there is no domain to bound")
    if any(s is not BoundStatus.BOUNDED for s in statuses):
        raise ValueError("feasible_set must be bounded to certify a variation bound")
    return np.asarray(series.lower[0]), np.asarray(series.upper[0])


def _variation_bound(
    dec: Decomposition,
    annihilation: float,
    fs: FeasibleSet,
    radius: float,
    center: Any,
) -> VariationBound:
    if not isinstance(fs, FeasibleSet):
        raise TypeError(f"feasible_set must be FeasibleSet, got {type(fs).__name__}")
    if fs.n_z != dec.nullity:
        raise ValueError(f"feasible_set has n_z={fs.n_z}, nullity={dec.nullity}")
    if fs.decomposition is not None and fs.decomposition is not dec:
        raise ValueError("feasible_set was built from a different decomposition")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"enclosing_radius must be positive and finite, got {radius}")
    z0 = np.zeros(dec.nullity) if center is None else np.asarray(center, float)
    if z0.shape != (dec.nullity,) or not np.all(np.isfinite(z0)):
        raise ValueError(f"center must be finite with shape ({dec.nullity},)")
    if dec.nullity > 0:
        lo, hi = _coordinate_box(fs)
        reach = float(np.max(np.maximum(np.abs(lo - z0), np.abs(hi - z0))))
        if radius < reach * (1.0 - _RADIUS_SLACK):
            raise ValueError(
                f"enclosing_radius {radius:g} does not enclose the set about "
                f"center (a coordinate reaches {reach:g})"
            )
    dev = annihilation * radius
    return VariationBound(float(radius), readonly_float(z0, "center"), dev, 2.0 * dev)


def classify_functional(
    decomposition: Decomposition,
    selector: Any,
    *,
    tol: float,
    feasible_set: FeasibleSet | None = None,
    enclosing_radius: float | None = None,
    center: Any = None,
) -> FunctionalIdentifiability:
    """Label ``s . x`` numerically invariant or family dependent over the fibre.

    ``rho = ||N^T s||_2 / ||s_x * s||_2`` with ``N`` already scaled by
    ``variable_scale`` (no second scaling); ``s = 0`` is invariant with
    residual 0.  See the module docstring for the numerical-versus-exact
    distinction, the unbounded-fibre caveat and the optional absolute bound
    ``|s . (x - x0)| <= ||N^T s|| R`` (``feasible_set`` and ``enclosing_radius``
    must be given together; ``center`` is ``z0``, default 0).

    Preconditions: ``decomposition`` is a :class:`Decomposition` carrying
    ``variable_scale`` (``TypeError``/``ValueError``); ``selector`` is a finite
    ``(dim_x,)`` vector; ``tol`` positive and finite; for the bound, a *bounded*
    feasible set of the same nullity and ``enclosing_radius`` positive, finite
    and at least the per-coordinate reach of the set about ``center`` (a
    necessary condition for enclosure; the caller certifies the rest).
    Postconditions: ``0 <= residual <= 1``; status is invariant iff
    ``residual <= tol``; rescaling ``selector`` leaves status and residual
    unchanged and scales ``annihilation_norm`` and the bound linearly.
    """
    if not isinstance(decomposition, Decomposition):
        raise TypeError(
            f"decomposition must be Decomposition, got {type(decomposition).__name__}"
        )
    tolerance = _check_tol(tol)
    if decomposition.variable_scale is None:
        raise ValueError("decomposition carries no variable_scale; use decompose()")
    if (feasible_set is None) != (enclosing_radius is None):
        raise ValueError("feasible_set and enclosing_radius must be given together")
    sel = _check_selector(decomposition, selector)
    scaled = float(np.linalg.norm(decomposition.variable_scale * sel))
    annihilation = float(np.linalg.norm(decomposition.N.T @ sel))
    rho = 0.0 if scaled == 0.0 else min(annihilation / scaled, 1.0)
    bound = None
    if feasible_set is not None and enclosing_radius is not None:
        bound = _variation_bound(
            decomposition, annihilation, feasible_set, enclosing_radius, center
        )
    status = (
        FunctionalStatus.NUMERICALLY_INVARIANT
        if rho <= tolerance
        else FunctionalStatus.FAMILY_DEPENDENT
    )
    return FunctionalIdentifiability(
        status, rho, tolerance, annihilation, scaled, bound
    )


def functional_record(
    identifiability: FunctionalIdentifiability,
    values: Any,
    *,
    units: str,
    model: ModelProvenance,
    source: InputProvenance,
    hypotheses: Sequence[str] = (),
    conventions: Sequence[str] = (),
) -> QuantityRecord:
    """Tag ``values`` as ``INVARIANT`` or ``FAMILY_DEPENDENT`` (epic rule R1).

    A numerically invariant functional becomes an ``INVARIANT`` record carrying
    the numerical-invariance and scale conventions (and no hypotheses).  A
    family-dependent one becomes ``FAMILY_DEPENDENT`` and needs at least one
    allocation hypothesis (``ValueError`` otherwise).
    """
    if not isinstance(identifiability, FunctionalIdentifiability):
        raise TypeError("identifiability must be a FunctionalIdentifiability")
    invariant = identifiability.status is FunctionalStatus.NUMERICALLY_INVARIANT
    return QuantityRecord(
        values=np.atleast_1d(np.asarray(values, dtype=float)),
        units=units,
        allocation_dependence=(
            AllocationDependence.INVARIANT
            if invariant
            else AllocationDependence.FAMILY_DEPENDENT
        ),
        model_provenance=model,
        input_provenance=source,
        conventions=(NUMERIC_INVARIANCE_CONVENTION, SCALE_CONVENTION, *conventions),
        hypotheses=() if invariant else tuple(hypotheses),
    )


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _check_partition(partition: Mapping[str, Any], dim: int) -> dict[str, Array]:
    mats = {k: np.asarray(v, dtype=float) for k, v in partition.items()}
    for key, mat in mats.items():
        if mat.shape != (dim, dim) or not np.all(np.isfinite(mat)):
            raise ValueError(f"partition {key!r} must be finite ({dim}, {dim})")
    total = sum(mats.values())
    if np.max(np.abs(total - np.eye(dim))) > _PARTITION_ATOL:
        raise ValueError("source partition matrices must sum to the identity")
    return mats


def attribution_quantities(
    names: Sequence[str],
    task_selectors_by_source: Mapping[str, Any],
    *,
    partition: Mapping[str, Any],
    units: str,
    scale: float,
    task_map: Any = None,
) -> QuantityFactory:
    """Factory of per-source attribution quantities ``J_task L E_k x``.

    ``task_selectors_by_source[k]`` is the ``(len(names), dim_x)`` selector
    ``J_task L E_k`` of source ``k`` and ``partition[k]`` the ``(dim_x, dim_x)``
    source partition matrix ``E_k``.  The result is a
    :data:`~.envelopes.QuantityFactory` built on
    :func:`~.envelopes.x_functional_quantities`, so the intervals come from
    :func:`~.envelopes.static_envelopes` with their existing certificates (LP
    extrema over the declared polygonal cone approximation; outer
    approximations give certified outer bounds, inner ones inner bounds only).
    Quantities are named ``"<name>|<source>"``, grouped by source.

    Preconditions: both mappings share the same non-empty source keys
    (``ValueError``; ``TypeError`` for non-mappings); the partition matrices
    sum to the identity; one name per selector row; if ``task_map`` (``J_task
    L``) is given, each selector equals ``task_map @ E_k``.
    Postcondition: the sum of a task's quantities over the sources equals
    ``task_map x``; whenever ``task_map`` rows lie in ``range(A^T)`` (a total
    induced acceleration is fixed by the balance) that sum is invariant across
    the fibre.
    """
    selectors = _require_mapping(task_selectors_by_source, "task_selectors_by_source")
    parts = _require_mapping(partition, "partition")
    if not selectors:
        raise ValueError("at least one source is required")
    if set(selectors) != set(parts):
        raise ValueError("selectors and partition must name the same sources")
    n_task = len(names)
    blocks = {k: np.asarray(v, dtype=float) for k, v in selectors.items()}
    dim = next(iter(blocks.values())).shape[-1]
    if any(b.shape != (n_task, dim) for b in blocks.values()):
        raise ValueError(f"names must have one entry per selector row ({n_task})")
    mats = _check_partition({k: parts[k] for k in blocks}, dim)
    if task_map is not None:
        tmap = np.asarray(task_map, dtype=float)
        for key, block in blocks.items():
            if tmap.shape != (n_task, dim) or not np.allclose(
                block, tmap @ mats[key], atol=_PARTITION_ATOL
            ):
                raise ValueError(
                    f"selector of source {key!r} differs from task_map @ E_k"
                )
    labels = [f"{t}|{k}" for k in blocks for t in names]
    stacked = np.vstack(list(blocks.values()))
    return x_functional_quantities(labels, stacked, units=units, scale=scale)
