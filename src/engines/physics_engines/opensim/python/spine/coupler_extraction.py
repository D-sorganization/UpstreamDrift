"""Extract ``CoordinateCouplerConstraint``s from an OpenSim model (SPN-8b, #12754).

Surfaces a model's own kinematic assumptions as a :class:`CouplingLaw` so
studies can report them.  Each enforced coupler ``q_dep = f(q_ind...)`` whose
dependent coordinate is a mapped spine coordinate (``SpineInventory``) becomes
one ``(level, axis)`` row of the law, axis 0/1/2 = flexion-extension, lateral
bending, axial rotation (ISB ZXY, radians).

Function support (anything else raises :class:`UnsupportedCouplerError`, never
approximated silently):

* ``LinearFunction``, ``PolynomialFunction`` of degree <= 1 and
  ``MultivariatePolynomialFunction`` of order <= 1 with zero intercept:
  ``theta = c . phi``; exact.
* Only linear couplers present: :class:`LinearCouplingLaw`.
* Any ``SimmSpline`` / ``PiecewiseLinearFunction`` coupler of one variable
  ``s``: :class:`StateDependentCouplingLaw` scheduled on ``s``.  The law form
  ``theta = C(s) phi`` needs ``f(0) = 0``; ``C`` at a knot ``s_j`` is
  ``f(s_j) / s_j`` (``f'(0)`` at ``s_j = 0``) so the law is exact at the knots
  and a linear-in-``C`` interpolation between them (the spline shape between
  knots and its extrapolation outside the knot range are NOT reproduced).
* Only single-variable polynomials (degree <= 4, linear ones allowed too):
  :class:`PolynomialCouplingLaw`.

Disabled constraints (``isEnforced`` false) prescribe nothing and are skipped.
The ``scale_factor`` element written by the fixture is not an OpenSim property
and is ignored.  The law covers the contiguous level range spanned by the
enforced couplers; coordinates without a coupler are zero rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.engines.physics_engines.opensim.python.spine.inventory import SpineInventory
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.coupling import (
    CouplingLaw,
    LinearCouplingLaw,
    PolynomialCouplingLaw,
    StateDependentCouplingLaw,
)
from src.shared.python.spine.levels import SpinalLevel

__all__ = ["UnsupportedCouplerError", "extract_coupling_law"]

logger = logging.getLogger(__name__)

_ZERO_TOL = 1e-12
_MAX_POLY_DEGREE = 4


class UnsupportedCouplerError(NotImplementedError, ValueError):
    """A coupler function or form that cannot be expressed as a spine law."""


@dataclass(frozen=True)
class _Coupler:
    name: str
    level: SpinalLevel
    axis: int
    independent: tuple[str, ...]
    kind: str  # "linear" | "poly" | "table"
    slopes: np.ndarray | None = None  # linear: one slope per independent
    poly: np.ndarray | None = None  # poly: ascending coefficients (c0..c4)
    knots: tuple[float, ...] = ()  # table
    function: object | None = None  # table: OpenSim function


# ----------------------------------------------------------- SWIG adapters --


def _constraints(model: object) -> list[object]:
    osim = import_opensim()
    cset = model.getConstraintSet()  # type: ignore[attr-defined]
    items = (cset.get(i) for i in range(cset.getSize()))
    found = (osim.CoordinateCouplerConstraint.safeDownCast(c) for c in items)
    return [c for c in found if c is not None]


def _name_list(array: object) -> tuple[str, ...]:
    return tuple(array.get(i) for i in range(array.getSize()))  # type: ignore[attr-defined]


def _is_enforced(constraint: object) -> bool:
    return bool(constraint.get_isEnforced())  # type: ignore[attr-defined]


def _function_of(constraint: object) -> object:
    return constraint.getFunction()  # type: ignore[attr-defined]


def _numbers(vector: object) -> np.ndarray:
    return np.array(
        [vector.get(i) for i in range(vector.size())],  # type: ignore[attr-defined]
        dtype=float,
    )


def _eval(function: object, point: np.ndarray) -> float:
    vec = import_opensim().Vector(int(point.size), 0.0)
    for i, value in enumerate(point):
        vec.set(i, float(value))
    return float(function.calcValue(vec))  # type: ignore[attr-defined]


def _eval_slope(function: object) -> float:
    osim = import_opensim()
    order = osim.StdVectorInt()
    order.push_back(0)
    return float(function.calcDerivative(order, osim.Vector(1, 0.0)))  # type: ignore[attr-defined]


def _cast(function: object, cls_name: str) -> object:
    return getattr(import_opensim(), cls_name).safeDownCast(function)


def _spline_knots(function: object) -> tuple[float, ...]:
    count = function.getNumberOfPoints()  # type: ignore[attr-defined]
    return tuple(float(function.getX(i)) for i in range(count))  # type: ignore[attr-defined]


# ----------------------------------------------------------------- parsing --


def _unsupported(name: str, why: str) -> UnsupportedCouplerError:
    return UnsupportedCouplerError(f"coupler {name!r}: {why}")


def _check_intercept(name: str, intercept: float) -> None:
    if abs(intercept) > _ZERO_TOL:
        raise ValueError(
            f"coupler {name!r}: nonzero intercept {intercept!r} is not "
            "representable as theta = C phi"
        )


def _parse_linear(name: str, fn: object, indep: tuple[str, ...]) -> dict:
    coefs = _numbers(fn.getCoefficients())  # type: ignore[attr-defined]
    if coefs.size != len(indep) + 1:
        raise ValueError(
            f"coupler {name!r}: LinearFunction has {coefs.size} coefficients for "
            f"{len(indep)} independent coordinates (expected {len(indep) + 1})"
        )
    _check_intercept(name, float(coefs[-1]))
    return {"kind": "linear", "slopes": coefs[:-1]}


def _parse_polynomial(name: str, fn: object, indep: tuple[str, ...]) -> dict:
    if len(indep) != 1:
        raise _unsupported(name, "PolynomialFunction needs exactly one independent")
    coefs = _numbers(fn.getCoefficients())  # type: ignore[attr-defined]
    ascending = np.trim_zeros(coefs, "f")[::-1]
    if ascending.size == 0:
        ascending = np.zeros(1)
    if ascending.size - 1 > _MAX_POLY_DEGREE:
        raise _unsupported(name, f"polynomial degree above {_MAX_POLY_DEGREE}")
    _check_intercept(name, float(ascending[0]))
    if ascending.size <= 2:
        slope = float(ascending[1]) if ascending.size == 2 else 0.0
        return {"kind": "linear", "slopes": np.array([slope])}
    padded = np.zeros(_MAX_POLY_DEGREE + 1)
    padded[: ascending.size] = ascending
    return {"kind": "poly", "poly": padded}


def _parse_multivariate(name: str, fn: object, indep: tuple[str, ...]) -> dict:
    if int(fn.getOrder()) > 1 or int(fn.getDimension()) != len(indep):  # type: ignore[attr-defined]
        raise _unsupported(
            name, "MultivariatePolynomialFunction supported for order <= 1 only"
        )
    origin = np.zeros(len(indep))
    intercept = _eval(fn, origin)
    _check_intercept(name, intercept)
    slopes = np.array(
        [_eval(fn, np.eye(len(indep))[j]) - intercept for j in range(len(indep))]
    )
    return {"kind": "linear", "slopes": slopes}


def _parse_table(name: str, fn: object, indep: tuple[str, ...]) -> dict:
    if len(indep) != 1:
        raise _unsupported(name, "tabulated coupler needs exactly one independent")
    at_origin = _eval(fn, np.zeros(1))
    if abs(at_origin) > _ZERO_TOL:
        raise ValueError(
            f"coupler {name!r}: f(0) = {at_origin!r} must be 0 to be expressed "
            "as theta = C(s) phi"
        )
    return {"kind": "table", "knots": _spline_knots(fn), "function": fn}


_PARSERS = {
    "LinearFunction": _parse_linear,
    "PolynomialFunction": _parse_polynomial,
    "MultivariatePolynomialFunction": _parse_multivariate,
    "SimmSpline": _parse_table,
    "PiecewiseLinearFunction": _parse_table,
}


def _parse_function(name: str, fn: object, indep: tuple[str, ...]) -> dict:
    class_name = str(fn.getConcreteClassName())  # type: ignore[attr-defined]
    parser = _PARSERS.get(class_name)
    if parser is None:
        raise _unsupported(name, f"unsupported function type {class_name!r}")
    return parser(name, _cast(fn, class_name), indep)


def _locate(inventory: SpineInventory) -> dict[str, tuple[SpinalLevel, int]]:
    out: dict[str, tuple[SpinalLevel, int]] = {}
    for level in inventory.levels:
        for axis, coord in enumerate(inventory.coordinate_names[level]):
            out[coord] = (level, axis)
    return out


def _read_couplers(model: object, inventory: SpineInventory) -> list[_Coupler]:
    constraints = _constraints(model)
    if not constraints:
        raise ValueError(
            "model has no CoordinateCouplerConstraint; nothing to extract a "
            "coupling law from"
        )
    where = _locate(inventory)
    found: list[_Coupler] = []
    for con in constraints:
        name = str(con.getName())  # type: ignore[attr-defined]
        if not _is_enforced(con):
            logger.info("skipping disabled coupler %s", name)
            continue
        dep = str(con.getDependentCoordinateName())  # type: ignore[attr-defined]
        if dep not in where:
            raise ValueError(
                f"coupler {name!r}: dependent coordinate {dep!r} is not a spine "
                f"coordinate of inventory {inventory.model_id!r}"
            )
        indep = _name_list(con.getIndependentCoordinateNames())  # type: ignore[attr-defined]
        fields = _parse_function(name, _function_of(con), indep)
        level, axis = where[dep]
        found.append(_Coupler(name, level, axis, indep, **fields))
    if not found:
        raise ValueError("model has no enforced CoordinateCouplerConstraint")
    seen: set[tuple[SpinalLevel, int]] = set()
    for item in found:
        if (item.level, item.axis) in seen:
            raise ValueError(
                f"coupler {item.name!r}: dependent coordinate coupled twice"
            )
        seen.add((item.level, item.axis))
    return found


# ---------------------------------------------------------------- assembly --


def _independent_union(couplers: list[_Coupler]) -> tuple[str, ...]:
    names: list[str] = []
    for item in couplers:
        names.extend(n for n in item.independent if n not in names)
    return tuple(names)


def _level_range(
    couplers: list[_Coupler], inventory: SpineInventory
) -> tuple[SpinalLevel, ...]:
    ords = [c.level.ordinal for c in couplers]
    return tuple(lv for lv in inventory.levels if min(ords) <= lv.ordinal <= max(ords))


def _required_array(value: np.ndarray | None, name: str, what: str) -> np.ndarray:
    if value is None:
        raise _unsupported(name, f"coupler lacks {what}")
    return value


def _linear_rows(
    couplers: list[_Coupler], names: tuple[str, ...], levels: tuple[SpinalLevel, ...]
) -> dict[SpinalLevel, np.ndarray]:
    mats = {lv: np.zeros((3, len(names))) for lv in levels}
    for item in couplers:
        if item.kind != "linear":
            continue
        slopes = _required_array(item.slopes, item.name, "slopes")
        for slope, ind in zip(slopes, item.independent, strict=True):
            mats[item.level][item.axis, names.index(ind)] = slope
    return mats


def _build_polynomial(couplers: list[_Coupler], names, ident: tuple) -> CouplingLaw:
    terms = []
    for item in couplers:
        if item.kind == "poly":
            coef = _required_array(item.poly, item.name, "poly")
        elif len(item.independent) == 1:
            coef = np.zeros(_MAX_POLY_DEGREE + 1)
            coef[1] = _required_array(item.slopes, item.name, "slopes")[0]
        else:
            raise _unsupported(
                item.name, "multivariate linear cannot mix with polynomial couplers"
            )
        offset, coef = float(coef[0]), np.concatenate(([0.0], coef[1:]))
        driver = names.index(item.independent[0])
        terms.append((item.level, item.axis, driver, offset, 0.0, coef))
    law_id, version, citation = ident
    return PolynomialCouplingLaw(law_id, version, names, tuple(terms), citation)


def _build_state_dependent(
    couplers: list[_Coupler],
    names: tuple[str, ...],
    levels: tuple[SpinalLevel, ...],
    ident: tuple,
) -> CouplingLaw:
    tables = [c for c in couplers if c.kind == "table"]
    sched = tables[0].independent[0]
    for item in couplers:
        if item.kind == "poly":
            raise _unsupported(item.name, "polynomial cannot mix with tabulated")
        if item.kind == "table" and item.independent[0] != sched:
            raise _unsupported(
                item.name, f"tabulated couplers must share scheduling {sched!r}"
            )
    knots = np.unique(np.concatenate([np.array(t.knots) for t in tables]))
    base = _linear_rows(couplers, names, levels)
    col = names.index(sched)
    mats = {lv: np.repeat(base[lv][None], knots.size, axis=0) for lv in levels}
    for item in tables:
        for j, s in enumerate(knots):
            mats[item.level][j, item.axis, :] = 0.0
            mats[item.level][j, item.axis, col] = _table_gain(item, float(s))
    law_id, version, citation = ident
    return StateDependentCouplingLaw(
        law_id, version, names, sched, knots, mats, citation
    )


def _table_gain(item: _Coupler, s: float) -> float:
    if abs(s) <= _ZERO_TOL:
        return _eval_slope(item.function)
    return _eval(item.function, np.array([s])) / s


def extract_coupling_law(
    model: str | Path | object,
    inventory: SpineInventory,
    *,
    law_id: str,
    version: str,
    citation: str,
) -> CouplingLaw:
    """Read a model's spine couplers into a :class:`CouplingLaw`.

    Preconditions: ``model`` is an existing ``.osim`` path or a loaded
    ``opensim.Model``; ``inventory`` is a :class:`SpineInventory`; ``law_id``,
    ``version`` and ``citation`` are non-empty strings; the model has at least
    one enforced ``CoordinateCouplerConstraint`` whose dependent coordinates
    are mapped by ``inventory``.  Postconditions: the law's independent names
    are the union of the couplers' independent coordinates in first-use order;
    ``expand`` reproduces each coupler (exactly for linear and polynomial
    forms, at the knots for tabulated ones, see the module docstring).

    Raises:
        TypeError: ``inventory`` is not a :class:`SpineInventory`.
        ValueError: empty strings, missing file, no (enforced) couplers, an
            unmapped or doubly coupled dependent coordinate, a nonzero
            intercept or ``f(0)``.
        UnsupportedCouplerError: an unsupported function type or mixture
            (also a ``NotImplementedError``), naming the constraint.
    """
    if not isinstance(inventory, SpineInventory):
        raise TypeError(
            f"inventory must be a SpineInventory, got {type(inventory).__name__}"
        )
    for label, text in (
        ("law_id", law_id),
        ("version", version),
        ("citation", citation),
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} must be a non-empty string")
    if isinstance(model, (str, Path)):
        if not Path(model).is_file():
            raise ValueError(f"model file not found: {model}")
        model = import_opensim().Model(str(model))
    couplers = _read_couplers(model, inventory)
    names = _independent_union(couplers)
    levels = _level_range(couplers, inventory)
    ident = (law_id, version, citation)
    kinds = {c.kind for c in couplers}
    if "table" in kinds:
        return _build_state_dependent(couplers, names, levels, ident)
    if "poly" in kinds:
        return _build_polynomial(couplers, names, ident)
    return LinearCouplingLaw(
        law_id, version, _linear_rows(couplers, names, levels), names, citation
    )
