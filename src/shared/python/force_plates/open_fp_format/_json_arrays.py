"""Arrays of the LAN JSON form (``arrays`` object) as validated ``SwingArrays``.

The JSON form lists every array as nested lists; NaN is serialised as ``null``.
``null`` is accepted only for CoP (undefined below the load threshold); anywhere
else it is refused.  Everything after conversion goes through the same checks as
the npz and CSV forms (:func:`_arrays.build_swing_arrays`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.force_plates.contracts import loaded_net_cop
from src.shared.python.force_plates.open_fp_format import _arrays, _meta
from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = ["arrays_from_json", "reconcile_rounding"]

#: Largest error [N] of a load rounded to the JSON quantum.
_HALF_LOAD_N = 0.5 * fmt.HTTP_FORCE_QUANTUM_N
#: Largest error [mm] of a CoP coordinate rounded to the JSON quantum.
_HALF_COP_MM = 0.5 * fmt.HTTP_COP_QUANTUM_MM
#: Float allowance [mm] on the net-CoP rounding bound.
_NET_FLOAT_MM = 1e-9

#: Arrays in which ``null`` (NaN) is meaningful.
_NULLABLE = frozenset({"cop_L", "cop_R", "cop_C"})


def _element(value: object, where: str, nullable: bool) -> float:
    if value is None:
        if nullable:
            return float("nan")
        raise ValueError(f"{where} is null; null is only allowed for CoP arrays")
    return _meta.finite_number(value, where)


def _row(row: object, width: int, where: str, nullable: bool) -> list[float]:
    if not isinstance(row, list) or len(row) != width:
        raise ValueError(f"{where} must be a list of {width} numbers")
    return [_element(v, f"{where}[{i}]", nullable) for i, v in enumerate(row)]


def _column(raw: object, key: str, where: str) -> np.ndarray:
    name = f"{where}: arrays[{key!r}]"
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a list")
    nullable = key in _NULLABLE
    width = fmt.ARRAY_KEYS[key]
    if width is None:
        values = [_element(v, f"{name}[{i}]", nullable) for i, v in enumerate(raw)]
    else:
        values = [_row(r, width, f"{name}[{i}]", nullable) for i, r in enumerate(raw)]
    return np.array(values, dtype=np.float64)


def arrays_from_json(raw: object, where: str) -> _arrays.SwingArrays:
    """Convert the parsed ``arrays`` object of a swing into ``SwingArrays``.

    Raises ``ValueError`` for a key set other than the documented one, ragged or
    non-numeric lists, ``null`` outside the CoP arrays, and everything
    :func:`_arrays.build_swing_arrays` refuses.  Values stay in file units.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: 'arrays' must be an object")
    arrays: Any = dict(raw)
    _meta.check_keys(arrays, frozenset(fmt.ARRAY_KEYS), f"{where}: arrays")
    columns = {key: _column(arrays[key], key, where) for key in fmt.ARRAY_KEYS}
    return _arrays.build_swing_arrays(columns, where)


def _settle_threshold(load: np.ndarray, cop: np.ndarray) -> np.ndarray:
    """Load whose side of the CoP threshold agrees with the source CoP flag.

    The source sets CoP to NaN from the unrounded load.  A rounded load within
    half a quantum of the threshold can land on the wrong side of it; there
    the source flag decides, and the load moves to the threshold edge (by at
    most half a quantum, inside the declared rounding).  Larger conflicts are
    left for the trial contract to refuse.
    """
    thr = fmt.SOURCE_COP_MIN_LOAD_N
    unloaded = np.isnan(cop[:, 0])
    out = load.copy()
    below = unloaded & (load >= thr) & (load <= thr + _HALF_LOAD_N)
    out[below] = np.nextafter(thr, -np.inf)
    out[~unloaded & (load < thr) & (load >= thr - _HALF_LOAD_N)] = thr
    return out


def _settle_net(arrays: _arrays.SwingArrays, where: str) -> np.ndarray:
    """Net CoP recomputed from the rounded feet, after a rounding-bound check.

    The source net CoP and the per-foot CoPs are rounded independently, so they
    can disagree by more than the contract's 0.1 mm.  Per axis the disagreement
    is at most ``2 * h_c + h_F * |c_L - c_R| / (F_L + F_R)`` (each CoP off by
    ``h_c``; each load weight off by ``h_F``, to first order).  Within that
    bound the net CoP is recomputed from the feet; beyond it the swing is
    refused.
    """
    loads = {"left": arrays.fz_left, "right": arrays.fz_right}
    cops = {"left": arrays.cop_left, "right": arrays.cop_right}
    recomputed = loaded_net_cop(loads, cops)
    source = arrays.cop_combined
    if not np.array_equal(np.isnan(source), np.isnan(recomputed)):
        return source  # a NaN-pattern conflict is the contract's to refuse
    spread = np.nan_to_num(np.abs(arrays.cop_left - arrays.cop_right))
    total = (arrays.fz_left + arrays.fz_right)[:, None]
    bound = 2.0 * _HALF_COP_MM + _HALF_LOAD_N * spread / total + _NET_FLOAT_MM
    if np.any(np.nan_to_num(np.abs(source - recomputed)) > bound):
        raise ValueError(
            f"{where}: net CoP differs from the load-weighted feet by more than "
            "the JSON rounding allows"
        )
    return recomputed


def reconcile_rounding(arrays: _arrays.SwingArrays, where: str) -> _arrays.SwingArrays:
    """Resolve the JSON form's rounding artefacts before the trial contract.

    See :func:`_settle_threshold` and :func:`_settle_net`.  Every change stays
    within the declared rounding, which the client adds to the uncertainty.
    """
    settled = dataclasses.replace(
        arrays,
        fz_left=_settle_threshold(arrays.fz_left, arrays.cop_left),
        fz_right=_settle_threshold(arrays.fz_right, arrays.cop_right),
    )
    return dataclasses.replace(settled, cop_combined=_settle_net(settled, where))
