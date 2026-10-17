"""Arrays of the LAN JSON form (``arrays`` object) as validated ``SwingArrays``.

The JSON form lists every array as nested lists; NaN is serialised as ``null``.
``null`` is accepted only for CoP (undefined below the load threshold); anywhere
else it is refused.  Everything after conversion goes through the same checks as
the npz and CSV forms (:func:`_arrays.build_swing_arrays`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.force_plates.open_fp_format import _arrays, _meta
from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = ["arrays_from_json"]

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
