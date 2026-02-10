"""Backward compatibility shim - module moved to core.type_utils."""

import sys as _sys

from .core import type_utils as _real_module  # noqa: E402
from .core.type_utils import (  # noqa: F401
    T,
    clamp,
    coerce_numeric,
    ensure_list,
    ensure_tuple,
    first,
    flatten,
    is_integer,
    is_numeric,
    parse_slice,
    safe_bool,
    safe_float,
    safe_int,
    safe_str,
    to_numpy_array,
)

_sys.modules[__name__] = _real_module
