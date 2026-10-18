"""Cache identity of the sensitivity stage (NSX-99, #12728).

:func:`assumption_digest` is the SHA-256 of a canonical encoding of every
declaration the stage consumed: the user's options (covariances, gains, bounds,
mechanisms, strategies, outcome priority), the variance problem (dynamics, input
covariances, outcome rows, input classes), the variance declaration, the gate's
numbers, the comparison settings and arm settings, the labels and clubhead
speeds of the comparison contexts and the provider's own ``provenance``.  A change
of any one of them changes the digest, so a report computed under another prior
can never be mistaken for this one.

Every float, scalar or array element, is encoded at ``_SIGNIFICANT`` significant
digits (signed zero folded to zero) before hashing, so last-bit noise from a
different BLAS or numpy build does not change the digest while any change a
declaration can carry does.

Limit, stated rather than hidden: a context's bulky payload (feasible sets,
muscle models) is identified by the provider's ``provenance`` mapping, not hashed
here, and a callable (the nonlinear rollout) is identified by its qualified name.
A value lying exactly on a 12-digit rounding boundary can still flip; that is
far below any declared assumption's precision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = ["assumption_digest", "canonical"]


_SIGNIFICANT = 12


def _float(value: float) -> str:
    """``value`` at ``_SIGNIFICANT`` significant digits; non-finite as its name."""
    if not math.isfinite(value):
        return repr(value)
    return format(value + 0.0, f".{_SIGNIFICANT}g")


def _opaque(obj: Any) -> dict[str, str]:
    name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return {"opaque": f"{getattr(obj, '__module__', type(obj).__module__)}.{name}"}


def canonical(obj: Any) -> Any:
    """A JSON-safe, order-stable encoding of ``obj`` (arrays hashed by content)."""
    if isinstance(obj, np.ndarray):
        flat = np.asarray(obj, dtype=np.float64).ravel()
        text = ",".join(_float(float(v)) for v in flat)
        return {
            "ndarray": list(obj.shape),
            "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        }
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bool | int | str) or obj is None:
        return obj
    if isinstance(obj, float | np.floating):
        return _float(float(obj))
    if is_dataclass(obj) and not isinstance(obj, type):
        body = {f.name: canonical(getattr(obj, f.name)) for f in fields(obj)}
        return {"type": type(obj).__qualname__, "fields": body}
    if isinstance(obj, Mapping):
        return {str(k): canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, frozenset | set):
        return sorted((canonical(x) for x in obj), key=lambda x: json.dumps(x))
    if isinstance(obj, tuple | list):
        return [canonical(x) for x in obj]
    return _opaque(obj)


def assumption_digest(parts: Mapping[str, Any]) -> str:
    """SHA-256 (hex) of ``canonical(parts)``; ``parts`` must not be empty."""
    if not isinstance(parts, Mapping) or len(parts) == 0:
        raise ValueError("the assumptions to hash must be a non-empty mapping")
    text = json.dumps(canonical(parts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
