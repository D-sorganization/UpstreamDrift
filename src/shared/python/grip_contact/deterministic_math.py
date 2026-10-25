"""Platform-independent scalar math for generated grip-contact goldens (#13552).

The Simscape golden vectors are regenerated and compared bit-for-bit on every
CI runner, but the obvious numpy/stdlib calls are not reproducible across
machines:

* ``@``, ``np.dot`` and ``np.linalg.norm`` dispatch to BLAS, whose rounding
  depends on the CPU kernel OpenBLAS selects. The rest cases leave ~1e-9 N
  of ~40 N opposing forces, so that noise exceeded the 1e-12 golden tolerance.
* ``math.exp``, ``math.tanh``, ``math.sin``, ``math.cos`` and ``math.atan2``
  come from the platform C library; glibc and the MSVC runtime differ in the
  last ulp, so a fixture generated on Windows did not regenerate on Linux.

Everything here uses only IEEE-754 basic operations (``+ - * /``, ``sqrt``),
``floor`` and ``ldexp`` in a fixed order, which are correctly rounded and
therefore bit-identical on every platform. Accuracy is within a few ulp of
the library functions, far inside every tolerance that consumes the results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "closest_point_segment3",
    "det_atan2",
    "det_exp",
    "det_sin_cos",
    "det_tanh",
    "dot3",
    "matmul3",
    "matvec3",
    "norm3",
]

_LN2_HI = 6.93147180369123816490e-01  # fdlibm split: k * _LN2_HI is exact
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e00
_EXPM1_TERMS = 18  # remainder < 1e-18 relative for |r| <= ln2 / 2
_PIO2_HI = 1.57079632673412561417e00  # fdlibm split: k * _PIO2_HI is exact
_PIO2_LO = 6.07710050650619224932e-11
_INV_PIO2 = 6.36619772367581382433e-01
_PI_HI = 3.14159265358979311600e00  # float(pi)
_PI_LO = 1.22464679914735317723e-16  # pi - float(pi)
_HALF_PI_HI = 1.57079632679489655800e00
_HALF_PI_LO = 6.12323399573676588613e-17
_TRIG_TERMS = 13  # sin/cos Taylor terms; remainder < 1e-30 for |r| <= pi / 4
_ATAN_TERMS = 30  # atan series terms; remainder < 1e-40 for |z| <= 0.2


def dot3(a: Any, b: Any) -> float:
    """``a . b`` for 3-vectors, summed left to right."""
    return (float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])) + float(
        a[2]
    ) * float(b[2])


def norm3(a: Any) -> float:
    """Euclidean norm of a 3-vector via :func:`dot3`."""
    return math.sqrt(dot3(a, a))


def matvec3(m: Any, v: Any) -> np.ndarray:
    """``m @ v`` for a 3x3 matrix and a 3-vector via :func:`dot3`."""
    return np.array([dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)], dtype=float)


def matmul3(a: Any, b: Any) -> np.ndarray:
    """``a @ b`` for 3x3 matrices via :func:`dot3`."""
    bt = np.asarray(b, dtype=float).T
    return np.array([[dot3(a[i], bt[j]) for j in range(3)] for i in range(3)])


def closest_point_segment3(
    point: np.ndarray, p0: np.ndarray, p1: np.ndarray
) -> np.ndarray:
    """Closest point on segment ``[p0, p1]`` (``pair_contact`` math)."""
    d = p1 - p0
    d_sq = dot3(d, d)
    if d_sq < 1e-14:
        return p0.copy()
    t = min(1.0, max(0.0, dot3(point - p0, d) / d_sq))
    return p0 + t * d


def _expm1_small(r: float) -> float:
    """``exp(r) - 1`` for ``|r| <= ln2 / 2`` by a fixed-length Horner series."""
    acc = 0.0
    for n in range(_EXPM1_TERMS, 0, -1):
        acc = r / n * (1.0 + acc)
    return acc


def det_exp(x: float) -> float:
    """``exp(x)`` by Cody-Waite reduction and :func:`_expm1_small` (~1 ulp)."""
    if x < -745.0:
        return 0.0
    k = math.floor(x * _INV_LN2 + 0.5)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    return math.ldexp(1.0 + _expm1_small(r), int(k))


def det_tanh(x: float) -> float:
    """``tanh(x)`` from :func:`det_exp`, accurate for small ``|x|`` too (~2 ulp)."""
    ax = abs(x)
    if ax > 20.0:
        t = 1.0
    else:
        two = 2.0 * ax
        e = _expm1_small(two) if two <= _LN2_HI / 2.0 else det_exp(two) - 1.0
        t = e / (e + 2.0)
    return t if x >= 0.0 else -t


def _sin_cos_reduced(r: float) -> tuple[float, float]:
    """``(sin r, cos r)`` for ``|r| <= pi / 4`` by fixed-length Horner series."""
    r2 = r * r
    s = c = 0.0
    for n in range(_TRIG_TERMS, 0, -1):
        s = 1.0 - r2 / ((2 * n) * (2 * n + 1)) * s
        c = 1.0 - r2 / ((2 * n - 1) * (2 * n)) * c
    return r * s, c


def det_sin_cos(x: float) -> tuple[float, float]:
    """``(sin x, cos x)`` for ``|x| < 1e5`` by quadrant reduction (~1 ulp).

    Preconditions:
        ``x`` is finite and ``|x| < 1e5`` (golden angles are below ``pi``).
    """
    if not (math.isfinite(x) and abs(x) < 1.0e5):
        raise ValueError(f"det_sin_cos needs finite |x| < 1e5, got {x!r}")
    k = math.floor(x * _INV_PIO2 + 0.5)
    r = (x - k * _PIO2_HI) - k * _PIO2_LO
    s, c = _sin_cos_reduced(r)
    quadrant = int(k) % 4
    if quadrant == 0:
        return s, c
    if quadrant == 1:
        return c, -s
    if quadrant == 2:
        return -s, -c
    return -c, s


def _atan_unit(z: float) -> float:
    """``atan(z)`` for ``0 <= z <= 1``: two half-angle steps, then a series."""
    for _ in range(2):
        z = z / (1.0 + math.sqrt(1.0 + z * z))
    z2 = z * z
    acc = 0.0
    for n in range(_ATAN_TERMS, 0, -1):
        acc = 1.0 / (2 * n - 1) - z2 * acc
    return 4.0 * (z * acc)


def det_atan2(y: float, x: float) -> float:
    """``atan2(y, x)`` with :func:`_atan_unit` (~2 ulp)."""
    ay, ax = abs(y), abs(x)
    if ax == 0.0 and ay == 0.0:
        return math.copysign(0.0, y) if x >= 0.0 else math.copysign(_PI_HI, y)
    if ay <= ax:
        a = _atan_unit(ay / ax)
    else:
        a = (_HALF_PI_HI - _atan_unit(ax / ay)) + _HALF_PI_LO
    if x < 0.0:
        a = (_PI_HI - a) + _PI_LO
    return a if y >= 0.0 else -a
