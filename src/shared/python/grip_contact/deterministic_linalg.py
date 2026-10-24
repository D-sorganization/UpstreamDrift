"""Platform-independent small dense linear algebra for the damping design (#13720).

``design_damping`` feeds the ``solref`` damping of the generated MJCF text,
which is compared byte-for-byte. Its ``scipy.linalg.eigh`` /
``scipy.optimize.nnls`` and the ``@`` products around them dispatch to
OpenBLAS/LAPACK, whose kernel (chosen from the CPU, or ``OPENBLAS_CORETYPE``)
changes the last ulp. This module re-implements the 6x6 pieces with only
IEEE-754 ``+ - * /`` and ``sqrt`` in a fixed order (the contract of
:mod:`src.shared.python.grip_contact.deterministic_math`, whose ``dot3`` is
reused for 3-vectors), so the result is bit-identical on every platform.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.shared.python.grip_contact.deterministic_math import dot3

__all__ = ["det_eigh_generalized", "det_matmul", "det_nnls", "det_quad_form"]

_JACOBI_SWEEPS = 60
_NNLS_TOL = 1e-12


def _rows(a: Any) -> list[list[float]]:
    return [[float(v) for v in row] for row in a]


def _dot(a: Any, b: Any) -> float:
    if len(a) == 3 and len(b) == 3:
        return dot3(a, b)
    acc = 0.0
    for x, y in zip(a, b, strict=True):
        acc += float(x) * float(y)
    return acc


def det_matmul(a: Any, b: Any) -> np.ndarray:
    """``a @ b`` for small 2-D arrays, each entry summed left to right."""
    cols = list(zip(*_rows(b), strict=True))
    return np.array([[_dot(row, c) for c in cols] for row in _rows(a)], dtype=float)


def det_quad_form(v: Any, m: Any) -> float:
    """``v @ m @ v`` for a vector and a square matrix, summed left to right."""
    return _dot(v, [_dot(row, v) for row in _rows(m)])


def _cholesky(m: list[list[float]]) -> list[list[float]]:
    n = len(m)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = m[i][j] - _dot(low[i][:j], low[j][:j])
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                low[i][i] = math.sqrt(s)
            else:
                low[i][j] = s / low[j][j]
    return low


def _forward(low: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``L x = b``."""
    x: list[float] = []
    for i, row in enumerate(low):
        x.append((b[i] - _dot(row[:i], x)) / row[i])
    return x


def _backward(low: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``L^T x = b``."""
    n = len(low)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        tail = _dot([low[k][i] for k in range(i + 1, n)], x[i + 1 :])
        x[i] = (b[i] - tail) / low[i][i]
    return x


def _jacobi(a: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Cyclic Jacobi eigen-decomposition of a symmetric matrix."""
    n = len(a)
    a = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(_JACOBI_SWEEPS):
        diag = sum(a[i][i] * a[i][i] for i in range(n))
        off = sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n))
        if off <= 1e-34 * diag:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def det_eigh_generalized(k: Any, m: Any) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of ``K x = w2 M x`` (``K`` symmetric, ``M`` positive definite).

    Returns ``(w2, vec)`` ascending with ``M``-orthonormal columns, like
    ``scipy.linalg.eigh(k, m)``; each column's largest entry is positive.
    """
    km, mm = _rows(k), _rows(m)
    n = len(km)
    low = _cholesky(mm)
    # C = L^-1 K L^-T: solve column-wise, then transpose and solve again.
    half = [_forward(low, [km[i][j] for i in range(n)]) for j in range(n)]
    cmat = [_forward(low, [half[j][i] for j in range(n)]) for i in range(n)]
    sym = [[0.5 * (cmat[i][j] + cmat[j][i]) for j in range(n)] for i in range(n)]
    vals, vecs = _jacobi(sym)
    order = sorted(range(n), key=lambda j: (vals[j], j))
    out = np.zeros((n, n))
    for dst, src in enumerate(order):
        phi = _backward(low, [vecs[i][src] for i in range(n)])
        big = max(range(n), key=lambda i: (abs(phi[i]), -i))
        if phi[big] < 0.0:
            phi = [-x for x in phi]
        out[:, dst] = phi
    return np.array([vals[j] for j in order], dtype=float), out


def _solve_square(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (lowest row wins ties)."""
    n = len(a)
    aug = [a[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: (abs(aug[r][c]), -r))
        aug[c], aug[piv] = aug[piv], aug[c]
        for r in range(c + 1, n):
            f = aug[r][c] / aug[c][c]
            for k in range(c, n + 1):
                aug[r][k] -= f * aug[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][n] - _dot(aug[i][i + 1 : n], x[i + 1 :])) / aug[i][i]
    return x


def _lstsq(a: list[list[float]], b: list[float], cols: list[int]) -> list[float]:
    """Least squares restricted to ``cols`` (normal equations)."""
    sub_t = [[row[c] for row in a] for c in cols]
    ata = [[_dot(x, y) for y in sub_t] for x in sub_t]
    return _solve_square(ata, [_dot(x, b) for x in sub_t])


def _residual(a: list[list[float]], b: list[float], x: list[float]) -> list[float]:
    return [b[i] - _dot(a[i], x) for i in range(len(a))]


def det_nnls(a: Any, b: Any) -> tuple[np.ndarray, float]:
    """Lawson-Hanson ``min ||a x - b||`` with ``x >= 0``; returns ``(x, ||r||)``."""
    am, bv = _rows(a), [float(v) for v in b]
    n = len(am[0])
    x = [0.0] * n
    passive: list[int] = []
    for _ in range(30 * n):
        r = _residual(am, bv, x)
        w = [_dot([row[j] for row in am], r) for j in range(n)]
        cand = [j for j in range(n) if j not in passive and w[j] > _NNLS_TOL]
        if not cand:
            break
        passive.append(max(cand, key=lambda j: (w[j], -j)))
        while True:
            idx = sorted(passive)
            s = [0.0] * n
            for j, val in zip(idx, _lstsq(am, bv, idx), strict=True):
                s[j] = val
            neg = [j for j in passive if s[j] <= 0.0]
            if not neg:
                x = s
                break
            alpha = min(x[j] / (x[j] - s[j]) for j in neg)
            x = [x[j] + alpha * (s[j] - x[j]) for j in range(n)]
            passive = [j for j in passive if x[j] > _NNLS_TOL]
            x = [x[j] if j in passive else 0.0 for j in range(n)]
    r = _residual(am, bv, x)
    return np.array(x, dtype=float), math.sqrt(_dot(r, r))
