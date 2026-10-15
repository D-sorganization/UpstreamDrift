"""Shared helpers for the closed-loop null-space (NSX-00) sympy derivations.

Every golden value in ``tests/fixtures/loop_nullspace/goldens`` is produced by a
module in this directory.  The helpers here are deliberately small: exact
sympy linear algebra, a rank tolerance that is stated once, and JSON encoding.
Nothing here imports repository production code, so a golden can never be
produced by the code it later checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp

GOLDEN_SCHEMA = "loop-nullspace-golden/1"
GOLDENS_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
)

# Relative singular-value cut used for numerical (float) ranks.  The exact
# fixtures (straight arm, spatial counts, envelope) use exact rank instead.
RANK_RTOL = 1e-10

WORKING_DPS = 50


def to_float(x: Any) -> float:
    """Round an exact or high-precision sympy number to a Python float."""
    return float(sp.N(x, WORKING_DPS))


def mat(m: sp.Matrix) -> list[list[float]]:
    """Matrix to nested lists of floats (row-major)."""
    return [[to_float(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def vec(m: sp.Matrix) -> list[float]:
    """Column or row vector to a flat list of floats."""
    return [to_float(x) for x in list(m)]


def exact_mat(m: sp.Matrix) -> list[list[str]]:
    """Matrix of exact rationals as strings (``"3/5"``)."""
    return [[_exact_str(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def _exact_str(x: Any) -> str:
    """Canonical string of an exact entry; refuse anything that is not a rational."""
    e = sp.expand(sp.sympify(x))
    if not e.is_Rational:
        e = sp.nsimplify(sp.simplify(e), rational=True)
    if not e.is_Rational:
        raise ValueError(f"entry {x!r} is not an exact rational")
    return str(e)


def numeric_rank(m: sp.Matrix) -> tuple[int, list[float]]:
    """Rank by SVD with the stated relative tolerance; also the singular values."""
    import numpy as np

    arr = np.array(mat(m), dtype=float)
    if arr.size == 0:
        return 0, []
    s = np.linalg.svd(arr, compute_uv=False)
    tol = RANK_RTOL * max(float(s[0]), 1.0) if s.size else 0.0
    return int((s > tol).sum()), [float(v) for v in s]


def dump(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the schema tag; the caller writes the file."""
    payload = {"schema": GOLDEN_SCHEMA, "name": name, **payload}
    return payload


def write_json(
    name: str, payload: dict[str, Any], directory: Path | None = None
) -> Path:
    target = (directory or GOLDENS_DIR) / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
