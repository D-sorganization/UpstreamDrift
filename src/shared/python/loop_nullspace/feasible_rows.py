"""Row, cone and form value objects for the feasible set (NSX-21, #12564).

Split from :mod:`feasible_set` to keep both modules small.  Every row and cone
block carries the id of the hypothesis that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.grip_split import closing_hand_map
from src.shared.python.loop_nullspace.solvers import (
    ConicResult,
    LPResult,
    SOCBlock,
    solve_lp,
    solve_socp,
)
from src.shared.python.motion_matching.force_nullspace.constraints import (
    FrictionCone,
)

__all__ = [
    "ConeBlock",
    "ContactCone",
    "FeasibleRow",
    "GripPreload",
    "LPForm",
    "SOCPForm",
    "cone_error_bound",
]

TORQUE_ID = "H-TORQUE-CAPACITY"
CONE_ID = "H-FRICTION-CONE"
GRIP_ID = "H-GRIP-PRELOAD"
DEFAULT_CONE_FACETS = 8
ROW_KINDS = ("torque", "cone_normal", "cone_linear", "grip")
FLAG_ASSUMED = "ASSUMED_CAPACITY"
FLAG_UNVERIFIED = "UNVERIFIED_HYPOTHESIS"


def _ro(a: Any, name: str) -> Array:
    arr = np.array(a, dtype=float, copy=True)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


def cone_error_bound(k: int, linearization: str = "outer") -> float:
    """Relative radius error of the ``k``-facet polygon of a circular cone.

    ``outer``: ``1/cos(pi/k) - 1`` (polygon circumscribes the circle);
    ``inner``: ``1 - cos(pi/k)`` (polygon inscribed, apothem ``cos(pi/k)``).
    Raises ``ValueError`` for ``k < 3`` or an unknown ``linearization``.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 3:
        raise ValueError(f"cone_facets must be an integer >= 3, got {k!r}")
    c = math.cos(math.pi / k)
    if linearization == "outer":
        return 1.0 / c - 1.0
    if linearization == "inner":
        return 1.0 - c
    raise ValueError(f"linearization must be 'outer' or 'inner', got {linearization!r}")


# ------------------------------------------------------------ value objects
@dataclass(frozen=True, eq=False)
class FeasibleRow:
    """One polyhedral row ``g . z <= h`` with its provenance.

    ``label`` is unique inside a set; ``hypothesis_id`` names the declaring
    hypothesis; ``flags`` lists evidence caveats (see ``FLAG_*``).
    """

    g: Array
    h: float
    label: str
    hypothesis_id: str
    kind: str
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        g = _ro(self.g, "g")
        if g.ndim != 1:
            raise ValueError("row g must be 1-D")
        if not math.isfinite(self.h):
            raise ValueError("row h must be finite")
        if not self.label or not self.hypothesis_id:
            raise ValueError("row needs a label and a hypothesis_id")
        if self.kind not in ROW_KINDS:
            raise ValueError(f"row kind {self.kind!r} not in {ROW_KINDS}")
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "h", float(self.h))


@dataclass(frozen=True, eq=False)
class ConeBlock:
    """Exact second-order cone: ``b - A z`` in ``{(t, u): t >= ||u||}``."""

    A: Array
    b: Array
    label: str
    hypothesis_id: str
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", _ro(self.A, "A"))
        object.__setattr__(self, "b", _ro(self.b, "b"))
        if self.A.ndim != 2 or self.b.shape != (self.A.shape[0],):
            raise ValueError("ConeBlock needs A (d, n_z) and b (d,)")
        if not self.label or not self.hypothesis_id:
            raise ValueError("cone block needs a label and a hypothesis_id")


@dataclass(frozen=True)
class ContactCone:
    """A named Coulomb cone over three entries of ``x`` (indices into ``x``)."""

    name: str
    cone: FrictionCone

    def __post_init__(self) -> None:
        if not isinstance(self.cone, FrictionCone):
            raise TypeError(f"cone must be FrictionCone, got {type(self.cone)!r}")
        if not self.name:
            raise ValueError("ContactCone requires a non-empty name")


@dataclass(frozen=True, eq=False)
class GripPreload:
    """Squeeze floor ``selector . x + offset >= minimum_squeeze_n`` for one hand.

    ``selector`` has length ``dim_x`` (``x = (tau, lambda, f_g)``).
    """

    name: str
    selector: Array
    offset: float
    minimum_squeeze_n: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GripPreload requires a non-empty name")
        sel = _ro(self.selector, "selector")
        if sel.ndim != 1:
            raise ValueError("selector must be 1-D")
        if not (math.isfinite(self.offset) and math.isfinite(self.minimum_squeeze_n)):
            raise ValueError("offset and minimum_squeeze_n must be finite")
        object.__setattr__(self, "selector", sel)

    def squeeze(self, x: Any) -> float:
        """Squeeze (N) at the full vector ``x``."""
        return float(self.selector @ np.asarray(x, float) + self.offset)

    @classmethod
    def from_closing_map(
        cls,
        hand: str,
        dim_x: int,
        *,
        lambda_start: int,
        d: Any,
        net: Any,
        minimum_squeeze_n: float,
    ) -> GripPreload:
        """Squeeze ``-s1`` of the closing hand from ``closing_hand_map`` (NSX-16).

        ``s = A lambda + s0`` with the six hand multipliers at
        ``x[lambda_start : lambda_start + 6]``; the axial split coordinate
        ``s1`` is positive for pull and negative for squeeze.
        """
        a_map, s0 = closing_hand_map(hand, d, net)
        if lambda_start < 0 or lambda_start + 6 > dim_x:
            raise ValueError("hand multipliers do not fit inside x")
        sel = np.zeros(dim_x)
        sel[lambda_start : lambda_start + 6] = -a_map[0]
        return cls(f"hand_{hand}", sel, -float(s0[0]), minimum_squeeze_n)


@dataclass(frozen=True, eq=False)
class LPForm:
    """``A_ub z <= b_ub`` with free ``z``: arguments for :func:`solve_lp`."""

    A_ub: Any
    b_ub: Array
    n_z: int

    def solve(self, c: Any) -> LPResult:
        """Minimise ``c . z`` over the polyhedron."""
        a = self.A_ub if self.A_ub.shape[0] else None
        b = self.b_ub if self.A_ub.shape[0] else None
        return solve_lp(c, None, None, a, b, [(None, None)] * self.n_z)


@dataclass(frozen=True, eq=False)
class SOCPForm:
    """Polyhedral rows plus exact second-order cones over free ``z``."""

    A_ub: Any
    b_ub: Array
    cones: tuple[SOCBlock, ...]
    n_z: int

    def solve(self, c: Any) -> ConicResult:
        """Minimise ``c . z`` with the exact cones."""
        a = self.A_ub if self.A_ub.shape[0] else None
        b = self.b_ub if self.A_ub.shape[0] else None
        return solve_socp(c, None, None, a, b, self.cones)
