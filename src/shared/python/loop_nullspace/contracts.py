"""Closed-loop snapshot and loop-topology contracts (NSX-10, ADR-0054 D2/D3).

``ClosedLoopSnapshot`` is the engine-agnostic state the null-space allocation
code consumes: ``M a + h = S^T tau + J_c^T lambda (+ J_g^T f)`` with ``J_c`` in
the canonical grip-frame convention (``closure_convention.py``).  All arrays
are copied, validated finite and read-only.  The evidence-axis types live in
``evidence.py`` and are re-exported here.  No engine is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    SYMMETRY_RTOL,
    AllocationDependence,
    Array,
    FootContact,
    GripFrame,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
    SpatialInertia,
    ValidationStatus,
    check_shape,
    check_symmetric,
    readonly_float,
)

__all__ = [
    "AllocationDependence",
    "ClosedLoopSnapshot",
    "ClosureKind",
    "FootContact",
    "GripFrame",
    "InputProvenance",
    "LoopTopology",
    "ModelProvenance",
    "QuantityRecord",
    "SpatialInertia",
    "ValidationStatus",
]


class ClosureKind(Enum):
    """Closure type; ``rows`` gives the number of closure rows."""

    WELD6 = "WELD6"
    SHAFT_FREE5 = "SHAFT_FREE5"
    BALL3 = "BALL3"
    PLANAR_POINT2 = "PLANAR_POINT2"
    PLANAR_WELD3 = "PLANAR_WELD3"

    @property
    def rows(self) -> int:
        """Number of closure (``J_c``) rows for this kind."""
        return _CLOSURE_ROWS[self.value]


_CLOSURE_ROWS = {
    "WELD6": 6,
    "SHAFT_FREE5": 5,
    "BALL3": 3,
    "PLANAR_POINT2": 2,
    "PLANAR_WELD3": 3,
}


@dataclass(frozen=True)
class LoopTopology:
    """Which side closes the loop and how coordinates are partitioned.

    ``actuated_indices`` and ``unactuated_indices`` must be disjoint and cover
    ``range(len(coordinate_order))``; ``loop_support_indices`` is a subset.
    """

    closure_kind: ClosureKind
    closing_side: str
    holding_side: str
    actuated_indices: tuple[int, ...]
    unactuated_indices: tuple[int, ...]
    loop_support_indices: tuple[int, ...]
    coordinate_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.closure_kind, ClosureKind):
            raise TypeError("closure_kind must be a ClosureKind")
        n = len(self.coordinate_order)
        act, unact = set(self.actuated_indices), set(self.unactuated_indices)
        if len(act) != len(self.actuated_indices) or len(unact) != len(
            self.unactuated_indices
        ):
            raise ValueError("topology index tuples must not repeat indices")
        if act & unact:
            raise ValueError("actuated and unactuated indices must be disjoint")
        if act | unact != set(range(n)):
            raise ValueError(
                "actuated and unactuated indices must cover every coordinate "
                f"0..{n - 1}"
            )
        if not set(self.loop_support_indices) <= set(range(n)):
            raise ValueError("loop_support_indices must be valid coordinate indices")

    @property
    def n(self) -> int:
        """Number of generalised coordinates."""
        return len(self.coordinate_order)

    @property
    def n_actuated(self) -> int:
        """Number of actuated coordinates."""
        return len(self.actuated_indices)


@dataclass(frozen=True)
class ClosedLoopSnapshot:
    """One validated closed-loop dynamic state (see module docstring)."""

    q: Array
    v: Array
    a: Array
    M: Array
    h: Array
    S: Array
    J_c: Array
    Jdot_v: Array
    grip_frame: GripFrame
    bare_club: SpatialInertia
    welded_hand_inertias: tuple[SpatialInertia, ...]
    model_provenance: ModelProvenance
    input_provenance: InputProvenance
    topology: LoopTopology
    J_g: Array | None = None
    foot_contacts: tuple[FootContact, ...] = ()
    grf_observed: Array | None = None

    def __post_init__(self) -> None:
        for name in ("q", "v", "a", "M", "h", "S", "J_c", "Jdot_v"):
            object.__setattr__(
                self, name, readonly_float(getattr(self, name), f"snapshot.{name}")
            )
        for name in ("J_g", "grf_observed"):
            val = getattr(self, name)
            if val is not None:
                object.__setattr__(self, name, readonly_float(val, f"snapshot.{name}"))
        object.__setattr__(self, "foot_contacts", tuple(self.foot_contacts))
        object.__setattr__(
            self, "welded_hand_inertias", tuple(self.welded_hand_inertias)
        )
        self._validate_shapes()
        check_symmetric(self.M, "snapshot.M")
        self._validate_mass_matrix()
        self._validate_selection()

    def _validate_shapes(self) -> None:
        topo = self.topology
        n, n_a, m = topo.n, topo.n_actuated, topo.closure_kind.rows
        for name in ("q", "v", "a", "h"):
            check_shape(getattr(self, name), (n,), f"snapshot.{name}")
        check_shape(self.M, (n, n), "snapshot.M")
        check_shape(self.S, (n_a, n), "snapshot.S")
        check_shape(self.J_c, (m, n), f"snapshot.J_c ({topo.closure_kind.name})")
        check_shape(self.Jdot_v, (m,), "snapshot.Jdot_v")
        if self.J_g is not None:
            if self.J_g.ndim != 2 or self.J_g.shape[1] != n:
                raise ValueError(f"snapshot.J_g must have {n} columns")
            if self.grf_observed is not None:
                check_shape(
                    self.grf_observed, (self.J_g.shape[0],), "snapshot.grf_observed"
                )
        elif self.grf_observed is not None or self.foot_contacts:
            raise ValueError("grf_observed and foot_contacts require J_g")

    def _validate_mass_matrix(self) -> None:
        sym = 0.5 * (self.M + self.M.T)
        try:
            np.linalg.cholesky(sym)
        except np.linalg.LinAlgError as exc:
            raise ValueError("snapshot.M must be positive definite") from exc

    def _validate_selection(self) -> None:
        expected = np.eye(self.topology.n)[list(self.topology.actuated_indices)]
        if np.max(np.abs(self.S - expected), initial=0.0) > SYMMETRY_RTOL:
            raise ValueError(
                "snapshot.S must select the actuated coordinates in the order "
                "of topology.actuated_indices"
            )

    def generalized_load(self) -> Array:
        """Return ``b = M a + h``, the generalised load the closure must carry."""
        return np.asarray(self.M @ self.a + self.h, dtype=np.float64)

    def closure_rows(self) -> Array:
        """Return the canonical closure Jacobian ``J_c`` (read-only)."""
        return self.J_c

    def actuated_rows(self) -> Array:
        """Return the actuated-coordinate selection matrix ``S`` (read-only)."""
        return self.S
