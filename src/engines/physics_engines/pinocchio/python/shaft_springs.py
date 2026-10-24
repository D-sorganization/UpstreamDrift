"""Chain-shaft spring and damper torque for the Pinocchio models (R11, #12691).

Pinocchio has no joint-spring element, so the R6 chain shaft's linear springs
and viscous dampers enter as a generalized torque added to the efforts before
the constrained solve:  ``tau_i = -k_i (q_i - q_ref,i) - c_i v_i`` on each
chain coordinate, the law MuJoCo's passive force and OpenSim's
``SpringGeneralizedForce`` apply.  Every constant is read from the spec
primitives (no per-engine parameter); a rigid club has no springs and leaves
the efforts untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_shaft_chain import (
    ShaftChainUnsupportedError,
    passive_coordinates,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ShaftSprings:
    """Spring constants of the chain coordinates, indexed in native order.

    ``q_index`` / ``v_index`` are the coordinates' Pinocchio slots; the
    stiffness (N m/rad), rest angle (rad) and damping (N m s/rad) arrays are
    aligned with ``names``.
    """

    names: tuple[str, ...]
    q_index: NDArray[np.int_]
    v_index: NDArray[np.int_]
    stiffness: Array
    rest: Array
    damping: Array

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, Any],
        q_slots: Mapping[str, int],
        v_slots: Mapping[str, int],
    ) -> ShaftSprings:
        """Springs of ``spec``'s chain coordinates; empty for a rigid club.

        Raises:
            ValueError: a chain coordinate has no primitive with spring keys.
        """
        names = passive_coordinates(spec)
        primitives = {
            p["coordinate"]: p for joint in spec["joints"] for p in joint["primitives"]
        }
        try:
            values = np.array(
                [
                    [
                        float(primitives[name]["stiffness_n_m_rad"]),
                        float(primitives[name]["springref_rad"]),
                        float(primitives[name]["damping_n_m_s_rad"]),
                    ]
                    for name in names
                ],
                dtype=float,
            ).reshape(len(names), 3)
        except KeyError as exc:
            raise ValueError(
                f"chain coordinate without spring primitive: {exc}"
            ) from exc
        return cls(
            names=names,
            q_index=np.array([q_slots[n] for n in names], dtype=int),
            v_index=np.array([v_slots[n] for n in names], dtype=int),
            stiffness=values[:, 0],
            rest=values[:, 1],
            damping=values[:, 2],
        )

    def __bool__(self) -> bool:
        return bool(self.names)

    def compiled(self) -> dict[str, tuple[float, float, float]]:
        """``{coordinate: (k, rest, c)}`` as applied by :meth:`effort`."""
        return {
            name: (float(k), float(r), float(c))
            for name, k, r, c in zip(
                self.names, self.stiffness, self.rest, self.damping, strict=True
            )
        }

    def effort(self, tau: Array, q: Array, v: Array) -> Array:
        """``tau`` with the chain coordinates unactuated plus the spring torque.

        All vectors are native-ordered.  Postcondition: a rigid club returns
        ``tau`` unchanged (same object).
        """
        if not self.names:
            return tau
        out = np.array(tau, dtype=float, copy=True)
        out[self.v_index] = (
            -self.stiffness * (q[self.q_index] - self.rest)
            - self.damping * (v[self.v_index])
        )
        return out

    def refuse(self, method: str) -> None:
        """DbC: a path that would omit the springs never sees a chain."""
        if self.names:
            raise ShaftChainUnsupportedError(
                f"Pinocchio {method} does not apply the chain-shaft springs (#12691)"
            )
