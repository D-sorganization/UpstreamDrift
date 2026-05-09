"""Per-engine pose-convention adapter Protocol.

A :class:`PoseConventionAdapter` knows how to translate between the
canonical pose convention (see :mod:`pose_interchange.canonical`) and a
specific physics engine's native ``q`` vector layout.

Subtask 1 (#4896) defines the Protocol; per-engine implementations land
in Subtask 2 (#4897) under :mod:`pose_interchange.adapters`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.shared.python.pose_interchange.canonical import CanonicalPose


@dataclass(frozen=True, slots=True)
class JointSlot:
    """Describes one joint's slot inside an engine's ``q`` vector.

    Adapters return one ``JointSlot`` per canonical joint name. The
    engine's full ``q`` vector layout is the union of these slots plus
    any free / floating-base prefix.

    Parameters
    ----------
    canonical_name
        Name of the canonical joint (a member of
        :data:`REFERENCE_GOLFER_FIELDS`).
    engine_name
        The native joint name in the engine's model file (URDF /
        MJCF / .osim / Simulink Parameter).
    start_index
        First index of this slot inside the engine's ``q`` vector.
    length
        Number of scalar position DOFs occupied. Always 1 for the
        canonical golfer joints (which are all 1-DOF revolute), but
        adapters can carry larger slots for free joints if needed.
    units
        ``"rad"`` or ``"deg"``. Adapters convert to canonical degrees
        in :meth:`PoseConventionAdapter.to_canonical`.
    sign
        ``+1`` if the engine and the canonical convention agree on
        sign for this joint, ``-1`` if the engine flips it. Captured
        once here so adapter code does not have to scatter sign-flips
        across function bodies.
    lower_limit
        Lower joint limit in the **engine's** units (rad or deg, per
        ``units``). ``-inf`` if unbounded.
    upper_limit
        Upper joint limit in the **engine's** units. ``+inf`` if
        unbounded.
    """

    canonical_name: str
    engine_name: str
    start_index: int
    length: int = 1
    units: str = "rad"
    sign: int = 1
    lower_limit: float = float("-inf")
    upper_limit: float = float("inf")

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"JointSlot.length must be >= 1, got {self.length}")
        if self.units not in {"rad", "deg"}:
            raise ValueError(
                f"JointSlot.units must be 'rad' or 'deg', got {self.units!r}"
            )
        if self.sign not in {1, -1}:
            raise ValueError(f"JointSlot.sign must be +1 or -1, got {self.sign}")
        if self.start_index < 0:
            raise ValueError(
                f"JointSlot.start_index must be >= 0, got {self.start_index}"
            )
        if not np.isfinite(self.lower_limit) and self.lower_limit > 0:
            raise ValueError("JointSlot.lower_limit cannot be +inf")
        if not np.isfinite(self.upper_limit) and self.upper_limit < 0:
            raise ValueError("JointSlot.upper_limit cannot be -inf")
        if self.lower_limit > self.upper_limit:
            raise ValueError(
                "JointSlot.lower_limit must be <= upper_limit "
                f"({self.lower_limit} > {self.upper_limit})"
            )


@runtime_checkable
class PoseConventionAdapter(Protocol):
    """Round-trip a :class:`CanonicalPose` through one engine's convention.

    Implementations live under :mod:`pose_interchange.adapters` (one
    file per engine). Each implementation declares a constant
    :attr:`engine_name` so the registry can dispatch on it.
    """

    engine_name: str

    def to_canonical(
        self,
        engine_q: npt.ArrayLike,
        *,
        model: Any | None = None,
    ) -> CanonicalPose:
        """Decode an engine ``q`` vector to a :class:`CanonicalPose`.

        Parameters
        ----------
        engine_q
            The engine's native position vector. Shape and ordering are
            engine-specific.
        model
            Optional engine-specific model handle. Some engines (Drake,
            MuJoCo, Pinocchio) need the model to interpret the ``q``
            layout because joint ordering is model-dependent.
        """

    def from_canonical(
        self,
        pose: CanonicalPose,
        *,
        model: Any | None = None,
    ) -> npt.NDArray[np.float64]:
        """Encode a :class:`CanonicalPose` as the engine's ``q`` vector."""

    def joint_layout(
        self,
        model: Any | None = None,
    ) -> Mapping[str, JointSlot]:
        """Return the joint slot layout, keyed by canonical joint name.

        The returned mapping must contain every canonical name the
        adapter knows how to map. Canonical names not in the map are
        silently dropped on ``from_canonical`` (those joints don't exist
        in the engine's model) and never produced by ``to_canonical``.
        """
