"""Drake AutoDiff derivatives of the weld KKT solve (NSX-86, #12715).

``DrakeAutoDiffDerivatives`` is the Drake provider of the NSX-84 derivative
seam: it differentiates the single KKT implementation with the *exact*
AutoDiffXd differentials of ``M, h, J, Jdot v`` taken from a re-templated
``MultibodyPlant`` (:meth:`FullBodyDrakeModel.closed_loop_differentials`), and
returns :class:`ClosedLoopDerivatives` in the closure tangent basis and the
canonical closure convention (NSX-02, epic rule R8), method ``AUTODIFF``.

Canonical convention.  Drake's native rows are mapped by the NSX-02 matrix ``C``
(``J_c = C J_native``).  ``C`` depends on ``q`` only through the grip frame
relative to the native frame; here the grip frame *is* the native LOCAL frame, so
``C`` is a constant signed permutation and ``dC/dq = 0``.  That is verified at the
state and anything else is refused (:class:`DerivativeCapabilityUnavailable`),
not approximated.

Scope and non-claims: rigid tree plus weld closure (the arrays of the closed-loop
snapshot); ground contact is not in those arrays and is not differentiated.  The
load ``tau`` is the generalised applied load in spec order.  A component AutoDiff
cannot evaluate raises a capability rejection; there is no finite-difference
fall-back.  Requires a real ``pydrake`` install.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.adapter_model import AdapterClosedLoopModel
from src.shared.python.loop_nullspace.autodiff_derivatives import (
    AutoDiffKKTDerivatives,
    ModelDifferentials,
)
from src.shared.python.loop_nullspace.configuration_manifold import (
    EuclideanManifold,
)
from src.shared.python.loop_nullspace.derivative_capability import (
    DerivativeCapabilityUnavailable,
)
from src.shared.python.loop_nullspace.snapshot import canonical_closure_matrix

__all__ = ["DrakeAutoDiffDerivatives", "DrakeClosedLoopModel"]

_FRAME_TOL = 1e-12


class DrakeClosedLoopModel(AdapterClosedLoopModel):
    """Canonical Drake ``M, h, J_c, Jdot_v`` plus their exact differentials."""

    def differentials(self, q: Any, v: Any) -> ModelDifferentials:
        """AutoDiffXd differentials of the canonical arrays at ``(q, v)``.

        Raises :class:`DerivativeCapabilityUnavailable` if the grip frame differs
        from the native closure frame (the row map would then vary with ``q``) or
        if AutoDiffXd cannot evaluate a component.
        """
        q_arr, v_arr = self._own(q, "q"), self._own(v, "v")
        adapter = self.adapter
        self._require_constant_row_map(q_arr)
        native = adapter.closed_loop_differentials(q_arr, v_arr)
        return native.map_closure_rows(canonical_closure_matrix(adapter, q_arr))

    def _require_constant_row_map(self, q: Any) -> None:
        adapter = self.adapter
        native_rot, native_pt = adapter.native_closure_frame(q)
        grip_rot, grip_pt = adapter.closure_wrench_frame(q)
        same = np.abs(np.asarray(native_rot) - np.asarray(grip_rot)).max() <= _FRAME_TOL
        same &= np.abs(np.asarray(native_pt) - np.asarray(grip_pt)).max() <= _FRAME_TOL
        if not same:
            raise DerivativeCapabilityUnavailable(
                "the grip frame differs from the native closure frame, so the "
                "canonical row map varies with q and its derivative is not "
                "provided; refusing to differentiate"
            )


class DrakeAutoDiffDerivatives(AutoDiffKKTDerivatives):
    """``constrained_acceleration_derivatives`` for a Drake force adapter.

    Args:
        adapter: A ``DrakeForceAdapter`` (``CLOSED_LOOP_SNAPSHOT`` FULL).
        **kwargs: Guard options of the derivative base (``step``, ``basis_field``,
            ``expected_rank``, ``velocity_tolerance``).  ``step`` only sizes the
            closure-rank guard probes; no derivative is differenced.
    """

    def __init__(self, adapter: Any, **kwargs: Any) -> None:
        model = DrakeClosedLoopModel(adapter)
        super().__init__(model, EuclideanManifold(model.nv), **kwargs)
