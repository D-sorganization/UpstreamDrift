"""OpenSim closed-loop constrained-dynamics derivatives (NSX-88, #12717).

OpenSim/Simbody exposes no analytic or automatic derivative of the weld-closed
forward dynamics, so the derivatives of ``(a, lambda)`` come from the NSX-84
central-difference routine (:class:`FiniteDifferenceDerivatives`) driven through
the engine-agnostic adapter model view
(:mod:`src.shared.python.loop_nullspace.adapter_derivatives`).  The capability
level is therefore ``PARTIAL`` with ``derivative_method=finite_difference``.

Coordinate-only derivatives are *insufficient* when the model carries
muscle-tendon actuators: the activation and fibre/tendon states are not
coordinates, so the coordinate tangent bundle ``(xi, eta)`` omits their
dynamics.  The result then says so explicitly
(:attr:`coordinate_only_sufficient`, :attr:`insufficiency_reason`) and the
provenance tag carries it; the flag is never silent.  The spec-built full-body
model has no muscles, so the flag is ``sufficient`` there.

Non-claim: agreement with Pinocchio is software conformance of one specification
in two engines, not physical validation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.adapter_derivatives import (
    adapter_closed_loop_derivatives,
    adapter_step_convergence,
)
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
)

__all__ = [
    "DerivativeNotConvergedError",
    "OpenSimClosedLoopDerivatives",
    "opensim_closed_loop_derivatives",
]

DEFAULT_STEP = 1e-5


class DerivativeNotConvergedError(ArithmeticError):
    """The finite-difference step-convergence report did not converge."""


def muscle_insufficiency_reason(muscle_count: int) -> str:
    """Why coordinate-only derivatives are insufficient (``""`` if sufficient)."""
    if muscle_count < 0:
        raise ValueError(f"muscle_count must be non-negative, got {muscle_count}")
    if muscle_count == 0:
        return ""
    return (
        f"{muscle_count} muscle-tendon actuator(s) present: activation and "
        "fibre/tendon states are outside the coordinate tangent bundle, so "
        "coordinate-only (q, v, tau) derivatives omit their dynamics"
    )


@dataclass(frozen=True)
class OpenSimClosedLoopDerivatives(ClosedLoopDerivatives):
    """:class:`ClosedLoopDerivatives` plus the OpenSim sufficiency/convergence flags.

    Attributes:
        coordinate_only_sufficient: ``False`` when muscles/tendons are present.
        insufficiency_reason: Non-empty exactly when not sufficient.
        converged: Step-convergence verdict, or ``None`` when not checked.
        observed_order: Fitted convergence order, or ``None`` when not checked.
    """

    coordinate_only_sufficient: bool = True
    insufficiency_reason: str = ""
    converged: bool | None = None
    observed_order: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coordinate_only_sufficient == bool(self.insufficiency_reason):
            raise ValueError(
                "insufficiency_reason must be set exactly when "
                "coordinate_only_sufficient is False"
            )

    def provenance_tag(self) -> str:
        """Base tag plus ``coordinate_only=...`` and ``converged=...``."""
        verdict = {True: "true", False: "false", None: "unchecked"}[self.converged]
        coord = "sufficient" if self.coordinate_only_sufficient else "insufficient"
        return f"{super().provenance_tag()};coordinate_only={coord};converged={verdict}"


def opensim_closed_loop_derivatives(
    adapter: Any,
    q: ArrayLike,
    v: ArrayLike,
    tau: ArrayLike,
    *,
    step: float = DEFAULT_STEP,
    check_convergence: bool = True,
) -> OpenSimClosedLoopDerivatives:
    """Converged central-difference derivatives of an OpenSim force adapter.

    ``adapter`` is an ``OpenSimNativeForceAdapter`` (any ``FULL`` snapshot
    adapter exposing ``muscle_count``).  ``tau`` is the generalised load.

    Preconditions (``ValueError``): finite vectors of length ``nv``,
    ``J_c v = 0``, ``step > 0``.  Raises :class:`DerivativeNotConvergedError`
    when the step-convergence report (NSX-84) does not converge, and
    ``ClosureRankChangeError`` at a rank change.  Postcondition: method
    ``FINITE_DIFFERENCE`` (``PARTIAL``); ``coordinate_only_sufficient`` is
    ``False`` with a reason whenever ``adapter.muscle_count > 0``.
    """
    reason = muscle_insufficiency_reason(int(adapter.muscle_count))
    base = adapter_closed_loop_derivatives(adapter, q, v, tau, step=step)
    converged = order = None
    if check_convergence:
        report = adapter_step_convergence(adapter, q, v, tau)
        if not report.converged:
            raise DerivativeNotConvergedError(
                f"finite-difference derivatives did not converge (errors "
                f"{report.errors}, observed order {report.observed_order:.2f})"
            )
        converged, order = True, float(report.observed_order)
    values = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return OpenSimClosedLoopDerivatives(
        **values,
        coordinate_only_sufficient=not reason,
        insufficiency_reason=reason,
        converged=converged,
        observed_order=order,
    )
