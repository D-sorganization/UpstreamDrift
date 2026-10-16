"""Derivative capability seam for the closed-loop constrained dynamics (NSX-84).

The tangent linearization ``(F_t, G_t)`` of ``loop_nullspace.tex``
(``sec:var-lin``) needs the derivatives of the single KKT implementation
(``constrained_dynamics.constrained_acceleration``) with the reaction re-solved.
This module is the engine-agnostic contract every engine adapter plugs into:

* :class:`DerivativeMethod` -- how an engine obtains the derivatives
  (analytic, autodiff, finite difference, or not at all) and the
  :class:`CapabilityLevel` it earns (``FULL`` / ``FULL`` / ``PARTIAL`` / ``NONE``).
* :class:`ClosedLoopDerivatives` -- the value an adapter returns from
  ``constrained_acceleration_derivatives(q, v, tau)``, expressed in tangent-bundle
  coordinates of ``ker J_c`` so ``J_c dq = 0`` holds by construction.
* :class:`ClosedLoopDerivativeProvider` -- the structural Protocol.
* :class:`DerivativeCapabilityRegistry` -- declarations per engine; asking for a
  missing capability raises :class:`DerivativeCapabilityUnavailable` with the
  reason.  There is no silent fall-back to finite differences: an engine opts in
  by declaring ``FINITE_DIFFERENCE`` explicitly.

Engine implementations are later children (NSX-85..89; Pinocchio is NSX-85,
Drake NSX-86).  No engine adapter is imported here; loaders are dotted
``module:attr`` paths resolved on demand.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
    capability_level_supported,
)
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)

__all__ = [
    "ClosedLoopDerivativeProvider",
    "ClosedLoopDerivatives",
    "DerivativeCapabilityDeclaration",
    "DerivativeCapabilityRegistry",
    "DerivativeCapabilityUnavailable",
    "DerivativeMethod",
    "default_registry",
    "require_adapter_derivatives",
    "unavailable_derivative_record",
]


class DerivativeCapabilityUnavailable(CapabilityUnavailable):
    """A closed-loop derivative capability is absent or below the required level."""


class DerivativeMethod(Enum):
    """How an engine obtains the closed-loop derivatives."""

    ANALYTIC = "analytic"
    AUTODIFF = "autodiff"
    FINITE_DIFFERENCE = "finite_difference"
    NONE = "none"

    @property
    def level(self) -> CapabilityLevel:
        """``FULL`` for analytic/autodiff, ``PARTIAL`` for finite difference."""
        if self is DerivativeMethod.NONE:
            return CapabilityLevel.NONE
        if self is DerivativeMethod.FINITE_DIFFERENCE:
            return CapabilityLevel.PARTIAL
        return CapabilityLevel.FULL


def _check(arr: ArrayLike, shape: tuple[int, ...], name: str) -> Array:
    out = np.array(arr, dtype=np.float64, copy=True)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class ClosedLoopDerivatives:
    """Derivatives of ``(a, lambda)`` on the tangent bundle of the closure.

    With ``P`` the orthonormal ``(nv, k)`` basis of ``ker J_c`` the perturbation
    is ``dq = P xi`` and ``dv = P eta + O xi`` where ``O xi`` is the minimum-norm
    velocity offset that keeps ``J_c dv + (dJ_c[dq]) v = 0``.  ``d_acc_d_xi``
    and ``d_acc_d_eta`` are ``d a / d xi`` and ``d a / d eta`` at fixed
    generalised load; ``d_acc_d_tau`` is the (exact, linear) response to the
    load.  The reaction ``lambda`` is re-solved on every perturbation.

    Attributes:
        acceleration: Nominal ``a``, shape ``(nv,)``.
        reaction: Nominal minimum-norm ``lambda``, shape ``(m,)``.
        basis: ``P``, shape ``(nv, k)``.
        projector: ``Pi = P P^T``, shape ``(nv, nv)`` (basis independent).
        d_acc_d_xi, d_acc_d_eta: shape ``(nv, k)``.
        d_acc_d_tau: shape ``(nv, nv)``.
        d_lambda_d_xi, d_lambda_d_eta: shape ``(m, k)``; ``d_lambda_d_tau``
            shape ``(m, nv)``.
        method: How the derivatives were obtained (provenance).
        scheme: Free-text sub-scheme, e.g. ``central_retraction``.
        step: Finite-difference step, or ``None`` for exact derivatives.
    """

    acceleration: Array
    reaction: Array
    basis: Array
    projector: Array
    d_acc_d_xi: Array
    d_acc_d_eta: Array
    d_acc_d_tau: Array
    d_lambda_d_xi: Array
    d_lambda_d_eta: Array
    d_lambda_d_tau: Array
    method: DerivativeMethod
    scheme: str = ""
    step: float | None = None

    def __post_init__(self) -> None:
        nv = int(np.shape(self.acceleration)[0])
        m = int(np.shape(self.reaction)[0])
        k = int(np.shape(self.basis)[1])
        spec = {
            "acceleration": (nv,),
            "reaction": (m,),
            "basis": (nv, k),
            "projector": (nv, nv),
            "d_acc_d_xi": (nv, k),
            "d_acc_d_eta": (nv, k),
            "d_acc_d_tau": (nv, nv),
            "d_lambda_d_xi": (m, k),
            "d_lambda_d_eta": (m, k),
            "d_lambda_d_tau": (m, nv),
        }
        for name, shape in spec.items():
            object.__setattr__(self, name, _check(getattr(self, name), shape, name))
        if self.method is DerivativeMethod.NONE:
            raise ValueError("a ClosedLoopDerivatives result cannot have method NONE")
        if self.step is not None and not (np.isfinite(self.step) and self.step > 0):
            raise ValueError(f"step must be positive and finite, got {self.step}")

    @property
    def nv(self) -> int:
        """Number of velocity coordinates."""
        return int(self.acceleration.shape[0])

    @property
    def tangent_dimension(self) -> int:
        """``k = dim ker J_c``."""
        return int(self.basis.shape[1])

    def provenance_tag(self) -> str:
        """``derivative_method=...;scheme=...;step=...`` for provenance strings."""
        parts = [f"derivative_method={self.method.value}"]
        if self.scheme:
            parts.append(f"scheme={self.scheme}")
        if self.step is not None:
            parts.append(f"step={self.step:g}")
        return ";".join(parts)


@runtime_checkable
class ClosedLoopDerivativeProvider(Protocol):
    """Engine-agnostic provider of :class:`ClosedLoopDerivatives`."""

    def constrained_acceleration_derivatives(
        self, q: ArrayLike, v: ArrayLike, tau: ArrayLike
    ) -> ClosedLoopDerivatives:
        """Tangent-bundle derivatives of ``(a, lambda)`` at ``(q, v, tau)``."""
        ...


@dataclass(frozen=True)
class DerivativeCapabilityDeclaration:
    """What one engine declares about its closed-loop derivatives."""

    engine: str
    method: DerivativeMethod
    reason: str = ""
    loader: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.engine or not isinstance(self.engine, str):
            raise ValueError("engine must be a non-empty string")
        if self.method is DerivativeMethod.NONE and not self.reason.strip():
            raise ValueError("a NONE declaration needs a reason")
        if self.loader is not None and ":" not in self.loader:
            raise ValueError(f"loader must look like 'module:attr', got {self.loader}")

    @property
    def level(self) -> CapabilityLevel:
        """Capability level earned by the declared method."""
        return self.method.level


class DerivativeCapabilityRegistry:
    """Per-engine derivative declarations; missing capability fails closed."""

    def __init__(self) -> None:
        self._entries: dict[str, DerivativeCapabilityDeclaration] = {}

    def declare(
        self,
        engine: str,
        method: DerivativeMethod,
        *,
        reason: str = "",
        loader: str | None = None,
    ) -> DerivativeCapabilityDeclaration:
        """Register a declaration; a repeated engine name raises ``ValueError``."""
        if engine in self._entries:
            raise ValueError(f"derivative capability for {engine!r} already declared")
        decl = DerivativeCapabilityDeclaration(engine, method, reason, loader)
        self._entries[engine] = decl
        return decl

    def declaration(self, engine: str) -> DerivativeCapabilityDeclaration:
        """Return the declaration of ``engine`` whatever its level.

        Raises:
            DerivativeCapabilityUnavailable: ``engine`` is not declared.
        """
        decl = self._entries.get(engine)
        if decl is None:
            raise DerivativeCapabilityUnavailable(
                f"closed-loop derivatives for {engine!r} are not declared"
            )
        return decl

    def require(
        self, engine: str, *, minimum: CapabilityLevel = CapabilityLevel.PARTIAL
    ) -> DerivativeCapabilityDeclaration:
        """Return the declaration or raise :class:`DerivativeCapabilityUnavailable`."""
        decl = self._entries.get(engine)
        if decl is None:
            raise DerivativeCapabilityUnavailable(
                f"closed-loop derivatives for {engine!r} are not declared"
            )
        if not capability_level_supported(decl.level, minimum=minimum):
            why = f": {decl.reason}" if decl.reason else ""
            raise DerivativeCapabilityUnavailable(
                f"closed-loop derivatives for {engine!r} are {decl.level.name} "
                f"({decl.method.value}) but {Capability.CLOSED_LOOP_DERIVATIVES.value}"
                f" requires {minimum.name}{why}"
            )
        return decl

    def load(
        self, engine: str, *, minimum: CapabilityLevel = CapabilityLevel.PARTIAL
    ) -> Any:
        """Import and return the declared provider factory (lazy, on demand)."""
        decl = self.require(engine, minimum=minimum)
        if decl.loader is None:
            raise DerivativeCapabilityUnavailable(
                f"{engine!r} declares {decl.method.value} derivatives but no loader"
            )
        module_name, _, attr = decl.loader.partition(":")
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise DerivativeCapabilityUnavailable(
                f"cannot import derivative provider {decl.loader!r} for "
                f"{engine!r}: {exc}"
            ) from exc


_ADAPTER_FD = (
    "src.shared.python.loop_nullspace.adapter_derivatives:"
    "adapter_closed_loop_derivatives"
)
_SIMSCAPE_REASON = (
    "the NSX-55 Simscape export is a snapshot grid and cannot be re-evaluated at "
    "perturbed states; a per-frame MATLAB R2025b linearize A/B export (model hash, "
    "solver settings, state mapping) is the deferred path, recorded in DV-12718 "
    "(R2026a is never acceptance evidence)"
)


def default_registry() -> DerivativeCapabilityRegistry:
    """Registry: reference, Pinocchio analytic, Drake AutoDiff, FD engines ``PARTIAL``."""
    reg = DerivativeCapabilityRegistry()
    reg.declare(
        "reference_fd",
        DerivativeMethod.FINITE_DIFFERENCE,
        loader=(
            "src.shared.python.loop_nullspace.tangent_linearization:"
            "FiniteDifferenceDerivatives"
        ),
    )
    # NSX-87: hard-KKT finite differences.  ``mjd_transitionFD`` (soft stepping)
    # is a separately characterised entry and is never conformance evidence.
    for engine in ("mujoco", "myosuite"):
        reg.declare(engine, DerivativeMethod.FINITE_DIFFERENCE, loader=_ADAPTER_FD)
    # NSX-88: OpenSim, the adapter itself exposes the derivative method.
    reg.declare(
        "opensim",
        DerivativeMethod.FINITE_DIFFERENCE,
        loader=(
            "src.engines.physics_engines.opensim.python.force_adapter:"
            "OpenSimNativeForceAdapter"
        ),
    )
    reg.declare(
        "pinocchio",
        DerivativeMethod.ANALYTIC,
        loader=(
            "src.engines.physics_engines.pinocchio.python.closed_loop_derivatives:"
            "PinocchioAnalyticDerivatives"
        ),
    )
    # NSX-86: Drake AutoDiffXd differentiates the weld KKT solve (exact).
    reg.declare(
        "drake",
        DerivativeMethod.AUTODIFF,
        loader=(
            "src.engines.physics_engines.drake.python.closed_loop_autodiff:"
            "DrakeAutoDiffDerivatives"
        ),
    )
    reg.declare("simscape", DerivativeMethod.NONE, reason=_SIMSCAPE_REASON)
    return reg


def unavailable_derivative_record(
    engine: str,
    *,
    shape: tuple[int, ...],
    units: str,
    model_provenance: ModelProvenance,
    input_provenance: InputProvenance,
    registry: DerivativeCapabilityRegistry | None = None,
    minimum: CapabilityLevel = CapabilityLevel.PARTIAL,
) -> QuantityRecord | None:
    """``UNAVAILABLE`` record for a derivative-based output, or ``None``.

    The one place a variability output asks the capability seam whether
    ``engine`` can supply derivatives.  When it cannot (undeclared, or below
    ``minimum``), the result is an all-NaN ``UNAVAILABLE`` :class:`QuantityRecord`
    whose reason is the registry's text; when it can, ``None`` so the caller
    computes the real quantity.  Never a zero or an interpolated stand-in.
    """
    try:
        (registry or default_registry()).require(engine, minimum=minimum)
    except DerivativeCapabilityUnavailable as exc:
        return QuantityRecord(
            values=np.full(shape, np.nan),
            units=units,
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            model_provenance=model_provenance,
            input_provenance=input_provenance,
            reason=str(exc),
        )
    return None


def require_adapter_derivatives(
    adapter: Any, *, minimum: CapabilityLevel = CapabilityLevel.FULL
) -> None:
    """Fail closed unless ``adapter`` declares and implements the derivatives.

    The adapter must answer ``capability(Capability.CLOSED_LOOP_DERIVATIVES)``
    with at least ``minimum`` *and* expose a callable
    ``constrained_acceleration_derivatives``.  A declaration without the method,
    or the method without a declaration, is refused.
    """
    ask = getattr(adapter, "capability", None)
    level = ask(Capability.CLOSED_LOOP_DERIVATIVES) if callable(ask) else None
    name = type(adapter).__name__
    if level is None or not capability_level_supported(level, minimum=minimum):
        shown = "undeclared" if level is None else level.name
        raise DerivativeCapabilityUnavailable(
            f"{Capability.CLOSED_LOOP_DERIVATIVES.value} requires {minimum.name}; "
            f"{name} declares {shown}"
        )
    if not callable(getattr(adapter, "constrained_acceleration_derivatives", None)):
        raise DerivativeCapabilityUnavailable(
            f"{name} declares closed_loop_derivatives but lacks "
            "constrained_acceleration_derivatives"
        )
