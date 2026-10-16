"""NSX-84 (#12713): derivative capability declaration and registry (fail closed)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace import derivative_capability as dc

pytestmark = pytest.mark.unit


def test_capability_enum_has_closed_loop_derivatives() -> None:
    assert Capability.CLOSED_LOOP_DERIVATIVES.value == "closed_loop_derivatives"


@pytest.mark.parametrize(
    ("method", "level"),
    [
        (dc.DerivativeMethod.ANALYTIC, CapabilityLevel.FULL),
        (dc.DerivativeMethod.AUTODIFF, CapabilityLevel.FULL),
        (dc.DerivativeMethod.FINITE_DIFFERENCE, CapabilityLevel.PARTIAL),
        (dc.DerivativeMethod.NONE, CapabilityLevel.NONE),
    ],
)
def test_method_maps_to_level(
    method: dc.DerivativeMethod, level: CapabilityLevel
) -> None:
    assert method.level is level


def test_registry_fails_closed_without_declaration() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    with pytest.raises(dc.DerivativeCapabilityUnavailable, match="not declared"):
        reg.require("pinocchio")


def test_none_declaration_carries_reason_and_never_falls_back() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    reg.declare("myosuite", dc.DerivativeMethod.NONE, reason="no analytic KKT rows")
    with pytest.raises(CapabilityUnavailable, match="no analytic KKT rows"):
        reg.require("myosuite")
    with pytest.raises(dc.DerivativeCapabilityUnavailable):
        reg.require("myosuite", minimum=CapabilityLevel.PARTIAL)


def test_none_declaration_needs_a_reason() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    with pytest.raises(ValueError, match="reason"):
        reg.declare("x", dc.DerivativeMethod.NONE)


def test_partial_does_not_satisfy_full() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    reg.declare("mujoco", dc.DerivativeMethod.FINITE_DIFFERENCE)
    assert reg.require("mujoco").method is dc.DerivativeMethod.FINITE_DIFFERENCE
    with pytest.raises(dc.DerivativeCapabilityUnavailable, match="requires FULL"):
        reg.require("mujoco", minimum=CapabilityLevel.FULL)


def test_duplicate_declaration_is_rejected() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    reg.declare("drake", dc.DerivativeMethod.AUTODIFF)
    with pytest.raises(ValueError, match="already declared"):
        reg.declare("drake", dc.DerivativeMethod.ANALYTIC)


def test_loader_is_lazy_and_resolved_on_demand() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    reg.declare(
        "fake",
        dc.DerivativeMethod.ANALYTIC,
        loader="definitely_not_a_module_nsx84:Factory",
    )
    assert "definitely_not_a_module_nsx84" not in sys.modules
    with pytest.raises(dc.DerivativeCapabilityUnavailable, match="cannot import"):
        reg.load("fake")


def test_loader_resolves_dotted_path() -> None:
    reg = dc.DerivativeCapabilityRegistry()
    reg.declare(
        "ref",
        dc.DerivativeMethod.FINITE_DIFFERENCE,
        loader=(
            "src.shared.python.loop_nullspace.tangent_linearization:"
            "FiniteDifferenceDerivatives"
        ),
    )
    factory = reg.load("ref")
    assert factory.__name__ == "FiniteDifferenceDerivatives"
    reg.declare("noloader", dc.DerivativeMethod.ANALYTIC)
    with pytest.raises(dc.DerivativeCapabilityUnavailable, match="no loader"):
        reg.load("noloader")


def test_default_registry_declares_the_reference_and_pending_engines() -> None:
    reg = dc.default_registry()
    assert reg.require("reference_fd").method is dc.DerivativeMethod.FINITE_DIFFERENCE
    assert reg.require("opensim").level is CapabilityLevel.PARTIAL
    assert reg.require("pinocchio").method is dc.DerivativeMethod.ANALYTIC
    for engine in ("drake",):
        with pytest.raises(dc.DerivativeCapabilityUnavailable):
            reg.require(engine)


class _Adapter:
    def __init__(self, level: CapabilityLevel | None, method: bool = True) -> None:
        self._level = level
        if method:
            self.constrained_acceleration_derivatives = lambda q, v, tau: None

    def capability(self, cap: Capability) -> CapabilityLevel | None:
        assert cap is Capability.CLOSED_LOOP_DERIVATIVES
        return self._level


def test_adapter_requires_declaration_and_method() -> None:
    dc.require_adapter_derivatives(_Adapter(CapabilityLevel.FULL))
    dc.require_adapter_derivatives(
        _Adapter(CapabilityLevel.PARTIAL), minimum=CapabilityLevel.PARTIAL
    )
    for bad in (
        _Adapter(None),
        _Adapter(CapabilityLevel.NONE),
        _Adapter(CapabilityLevel.PARTIAL),
        _Adapter(CapabilityLevel.FULL, method=False),
        object(),
    ):
        with pytest.raises(dc.DerivativeCapabilityUnavailable):
            dc.require_adapter_derivatives(bad)


def test_derivative_result_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        dc.ClosedLoopDerivatives(
            acceleration=np.zeros(2),
            reaction=np.zeros(1),
            basis=np.eye(2)[:, :1],
            projector=np.eye(2),
            d_acc_d_xi=np.zeros((2, 2)),
            d_acc_d_eta=np.zeros((2, 1)),
            d_acc_d_tau=np.zeros((2, 2)),
            d_lambda_d_xi=np.zeros((1, 1)),
            d_lambda_d_eta=np.zeros((1, 1)),
            d_lambda_d_tau=np.zeros((1, 2)),
            method=dc.DerivativeMethod.ANALYTIC,
        )
