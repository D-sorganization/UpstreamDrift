"""Corrective RED contracts for complete hybrid authority policy (#9236)."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.research.proximal_distal_energy import (
    run_articulated_manufactured_solution as runner,
)
from tests.research.test_articulated_manufactured_hybrid_semantics_red import (
    _profiled_records,
)

pytestmark = [pytest.mark.scientific]

REQUIRED_GATE_FIELDS = (
    "inverse_dynamics_relative_tolerance",
    "conservation_relative_tolerance",
    "cross_engine_relative_tolerance",
    "constraint_position_tolerance_m",
    "constraint_velocity_tolerance_m_s",
    "constraint_virtual_power_tolerance_w",
)
REQUIRED_COMPATIBILITY_FIELDS = (
    "free_body.inverse_dynamics_relative_error.lagrange_mujoco",
    "free_body.inverse_dynamics_relative_error.lagrange_pinocchio",
    "free_body.inverse_dynamics_relative_error.mujoco_pinocchio",
    "free_body.inverse_dynamics_relative_error.maximum",
    "free_body.integration_step_error_rad.0.0005",
    "free_body.integration_step_error_rad.0.001",
    "free_body.integration_step_error_rad.0.002",
    "free_body.richardson_orders.0",
    "free_body.richardson_orders.1",
    "free_body.gravity_free_zero_torque_relative_drift.linear_momentum",
    "free_body.gravity_free_zero_torque_relative_drift.angular_momentum",
    "free_body.gravity_free_zero_torque_relative_drift.kinetic_energy",
    "constrained_motion.position_residual_m",
    "constrained_motion.velocity_residual_m_s",
    "constrained_motion.virtual_power_residual_w",
    "constrained_motion.multiplier_relative_residual",
    "constrained_motion.cross_engine_multiplier_relative_residual",
    "constrained_motion.equilibrium_relative_residual",
)
POLICY_FIELD = "rolling_compatibility_absolute_tolerance_by_field"


def _set_design_value(
    authority: dict[str, Any],
    rolling: dict[str, Any],
    field: str,
    value: object,
) -> None:
    authority["design"][field] = value
    rolling["design"][field] = value


@pytest.mark.parametrize("field", REQUIRED_GATE_FIELDS)
def test_missing_governed_gate_tolerance_is_rejected(field: str) -> None:
    """Every gate tolerance is required even when both records omit it."""

    authority, rolling = _profiled_records()
    del authority["design"][field]
    del rolling["design"][field]

    with pytest.raises(ValueError, match="required|missing|tolerance|policy"):
        runner.compare_semantic_evidence(authority, rolling)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("inverse_dynamics_relative_tolerance", 0.0),
        ("conservation_relative_tolerance", -1.0),
        ("cross_engine_relative_tolerance", True),
        ("constraint_position_tolerance_m", "1e-10"),
        ("constraint_velocity_tolerance_m_s", float("nan")),
        ("constraint_virtual_power_tolerance_w", float("inf")),
    ),
)
def test_governed_gate_tolerance_requires_positive_finite_number(
    field: str, invalid: object
) -> None:
    """Missing validation cannot be replaced by falsey or nonnumeric values."""

    authority, rolling = _profiled_records()
    _set_design_value(authority, rolling, field, invalid)

    with pytest.raises(ValueError, match="finite|numeric|positive|tolerance|policy"):
        runner.compare_semantic_evidence(authority, rolling)


def test_missing_rolling_compatibility_policy_is_rejected() -> None:
    """An absent rolling policy cannot mean implicit exact comparison."""

    authority, rolling = _profiled_records()
    del authority["design"][POLICY_FIELD]
    del rolling["design"][POLICY_FIELD]

    with pytest.raises(ValueError, match="required|missing|compatib|policy"):
        runner.compare_semantic_evidence(authority, rolling)


@pytest.mark.parametrize("invalid", (None, [], "{}", True, {}))
def test_rolling_compatibility_policy_must_be_nonempty_mapping(
    invalid: object,
) -> None:
    """The policy boundary rejects missing, empty, and wrong-shaped values."""

    authority, rolling = _profiled_records()
    _set_design_value(authority, rolling, POLICY_FIELD, invalid)

    with pytest.raises(ValueError, match="compatib|policy|mapping|required"):
        runner.compare_semantic_evidence(authority, rolling)


@pytest.mark.parametrize("field", REQUIRED_COMPATIBILITY_FIELDS)
def test_missing_rolling_compatibility_field_is_rejected(field: str) -> None:
    """Every governed numeric result needs an explicit rolling tolerance."""

    authority, rolling = _profiled_records()
    del authority["design"][POLICY_FIELD][field]
    del rolling["design"][POLICY_FIELD][field]

    with pytest.raises(ValueError, match="complete|required|compatib|policy|field"):
        runner.compare_semantic_evidence(authority, rolling)


@pytest.mark.parametrize("invalid", (0.0, -1.0, True, "1e-8"))
def test_rolling_compatibility_field_requires_positive_finite_number(
    invalid: object,
) -> None:
    """Every field allowance is an explicit positive finite number."""

    authority, rolling = _profiled_records()
    field = REQUIRED_COMPATIBILITY_FIELDS[0]
    authority["design"][POLICY_FIELD][field] = invalid
    rolling["design"][POLICY_FIELD][field] = invalid

    with pytest.raises(ValueError, match="finite|numeric|positive|compatib|policy"):
        runner.compare_semantic_evidence(authority, rolling)


def test_unknown_rolling_compatibility_field_is_rejected() -> None:
    """A typo cannot create an unused tolerance that silently passes."""

    authority, rolling = _profiled_records()
    authority["design"][POLICY_FIELD]["free_body.typo"] = 1.0e-8
    rolling["design"][POLICY_FIELD]["free_body.typo"] = 1.0e-8

    with pytest.raises(ValueError, match="complete|unknown|compatib|policy|field"):
        runner.compare_semantic_evidence(authority, rolling)
