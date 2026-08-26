"""Cross-record integrity checks for the hybrid-system contract."""

from __future__ import annotations

from typing import Any

from scripts.research.proximal_distal_energy._hybrid_system_contract_records import (
    ActuatorDynamics,
    Control,
    EventUncertainty,
    Guard,
    Impact,
    Mode,
    Reset,
    StateBlock,
)


def validate_unique_ids(items: tuple[Any, ...], attribute: str) -> None:
    """Reject duplicate identifiers within one component registry."""
    values = tuple(getattr(item, attribute) for item in items)
    if len(values) != len(set(values)):
        raise ValueError(f"{attribute} values must be unique")


def validate_references(
    states: tuple[StateBlock, ...],
    controls: tuple[Control, ...],
    modes: tuple[Mode, ...],
    guards: tuple[Guard, ...],
    resets: tuple[Reset, ...],
    impacts: tuple[Impact, ...],
    actuators: tuple[ActuatorDynamics, ...],
    uncertainties: tuple[EventUncertainty, ...],
) -> None:
    """Reject guards, resets, impacts, controls, or actuators with dangling IDs."""
    state_ids = {item.block_id for item in states}
    mode_ids = {item.mode_id for item in modes}
    guard_ids = {item.guard_id for item in guards}
    actuator_ids = {item.actuator_id for item in actuators}
    uncertainty_ids = {item.uncertainty_id for item in uncertainties}
    for guard in guards:
        if guard.source_mode not in mode_ids or guard.target_mode not in mode_ids:
            raise ValueError(f"guard {guard.guard_id} references unknown mode")
        if guard.uncertainty_id not in uncertainty_ids:
            raise ValueError(f"guard {guard.guard_id} references unknown uncertainty")
    for reset in resets:
        if reset.guard_id not in guard_ids:
            raise ValueError(f"reset {reset.reset_id} references unknown guard")
    for impact in impacts:
        if impact.guard_id is not None and impact.guard_id not in guard_ids:
            raise ValueError(f"impact {impact.impact_id} references unknown guard")
    for uncertainty in uncertainties:
        if uncertainty.guard_id not in guard_ids:
            raise ValueError(
                f"uncertainty {uncertainty.uncertainty_id} references unknown guard"
            )
    for actuator in actuators:
        if set(actuator.state_block_ids) - state_ids:
            raise ValueError(
                f"actuator {actuator.actuator_id} references unknown state block"
            )
    for control in controls:
        if control.actuator_id is not None and control.actuator_id not in actuator_ids:
            raise ValueError(
                f"control {control.control_id} references unknown actuator"
            )
