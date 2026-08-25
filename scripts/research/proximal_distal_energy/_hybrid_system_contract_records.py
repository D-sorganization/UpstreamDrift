"""Immutable records for the proximal-distal hybrid-system contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "proximal-distal-hybrid-system/v1"
EXPECTED_TIER_IDS = (
    "analytical_double_pendulum",
    "forward_two_arm_planar",
    "moving_base_compliant_club",
    "articulated_spatial_whole_body",
    "neuromusculoskeletal",
    "impact_and_ball_flight",
    "participant_calibrated_digital_twin",
    "governed_human_validation",
)
AVAILABILITY = frozenset({"implemented", "partial", "not_applicable", "unavailable"})
AUTHORITY_STATUS = frozenset({"implemented", "partial", "unavailable"})
COMPARISON_STATUS = frozenset(
    {
        "eligible_for_declared_diagnostic_only",
        "eligible_for_declared_forward_comparison",
        "unavailable",
    }
)


@dataclass(frozen=True, slots=True)
class StateBlock:
    """Continuous or algebraic state block with declared coordinates."""

    block_id: str
    variables: tuple[str, ...]
    unit: str
    frame: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class Control:
    """One bounded control channel and its actuator binding."""

    control_id: str
    variable: str
    unit: str
    bounds: tuple[float, float] | None
    actuator_id: str | None
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class AlgebraicConstraint:
    """One equality, inequality, or complementarity constraint."""

    constraint_id: str
    kind: str
    expression: str
    residual_unit: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class Mode:
    """One hybrid mode and its invariant."""

    mode_id: str
    invariant: str
    contact_state: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class Guard:
    """Directed transition surface between registered modes."""

    guard_id: str
    source_mode: str
    target_mode: str
    surface: str
    direction: str
    uncertainty_id: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class Reset:
    """State reset associated with a registered guard."""

    reset_id: str
    guard_id: str
    state_map: str
    energy_accounting: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class Impact:
    """Impulsive event law or an explicit not-applicable boundary."""

    impact_id: str
    guard_id: str | None
    impulse_law: str
    energy_accounting: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class ActuatorDynamics:
    """Actuator state, delay, saturation, and differential-law contract."""

    actuator_id: str
    state_block_ids: tuple[str, ...]
    differential_law: str
    saturation: str
    delay_s: float | None
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class EventUncertainty:
    """Bounded uncertain guard-surface parameterization."""

    uncertainty_id: str
    guard_id: str
    parameterization: str
    bounds: tuple[float, float] | None
    distribution_status: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class Observable:
    """Cross-tier quantity with explicit unit and frame."""

    observable_id: str
    quantity: str
    unit: str
    frame: str
    availability: str
    reason: str


@dataclass(frozen=True, slots=True)
class HybridTier:
    """Complete topology and evidence boundary for one model-ladder tier."""

    tier_id: str
    title: str
    authority_status: str
    source_paths: tuple[str, ...]
    state_blocks: tuple[StateBlock, ...]
    controls: tuple[Control, ...]
    algebraic_constraints: tuple[AlgebraicConstraint, ...]
    modes: tuple[Mode, ...]
    guards: tuple[Guard, ...]
    resets: tuple[Reset, ...]
    impacts: tuple[Impact, ...]
    actuator_dynamics: tuple[ActuatorDynamics, ...]
    uncertain_event_surfaces: tuple[EventUncertainty, ...]
    observables: tuple[Observable, ...]
    limitations: tuple[str, ...]
    falsifiers: tuple[str, ...]
    comparison_eligibility: str
    comparison_blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HybridSystemContract:
    """Executable registry for all eight proximal-distal model tiers."""

    schema_version: str
    issue: str
    inference_boundary: str
    numerical_tolerances: dict[str, float]
    tiers: tuple[HybridTier, ...]

    @classmethod
    def from_json(
        cls, path: str | Path, *, repository_root: str | Path
    ) -> HybridSystemContract:
        """Load and validate a contract from canonical JSON."""
        from scripts.research.proximal_distal_energy.hybrid_system_contract import (
            load_contract_from_json,
        )

        return load_contract_from_json(path, repository_root=repository_root)

    @classmethod
    def from_dict(
        cls, record: dict[str, object], *, repository_root: str | Path
    ) -> HybridSystemContract:
        """Validate a decoded contract and all internal references."""
        from scripts.research.proximal_distal_energy.hybrid_system_contract import (
            load_contract_from_dict,
        )

        return load_contract_from_dict(record, repository_root=repository_root)
