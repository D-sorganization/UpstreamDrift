"""Golf simulator integration package for UpstreamDrift.

Provides canonical immutable shot contracts, capability ports, and conversion facades.
"""

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ClubData,
    ConnectionState,
    ConnectionStatus,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotMetadata,
    ShotQualification,
    SimulatorAdapter,
    SimulatorCapabilities,
    SimulatorEvent,
    SourceKind,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.launch_bridge import (
    launch_conditions_to_shot_envelope,
    pipeline_result_to_shot_envelope,
    post_impact_state_to_shot_envelope,
    rpm_to_rad_s,
)

__all__ = [
    "AimContext",
    "CapabilityDescriptor",
    "CapabilityState",
    "ClubData",
    "ConnectionState",
    "ConnectionStatus",
    "ContactStatus",
    "NumericalStatus",
    "ScientificStatus",
    "ShotEnvelope",
    "ShotMetadata",
    "ShotQualification",
    "SimulatorAdapter",
    "SimulatorCapabilities",
    "SimulatorEvent",
    "SourceKind",
    "SubmissionReceipt",
    "SubmissionState",
    "launch_conditions_to_shot_envelope",
    "pipeline_result_to_shot_envelope",
    "post_impact_state_to_shot_envelope",
    "rpm_to_rad_s",
]
