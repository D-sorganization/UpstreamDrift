"""Golf simulator integration package for UpstreamDrift.

Provides canonical immutable shot contracts, capability ports, and conversion facades.
"""

from src.shared.python.golf_simulator.adapters.local import (
    LocalReferenceAdapter,
    TrajectoryRecord,
)
from src.shared.python.golf_simulator.adapters.relay import (
    FlightRelayAdapter,
    FlightRelayConfig,
    UnsupportedDestinationError,
)
from src.shared.python.golf_simulator.contracts import (
    AimContext,
    CapabilityDescriptor,
    CapabilityState,
    ClubData,
    ConnectionState,
    ConnectionStatus,
    ContactStatus,
    NumericalStatus,
    PreparedShot,
    ScientificStatus,
    SessionState,
    ShotEnvelope,
    ShotMetadata,
    ShotQualification,
    SimulatorAdapter,
    SimulatorCapabilities,
    SimulatorEvent,
    SourceKind,
    SubmissionReceipt,
    SubmissionState,
    UnsupportedCapabilityError,
    assert_capability_supported,
)
from src.shared.python.golf_simulator.journal import (
    DeliveryStatus,
    JournalEntry,
    ShotJournal,
)
from src.shared.python.golf_simulator.launch_bridge import (
    launch_conditions_to_shot_envelope,
    pipeline_result_to_shot_envelope,
    post_impact_state_to_shot_envelope,
    rpm_to_rad_s,
    shot_envelope_to_launch_conditions,
)
from src.shared.python.golf_simulator.discovery import (
    SimulatorConfig,
    SimulatorEndpoint,
    SupportReceipt,
    discover_simulator_installation,
)
from src.shared.python.golf_simulator.logging_redaction import (
    SecretRedactionFilter,
    redact_mapping,
    redact_text,
)
from src.shared.python.golf_simulator.packaging import (
    EnvironmentReadinessReport,
    LicensingPolicyViolationError,
    assert_licensing_policy,
    check_environment_readiness,
)
from src.shared.python.golf_simulator.producer_lock import (
    ProducerConflictError,
    ProducerLock,
    ProducerLockManager,
)
from src.shared.python.golf_simulator.remote_bridge import (
    AuthenticationError,
    BridgeSecurityError,
    CancellationToken,
    LocalBridgeServer,
    QueueCapacityExceededError,
    RemoteBridgeClient,
)
from src.shared.python.golf_simulator.replay import (
    MonotonicReplayClock,
    ReplaySubmissionCoordinator,
)
from src.shared.python.golf_simulator.session import GolfSessionService

__all__ = [
    "AimContext",
    "AuthenticationError",
    "BridgeSecurityError",
    "CancellationToken",
    "CapabilityDescriptor",
    "CapabilityState",
    "ClubData",
    "ConnectionState",
    "ConnectionStatus",
    "ContactStatus",
    "DeliveryStatus",
    "EnvironmentReadinessReport",
    "FlightRelayAdapter",
    "FlightRelayConfig",
    "GolfSessionService",
    "JournalEntry",
    "LicensingPolicyViolationError",
    "LocalBridgeServer",
    "LocalReferenceAdapter",
    "MonotonicReplayClock",
    "NumericalStatus",
    "PreparedShot",
    "ProducerConflictError",
    "ProducerLock",
    "ProducerLockManager",
    "QueueCapacityExceededError",
    "RemoteBridgeClient",
    "ReplaySubmissionCoordinator",
    "ScientificStatus",
    "SecretRedactionFilter",
    "SessionState",
    "ShotEnvelope",
    "ShotJournal",
    "ShotMetadata",
    "ShotQualification",
    "SimulatorAdapter",
    "SimulatorCapabilities",
    "SimulatorConfig",
    "SimulatorEndpoint",
    "SimulatorEvent",
    "SourceKind",
    "SubmissionReceipt",
    "SubmissionState",
    "SupportReceipt",
    "TrajectoryRecord",
    "UnsupportedCapabilityError",
    "UnsupportedDestinationError",
    "assert_capability_supported",
    "assert_licensing_policy",
    "check_environment_readiness",
    "discover_simulator_installation",
    "launch_conditions_to_shot_envelope",
    "pipeline_result_to_shot_envelope",
    "post_impact_state_to_shot_envelope",
    "redact_mapping",
    "redact_text",
    "rpm_to_rad_s",
    "shot_envelope_to_launch_conditions",
]
