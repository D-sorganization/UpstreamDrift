"""UpstreamDrift adapters for the canonical Tools ground-model contracts."""

from .consumer_gateway import (
    EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION,
    EXPECTED_REQUEST_SCHEMA_VERSION,
    EXPECTED_RESULT_SCHEMA_VERSION,
    GROUND_CONTRACT_MODULE,
    GroundContractCapabilities,
    GroundContractCompatibilityError,
    GroundContractGateway,
    GroundContractUnavailableError,
    load_ground_contract_gateway,
    probe_ground_contracts,
)

__all__ = [
    "EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION",
    "EXPECTED_REQUEST_SCHEMA_VERSION",
    "EXPECTED_RESULT_SCHEMA_VERSION",
    "GROUND_CONTRACT_MODULE",
    "GroundContractCapabilities",
    "GroundContractCompatibilityError",
    "GroundContractGateway",
    "GroundContractUnavailableError",
    "load_ground_contract_gateway",
    "probe_ground_contracts",
]
