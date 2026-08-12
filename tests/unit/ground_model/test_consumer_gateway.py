"""Contract tests for the optional Tools ground-model consumer boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.shared.python.ground_model.consumer_gateway import (
    EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION,
    EXPECTED_REQUEST_SCHEMA_VERSION,
    EXPECTED_RESULT_SCHEMA_VERSION,
    GroundContractCompatibilityError,
    GroundContractCapabilities,
    GroundContractUnavailableError,
    GroundContractGateway,
    load_ground_contract_gateway,
    probe_ground_contracts,
)

pytestmark = pytest.mark.unit


class _WireResult:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def to_json(self) -> str:
        return self.payload


def _facade() -> SimpleNamespace:
    return SimpleNamespace(
        REQUEST_SCHEMA_VERSION=EXPECTED_REQUEST_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION=EXPECTED_RESULT_SCHEMA_VERSION,
        GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION=(
            EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION
        ),
        request_from_json=lambda text: ("request", text),
        result_from_json=lambda text: _WireResult(text),
        run_ground_reference=lambda request, execution=None: (
            "result",
            request,
            execution,
        ),
    )


def test_gateway_preserves_canonical_facade_objects_and_wire_text() -> None:
    gateway = GroundContractGateway.from_module(_facade())

    request = gateway.parse_request('{"request":1}')
    result = gateway.parse_result('{"result":1}')

    assert request == ("request", '{"request":1}')
    assert isinstance(result, _WireResult)
    assert gateway.serialize_result(result) == '{"result":1}'
    assert gateway.run_reference(request, execution="bounded") == (
        "result",
        request,
        "bounded",
    )
    assert gateway.capabilities.available is True
    assert gateway.capabilities.failure is None


def test_probe_is_import_safe_when_tools_ground_package_is_absent() -> None:
    def missing_importer(module_name: str) -> object:
        raise ModuleNotFoundError(f"No module named {module_name!r}", name=module_name)

    capabilities = probe_ground_contracts(importer=missing_importer)

    assert capabilities.available is False
    assert capabilities.request_schema_version is None
    assert capabilities.result_schema_version is None
    assert capabilities.reference_execution_schema_version is None
    assert "No module named" in (capabilities.failure or "")
    with pytest.raises(GroundContractUnavailableError, match="not available"):
        load_ground_contract_gateway(importer=missing_importer)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("RESULT_SCHEMA_VERSION", "flight-to-ground-result/v2", "incompatible"),
        ("request_from_json", None, "callable"),
        ("run_ground_reference", None, "callable"),
    ],
)
def test_gateway_rejects_malformed_or_incompatible_facades(
    field: str, value: object, message: str
) -> None:
    facade = _facade()
    setattr(facade, field, value)

    with pytest.raises(GroundContractCompatibilityError, match=message):
        GroundContractGateway.from_module(facade)

    capabilities = probe_ground_contracts(importer=lambda _name: facade)
    assert capabilities.available is False
    assert message in (capabilities.failure or "")


def test_gateway_validates_public_input_and_output_boundaries() -> None:
    gateway = GroundContractGateway.from_module(_facade())

    with pytest.raises(TypeError, match="request JSON must be text"):
        gateway.parse_request(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="result JSON must be text"):
        gateway.parse_result(b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="result must expose callable to_json"):
        gateway.serialize_result(object())

    malformed = _facade()
    malformed.result_from_json = lambda _text: SimpleNamespace(to_json=lambda: 3)
    malformed_gateway = GroundContractGateway.from_module(malformed)
    with pytest.raises(GroundContractCompatibilityError, match="return text"):
        malformed_gateway.serialize_result(malformed_gateway.parse_result("{}"))


def test_gateway_requires_a_callable_importer() -> None:
    with pytest.raises(TypeError, match="importer must be callable"):
        probe_ground_contracts(importer=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="importer must be callable"):
        load_ground_contract_gateway(importer=None)  # type: ignore[arg-type]


def test_capability_records_enforce_available_and_unavailable_invariants() -> None:
    with pytest.raises(ValueError, match="exact v1 schemas"):
        GroundContractCapabilities(
            available=True,
            module_name="shared.python.swing_sim.ground",
            request_schema_version=EXPECTED_REQUEST_SCHEMA_VERSION,
            result_schema_version="flight-to-ground-result/v2",
            reference_execution_schema_version=(
                EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION
            ),
        )
    with pytest.raises(ValueError, match="failure description"):
        GroundContractCapabilities(
            available=False,
            module_name="shared.python.swing_sim.ground",
            request_schema_version=None,
            result_schema_version=None,
            reference_execution_schema_version=None,
        )
