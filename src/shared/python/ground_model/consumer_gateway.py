"""Fail-closed UpstreamDrift gateway to the optional Tools ground contracts.

This module owns no ground physics.  It validates the versioned public façade
from ``shared.python.swing_sim.ground`` and keeps the returned Tools records
intact so provenance and deterministic wire behavior are not relabeled here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

GROUND_CONTRACT_MODULE = "shared.python.swing_sim.ground"
EXPECTED_REQUEST_SCHEMA_VERSION = "flight-to-ground-request/v1"
EXPECTED_RESULT_SCHEMA_VERSION = "flight-to-ground-result/v1"
EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION = "ground-reference-execution/v1"

_MISSING = object()
_SCHEMA_EXPECTATIONS = (
    ("REQUEST_SCHEMA_VERSION", EXPECTED_REQUEST_SCHEMA_VERSION),
    ("RESULT_SCHEMA_VERSION", EXPECTED_RESULT_SCHEMA_VERSION),
    (
        "GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION",
        EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION,
    ),
)
_CALLABLE_EXPORTS = (
    "request_from_json",
    "result_from_json",
    "run_ground_reference",
)
Importer = Callable[[str], object]


class GroundContractUnavailableError(ImportError):
    """Raised when the pinned or editable Tools ground façade is unavailable."""


class GroundContractCompatibilityError(ValueError):
    """Raised when an available Tools façade violates the supported contract."""


@dataclass(frozen=True)
class GroundContractCapabilities:
    """Immutable result of probing the optional Tools ground boundary."""

    available: bool
    module_name: str
    request_schema_version: str | None
    result_schema_version: str | None
    reference_execution_schema_version: str | None
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.module_name != GROUND_CONTRACT_MODULE:
            raise ValueError("module_name must identify the canonical Tools façade")
        versions = (
            self.request_schema_version,
            self.result_schema_version,
            self.reference_execution_schema_version,
        )
        if self.available:
            expected = (
                EXPECTED_REQUEST_SCHEMA_VERSION,
                EXPECTED_RESULT_SCHEMA_VERSION,
                EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION,
            )
            if versions != expected or self.failure is not None:
                raise ValueError("available capabilities require exact v1 schemas")
        elif any(version is not None for version in versions) or not self.failure:
            raise ValueError("unavailable capabilities require one failure description")


def _required_export(module: object, name: str) -> object:
    value = getattr(module, name, _MISSING)
    if value is _MISSING:
        raise GroundContractCompatibilityError(
            f"Tools ground façade is missing required export {name}"
        )
    return value


def _validate_schema_exports(module: object) -> None:
    for name, expected in _SCHEMA_EXPECTATIONS:
        actual = _required_export(module, name)
        if actual != expected:
            raise GroundContractCompatibilityError(
                f"Tools ground façade has incompatible {name}: "
                f"expected {expected!r}, got {actual!r}"
            )


def _required_callable(module: object, name: str) -> Callable[..., object]:
    value = _required_export(module, name)
    if not callable(value):
        raise GroundContractCompatibilityError(
            f"Tools ground façade export {name} must be callable"
        )
    return value


class GroundContractGateway:
    """Thin consumer of the exact Tools v1 request/result façade.

    Postconditions:
        Parsed and executed records are returned without wrapping or copying.
        Serialized results use the record's canonical ``to_json`` implementation.
    """

    def __init__(
        self,
        *,
        request_parser: Callable[[str], object],
        result_parser: Callable[[str], object],
        reference_runner: Callable[..., object],
    ) -> None:
        for value, name in (
            (request_parser, "request_parser"),
            (result_parser, "result_parser"),
            (reference_runner, "reference_runner"),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        self._request_parser = request_parser
        self._result_parser = result_parser
        self._reference_runner = reference_runner
        self._capabilities = _available_capabilities()

    @classmethod
    def from_module(cls, module: object) -> GroundContractGateway:
        """Validate ``module`` and bind only its stable public callables."""
        if module is None:
            raise TypeError("module must not be None")
        _validate_schema_exports(module)
        return cls(
            request_parser=_required_callable(module, "request_from_json"),
            result_parser=_required_callable(module, "result_from_json"),
            reference_runner=_required_callable(module, "run_ground_reference"),
        )

    @property
    def capabilities(self) -> GroundContractCapabilities:
        """Return the exact schemas validated when this gateway was constructed."""
        return self._capabilities

    def parse_request(self, text: str) -> object:
        """Parse one strict canonical request JSON document."""
        if not isinstance(text, str):
            raise TypeError("request JSON must be text")
        return self._request_parser(text)

    def parse_result(self, text: str) -> object:
        """Parse one strict canonical result JSON document."""
        if not isinstance(text, str):
            raise TypeError("result JSON must be text")
        return self._result_parser(text)

    def serialize_result(self, result: object) -> str:
        """Serialize a Tools result through its deterministic public wire method."""
        serializer = getattr(result, "to_json", None)
        if not callable(serializer):
            raise TypeError("result must expose callable to_json")
        text = serializer()
        if not isinstance(text, str):
            raise GroundContractCompatibilityError("result to_json must return text")
        return text

    def run_reference(self, request: object, execution: object | None = None) -> object:
        """Run the canonical Tools reference pipeline without re-labeling output."""
        if request is None:
            raise TypeError("request must not be None")
        return self._reference_runner(request, execution)


def _available_capabilities() -> GroundContractCapabilities:
    return GroundContractCapabilities(
        available=True,
        module_name=GROUND_CONTRACT_MODULE,
        request_schema_version=EXPECTED_REQUEST_SCHEMA_VERSION,
        result_schema_version=EXPECTED_RESULT_SCHEMA_VERSION,
        reference_execution_schema_version=(
            EXPECTED_REFERENCE_EXECUTION_SCHEMA_VERSION
        ),
    )


def _unavailable_capabilities(failure: str) -> GroundContractCapabilities:
    return GroundContractCapabilities(
        available=False,
        module_name=GROUND_CONTRACT_MODULE,
        request_schema_version=None,
        result_schema_version=None,
        reference_execution_schema_version=None,
        failure=failure,
    )


def _validate_importer(importer: Importer) -> None:
    if not callable(importer):
        raise TypeError("importer must be callable")


def load_ground_contract_gateway(
    importer: Importer = import_module,
) -> GroundContractGateway:
    """Load and strictly validate the configured Tools ground façade."""
    _validate_importer(importer)
    try:
        module = importer(GROUND_CONTRACT_MODULE)
    except ImportError as exc:
        raise GroundContractUnavailableError(
            f"Tools ground contracts are not available: {exc}"
        ) from exc
    return GroundContractGateway.from_module(module)


def probe_ground_contracts(
    importer: Importer = import_module,
) -> GroundContractCapabilities:
    """Return exact capability evidence without making optional imports fatal."""
    _validate_importer(importer)
    try:
        gateway = load_ground_contract_gateway(importer)
    except (GroundContractUnavailableError, GroundContractCompatibilityError) as exc:
        return _unavailable_capabilities(str(exc))
    return gateway.capabilities


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
