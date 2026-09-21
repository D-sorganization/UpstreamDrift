"""Simulator adapters package."""

from .fake import FakeSimulatorAdapter
from .local import LocalReferenceAdapter
from .relay import (
    FlightRelayAdapter,
    FlightRelayConfig,
    UnsupportedDestinationError,
)

__all__ = [
    "FakeSimulatorAdapter",
    "FlightRelayAdapter",
    "FlightRelayConfig",
    "LocalReferenceAdapter",
    "UnsupportedDestinationError",
]
