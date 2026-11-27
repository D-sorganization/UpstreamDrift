"""AIP-specific errors for JSON-RPC dispatch (#14940)."""

from __future__ import annotations

AIP_NOT_IMPLEMENTED_CODE = "not_implemented"
AIP_METHOD_NOT_IMPLEMENTED = -32000


class AIPMethodNotImplemented(Exception):
    """Raised when a registered AIP method has no backing implementation."""

    def __init__(self, method: str, message: str) -> None:
        self.method, self.message = method, message
        super().__init__(message)
