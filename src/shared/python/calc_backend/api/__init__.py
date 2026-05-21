"""API standardization modules for calc_backend.

Provides:
- StandardResponse: consistent response wrapper across all endpoints
- ErrorCode and ErrorDetail: standardized error handling
- ResponseMetadata: request tracking and observability
- StandardResponseBuilder: convenient builder for success/error responses
"""

from .standard_response import (
    ErrorCode,
    ErrorDetail,
    ResponseMetadata,
    StandardResponse,
    StandardResponseBuilder,
)

__all__ = [
    "StandardResponse",
    "ErrorCode",
    "ErrorDetail",
    "ResponseMetadata",
    "StandardResponseBuilder",
]
