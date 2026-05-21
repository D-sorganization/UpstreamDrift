"""API standardization module for upstream_drift_tools.

Provides standardized request/response schemas and utilities for all
upstream_drift_tools APIs, ensuring consistent interfaces across all tools.

Key exports:
    - StandardResponse: Unified response wrapper
    - ErrorDetail: Error detail structure
    - ErrorCode: Standard error codes
    - ResponseMetadata: Response metadata
"""

from __future__ import annotations

from .standard_response import (
    ErrorCode,
    ErrorDetail,
    ResponseMetadata,
    StandardResponse,
)

__all__ = [
    "StandardResponse",
    "ErrorDetail",
    "ErrorCode",
    "ResponseMetadata",
]
