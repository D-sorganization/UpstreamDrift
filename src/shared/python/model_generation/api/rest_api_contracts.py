"""Shared request, response, and route contracts for model_generation APIs.

This module re-exports the canonical protocol/data types defined in
``rest_api_types`` so that every adapter and route module shares a single
``HTTPMethod`` enum and ``APIRequest``/``APIResponse``/``Route`` class. Defining
duplicate copies here previously produced distinct enum identities, causing
route matching (which compares ``route.method != request.method``) to fail and
return 404 for valid requests.
"""

from __future__ import annotations

from .rest_api_types import (
    APIRequest,
    APIResponse,
    HTTPMethod,
    Route,
)

__all__ = ["APIRequest", "APIResponse", "HTTPMethod", "Route"]
