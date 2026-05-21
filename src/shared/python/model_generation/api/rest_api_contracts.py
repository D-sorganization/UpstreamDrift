"""Shared request, response, and route contracts for model_generation APIs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HTTPMethod(Enum):
    """Supported HTTP methods for framework-agnostic request handling."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIRequest:
    """Framework-neutral request container."""

    method: HTTPMethod
    path: str
    query_params: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    files: dict[str, bytes] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class APIResponse:
    """Framework-neutral response container."""

    status_code: int
    body: dict[str, Any] | str | bytes
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: dict[str, Any]) -> APIResponse:
        """Create a 200 response with a JSON payload."""
        return cls(status_code=200, body=data)

    @classmethod
    def created(cls, data: dict[str, Any]) -> APIResponse:
        """Create a 201 response with a JSON payload."""
        return cls(status_code=201, body=data)

    @classmethod
    def error(cls, message: str, status_code: int = 400) -> APIResponse:
        """Create an error response with a JSON error payload."""
        return cls(status_code=status_code, body={"error": message})

    @classmethod
    def not_found(cls, message: str = "Not found") -> APIResponse:
        """Create a 404 error response."""
        return cls(status_code=404, body={"error": message})

    @classmethod
    def file(
        cls,
        content: str | bytes,
        filename: str,
        content_type: str = "application/xml",
    ) -> APIResponse:
        """Create a file download response with a content-disposition header."""
        return cls(
            status_code=200,
            body=content if isinstance(content, bytes) else content.encode(),
            content_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


@dataclass
class Route:
    """HTTP route definition used by framework adapters."""

    method: HTTPMethod
    path: str
    handler: Callable[[APIRequest], APIResponse]
    description: str = ""
    tags: list[str] = field(default_factory=list)
