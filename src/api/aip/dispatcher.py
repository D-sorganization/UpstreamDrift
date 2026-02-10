"""JSON-RPC 2.0 method dispatcher for AIP server.

Implements the core dispatch logic for JSON-RPC method resolution,
parameter validation, and error handling according to the JSON-RPC 2.0
specification (https://www.jsonrpc.org/specification).

See issue #763
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def make_error(code: int, message: str, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error object.

    Args:
        code: Error code.
        message: Human-readable error message.
        data: Optional additional data.

    Returns:
        Error dictionary.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return error


def make_response(
    result: Any = None,
    error: dict[str, Any] | None = None,
    request_id: int | str | None = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 response object.

    Args:
        result: Method result (mutually exclusive with error).
        error: Error object (mutually exclusive with result).
        request_id: Matching request ID.

    Returns:
        Response dictionary.
    """
    resp: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result
    return resp


class MethodRegistry:
    """Registry for JSON-RPC methods.

    Methods are registered with a namespace (e.g., 'simulation.start')
    and can be dispatched by name.

    See issue #763
    """

    def __init__(self) -> None:
        """Initialize with empty method registry."""
        self._methods: dict[str, Callable[..., Any]] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self, name: str, handler: Callable[..., Any], description: str = ""
    ) -> None:
        """Register a method handler.

        Args:
            name: Method name (e.g., 'simulation.start').
            handler: Callable that implements the method.
            description: Human-readable description.
        """
        self._methods[name] = handler
        self._descriptions[name] = description

    def get_method(self, name: str) -> Callable[..., Any] | None:
        """Look up a method by name.

        Args:
            name: Method name.

        Returns:
            Handler callable or None if not found.
        """
        return self._methods.get(name)

    def list_methods(self) -> list[str]:
        """List all registered method names.

        Returns:
            Sorted list of method names.
        """
        return sorted(self._methods.keys())

    def get_description(self, name: str) -> str:
        """Get method description.

        Args:
            name: Method name.

        Returns:
            Description string.
        """
        return self._descriptions.get(name, "")

    def list_by_namespace(self) -> dict[str, list[str]]:
        """Group methods by namespace.

        Returns:
            Dictionary mapping namespace to method names.
        """
        namespaces: dict[str, list[str]] = {}
        for name in sorted(self._methods.keys()):
            parts = name.split(".", 1)
            ns = parts[0] if len(parts) > 1 else "root"
            namespaces.setdefault(ns, []).append(name)
        return namespaces


async def dispatch(
    registry: MethodRegistry,
    request: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Dispatch a JSON-RPC 2.0 request.

    Validates the request structure, resolves the method, and
    calls the handler with the provided parameters.

    Args:
        registry: Method registry to look up handlers.
        request: JSON-RPC request object.
        context: Optional context dict passed to handlers.

    Returns:
        JSON-RPC response object, or None for notifications.
    """
    request_id = request.get("id")

    # Validate JSON-RPC version
    if request.get("jsonrpc") != "2.0":
        return make_response(
            error=make_error(INVALID_REQUEST, "Invalid JSON-RPC version"),
            request_id=request_id,
        )

    # Validate method
    method_name = request.get("method")
    if not isinstance(method_name, str) or not method_name:
        return make_response(
            error=make_error(INVALID_REQUEST, "Missing or invalid method"),
            request_id=request_id,
        )

    # Look up handler
    handler = registry.get_method(method_name)
    if handler is None:
        return make_response(
            error=make_error(
                METHOD_NOT_FOUND,
                f"Method not found: {method_name}",
                data={"available": registry.list_methods()},
            ),
            request_id=request_id,
        )

    # Prepare parameters
    params = request.get("params")
    kwargs: dict[str, Any] = {}
    args: list[Any] = []

    if isinstance(params, dict):
        kwargs = params
    elif isinstance(params, list):
        args = params
    elif params is not None:
        return make_response(
            error=make_error(INVALID_PARAMS, "Params must be object or array"),
            request_id=request_id,
        )

    # Inject context if handler accepts it
    if context is not None:
        kwargs["_context"] = context

    # Call handler
    try:
        import asyncio

        if asyncio.iscoroutinefunction(handler):
            result = await handler(*args, **kwargs)
        else:
            result = handler(*args, **kwargs)
    except TypeError as exc:
        return make_response(
            error=make_error(
                INVALID_PARAMS,
                f"Invalid parameters: {str(exc)}",
            ),
            request_id=request_id,
        )
    except ImportError as exc:
        return make_response(
            error=make_error(
                INTERNAL_ERROR,
                f"Internal error: {str(exc)}",
            ),
            request_id=request_id,
        )

    # Notifications (no id) don't get a response
    if request_id is None:
        return None

    return make_response(result=result, request_id=request_id)
