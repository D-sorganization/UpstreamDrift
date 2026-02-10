"""AIP JSON-RPC 2.0 server routes.

Provides HTTP endpoints for JSON-RPC 2.0 method dispatch,
capability negotiation, and batch requests.

See issue #763

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..aip.dispatcher import (
    INVALID_REQUEST,
    PARSE_ERROR,
    dispatch,
    make_error,
    make_response,
)
from ..aip.methods import create_registry
from ..dependencies import get_engine_manager, get_logger
from ..models.responses import (
    AIPCapability,
    AIPHandshakeResponse,
)

router = APIRouter()

# Create the method registry once at module level (stateless, safe)
_registry = create_registry()


@router.get(
    "/aip/capabilities",
    response_model=AIPHandshakeResponse,
)
async def get_capabilities() -> AIPHandshakeResponse:
    """Get AIP server capabilities for handshake negotiation.

    Returns the list of supported methods grouped by namespace,
    enabling clients to discover available functionality before
    making RPC calls.

    Returns:
        AIP handshake response with capabilities.
    """
    namespaces = _registry.list_by_namespace()
    capabilities = [
        AIPCapability(
            name=ns,
            version="1.0",
            methods=methods,
        )
        for ns, methods in namespaces.items()
    ]

    return AIPHandshakeResponse(
        server_name="UpstreamDrift AIP Server",
        protocol_version="2.0",
        capabilities=capabilities,
        supported_methods=_registry.list_methods(),
    )


@router.post("/aip/rpc")
async def handle_rpc(
    request: Request,
    engine_manager: Any = Depends(get_engine_manager),
    logger: Any = Depends(get_logger),
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle a JSON-RPC 2.0 request or batch request.

    Accepts single requests or arrays of requests (batch mode).
    Each request is dispatched independently.

    Args:
        request: FastAPI request (raw JSON body).
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        JSON-RPC response(s).
    """
    # Parse request body
    try:
        body = await request.json()
    except Exception as exc:
        return make_response(
            error=make_error(PARSE_ERROR, f"Parse error: {str(exc)}"),
            request_id=None,
        )

    # Build context for method handlers
    context: dict[str, Any] = {
        "engine_manager": engine_manager,
        "logger": logger,
    }

    # Handle batch requests
    if isinstance(body, list):
        if not body:
            return make_response(
                error=make_error(INVALID_REQUEST, "Empty batch"),
                request_id=None,
            )

        responses: list[dict[str, Any]] = []
        for item in body:
            if not isinstance(item, dict):
                responses.append(
                    make_response(
                        error=make_error(INVALID_REQUEST, "Invalid request in batch"),
                        request_id=None,
                    )
                )
                continue

            result = await dispatch(_registry, item, context)
            if result is not None:  # Notifications return None
                responses.append(result)

        return (
            responses
            if responses
            else make_response(
                error=make_error(INVALID_REQUEST, "All requests were notifications"),
                request_id=None,
            )
        )

    # Handle single request
    if not isinstance(body, dict):
        return make_response(
            error=make_error(INVALID_REQUEST, "Request must be an object or array"),
            request_id=None,
        )

    result = await dispatch(_registry, body, context)
    if result is None:
        # Notification: return empty 204
        return make_response(
            result=None,
            request_id=None,
        )

    return result


@router.get("/aip/methods")
async def list_methods() -> dict[str, Any]:
    """List all available JSON-RPC methods.

    Returns:
        Dictionary with method names, descriptions, and namespaces.
    """
    methods = []
    for name in _registry.list_methods():
        methods.append(
            {
                "name": name,
                "description": _registry.get_description(name),
            }
        )

    return {
        "methods": methods,
        "namespaces": _registry.list_by_namespace(),
        "total": len(methods),
    }
