"""FastAPI adapter for the model generation REST API.

Registers ``ModelGenerationAPI`` routes with a FastAPI application instance.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from .rest_api_routes import ModelGenerationAPI
from .rest_api_types import APIRequest, HTTPMethod, Route

logger = logging.getLogger(__name__)


class FastAPIAdapter:
    """Adapter for FastAPI framework."""

    def __init__(self, api: ModelGenerationAPI) -> None:
        if api is None:
            raise ValueError("api must be provided")
        self.api = api

    def register(self, app: Any) -> None:
        """Register routes with FastAPI app."""
        from fastapi import Request, Response
        from fastapi.responses import JSONResponse

        def make_handler(r: Route) -> Callable[..., Any]:
            """Build the async endpoint for *r*.

            This is a *plain* (non-async) factory: it returns the inner
            ``handler`` coroutine function so FastAPI receives a callable
            endpoint, not an un-awaited coroutine object. ``r`` is bound as a
            default-free closure argument to avoid late-binding across the
            registration loop.
            """

            async def handler(request: Request) -> Any:
                body = None
                try:
                    body = await request.json()
                except (ValueError, UnicodeDecodeError) as e:
                    logger.debug("Failed to parse request JSON body: %s", e)

                files = {}
                content_type = request.headers.get("content-type", "").lower()
                if content_type.startswith(
                    ("multipart/form-data", "application/x-www-form-urlencoded")
                ):
                    try:
                        form = await request.form()
                    except AssertionError as e:
                        logger.debug("Failed to parse request form body: %s", e)
                    else:
                        for key, value in form.items():
                            if hasattr(value, "read"):
                                files[key] = await value.read()

                # FastAPI inspects the endpoint signature to resolve ``{path}``
                # params. A catch-all ``**kwargs`` is invisible to it (and a
                # bare ``request: Request`` param is the one signature FastAPI
                # recognises for the raw request), so path params must be read
                # from the resolved request rather than from kwargs.
                path_params = dict(getattr(request, "path_params", {}))
                api_request = APIRequest(
                    method=HTTPMethod(request.method),
                    path=request.url.path,
                    query_params={
                        **request.query_params,
                        **path_params,
                    },
                    body=body,
                    files=files,
                    headers=dict(request.headers),
                )

                response = self.api.handle_request(api_request)

                if isinstance(response.body, bytes):
                    return Response(
                        content=response.body,
                        status_code=response.status_code,
                        media_type=response.content_type,
                        headers=response.headers,
                    )
                return JSONResponse(
                    content=response.body,
                    status_code=response.status_code,
                    headers=response.headers,
                )

            # This module uses ``from __future__ import annotations``, so the
            # ``request: Request`` annotation is a *string* that FastAPI would
            # try (and fail) to resolve against ``handler.__globals__`` — where
            # ``Request`` is not defined (it is imported locally). Attach an
            # explicit signature whose annotation is the real ``Request`` class
            # so FastAPI injects the raw request instead of treating it as a
            # required query parameter (which previously produced a 422).
            handler.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                parameters=[
                    inspect.Parameter(
                        "request",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=Request,
                    )
                ]
            )
            return handler

        for route in self.api.get_routes():
            # FastAPI uses {param} format already
            app.add_api_route(
                route.path,
                make_handler(route),
                methods=[route.method.value],
                tags=route.tags,
                summary=route.description,
            )
