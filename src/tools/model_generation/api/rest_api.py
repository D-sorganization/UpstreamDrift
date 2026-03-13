"""Compatibility shim — delegates to the canonical REST API implementation.

The canonical ``ModelGenerationAPI`` and related classes live in
``src/shared/python/model_generation/api/rest_api.py``.  Because
``src/shared/python`` appears *before* ``src/tools`` in ``pythonpath``,
the ``model_generation.api.rest_api`` module always resolves to the shared
version.  This file exists only to satisfy tools that enumerate the
``src/tools`` directory and would otherwise report a missing module.

Do **not** add implementation code here — changes must go in the shared file.
"""

# Re-export everything from the canonical shared implementation so that
# any path-based import of this module still works.
from model_generation.api.rest_api import (  # noqa: F401
    APIRequest,
    APIResponse,
    FastAPIAdapter,
    FlaskAdapter,
    HTTPMethod,
    ModelGenerationAPI,
    Route,
)

__all__ = [
    "APIRequest",
    "APIResponse",
    "FastAPIAdapter",
    "FlaskAdapter",
    "HTTPMethod",
    "ModelGenerationAPI",
    "Route",
]
