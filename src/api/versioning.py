"""Shared API version resolution and URI-path version routing helpers.

This module owns two concerns that share the same domain:

1. Reporting the application's semantic version (``get_app_version``) for
   use in OpenAPI metadata, ``/healthz``-style endpoints, and logging.
2. Constructing URI-path versioned ``APIRouter`` instances
   (``make_versioned_router``) that optionally signal deprecation via
   RFC 8594 ``Sunset`` and RFC 9745 ``Deprecation`` response headers.

See ``docs/adr/api-versioning.md`` for the full versioning policy.

Design by Contract:
    - ``get_app_version`` is defensive and returns ``"0.0.0"`` on any error.
    - ``make_versioned_router`` validates its inputs and raises
      ``ValueError``/``TypeError`` on misuse rather than failing later.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib

# Public re-exports for the versioning policy helpers.
__all__ = [
    "get_app_version",
    "make_versioned_router",
]

# Major-version segment, e.g. "v1", "v2", "v17". Minor/patch revisions are
# expressed in the application version, not the URI.
_VERSION_PATTERN = re.compile(r"^v\d+$")


@lru_cache(maxsize=1)
def get_app_version() -> str:
    """Return the canonical application version string.

    Attempts to resolve version from installed package metadata first, then falls
    back to reading pyproject.toml. Returns "0.0.0" if all resolution methods fail.

    Returns:
        Semantic version string (e.g., "2.1.0", "1.0.0-beta", or "0.0.0" as fallback)

    Note:
        Result is cached for performance since version resolution involves file I/O.
    """
    for package_name in ("upstream-drift", "golf-modeling-suite"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)
        return str(data["project"]["version"])
    except (FileNotFoundError, KeyError, OSError, TypeError):
        return "0.0.0"


def _validate_version(value: object) -> str:
    """Validate a URI-path version segment.

    Accepts strings of the form ``vN`` where ``N`` is one or more digits.
    Anything else raises ``ValueError``. Non-string inputs raise
    ``TypeError`` so misuse is distinguishable from a malformed string.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"version must be a string of the form 'vN' (e.g. 'v1'); "
            f"got {type(value).__name__!s}"
        )
    if not _VERSION_PATTERN.match(value):
        raise ValueError(
            f"version must match '^v\\d+$' (e.g. 'v1', 'v2'); got {value!r}"
        )
    return value


def _format_sunset(sunset: str | None) -> str | None:
    """Validate and normalize a ``Sunset`` header value.

    The value is treated as opaque text; we only require it to be a
    non-empty string when provided, so callers can supply RFC 1123 dates
    or any other RFC 8594-compliant representation.
    """
    if sunset is None:
        return None
    if not isinstance(sunset, str):
        raise TypeError(
            f"sunset must be a string (RFC 1123 date) or None; "
            f"got {type(sunset).__name__!s}"
        )
    stripped = sunset.strip()
    if not stripped:
        raise ValueError("sunset must be a non-empty RFC 1123 date string")
    return stripped


def make_versioned_router(
    version: str = "v1",
    *,
    deprecated: bool = False,
    sunset: str | None = None,
) -> APIRouter:
    """Create a versioned ``APIRouter`` with optional deprecation headers.

    Args:
        version: URI-path version segment, e.g. ``"v1"`` or ``"v2"``. Must
            match ``^v\\d+$``.
        deprecated: If ``True``, every response from routes registered on
            the returned router will carry ``Deprecation: true`` (per
            RFC 9745).
        sunset: Optional RFC 1123 date string. When provided alongside
            ``deprecated=True``, every response also carries
            ``Sunset: <date>`` (per RFC 8594). Ignored when
            ``deprecated`` is ``False``.

    Returns:
        A ``fastapi.APIRouter`` whose ``prefix`` is ``f"/{version}"`` and
        whose ``deprecated`` flag mirrors the argument. When deprecated,
        a router-level dependency injects the appropriate response
        headers on every request.

    Raises:
        ValueError: If ``version`` does not match ``^v\\d+$`` or if
            ``sunset`` is an empty string.
        TypeError: If ``version`` is not a string, or ``sunset`` is not
            a string or ``None``.

    Example:
        >>> router = make_versioned_router("v1")
        >>> router.prefix
        '/v1'
    """
    normalized_version = _validate_version(version)
    normalized_sunset = _format_sunset(sunset)

    # Imported lazily so that importing this module does not require
    # FastAPI to be installed (e.g. for ``get_app_version`` consumers in
    # tooling/scripts that run without the API extras).
    from fastapi import APIRouter, Depends, Response

    def _deprecation_headers(response: Response) -> None:
        """Inject RFC 9745/RFC 8594 headers on deprecated routers."""
        response.headers["Deprecation"] = "true"
        if normalized_sunset is not None:
            response.headers["Sunset"] = normalized_sunset

    # The module uses ``from __future__ import annotations`` so the
    # ``response: Response`` hint above is a string at runtime. FastAPI
    # inspects ``__annotations__`` to recognise the special ``Response``
    # parameter; without a live class object it would mis-classify the
    # parameter as a query field and reject the request with HTTP 422.
    # Pin the annotation to the actual class so the dependency injector
    # passes the live ``Response`` instance.
    _deprecation_headers.__annotations__ = {"response": Response, "return": None}

    dependencies: list = []
    if deprecated:
        dependencies.append(Depends(_deprecation_headers))

    return APIRouter(
        prefix=f"/{normalized_version}",
        deprecated=deprecated,
        dependencies=dependencies,
    )
