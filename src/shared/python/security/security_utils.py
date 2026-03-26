"""Security utilities for path validation and subprocess hardening.

URL Scheme Policy
-----------------
By default :func:`validate_url_scheme` allows both ``http`` and ``https``
schemes.  For remote model downloads and repository access the recommended
policy is **https only** -- pass ``allowed_schemes=("https",)`` to enforce
this.  The following schemes are always blocked unless explicitly allowed:

* ``file`` -- prevents local file disclosure
* ``ftp`` -- no TLS, credential leakage risk
* ``data`` -- can be used to smuggle payloads
* ``gopher`` -- classic SSRF vector
"""

from pathlib import Path
from urllib.parse import urlparse


def validate_path(path: str | Path, allowed_roots: list[Path], strict: bool = True) -> Path:
    """Validate that a path is within allowed root directories.

    Args:
        path: The path to validate.
        allowed_roots: A list of allowed root directories.
        strict: If True, raises ValueError on violation.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If path is outside allowed roots and strict is True.
    """
    try:
        resolved_path = Path(path).resolve()
    except Exception as e:  # noqa: BLE001 — catch any resolve() failure
        if strict:
            raise ValueError(f"Invalid path format: {path}") from e
        return Path(path)

    is_allowed = False
    for root in allowed_roots:
        try:
            resolved_root = root.resolve()
            if str(resolved_path).startswith(str(resolved_root)):
                is_allowed = True
                break
        except (RuntimeError, TypeError, ValueError):
            continue

    if not is_allowed and strict:
        raise ValueError(
            f"Path traversal blocked: {path} is not within allowed roots: "
            f"{[str(r) for r in allowed_roots]}"
        )

    return resolved_path


def validate_url_scheme(
    url: str,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
) -> str:
    """Validate that a URL uses an allowed scheme (SSRF prevention).

    For remote model repositories and downloads, callers should pass
    ``allowed_schemes=("https",)`` to restrict to TLS-only connections.

    Args:
        url: The URL to validate.
        allowed_schemes: Tuple of allowed URL schemes.  Defaults to
            ``("http", "https")``.  Use ``("https",)`` for remote
            model access.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL scheme is not in *allowed_schemes*.
    """
    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Allowed schemes: {', '.join(allowed_schemes)}"
        )
    return url


def validate_url_https_only(url: str) -> str:
    """Convenience wrapper that restricts URLs to ``https`` only.

    This is the recommended validator for all remote model downloads and
    repository access.

    Args:
        url: The URL to validate.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL does not use the ``https`` scheme.
    """
    return validate_url_scheme(url, allowed_schemes=("https",))
