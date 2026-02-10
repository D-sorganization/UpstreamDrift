"""Security utilities for path validation and subprocess hardening."""

from pathlib import Path
from urllib.parse import urlparse


def validate_path(
    path: str | Path, allowed_roots: list[Path], strict: bool = True
) -> Path:
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
    except Exception as e:
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
        except Exception:
            continue

    if not is_allowed:
        if strict:
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

    Args:
        url: The URL to validate.
        allowed_schemes: Tuple of allowed URL schemes.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL scheme is not in allowed_schemes.
    """
    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Allowed schemes: {', '.join(allowed_schemes)}"
        )
    return url
