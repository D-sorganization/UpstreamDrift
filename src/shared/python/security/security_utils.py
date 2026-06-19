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

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

#: Default bound (seconds) for all outbound network downloads.  ``urlretrieve``
#: has no timeout parameter at all, so unbounded calls can hang a worker thread
#: or the GUI indefinitely on a slow/half-open connection (issue #7184).
DOWNLOAD_TIMEOUT_SECONDS = 30


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
    except Exception as e:  # noqa: BLE001 — catch any resolve() failure
        if strict:
            raise ValueError(f"Invalid path format: {path}") from e
        return Path(path)

    is_allowed = False
    for root in allowed_roots:
        try:
            resolved_root = root.resolve()
            # Separator-aware containment: a plain startswith() admits
            # sibling directories sharing a string prefix (e.g.
            # /data/models-evil under allowed root /data/models). Use path
            # ancestry instead (issue #7689).
            if resolved_path == resolved_root or resolved_path.is_relative_to(
                resolved_root
            ):
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


def download_to_file(
    url: str,
    dest: str | Path,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Stream *url* to *dest* with a bounded socket timeout.

    ``urllib.request.urlretrieve`` accepts no ``timeout`` argument, so it can
    block forever on a hung server.  This helper streams the response via
    :func:`urllib.request.urlopen`, which honours *timeout*, into *dest*.

    The URL scheme is validated before opening the request so local files,
    data URLs, and other custom schemes cannot reach ``urlopen``.

    Args:
        url: The URL to download.  Must already be scheme-validated.
        dest: Destination file path.
        timeout: Per-operation socket timeout in seconds (must be > 0).

    Returns:
        The destination path as a :class:`~pathlib.Path`.

    Raises:
        ValueError: If *timeout* is not positive.
        TimeoutError: If the connection or read exceeds *timeout*.
        OSError: For other network/IO failures.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    validated_url = validate_url_scheme(url)
    dest_path = Path(dest)
    req = urllib.request.Request(validated_url)
    # The request URL has already passed validate_url_scheme above.
    with (
        urllib.request.urlopen(req, timeout=timeout) as response,  # noqa: S310  # nosec B310
        open(dest_path, "wb") as out,
    ):
        shutil.copyfileobj(response, out)
    return dest_path


def safe_extract_zip(zip_file: zipfile.ZipFile, dest: str | Path) -> None:
    """Extract *zip_file* into *dest*, rejecting path-traversal members.

    Guards against Zip Slip (issue #7183): a member named ``../evil`` or an
    absolute path would otherwise let :meth:`zipfile.ZipFile.extractall` write
    outside *dest*.  Every member's resolved target must stay within *dest*.

    Args:
        zip_file: An open :class:`zipfile.ZipFile`.
        dest: Destination directory.  Created if missing.

    Raises:
        ValueError: If any member resolves outside *dest* (absolute path,
            ``..`` traversal, or a symlink-style escape).
    """
    dest_path = Path(dest).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    dest_str = str(dest_path)
    for member in zip_file.namelist():
        # Reject absolute paths and explicit parent traversal up front so the
        # error message is precise even on exotic platforms.
        member_path = Path(member)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in archive: {member!r}")
        target = (dest_path / member).resolve()
        if target != dest_path and not str(target).startswith(dest_str + os.sep):
            raise ValueError(f"Unsafe path in archive: {member!r}")
    zip_file.extractall(dest_path)  # noqa: S202 - members validated above
