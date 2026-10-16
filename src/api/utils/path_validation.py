"""Path validation helpers for API inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import HTTPException

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Image extensions the launcher logo routes may serve (issue #12964).
LOGO_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".ico")

ALLOWED_MODEL_DIRS = [
    (_PROJECT_ROOT / "shared/models").resolve(),
    (_PROJECT_ROOT / "models").resolve(),
    (_PROJECT_ROOT / "data").resolve(),
]


def _candidate_traverses_symlink(candidate: Path, allowed_dir: Path) -> bool:
    """Return True if any component of ``candidate`` under ``allowed_dir`` is a symlink.

    A symlink anywhere in the path under the allowed root can redirect outside
    the intended directory tree even when ``Path.resolve()`` happens to keep
    the resolved path inside that root on some platforms. Rejecting any
    symlink traversal is a defense-in-depth measure recommended by the
    Category K Data Handling review (issue #5918).
    """
    try:
        rel = candidate.relative_to(allowed_dir)
    except ValueError:
        # candidate is not under allowed_dir; let the contains-check decide.
        return False

    current = allowed_dir
    for part in rel.parts:
        current = current / part
        try:
            if current.is_symlink():
                return True
        except OSError:
            # Treat unreadable components as suspicious.
            return True
    return False


def resolve_contained_path(candidate: Path, allowed_dirs: Iterable[Path]) -> Path:
    """Resolve a candidate path and ensure it stays under an allowed root.

    Hardening (issue #5918): symlink traversal is rejected explicitly so a
    symlink inside an allowed root pointing outside it cannot bypass the
    containment check.
    """
    try:
        resolved_candidate = candidate.resolve()
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid path format",
        ) from exc

    for allowed_dir in allowed_dirs:
        try:
            resolved_allowed_dir = allowed_dir.resolve()
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid path format",
            ) from exc

        if _candidate_traverses_symlink(candidate, allowed_dir):
            raise HTTPException(
                status_code=400,
                detail="Symlink traversal is not permitted in model paths",
            )

        try:
            resolved_candidate.relative_to(resolved_allowed_dir)
        except ValueError:
            continue

        if resolved_candidate.exists():
            return resolved_candidate

    raise HTTPException(
        status_code=404,
        detail="Model file not found in allowed directories",
    )


def resolve_output_path(candidate: Path, allowed_dirs: Iterable[Path]) -> Path:
    """Resolve an *output* path and ensure it stays under an allowed root.

    Unlike :func:`resolve_contained_path`, the target file is not required to
    exist yet (it is about to be written). Containment is checked against the
    resolved candidate, and symlink traversal of any existing parent component
    is rejected as defense-in-depth (issue #6926).

    Returns:
        The resolved candidate path (which may not exist yet).

    Raises:
        HTTPException: 400 if the path is malformed, traverses a symlink, or
            escapes every allowed root.

    Direct test coverage for this security-critical guard lives in
    ``tests/unit/utils/test_path_validation.py`` (issue #7710).
    """
    try:
        resolved_candidate = candidate.resolve()
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid path format",
        ) from exc

    for allowed_dir in allowed_dirs:
        try:
            resolved_allowed_dir = allowed_dir.resolve()
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid path format",
            ) from exc

        try:
            resolved_candidate.relative_to(resolved_allowed_dir)
        except ValueError:
            continue

        if _candidate_traverses_symlink(candidate, allowed_dir):
            raise HTTPException(
                status_code=400,
                detail="Symlink traversal is not permitted in output paths",
            )

        return resolved_candidate

    raise HTTPException(
        status_code=400,
        detail="Invalid path: escapes allowed output directories",
    )


def validate_model_path(model_path: str) -> str:
    """Validate model path to prevent path traversal attacks."""
    try:
        user_path = Path(model_path)
    except TypeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid path format",
        ) from exc

    # Check both POSIX and Windows-style absolute paths. Path.is_absolute()
    # on Windows returns False for POSIX-style leading-slash paths such as
    # "/etc/passwd", so we also reject any path that begins with "/" or "\\"
    # to defend against path-traversal regardless of host OS.
    if (
        user_path.is_absolute()
        or (len(model_path) >= 2 and model_path[1] == ":")
        or model_path.startswith(("/", "\\"))
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid path: absolute paths are not allowed",
        )

    if ".." in user_path.parts:
        raise HTTPException(
            status_code=400,
            detail="Invalid path: parent directory references not allowed",
        )

    for allowed_dir in ALLOWED_MODEL_DIRS:
        candidate = allowed_dir / user_path
        try:
            return str(resolve_contained_path(candidate, [allowed_dir]))
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            continue

    raise HTTPException(
        status_code=404,
        detail="Model file not found in allowed directories",
    )


def resolve_file_within(
    base_dir: Path,
    filename: str,
    allowed_suffixes: Iterable[str] | None = None,
) -> Path | None:
    """Resolve a caller-supplied ``filename`` to a regular file inside ``base_dir``.

    Single shared guard for routes that serve files by a user-supplied path
    (issue #12964). ``..`` alone is not enough: an absolute path such as
    ``C:/Windows/win.ini`` or ``/etc/passwd`` replaces ``base_dir`` entirely
    when joined, so the final resolved path is checked for containment.

    Args:
        base_dir: Trusted directory the file must live under.
        filename: Caller-supplied path fragment (URL-decoded by the framework).
        allowed_suffixes: Optional case-insensitive extension allow-list
            (e.g. ``(".svg", ".png")``). ``None`` means any extension.

    Returns:
        The resolved path, or ``None`` if the request is unsafe, escapes
        ``base_dir`` (``..``, absolute path, backslash traversal, symlink),
        is not a regular file, or has a disallowed extension.
    """
    if not isinstance(filename, str) or not filename.strip() or "\x00" in filename:
        return None
    # Backslashes are separators on Windows; normalise so the same input is
    # rejected identically on every platform.
    normalised = filename.replace("\\", "/")
    # Reject absolute requests outright, including a Windows drive/root form
    # that POSIX hosts would not flag as absolute.
    parsed = PurePosixPath(normalised)
    if parsed.is_absolute() or PureWindowsPath(normalised).drive:
        return None
    try:
        resolved_base = base_dir.resolve()
        resolved = (resolved_base / normalised).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_relative_to(resolved_base):
        return None
    if not resolved.is_file():
        return None
    if allowed_suffixes is not None:
        allowed = {s.lower() for s in allowed_suffixes}
        if resolved.suffix.lower() not in allowed:
            return None
    return resolved
