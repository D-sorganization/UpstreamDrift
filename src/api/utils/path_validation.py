"""Path validation helpers for API inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fastapi import HTTPException

ALLOWED_MODEL_DIRS = [
    Path("shared/models").resolve(),
    Path("models").resolve(),
    Path("data").resolve(),
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

        try:
            resolved_candidate.relative_to(resolved_allowed_dir)
        except ValueError:
            continue

        if _candidate_traverses_symlink(candidate, allowed_dir):
            raise HTTPException(
                status_code=400,
                detail="Symbolic links are not permitted in model paths",
            )

        if resolved_candidate.exists():
            return resolved_candidate

    raise HTTPException(
        status_code=404,
        detail="Model file not found in allowed directories",
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

    if ".." in user_path.parts or ".." in model_path:
        raise HTTPException(
            status_code=400,
            detail="Invalid path: parent directory references not allowed",
        )

    for allowed_dir in ALLOWED_MODEL_DIRS:
        candidate = allowed_dir / user_path
        try:
            return str(resolve_contained_path(candidate, [allowed_dir]))
        except HTTPException:
            continue

    raise HTTPException(
        status_code=404,
        detail="Model file not found in allowed directories",
    )
