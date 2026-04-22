"""Path validation helpers for API inputs."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

ALLOWED_MODEL_DIRS = [
    Path("shared/models").resolve(),
    Path("models").resolve(),
    Path("data").resolve(),
]


def validate_path_within_root(candidate_path: Path, root_dir: Path) -> Path:
    """Validate that a resolved path stays within a root directory.

    Args:
        candidate_path: Path to validate.
        root_dir: Approved root directory for the candidate.

    Returns:
        The resolved candidate path when it remains within the approved root.

    Raises:
        HTTPException: If the path cannot be resolved or escapes the root.
    """
    if not isinstance(candidate_path, Path) or not isinstance(root_dir, Path):
        raise HTTPException(status_code=400, detail="Invalid path format")

    try:
        resolved_candidate = candidate_path.resolve(strict=False)
        resolved_root = root_dir.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid path format") from exc

    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Model path escapes approved model roots",
        ) from exc

    return resolved_candidate


def validate_model_path(model_path: str) -> str:
    """Validate model path to prevent path traversal attacks."""
    try:
        user_path = Path(model_path)
    except TypeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid path format",
        ) from exc

    # Check both POSIX and Windows-style absolute paths
    if user_path.is_absolute() or (len(model_path) >= 2 and model_path[1] == ":"):
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
        try:
            candidate = (allowed_dir / user_path).resolve()
        except (ValueError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid path format",
            ) from exc

        try:
            candidate.relative_to(allowed_dir)
        except ValueError:
            continue

        if candidate.exists():
            return str(candidate)

    raise HTTPException(
        status_code=404,
        detail="Model file not found in allowed directories",
    )
