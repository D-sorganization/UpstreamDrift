"""Path validation helpers for API inputs."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

ALLOWED_MODEL_DIRS = [
    Path("shared/models").resolve(),
    Path("models").resolve(),
    Path("data").resolve(),
]


def validate_model_path(model_path: str) -> str:
    """Validate model path to prevent path traversal attacks."""
    try:
        user_path = Path(model_path)
    except TypeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid path format",
        ) from exc

    if user_path.is_absolute():
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
