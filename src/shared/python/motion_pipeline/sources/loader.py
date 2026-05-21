"""Mocap source loading helper."""

from __future__ import annotations

from pathlib import Path
from src.shared.python.motion_pipeline.sources.registry import load_any, registered_adapters
from src.shared.python.motion_pipeline.sources.base import LoadedPayload


def load_source(path: Path, format_hint: str | None = None) -> LoadedPayload:
    """Load a motion capture source file, optionally guided by a format hint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {path}")

    # If format_hint is specified and not auto/passthrough, find matching adapter
    if format_hint and format_hint.lower() not in ("auto", "passthrough"):
        for cls in registered_adapters():
            if cls.format_name.lower() == format_hint.lower():
                adapter = cls()
                return adapter.load_checked(path)

    return load_any(path)
