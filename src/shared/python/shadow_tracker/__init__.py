"""Shadow Tracker silhouette-to-forward-dynamics package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .source_records import (
        FrameIdentity,
        RightsStatus,
        SourceAsset,
        validate_frame_sequence,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "SourceAsset": ".source_records",
    "FrameIdentity": ".source_records",
    "RightsStatus": ".source_records",
    "validate_frame_sequence": ".source_records",
}

__all__ = [
    "FrameIdentity",
    "RightsStatus",
    "SourceAsset",
    "validate_frame_sequence",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
