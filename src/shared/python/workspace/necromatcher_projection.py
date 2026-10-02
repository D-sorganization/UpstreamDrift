"""Native image-space review through the shared verified research binding."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .necromatcher_native import load_native_fit_binding

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary


def project_fit_frame(
    library: NecromatcherLibrary, fit_id: str, frame_index: int
) -> dict[str, Any]:
    """Project exact stored frames without certifying camera, timing or motion."""
    return load_native_fit_binding(library, fit_id).project(frame_index)
