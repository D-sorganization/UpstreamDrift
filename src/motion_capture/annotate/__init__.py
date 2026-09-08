"""Guided manual landmark annotation (epic #9791).

:mod:`.store` holds the sparse per-view clicks and skips, :mod:`.guide`
walks the user frame by frame and joint by joint, and
:mod:`.to_observations` turns the result into a ``view-observations``
set the reconstruction and model fit consume unchanged. All Qt-free; the
Capture Rig tile's dialog is a thin layer over these.
"""

from .guide import KEYMAP, Guide, Prompt
from .store import (
    ANNOTATIONS_DIR,
    SCHEMA_VERSION,
    AnnotationSet,
    Point,
    annotation_path,
)

__all__ = [
    "ANNOTATIONS_DIR",
    "KEYMAP",
    "SCHEMA_VERSION",
    "AnnotationSet",
    "Guide",
    "Point",
    "Prompt",
    "annotation_path",
]
