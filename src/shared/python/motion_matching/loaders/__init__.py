"""Source-format loaders; optional backends load only when requested."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .body_json import load_body_target_json
    from .c3d import load_club_target_c3d
    from .c3d_body import (
        DEFAULT_BODY_MARKER_EXCLUDES,
        default_anatomical_marker_set,
        load_body_target_c3d,
    )
    from .excel import load_club_target_excel
    from .matlab_dataset import load_club_target_mat
    from .synthetic import synthesize_target_from_coefficients

_LAZY_EXPORTS = {
    "load_body_target_json": ".body_json",
    "load_club_target_c3d": ".c3d",
    "DEFAULT_BODY_MARKER_EXCLUDES": ".c3d_body",
    "default_anatomical_marker_set": ".c3d_body",
    "load_body_target_c3d": ".c3d_body",
    "load_club_target_excel": ".excel",
    "load_club_target_mat": ".matlab_dataset",
    "synthesize_target_from_coefficients": ".synthetic",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        return getattr(import_module(_LAZY_EXPORTS[name], __package__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_BODY_MARKER_EXCLUDES",
    "default_anatomical_marker_set",
    "load_body_target_c3d",
    "load_body_target_json",
    "load_club_target_c3d",
    "load_club_target_excel",
    "load_club_target_mat",
    "synthesize_target_from_coefficients",
]
