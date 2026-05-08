"""Source-format loaders for the motion-matching pipeline."""

from .c3d import load_club_target_c3d
from .c3d_body import (
    DEFAULT_BODY_MARKER_EXCLUDES,
    default_anatomical_marker_set,
    load_body_target_c3d,
)
from .excel import load_club_target_excel
from .matlab_dataset import load_club_target_mat
from .synthetic import synthesize_target_from_coefficients

__all__ = [
    "DEFAULT_BODY_MARKER_EXCLUDES",
    "default_anatomical_marker_set",
    "load_body_target_c3d",
    "load_club_target_c3d",
    "load_club_target_excel",
    "load_club_target_mat",
    "synthesize_target_from_coefficients",
]
