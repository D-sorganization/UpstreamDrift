"""Source-format loaders for the motion-matching pipeline."""

from .c3d import load_club_target_c3d
from .excel import load_club_target_excel
from .synthetic import synthesize_target_from_coefficients

__all__ = [
    "load_club_target_c3d",
    "load_club_target_excel",
    "synthesize_target_from_coefficients",
]
