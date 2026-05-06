"""Motion matching: club-target ingestion and trajectory comparison.

Public API:
    ClubTarget       -- canonical frozen dataclass for a measured swing.
    SourceProvenance -- file/format/sha256 metadata.
    AlignOptions     -- resampling and impact-alignment options.
    load_club_target_excel -- Wiffle/ProV1 xlsx loader.
    load_club_target_c3d   -- Optical C3D loader.
    synthesize_target_from_coefficients -- stub; awaiting #014/#018.

See ``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
shared/CLUB_IK_SPEC.md`` for the canonical schema.
"""

from .club_target import AlignOptions, ClubTarget, SourceProvenance
from .loaders.c3d import load_club_target_c3d
from .loaders.excel import load_club_target_excel
from .loaders.synthetic import synthesize_target_from_coefficients

__all__ = [
    "AlignOptions",
    "ClubTarget",
    "SourceProvenance",
    "load_club_target_c3d",
    "load_club_target_excel",
    "synthesize_target_from_coefficients",
]
