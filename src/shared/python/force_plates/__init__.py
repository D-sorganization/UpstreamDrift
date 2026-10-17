"""Vertical-only force-plate data contracts (epic #13061).

Public façade: the vertical plate trial and its declared uncertainty and
provenance (:mod:`.contracts`), the stance-to-world placement
(:mod:`.placement`), the view as a shared ground-reaction series
(:mod:`.conversion`), the clock alignment to model time (:mod:`.alignment`)
and readings at a model time with an uncertainty budget (:mod:`.readings`).
Device readers live in subpackages and depend on this package, never the
reverse.
"""

from __future__ import annotations

from src.shared.python.force_plates.alignment import AlignmentKind, PlateTimeAlignment
from src.shared.python.force_plates.contracts import (
    FOOT_LABELS,
    NEGATIVE_LOAD_SIGMAS,
    NET_COP_TOL_M,
    NET_LABEL,
    Handedness,
    PlateProvenance,
    PlateUncertainty,
    VerticalPlateTrial,
    loaded_net_cop,
)
from src.shared.python.force_plates.conversion import (
    ObservedGroundReaction,
    observability,
    to_ground_reaction_series,
)
from src.shared.python.force_plates.placement import ROTATION_ATOL, PlatePlacement
from src.shared.python.force_plates.readings import (
    FootSample,
    PlacedTrial,
    PlateReading,
    ReadingOptions,
    reading_at,
)

__all__ = [
    "FOOT_LABELS",
    "NEGATIVE_LOAD_SIGMAS",
    "NET_COP_TOL_M",
    "NET_LABEL",
    "ROTATION_ATOL",
    "AlignmentKind",
    "FootSample",
    "Handedness",
    "ObservedGroundReaction",
    "PlacedTrial",
    "PlatePlacement",
    "PlateProvenance",
    "PlateReading",
    "PlateTimeAlignment",
    "PlateUncertainty",
    "ReadingOptions",
    "VerticalPlateTrial",
    "loaded_net_cop",
    "observability",
    "reading_at",
    "to_ground_reaction_series",
]
