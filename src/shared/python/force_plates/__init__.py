"""Vertical-only force-plate data contracts (epic #13061).

Public façade: the vertical plate trial and its declared uncertainty and
provenance (:mod:`.contracts`), the stance-to-world placement
(:mod:`.placement`) and its derivation from a model's target frame
(:mod:`.model_frame`), the view as a shared ground-reaction series
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
from src.shared.python.force_plates.events import (
    FORCE_EVENT_PREFIX,
    force_event_markers,
    impact_anchor,
    merge_event_markers,
)
from src.shared.python.force_plates.model_frame import (
    TargetedPlacement,
    placement_for_target,
)
from src.shared.python.force_plates.overlay import (
    MeasuredOverlay,
    attach_measured_overlay,
    measured_overlay,
    overlay_on_series,
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
    "FORCE_EVENT_PREFIX",
    "FOOT_LABELS",
    "NEGATIVE_LOAD_SIGMAS",
    "NET_COP_TOL_M",
    "NET_LABEL",
    "ROTATION_ATOL",
    "AlignmentKind",
    "FootSample",
    "Handedness",
    "MeasuredOverlay",
    "ObservedGroundReaction",
    "PlacedTrial",
    "PlatePlacement",
    "PlateProvenance",
    "PlateReading",
    "PlateTimeAlignment",
    "PlateUncertainty",
    "ReadingOptions",
    "TargetedPlacement",
    "VerticalPlateTrial",
    "attach_measured_overlay",
    "force_event_markers",
    "impact_anchor",
    "loaded_net_cop",
    "measured_overlay",
    "merge_event_markers",
    "observability",
    "overlay_on_series",
    "reading_at",
    "placement_for_target",
    "to_ground_reaction_series",
]
