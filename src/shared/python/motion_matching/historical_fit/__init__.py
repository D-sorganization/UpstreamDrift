"""Historical image-to-native fitting with explicit camera and pose assumptions."""

from .contracts import (
    CameraProjection,
    ImageFitConfig,
    ImageFitInputs,
    ImageFitResult,
    ImageSplineStart,
)
from .shaft_observations import (
    ShaftAxisEvidence,
    ShaftAxisSegment,
    SourceBoundShaftFrame,
)
from .shaft_geometry import AuthoredShaftAxis, resolve_authored_shaft_axis
from .shaft_residuals import (
    AdditionalImageResiduals,
    ImageResidualTerm,
    ImageResidualAssessment,
    ImageSourceIdentity,
    ShaftAxisAssessment,
    ShaftAxisResidualTerm,
    project_authored_shaft_line,
)
from .shaft_row_timing import (
    ShaftRowTiming,
    RowTimedShaftAssessment,
    assess_row_timed_shaft,
)
from .solver import fit_image_trajectory, initialize_image_trajectory
from .capture import CaptureImageEvidence, read_capture_evidence
from .camera import initialize_camera_hypothesis

__all__ = [
    "AdditionalImageResiduals",
    "ImageResidualTerm",
    "ImageResidualAssessment",
    "ImageSourceIdentity",
    "ShaftAxisAssessment",
    "ShaftAxisResidualTerm",
    "project_authored_shaft_line",
    "ShaftRowTiming",
    "RowTimedShaftAssessment",
    "assess_row_timed_shaft",
    "AuthoredShaftAxis",
    "resolve_authored_shaft_axis",
    "ShaftAxisEvidence",
    "ShaftAxisSegment",
    "SourceBoundShaftFrame",
    "SplineCoordinateExpansion",
    "expand_image_spline_coordinates",
    "ContactPinPhase",
    "ContactPinSchedule",
    "ScheduledConstraintOptions",
    "CaptureImageEvidence",
    "read_capture_evidence",
    "CameraProjection",
    "ImageFitConfig",
    "ImageFitInputs",
    "ImageFitResult",
    "ImageSplineStart",
    "fit_image_trajectory",
    "initialize_image_trajectory",
    "initialize_camera_hypothesis",
]

from .contact_schedule import (
    ContactPinPhase,
    ContactPinSchedule,
    ScheduledConstraintOptions,
)

from .spline_expansion import SplineCoordinateExpansion, expand_image_spline_coordinates
