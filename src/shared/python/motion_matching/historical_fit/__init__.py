"""Historical image-to-native fitting with explicit camera and pose assumptions."""

from .contracts import (
    CameraProjection,
    ImageFitConfig,
    ImageFitInputs,
    ImageFitResult,
    ImageSplineStart,
)
from .solver import fit_image_trajectory, initialize_image_trajectory
from .capture import CaptureImageEvidence, read_capture_evidence
from .camera import initialize_camera_hypothesis

__all__ = [
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
