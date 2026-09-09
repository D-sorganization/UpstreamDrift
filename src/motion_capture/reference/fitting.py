"""Explicit C3D marker profiles for the shared articulated fitter (#9914)."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

from .importers import MotionDraft, finish_motion_import
from .model import Axis
from .registration import canonical_z_up_to_adr0041_world


class MarkerProfile(BaseModel):
    """Source axes assigned to canonical XYZ; each joint is a marker centroid.

    Every member must be present at a sample. No occlusion filling or guessed
    marker aliases occurs. Mapping surface markers to joint centers is an
    approximation explicitly retained in the profile's notes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["reference-marker-profile/1.0"] = (
        "reference-marker-profile/1.0"
    )
    name: str = Field(min_length=1)
    axes: tuple[Axis, Axis, Axis] = ("+X", "-Z", "+Y")
    units: Literal["m", "cm", "mm"] = "m"
    joints: dict[str, tuple[str, ...]]
    notes: str = "Surface markers approximate joint centers."

    @model_validator(mode="after")
    def valid_profile(self) -> Self:
        if not self.joints or set(self.joints) - set(JOINT_NAMES):
            raise ValueError("Profile joints must use reconstruction joint names")
        if any(
            not names
            or len(set(names)) != len(names)
            or any(not n.strip() for n in names)
            for names in self.joints.values()
        ):
            raise ValueError("Each joint needs unique nonempty marker names")
        basis = np.array(
            [
                np.eye(3)["XYZ".index(axis[-1])] * (1 if axis[0] == "+" else -1)
                for axis in self.axes
            ]
        )
        if not np.isclose(np.linalg.det(basis), 1):
            raise ValueError("Profile axes must define a right-handed rotation")
        return self


TOUR_AVERAGE_PROFILE = MarkerProfile(
    name="tour-average-surface-proxies/1.0",
    joints={
        "mid_hip": ("WaistLeft", "WaistRight"),
        "neck": ("BackTop",),
        "nose": ("HeadFront",),
        "left_shoulder": ("LShoulderBack",),
        "right_shoulder": ("RShoulderBack",),
        "left_elbow": ("LElbowOut",),
        "right_elbow": ("RElbowOut",),
        "left_wrist": ("LWristTop",),
        "right_wrist": ("RWristTop",),
        "left_hip": ("WaistLeft",),
        "right_hip": ("WaistRight",),
        "left_knee": ("LKneeOut",),
        "right_knee": ("RKneeOut",),
        "left_ankle": ("LAnkleOut",),
        "right_ankle": ("RAnkleOut",),
    },
    notes="Surface proxies, not anatomical centers. Waist markers approximate hips; "
    "BackTop approximates neck. Posterior shoulder markers avoid the occluded "
    "RShoulderTop. Club and sentinel markers are excluded.",
)


def map_markers(draft: MotionDraft, profile: MarkerProfile) -> np.ndarray:
    """Return (T, 15, 3) metres in the fitter's Y-up world, with NaN gaps."""
    missing = {n for names in profile.joints.values() for n in names} - set(draft.names)
    if missing:
        raise ValueError(f"Profile markers absent from source: {sorted(missing)}")
    motion = finish_motion_import(
        draft,
        title=profile.name,
        units=profile.units,
        axes=profile.axes,
        joint_names=draft.names,
    )
    canonical = np.array(
        [
            [point if point is not None else (np.nan,) * 3 for point in row]
            for row in motion.points_m
        ]
    )
    world = canonical_z_up_to_adr0041_world(canonical)
    mapped = np.full((len(draft.time_s), len(JOINT_NAMES), 3), np.nan)
    for name, markers in profile.joints.items():
        mapped[:, JOINT_NAMES.index(name)] = world[
            :, [draft.names.index(m) for m in markers]
        ].mean(axis=1)
    return mapped
