"""Compatibility profile for GSPro Open Connect v1 (GS-00 #10189).

Characterizes observed vs documented vs unresolved vendor protocol semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FieldObservationStatus(str, Enum):
    """Observation status of protocol fields."""

    OBSERVED = "observed"
    DOCUMENTED = "documented"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


class ResponseCategory(str, Enum):
    """Categorized response type from GSPro."""

    CONFIRMED_ACCEPTED = "confirmed_accepted"
    PLAYER_UPDATE = "player_update"
    ERROR_REJECTED = "error_rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GSProProfile:
    """Characterized compatibility profile for a GSPro release."""

    profile_id: str = "gspro_open_connect_v1"
    version: str = "1.0.0"
    app_release: str = "v1"
    port: int = 921
    host: str = "127.0.0.1"
    speed_unit: str = "mph"
    distance_unit: str = "Yards"
    hla_sign_positive: str = "right"
    spin_axis_sign_positive: str = "left_tilt"
    max_message_bytes: int = 65536
    loss_tolerance_rad_s: float = 10.0
    evidence_digest: str = (
        "sha256:8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"
    )
    field_matrix: dict[str, FieldObservationStatus] = field(
        default_factory=lambda: {
            "BallData.Speed": FieldObservationStatus.OBSERVED,
            "BallData.TotalSpin": FieldObservationStatus.OBSERVED,
            "BallData.VLA": FieldObservationStatus.OBSERVED,
            "BallData.HLA": FieldObservationStatus.OBSERVED,
            "BallData.SpinAxis": FieldObservationStatus.OBSERVED,
            "BallData.BackSpin": FieldObservationStatus.DOCUMENTED,
            "BallData.SideSpin": FieldObservationStatus.DOCUMENTED,
            "ClubData.Speed": FieldObservationStatus.OBSERVED,
            "ClubData.AngleOfAttack": FieldObservationStatus.OBSERVED,
            "ClubData.FaceToTarget": FieldObservationStatus.DOCUMENTED,
            "ClubData.Path": FieldObservationStatus.DOCUMENTED,
            "ShotDataOptions.ContainsBallData": FieldObservationStatus.OBSERVED,
            "ShotDataOptions.ContainsClubData": FieldObservationStatus.OBSERVED,
            "ShotDataOptions.IsHeartBeat": FieldObservationStatus.OBSERVED,
            "Course.AimAngle": FieldObservationStatus.UNSUPPORTED,
            "NativeAvatar": FieldObservationStatus.UNSUPPORTED,
        }
    )
    code_map: dict[int, ResponseCategory] = field(
        default_factory=lambda: {
            200: ResponseCategory.CONFIRMED_ACCEPTED,
            201: ResponseCategory.PLAYER_UPDATE,
            501: ResponseCategory.ERROR_REJECTED,
            502: ResponseCategory.ERROR_REJECTED,
        }
    )

    def get_field_status(self, field_path: str) -> FieldObservationStatus:
        """Return observation status of field in this profile."""
        return self.field_matrix.get(field_path, FieldObservationStatus.UNRESOLVED)

    def categorize_code(self, code: int) -> ResponseCategory:
        """Map a numeric response code to its response category."""
        return self.code_map.get(code, ResponseCategory.UNKNOWN)


DEFAULT_GSPRO_PROFILE = GSProProfile()
