"""Exact zero-MTP CustomJoint profile for the assembled BUET–Hamner source.

The earlier PinJoint reducer remains strict and independent. This profile
reuses the native CustomJoint admission and mechanical lift from subtalar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .native_subtalar_reduction import (
    ZeroCustomJointProfile,
    ZeroSubtalarReductionReceipt,
    _sha,
    derive_zero_custom_joint_model,
)

_PROFILE = ZeroCustomJointProfile(
    "MTP",
    ("mtp_r", "mtp_l"),
    ("mtp_angle_r", "mtp_angle_l"),
)


@dataclass(frozen=True)
class ZeroCustomMtpReductionRequest:
    """Source-bound ordered bilateral zero lock target and fresh destination."""

    source_model_path: Path
    source_sha256: str
    derived_model_path: Path
    declared_target_rad: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.declared_target_rad, tuple) or (
            self.declared_target_rad != (("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0))
        ):
            raise ValueError("exact bilateral zero-MTP target declaration required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source SHA-256 must be lowercase hex")
        if self.source_model_path.resolve() == self.derived_model_path.resolve():
            raise ValueError("source and derived paths must differ")


@dataclass(frozen=True)
class ZeroCustomMtpReductionReceipt(ZeroSubtalarReductionReceipt):
    """Sampled mechanical result and exact profile-wrapper source identity."""

    profile_source_sha256: str


def derive_zero_custom_mtp_model(
    request: ZeroCustomMtpReductionRequest,
) -> ZeroCustomMtpReductionReceipt:
    """Reduce only the exact two native CustomJoints after zero-law admission."""
    profile_source = Path(__file__)
    profile_hash = _sha(profile_source)
    base = derive_zero_custom_joint_model(request, _PROFILE)
    if _sha(profile_source) != profile_hash:
        request.derived_model_path.unlink(missing_ok=True)
        raise ValueError("MTP reduction profile source changed during verification")
    return ZeroCustomMtpReductionReceipt(
        **asdict(base),
        profile_source_sha256=profile_hash,
    )
