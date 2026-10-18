"""Immutable identity of the camera source used by a reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.motion_capture.provenance import sha256_of
from src.shared.python.core.contracts import PreconditionError, require

from .cameras import PinholeCamera


@dataclass(frozen=True)
class CameraSourceEvidence:
    """Bind loaded camera inputs to their source bytes, including optical zoom.

    Capture before parsing camera and lens records; verify after parsing and
    before publishing the reconstruction summary. This does not certify the
    calibration's physical accuracy.
    """

    path: Path
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> CameraSourceEvidence:
        """Snapshot the selected source using the shared provenance hash."""
        resolved = path.resolve()
        return cls(resolved, sha256_of(resolved))

    def verify(self) -> None:
        """Reject missing or changed inputs instead of publishing false lineage."""
        try:
            current = sha256_of(self.path)
        except (OSError, PreconditionError) as exc:
            raise ValueError(
                "Camera calibration source is unavailable; select it again"
            ) from exc
        if current != self.sha256:
            raise ValueError("Camera calibration source changed; reconstruct again")


def validate_camera_baselines(
    cameras: Sequence[PinholeCamera], *, min_baseline_m: float = 0.05
) -> None:
    """Validate that every camera pair in the starting rig has an observable baseline.

    Preconditions: at least two cameras; min_baseline_m > 0.
    Raises ValueError with an actionable message if any camera pair is
    coincident or closer than min_baseline_m.
    """
    require(len(cameras) >= 2, "camera rig requires at least two cameras")
    require(min_baseline_m > 0, "min_baseline_m must be positive")
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            c1, c2 = cameras[i], cameras[j]
            baseline = float(np.linalg.norm(c1.position_m - c2.position_m))
            if baseline < min_baseline_m:
                raise ValueError(
                    f"Cameras '{c1.camera_id}' and '{c2.camera_id}' have a degenerate "
                    f"baseline ({baseline:.4f} m < {min_baseline_m:.4f} m); check camera poses."
                )
