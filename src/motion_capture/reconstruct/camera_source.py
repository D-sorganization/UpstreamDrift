"""Immutable identity of the camera source used by a reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.motion_capture.provenance import sha256_of
from src.shared.python.core.contracts import PreconditionError


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
