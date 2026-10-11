"""OpenCap output adapter (#11406).

Provides an adapter for reading OpenCap session outputs and exposing them
to the UpstreamDrift motion pipeline and engines.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.shared.python.motion_pipeline.sources.opencap_session import (
    OpenCapSession,
    OpenCapSessionMetadata,
    inspect_opencap_session,
    load_opencap_session,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenCapOutputAdapter"]


class OpenCapOutputAdapter:
    """Adapter for inspecting and loading completed OpenCap sessions."""

    def load(self, session_dir: Path | str, trial: str | None = None) -> OpenCapSession:
        """Load one trial from an OpenCap session directory.

        Args:
            session_dir: Path to the OpenCap session directory.
            trial: Optional trial name. If omitted, loads default trial.

        Returns:
            OpenCapSession with observations, scaled model, and kinematics.
        """
        path = Path(session_dir).expanduser().resolve()
        return load_opencap_session(path, trial=trial)

    def inspect(self, session_dir: Path | str) -> OpenCapSessionMetadata:
        """Inspect session metadata without full marker/motion decoding.

        Args:
            session_dir: Path to the OpenCap session directory.

        Returns:
            OpenCapSessionMetadata summarizing trials, model, and subject.
        """
        path = Path(session_dir).expanduser().resolve()
        return inspect_opencap_session(path)
