"""Cheap, bounded inspection of detector identity and edit association (#9913)."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

from src.motion_capture.rig.edits import load_edits

MAX_INDEX_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ObservationEvidence:
    detector: str
    description: str
    edits_match: bool | None


def inspect_observations(root: Path, output: Path) -> ObservationEvidence:
    """Describe metadata honestly, without reading videos or certifying quality."""
    if not output.is_file():
        return ObservationEvidence("Unknown Detector", "Output Missing", False)
    try:
        with (output.parent / "observations.json").open("rb") as stream:
            content = stream.read(MAX_INDEX_BYTES + 1)
        if len(content) > MAX_INDEX_BYTES:
            raise ValueError("Detector index is too large")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Detector index must be an object")
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("Detector provenance must be an object")
        detector = str(provenance.get("estimator") or "Unknown Detector")
        recorded = provenance.get("edits")
        if recorded is None:
            return ObservationEvidence(
                detector, "Available — Edit Association Unverified", None
            )
        if recorded != load_edits(root).model_dump(mode="json"):
            return ObservationEvidence(
                detector, "Stale — Swing Edits Changed; Detect Again", False
            )
        return ObservationEvidence(
            detector, "Available — Swing Edits Match; Review Quality", True
        )
    except (ValueError, OSError) as exc:
        return ObservationEvidence(
            "Unknown Detector", f"Available — Metadata Needs Review: {exc}", None
        )


def observation_evidence(root: Path, output: Path) -> tuple[str, str]:
    """Compatibility facade for the capture status table."""
    evidence = inspect_observations(root, output)
    return evidence.detector, evidence.description
