"""Bounded input-lineage checks for saved capture results."""

from __future__ import annotations

from hashlib import sha256

from .session import SessionMedia
from .wizard_storage import read_document


def model_revision_problem(media: SessionMedia) -> str | None:
    """Describe stale triangulated-model lineage without following arbitrary paths.

    The existing model writer hashes its reconstruction summary. Only the known
    capture-relative summary is read here; source paths from metadata are never
    opened. Image-space fits have a separate observation/camera input contract.
    """
    if media.model_fit is None:
        return None
    provenance = media.model_fit.get("provenance", {})
    if not isinstance(provenance, dict):
        return "Model input association is unverified; fit again."
    parameters = provenance.get("parameters", {})
    source = parameters.get("source", {}) if isinstance(parameters, dict) else {}
    if isinstance(source, dict) and source.get("kind") == "image_space":
        return None
    inputs = provenance.get("inputs", [])
    expected = "reconstruct/session_reconstruction.json"
    records = (
        [
            item
            for item in inputs
            if isinstance(item, dict) and item.get("path") == expected
        ]
        if isinstance(inputs, list)
        else []
    )
    if len(records) != 1:
        return "Model reconstruction association is unverified; fit again."
    try:
        current = sha256(read_document(media.root / expected)).hexdigest()
    except (ValueError, OSError):
        return "Reconstruction evidence is unavailable; reconstruct and fit again."
    if records[0].get("sha256") != current:
        return "Reconstruction changed since this model fit; fit again."
    return None
