"""Serialization and deserialization for MatchedSwingCandidate packages (MS-15, #10334).

Saves and loads self-contained .npz packages containing manifest metadata and
immutable numpy arrays without pickle dependencies (allow_pickle=False).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    MatchedSwingCandidate,
)

logger = logging.getLogger(__name__)

MANIFEST_KEY = "manifest_json"


@precondition(
    lambda candidate, path: isinstance(candidate, MatchedSwingCandidate),
    "candidate must be MatchedSwingCandidate",
)
def save_candidate(candidate: MatchedSwingCandidate, path: Path | str) -> None:
    """Save a MatchedSwingCandidate package to an .npz archive."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure checksums are up to date in metadata
    checksums = candidate.compute_checksums()
    meta_dict = candidate.metadata.to_dict()
    meta_dict["checksums"] = checksums

    manifest_str = json.dumps(meta_dict, indent=2)

    arrays_to_save: dict[str, np.ndarray] = {
        MANIFEST_KEY: np.array(manifest_str),
        "time_s": np.ascontiguousarray(candidate.time_s),
        "q": np.ascontiguousarray(candidate.q),
    }
    if candidate.v is not None:
        arrays_to_save["v"] = np.ascontiguousarray(candidate.v)
    if candidate.tau is not None:
        arrays_to_save["tau"] = np.ascontiguousarray(candidate.tau)
    if candidate.actuator_states is not None:
        arrays_to_save["actuator_states"] = np.ascontiguousarray(
            candidate.actuator_states
        )
    if candidate.model_markers_m is not None:
        arrays_to_save["model_markers_m"] = np.ascontiguousarray(
            candidate.model_markers_m
        )
    if candidate.target_markers_m is not None:
        arrays_to_save["target_markers_m"] = np.ascontiguousarray(
            candidate.target_markers_m
        )
    if candidate.marker_validity is not None:
        arrays_to_save["marker_validity"] = np.ascontiguousarray(
            candidate.marker_validity
        )
    if candidate.external_forces is not None:
        arrays_to_save["external_forces"] = np.ascontiguousarray(
            candidate.external_forces
        )
    if candidate.root_forces is not None:
        arrays_to_save["root_forces"] = np.ascontiguousarray(candidate.root_forces)
    if candidate.contact_modes is not None:
        arrays_to_save["contact_modes"] = np.ascontiguousarray(candidate.contact_modes)
    if candidate.grip_wrench is not None:
        arrays_to_save["grip_wrench"] = np.ascontiguousarray(candidate.grip_wrench)

    np.savez(target_path, **arrays_to_save)  # type: ignore[arg-type]
    meta = candidate.metadata
    logger.debug(
        "Saved candidate package (%s) -> %s",
        meta.profile.value,
        target_path,
    )


@precondition(
    lambda path, validate_checksums=True: Path(path).is_file(),
    "candidate archive file must exist",
)
@postcondition(
    lambda r: isinstance(r, MatchedSwingCandidate), "must return MatchedSwingCandidate"
)
def load_candidate(
    path: Path | str, validate_checksums: bool = True
) -> MatchedSwingCandidate:
    """Load and validate a MatchedSwingCandidate package from an .npz archive."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        if MANIFEST_KEY not in data:
            raise ValueError(
                f"Archive {p} is not a valid MatchedSwingCandidate package (missing {MANIFEST_KEY})"
            )

        manifest_raw = data[MANIFEST_KEY]
        manifest_str = str(manifest_raw)
        meta_dict = json.loads(manifest_str)

        schema_version = meta_dict.get("schema_version")
        if schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {schema_version!r} (expected {CANDIDATE_SCHEMA_VERSION!r})"
            )

        metadata = CandidateMetadata.from_dict(meta_dict)

        time_s = data["time_s"]
        q = data["q"]
        v = data.get("v", None)
        tau = data.get("tau", None)
        actuator_states = data.get("actuator_states", None)
        model_markers_m = data.get("model_markers_m", None)
        target_markers_m = data.get("target_markers_m", None)
        marker_validity = data.get("marker_validity", None)
        external_forces = data.get("external_forces", None)
        root_forces = data.get("root_forces", None)
        contact_modes = data.get("contact_modes", None)
        grip_wrench = data.get("grip_wrench", None)

        markers = CandidateMarkers(
            model_markers_m=model_markers_m,
            target_markers_m=target_markers_m,
            marker_validity=marker_validity,
        )
        auxiliary = CandidateAuxiliary(
            actuator_states=actuator_states,
            external_forces=external_forces,
            root_forces=root_forces,
            contact_modes=contact_modes,
            grip_wrench=grip_wrench,
        )

        candidate = MatchedSwingCandidate(
            metadata=metadata,
            time_s=time_s,
            q=q,
            v=v,
            tau=tau,
            markers=markers,
            auxiliary=auxiliary,
            compute_checksums=False,
        )

        if validate_checksums:
            candidate.verify_checksums()

        return candidate
