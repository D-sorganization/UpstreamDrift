"""COV-8 Necromatcher owner project with marker-anchored anthropometry (#11276).

Provides privacy-guarded project initialization, marker-anchored anthropometry
fitting from joint centres, and immutable capture/swing import for subject-O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.motion_matching.anthropometry import (
    DE_LEVA_MALE,
    segment_parameters,
)
from src.shared.python.workspace.necromatcher import NecromatcherLibrary

VALID_PROVENANCE_TAGS = frozenset(("observed (marker-derived)", "population-prior"))

BILATERAL_PAIRS = (
    ("thigh_r", "thigh_l"),
    ("shank_r", "shank_l"),
    ("upper_arm_r", "upper_arm_l"),
    ("forearm_r", "forearm_l"),
    ("hand_r", "hand_l"),
    ("foot_r", "foot_l"),
)


@dataclass(frozen=True)
class SegmentLengthEstimate:
    """Estimated length and variation for a single human body segment.

    Attributes:
        segment: Canonical segment identifier (e.g. 'thigh_r', 'trunk').
        length_m: Estimated segment length in metres (SI, strictly positive).
        spread_m: Spread across observations in metres (SI, non-negative).
        sample_count: Number of observation samples used.
        provenance: Provenance tag ('observed (marker-derived)' or 'population-prior').
        uncertainty_m: Estimated measurement uncertainty in metres.
    """

    segment: str
    length_m: float
    spread_m: float
    sample_count: int
    provenance: str
    uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        require(
            isinstance(self.segment, str) and bool(self.segment.strip()),
            "segment must be a non-empty string",
        )
        require(
            isinstance(self.length_m, (int, float))
            and math.isfinite(self.length_m)
            and self.length_m > 0,
            f"Segment length for {self.segment} must be a positive finite number, got {self.length_m!r}",
        )
        require(
            isinstance(self.spread_m, (int, float))
            and math.isfinite(self.spread_m)
            and self.spread_m >= 0,
            f"Segment spread for {self.segment} must be a non-negative finite number",
        )
        require(
            isinstance(self.sample_count, int) and self.sample_count >= 0,
            f"sample_count for {self.segment} must be a non-negative integer",
        )
        require(
            isinstance(self.provenance, str)
            and self.provenance in VALID_PROVENANCE_TAGS,
            f"Missing or invalid provenance tag for segment {self.segment}: {self.provenance!r}",
        )
        require(
            isinstance(self.uncertainty_m, (int, float))
            and math.isfinite(self.uncertainty_m)
            and self.uncertainty_m >= 0,
            f"Segment uncertainty for {self.segment} must be non-negative",
        )


@dataclass(frozen=True)
class MarkerAnchoredAnthropometry:
    """Anthropometric profile anchored to marker-derived joint centres.

    Attributes:
        subject_id: Subject identifier (e.g. 'subject-O').
        height_m: Total body height in metres.
        mass_kg: Total body mass in kilograms.
        height_source: Provenance of height measurement.
        mass_source: Provenance of mass measurement.
        segments: Per-segment length estimates and provenances.
        max_bilateral_asymmetry: Maximum allowed relative left/right asymmetry.
    """

    subject_id: str
    height_m: float
    mass_kg: float
    height_source: str = "owner_reported"
    mass_source: str = "owner_reported"
    segments: Mapping[str, SegmentLengthEstimate] = field(default_factory=dict)
    max_bilateral_asymmetry: float = 0.10

    def __post_init__(self) -> None:
        require(
            isinstance(self.subject_id, str) and bool(self.subject_id.strip()),
            "subject_id must be a non-empty string",
        )
        require(
            isinstance(self.height_m, (int, float))
            and math.isfinite(self.height_m)
            and self.height_m > 0,
            "height_m must be a positive finite number",
        )
        require(
            isinstance(self.mass_kg, (int, float))
            and math.isfinite(self.mass_kg)
            and self.mass_kg > 0,
            "mass_kg must be a positive finite number",
        )
        require(
            isinstance(self.height_source, str) and bool(self.height_source.strip()),
            "height_source must be a non-empty string",
        )
        require(
            isinstance(self.mass_source, str) and bool(self.mass_source.strip()),
            "mass_source must be a non-empty string",
        )
        require(
            isinstance(self.segments, Mapping) and len(self.segments) > 0,
            "segments must be a non-empty mapping",
        )
        self._validate_segments_and_symmetry()

    def _validate_segments_and_symmetry(self) -> None:
        for name, est in self.segments.items():
            require(
                isinstance(est, SegmentLengthEstimate),
                f"Segment {name} must be a SegmentLengthEstimate instance",
            )
            require(
                est.provenance in VALID_PROVENANCE_TAGS,
                f"Missing or invalid provenance tag for segment {name}: {est.provenance!r}",
            )

        for right, left in BILATERAL_PAIRS:
            if right in self.segments and left in self.segments:
                len_r = self.segments[right].length_m
                len_l = self.segments[left].length_m
                denom = max(len_r, len_l)
                asym = abs(len_r - len_l) / denom if denom > 0 else 0.0
                require(
                    asym <= self.max_bilateral_asymmetry,
                    f"Bilateral length asymmetry for {right}/{left} ({asym:.1%}) "
                    f"exceeds declared bound of {self.max_bilateral_asymmetry:.1%}",
                )


@dataclass
class OwnerPlayerProject:
    """Private owner player project bound to Necromatcher library."""

    library: NecromatcherLibrary
    player_id: str
    player_name: str
    library_root: Path
    anthropometry: MarkerAnchoredAnthropometry | None = None
    swings: dict[str, Any] = field(default_factory=dict)
    captures: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            isinstance(self.library, NecromatcherLibrary),
            "library must be NecromatcherLibrary",
        )
        require(
            isinstance(self.player_id, str) and bool(self.player_id.strip()),
            "player_id must be non-empty",
        )
        require(
            isinstance(self.player_name, str) and bool(self.player_name.strip()),
            "player_name must be non-empty",
        )
        require(isinstance(self.library_root, Path), "library_root must be a Path")


def create_owner_project(
    library_root: Path | str,
    *,
    capture_data_dir: Path | str | None = None,
    private: bool = True,
    player_id: str = "subject-O",
    player_name: str = "Subject O",
) -> OwnerPlayerProject:
    """Create or load a private owner project rooted strictly inside CAPTURE_DATA_DIR."""
    resolved_root = Path(library_root).resolve()
    if private:
        cap_dir = capture_data_dir or os.environ.get("CAPTURE_DATA_DIR")
        require(
            cap_dir is not None and bool(str(cap_dir).strip()),
            "Private owner library requires CAPTURE_DATA_DIR environment variable or parameter",
        )
        if not cap_dir:
            raise ValueError(
                "Private owner library requires CAPTURE_DATA_DIR environment variable or parameter"
            )
        resolved_cap = Path(cap_dir).resolve()
        try:
            resolved_root.relative_to(resolved_cap)
        except ValueError as exc:
            raise ValueError(
                f"Privacy guard: library root {resolved_root} is outside CAPTURE_DATA_DIR {resolved_cap}"
            ) from exc

    if (resolved_root / "project.json").exists():
        library = NecromatcherLibrary(resolved_root)
    else:
        library = NecromatcherLibrary.create(resolved_root)

    existing_ids = {p.subject_id for p in library.players()}
    if player_id not in existing_ids:
        library.add_player(player_id, player_name)

    return OwnerPlayerProject(
        library=library,
        player_id=player_id,
        player_name=player_name,
        library_root=resolved_root,
    )


def compute_marker_anchored_anthropometry(
    marker_data: Mapping[str, Any],
    *,
    subject_id: str = "subject-O",
    height_m: float = 1.83,
    mass_kg: float = 84.0,
    height_source: str = "owner_reported",
    mass_source: str = "owner_reported",
    max_bilateral_asymmetry: float = 0.10,
    sample_method: str = "median",
) -> MarkerAnchoredAnthropometry:
    """Compute marker-anchored segment lengths with variation and population priors."""
    require(isinstance(marker_data, Mapping), "marker_data must be a mapping")
    estimates: dict[str, SegmentLengthEstimate] = {}

    for seg_name, raw_val in marker_data.items():
        if isinstance(raw_val, SegmentLengthEstimate):
            estimates[seg_name] = raw_val
            continue

        arr = np.asarray(raw_val, dtype=float)
        require(
            bool(np.all(np.isfinite(arr))),
            f"Segment length for {seg_name} must be finite",
        )
        require(
            bool(np.all(arr > 0)),
            f"Segment length for {seg_name} must be positive",
        )

        length = (
            float(np.median(arr)) if sample_method == "median" else float(np.mean(arr))
        )
        spread = float(np.std(arr)) if arr.size > 1 else 0.0
        estimates[seg_name] = SegmentLengthEstimate(
            segment=seg_name,
            length_m=length,
            spread_m=spread,
            sample_count=int(arr.size),
            provenance="observed (marker-derived)",
            uncertainty_m=spread / math.sqrt(arr.size) if arr.size > 1 else 0.0,
        )

    for canon_seg in DE_LEVA_MALE:
        has_direct = canon_seg in estimates
        has_bilateral = any(k.startswith(canon_seg) for k in estimates)
        if not (has_direct or has_bilateral):
            params = segment_parameters(height_m, mass_kg, canon_seg)
            estimates[canon_seg] = SegmentLengthEstimate(
                segment=canon_seg,
                length_m=params.length_m,
                spread_m=0.0,
                sample_count=0,
                provenance="population-prior",
                uncertainty_m=0.0,
            )

    return MarkerAnchoredAnthropometry(
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        height_source=height_source,
        mass_source=mass_source,
        segments=estimates,
        max_bilateral_asymmetry=max_bilateral_asymmetry,
    )


def import_video_swings_to_owner_project(
    project: OwnerPlayerProject,
    swings: Sequence[Mapping[str, Any]],
    *,
    skip_rejected: bool = True,
) -> dict[str, dict[str, Any]]:
    """Import graded video swings and captures preserving immutable versions."""
    require(
        isinstance(project, OwnerPlayerProject), "project must be OwnerPlayerProject"
    )
    imported: dict[str, dict[str, Any]] = {}

    for spec in swings:
        swing_id = str(spec["swing_id"])
        grade = str(spec.get("grade", "A"))
        if skip_rejected and grade == "R":
            continue

        swing_name = str(spec.get("name", spec.get("swing_name", swing_id)))
        existing_swings = {
            s.session_id for s in project.library.swings(project.player_id)
        }
        if swing_id not in existing_swings:
            project.library.add_swing(swing_id, project.player_id, swing_name)

        capture_dir = spec.get("capture_dir")
        if capture_dir is not None:
            capture_id = str(spec.get("capture_id", f"{swing_id}-capture"))
            existing_assets = {
                a.dataset_id: a for a in project.library.assets(swing_id)
            }
            if capture_id in existing_assets:
                asset = existing_assets[capture_id]
            else:
                asset = project.library.add_capture(
                    capture_id, swing_id, Path(capture_dir)
                )
            project.captures[capture_id] = asset

        project.swings[swing_id] = dict(spec)
        imported[swing_id] = dict(spec)

    return imported
