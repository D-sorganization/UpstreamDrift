"""Dynamic Center of Mass Computation from Marker Data.

Provides the `BiomechanicalModel` class for computing full-body dynamic Center
of Mass (COM) trajectories from raw motion capture marker data, integrating
with the anthropometric datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    estimate_segment_masses,
    get_anthropometry_key,
)


@dataclass
class SegmentDefinition:
    """Defines how to compute a segment's position from markers."""

    name: str
    proximal_markers: list[str]
    distal_markers: list[str]


class BiomechanicalModel:
    """Computes full-body dynamic COM from marker trajectories and anthropometry."""

    def __init__(self, total_mass_kg: float, gender_factor: float = 0.5) -> None:
        """Initialize the model with subject parameters.

        Args:
            total_mass_kg: Total body mass in kg.
            gender_factor: 0.0 = female, 1.0 = male, 0.5 = neutral.
        """
        require(total_mass_kg > 0, "Total mass must be positive")
        require(0.0 <= gender_factor <= 1.0, "Gender factor must be between 0 and 1")

        self.total_mass_kg = total_mass_kg
        self.gender_factor = gender_factor
        self.segment_masses = estimate_segment_masses(total_mass_kg, gender_factor)
        self.segments: list[SegmentDefinition] = []

    def add_segment(
        self, name: str, proximal_markers: list[str], distal_markers: list[str]
    ) -> None:
        """Add a segment definition to the model.

        Args:
            name: Anthropometric segment name (e.g., 'thigh', 'pelvis').
            proximal_markers: List of marker names defining the proximal joint.
            distal_markers: List of marker names defining the distal joint.
        """
        require(len(proximal_markers) > 0, "Must have at least one proximal marker")
        require(len(distal_markers) > 0, "Must have at least one distal marker")

        key = get_anthropometry_key(name)
        require(key in self.segment_masses, f"Unknown segment name: {name} (mapped to {key})")

        self.segments.append(SegmentDefinition(name, proximal_markers, distal_markers))

    def compute_dynamic_com(
        self, marker_trajectories: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Compute the dynamic full-body COM trajectory.

        Args:
            marker_trajectories: Dict mapping marker name to (N, 3) trajectory array.

        Returns:
            (N, 3) array of the full-body COM trajectory.
        """
        if not self.segments:
            raise ValueError("No segments defined in the model")

        required_markers = set()
        for seg in self.segments:
            required_markers.update(seg.proximal_markers)
            required_markers.update(seg.distal_markers)

        for marker in required_markers:
            require(marker in marker_trajectories, f"Missing required marker: {marker}")

        first_marker = next(iter(required_markers))
        n_frames = len(marker_trajectories[first_marker])

        for marker in required_markers:
            require(
                len(marker_trajectories[marker]) == n_frames,
                f"Marker {marker} has inconsistent frame count",
            )

        full_body_com = np.zeros((n_frames, 3))
        total_modeled_mass = 0.0

        for seg in self.segments:
            proximal_pos = np.zeros((n_frames, 3))
            for m in seg.proximal_markers:
                proximal_pos += marker_trajectories[m]
            proximal_pos /= len(seg.proximal_markers)

            distal_pos = np.zeros((n_frames, 3))
            for m in seg.distal_markers:
                distal_pos += marker_trajectories[m]
            distal_pos /= len(seg.distal_markers)

            key = get_anthropometry_key(seg.name)
            data = DE_LEVA_DATA.get_segment_data(key, self.gender_factor)
            com_ratio = data.com_proximal_ratio

            # Segment COM is interpolated along the longitudinal axis
            segment_com = proximal_pos + com_ratio * (distal_pos - proximal_pos)

            mass = self.segment_masses[seg.name]
            full_body_com += segment_com * mass
            total_modeled_mass += mass

        require(total_modeled_mass > 0, "Total modeled mass must be positive")

        return full_body_com / total_modeled_mass
