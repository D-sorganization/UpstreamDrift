"""Declared plate facts and stance placement for a trial import (OF-07, #13087).

A swing file cannot carry the plate's uncertainty, sensing depth or capacity,
nor where the stance frame sits in the model's world.  The desktop
declarations dialog and the plate-overlay API both collect those as plain
values; this module is the one place that turns them into the domain objects,
which validate them again.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.force_plates.contracts import PlateUncertainty
from src.shared.python.force_plates.model_frame import (
    TargetedPlacement,
    placement_for_target,
)
from src.shared.python.force_plates.open_fp_format.reader import PlateDeclarations
from src.shared.python.impact_parameters.target_frame import (
    UD_DEFAULT_TARGET_DIR,
    TargetFrame,
)

__all__ = ["DeclaredPlate", "DeclaredStance"]

_FRAME_HANDEDNESS = {"RH": "right", "LH": "left"}


@dataclass(frozen=True)
class DeclaredPlate:
    """Plain declared plate facts; see :class:`PlateUncertainty` for meanings.

    ``device`` ``None`` keeps the reader's default provenance label.
    """

    force_floor_n: float
    force_fraction: float
    cop_m: float
    sensing_depth_sigma_m: float
    uncertainty_source: str
    sensing_depth_m: float
    capacity_n: float
    device: str | None = None

    def to_declarations(self) -> PlateDeclarations:
        """The validated :class:`PlateDeclarations` (``ValueError`` if invalid)."""
        uncertainty = PlateUncertainty(
            force_floor_n=self.force_floor_n,
            force_fraction=self.force_fraction,
            cop_m=self.cop_m,
            sensing_depth_sigma_m=self.sensing_depth_sigma_m,
            source=self.uncertainty_source,
        )
        extra = {} if self.device is None else {"device": self.device}
        return PlateDeclarations(
            uncertainty=uncertainty,
            sensing_depth_m=self.sensing_depth_m,
            capacity_n=self.capacity_n,
            **extra,
        )


@dataclass(frozen=True)
class DeclaredStance:
    """Where the stance frame sits: handedness, world origin [m], target line.

    ``origin_m`` is the world point midway between the plate centres on the
    plate top; ``target_dir`` defaults to the ADR-0041 target line (``-Y``).
    """

    handedness: str
    origin_m: tuple[float, float, float]
    target_dir: tuple[float, float, float] = UD_DEFAULT_TARGET_DIR

    def to_placement(self, *, frame_label: str) -> TargetedPlacement:
        """The :class:`TargetedPlacement`, its frame named ``<label>:target_<dir>``.

        Raises ``ValueError`` for an unknown handedness, an empty
        ``frame_label``, a non-horizontal target line or a bad origin.
        """
        frame_handedness = _FRAME_HANDEDNESS.get(self.handedness)
        if frame_handedness is None:
            raise ValueError(f"handedness must be 'RH' or 'LH', got {self.handedness!r}")
        if not isinstance(frame_label, str) or not frame_label.strip():
            raise ValueError("frame_label must be a non-empty string")
        target_dir = tuple(float(v) for v in self.target_dir)
        frame = TargetFrame(
            target_dir=target_dir,  # type: ignore[arg-type]
            handedness=frame_handedness,
            frame_id=f"{frame_label}:target_{target_dir}",
        )
        return placement_for_target(
            self.handedness,  # type: ignore[arg-type]
            frame,
            origin_m=self.origin_m,
        )
