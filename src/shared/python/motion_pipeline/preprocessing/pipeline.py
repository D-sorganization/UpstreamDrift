"""
Composable preprocessing pipeline for motion capture data.

Part of issue #4564. Pipeline API for chaining preprocessing steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import KeypointSequence, MarkerTrajectory


class PreprocessingStep(Protocol):
    """Protocol for a preprocessing step."""

    def apply(
        self,
        data: KeypointSequence | MarkerTrajectory,
    ) -> KeypointSequence | MarkerTrajectory:
        """
        Apply preprocessing step to data.

        Args:
            data: Input data

        Returns:
            Preprocessed data
        """
        ...


@dataclass
class PreprocessingPipeline:
    """
    Composable preprocessing pipeline.

    Chains multiple preprocessing steps in order.

    Example:
        pipeline = PreprocessingPipeline([
            GapFillStep(strategy=GapFillStrategy.LINEAR),
            FilterStep(filter_type=FilterType.BUTTERWORTH, cutoff=6.0),
            ResampleStep(target_fps=100.0),
            NormalizeStep(target_up=UpAxis.Y_UP, center_origin=True),
        ])
        result = pipeline.apply(data)
    """

    steps: list[PreprocessingStep] = field(default_factory=list)

    def add_step(self, step: PreprocessingStep) -> None:
        """Add a preprocessing step to the pipeline."""
        self.steps.append(step)

    def apply(
        self,
        data: KeypointSequence | MarkerTrajectory,
    ) -> KeypointSequence | MarkerTrajectory:
        """
        Apply all preprocessing steps in order.

        Args:
            data: Input data

        Returns:
            Fully preprocessed data
        """
        result = data
        for step in self.steps:
            result = step.apply(result)
        return result

    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self.steps)

    def __iter__(self):
        """Iterate over steps."""
        return iter(self.steps)


@dataclass
class GapFillStep:
    """Gap-filling preprocessing step."""

    strategy: GapFillStrategy = None  # type: ignore
    max_gap: int = 10

    def __post_init__(self):
        from .gap_fill import GapFillStrategy

        if self.strategy is None:
            self.strategy = GapFillStrategy.LINEAR

    def apply(
        self,
        data: KeypointSequence | MarkerTrajectory,
    ) -> KeypointSequence | MarkerTrajectory:
        """Apply gap-filling to data."""
        from .gap_fill import gap_fill

        return gap_fill(data, strategy=self.strategy, max_gap=self.max_gap)


@dataclass
class FilterStep:
    """Filtering preprocessing step."""

    filter_type: FilterType = None  # type: ignore
    cutoff: float = 6.0
    order: int = 2
    fps: float | None = None

    def __post_init__(self):
        from .filter import FilterType

        if self.filter_type is None:
            self.filter_type = FilterType.BUTTERWORTH

    def apply(
        self,
        data: KeypointSequence | MarkerTrajectory,
    ) -> KeypointSequence | MarkerTrajectory:
        """Apply filter to data."""
        from .filter import apply_filter

        return apply_filter(
            data,
            filter_type=self.filter_type,
            cutoff=self.cutoff,
            order=self.order,
            fps=self.fps,
        )


@dataclass
class ResampleStep:
    """Resampling preprocessing step."""

    target_fps: float = 100.0
    source_fps: float | None = None

    def apply(
        self,
        data: KeypointSequence | MarkerTrajectory,
    ) -> KeypointSequence | MarkerTrajectory:
        """Apply resampling to data."""
        from .resample import resample

        return resample(data, target_fps=self.target_fps, source_fps=self.source_fps)


@dataclass
class NormalizeStep:
    """Coordinate normalization preprocessing step."""

    target_up: UpAxis = None  # type: ignore
    source_up: UpAxis | None = None
    center_origin: bool = True
    target_unit: UnitSystem | None = None

    def __post_init__(self):
        from .normalize import UpAxis

        if self.target_up is None:
            self.target_up = UpAxis.Y_UP

    def apply(
        self,
        data: KeypointSequence | MarkerTrajectory,
    ) -> KeypointSequence | MarkerTrajectory:
        """Apply coordinate normalization to data."""
        from .normalize import normalize_coordinates, convert_units

        # Apply coordinate normalization
        result = normalize_coordinates(
            data,
            target_up=self.target_up,
            source_up=self.source_up,
            center_origin=self.center_origin,
        )

        # Apply unit conversion if specified
        if self.target_unit is not None:
            result = convert_units(result, target_unit=self.target_unit)

        return result


# Type hints for post_init
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gap_fill import GapFillStrategy
    from .filter import FilterType
    from .normalize import UpAxis, UnitSystem
