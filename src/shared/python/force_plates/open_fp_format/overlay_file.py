"""Overlay an open-fp swing file on a ground-reaction plot series (OF-07, #13087).

One entry point shared by the PyQt tile and the API service so neither
re-implements file reading, placement or overlaying.
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    GroundReactionPlotSeries,
)
from src.shared.python.force_plates.alignment import PlateTimeAlignment
from src.shared.python.force_plates.model_frame import TargetedPlacement
from src.shared.python.force_plates.open_fp_format.reader import (
    PlateDeclarations,
    read_swing,
)
from src.shared.python.force_plates.overlay import overlay_on_series
from src.shared.python.force_plates.readings import PlacedTrial, ReadingOptions

__all__ = ["DEFAULT_COVERAGE", "overlay_swing_file"]

#: Band factor used to turn the interpolation bias estimate into a sigma.
DEFAULT_COVERAGE = 2.0


def overlay_swing_file(
    series: GroundReactionPlotSeries,
    json_path: str | Path,
    *,
    declared: PlateDeclarations,
    placement: TargetedPlacement,
    alignment: PlateTimeAlignment,
    options: ReadingOptions | None = None,
) -> GroundReactionPlotSeries:
    """``series`` with the measured traces of the swing at ``json_path``.

    ``placement`` is the :class:`TargetedPlacement` of the stance frame, and its
    trial handedness must equal the file's (a swing is never mirrored).
    ``options`` defaults to ``ReadingOptions(coverage=2.0)``.

    Raises ``TypeError`` for wrong types and ``ValueError`` for anything the
    reader, the placement check or the overlay refuses.
    """
    if not isinstance(placement, TargetedPlacement):
        raise TypeError("placement must be a TargetedPlacement")
    trial = read_swing(json_path, declared=declared)
    if trial.handedness != placement.handedness:
        raise ValueError(
            f"file handedness {trial.handedness!r} differs from the placement's "
            f"{placement.handedness!r}"
        )
    placed = PlacedTrial(trial, placement.placement)
    use = options if options is not None else ReadingOptions(coverage=DEFAULT_COVERAGE)
    return overlay_on_series(series, placed, alignment, use)
