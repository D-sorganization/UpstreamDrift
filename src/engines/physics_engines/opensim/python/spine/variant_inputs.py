"""Movement variants as golf trunk inputs and model range checks (SPN-13b, #12767).

Bridges :mod:`src.shared.python.spine.movement_variants` to the OpenSim spine
pipeline. :func:`variant_golf_input` returns a new :class:`GolfTrunkInput`
whose ``pelvis_to_thorax`` is the varied series; time, pelvis pose, events and
every other coordinate are held (``thorax_relative_hold``: arm coordinates are
thorax-relative, so the arms follow the thorax). The input ``source`` keeps the
base file name and hash and adds ``variant`` = the variant's
``distribution_source``, which :func:`drive_segmental_model` carries into
``DriveResult.input_source``.

:func:`check_model_ranges` checks driven level coordinates against the target
model's own coordinate ranges (``Coordinate.getRangeMin/Max``, read natively).
Violations are reported, never clipped. Units: radians/metres, seconds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.engines.physics_engines.opensim.python.spine.golf_input import (
    GolfTrunkInput,
)
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.movement_variants import (
    MovementVariant,
    RangeCheck,
    VariantResult,
    apply_variant,
    check_coordinate_ranges,
)

__all__ = [
    "VariantGolfInput",
    "check_model_ranges",
    "event_indices",
    "model_coordinate_ranges",
    "variant_golf_input",
]


@dataclass(frozen=True)
class VariantGolfInput:
    """A varied golf trunk input and the :class:`VariantResult` that made it."""

    golf_input: GolfTrunkInput
    result: VariantResult


def _check_input(golf_input: object) -> GolfTrunkInput:
    if not isinstance(golf_input, GolfTrunkInput):
        raise TypeError(
            f"golf_input must be a GolfTrunkInput, got {type(golf_input).__name__}"
        )
    return golf_input


def event_indices(golf_input: GolfTrunkInput) -> dict[str, int]:
    """Map each event time of ``golf_input`` to its nearest sample index.

    Raises:
        TypeError: ``golf_input`` is not a :class:`GolfTrunkInput`.
        ValueError: an event time lies outside ``[time_s[0], time_s[-1]]``.
    """
    golf = _check_input(golf_input)
    times = np.asarray(golf.time_s)
    out: dict[str, int] = {}
    for name, when in golf.events.items():
        if not times[0] <= when <= times[-1]:
            raise ValueError(
                f"event {name!r} at {when} s is outside the samples "
                f"[{times[0]}, {times[-1]}] s"
            )
        out[name] = int(np.argmin(np.abs(times - when)))
    return out


def variant_golf_input(
    golf_input: GolfTrunkInput,
    variant: MovementVariant,
    *,
    handedness: Literal["right", "left"],
    parameter: float | int | None = None,
) -> VariantGolfInput:
    """Apply ``variant`` to the trunk rotation of ``golf_input``.

    Events are mapped to their nearest samples (:func:`event_indices`).

    Preconditions: ``golf_input`` is a :class:`GolfTrunkInput` (``TypeError``);
    the preconditions of :func:`apply_variant` hold (``TypeError`` /
    ``ValueError``).

    Postconditions: only ``pelvis_to_thorax`` and ``source`` differ from the
    input; ``source["variant"]`` equals ``result.distribution_source``.
    """
    golf = _check_input(golf_input)
    result = apply_variant(
        variant,
        golf.pelvis_to_thorax,
        golf.time_s,
        event_indices(golf),
        handedness=handedness,
        parameter=parameter,
    )
    source = {**golf.source, "variant": result.distribution_source}
    varied = GolfTrunkInput(
        time_s=golf.time_s,
        pelvis_to_thorax=result.rotations,
        pelvis_pose_ground=golf.pelvis_pose_ground,
        other_coordinates=golf.other_coordinates,
        events=golf.events,
        source=source,
    )
    ensure(varied.source["variant"] != "", "variant source must be non-empty")
    return VariantGolfInput(varied, result)


def _coordinate_range(model: object, name: str) -> tuple[float, float]:
    coords = model.getCoordinateSet()  # type: ignore[attr-defined]
    if not coords.getIndex(name) >= 0:
        raise ValueError(f"model has no coordinate {name!r}")
    coord = coords.get(name)
    return float(coord.getRangeMin()), float(coord.getRangeMax())


def model_coordinate_ranges(
    model_path: str | Path, names: Iterable[str]
) -> dict[str, tuple[float, float]]:
    """Read ``(min, max)`` ranges of the named coordinates from a model file.

    Raises:
        ValueError: the file does not exist or a coordinate is unknown.
    """
    path = Path(model_path)
    if not path.is_file():
        raise ValueError(f"model file not found: {path.name}")
    model = import_opensim().Model(str(path))
    return {str(name): _coordinate_range(model, str(name)) for name in names}


def check_model_ranges(
    model_path: str | Path, coordinates: Mapping[str, np.ndarray]
) -> RangeCheck:
    """Check coordinate series (e.g. a driven variant) against the model ranges.

    Same contract as :func:`check_coordinate_ranges` with the ranges read by
    :func:`model_coordinate_ranges` (``ValueError`` for unknown coordinates).
    """
    if not isinstance(coordinates, Mapping):
        raise TypeError("coordinates must be a mapping of name -> series")
    ranges = model_coordinate_ranges(model_path, coordinates)
    return check_coordinate_ranges(coordinates, ranges)
