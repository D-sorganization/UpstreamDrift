"""Flex sweep: one hand motion through every shaft flex (#15684).

SAME HAND MOTION: each flex gets the identical grip track, so the rows show
what the bend does to the head for that fixed motion. The golfer does not
adapt (no re-timed release, no speed-potential change), so this is a
groundwork answer to "same hands, softer or stiffer shaft", not a model of how
fast a given player would swing each shaft.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import logging
import math
from typing import Any

from src.shared.python.club_fitting.bend_trace import BendTrace, bend_trace
from src.shared.python.club_fitting.flight_trace import FLIGHT_KEY, FlightTrace
from src.shared.python.club_fitting.grip_driven_shaft import (
    HeadGeometry,
    ShaftRun,
    simulate_grip_driven_shaft,
)
from src.shared.python.club_fitting.grip_motion import (
    ClubPoseTrack,
    FittingSettings,
    load_settings,
)
from src.shared.python.motion_matching import club_models as cm
from src.shared.python.motion_matching import club_shaft_chain as cs

logger = logging.getLogger(__name__)
RIGID = "Rigid"
MPH_PER_MPS = 2.2369362920544
ShotFn = Callable[["ShaftRun | None"], dict[str, Any]]


@dataclass(frozen=True)
class FlexRow:
    """Result of one flex. ``shot`` holds ball speed, launch and carry, if known.

    Deltas are bent minus straight club in the same hand motion, at impact.
    ``peak_tip_deflection_mm`` is the peak up to impact; the extrapolated
    window after it is not physical. ``bend`` is the lead and droop up to
    impact (for the shaft report) and ``flight`` the ball flight (for the flex
    comparison); neither is part of :meth:`record`.
    """

    flex: str
    flex_cpm: float | None
    peak_head_speed_mph: float
    impact_head_speed_mph: float
    rigid_impact_head_speed_mph: float
    loft_delta_deg: float
    face_delta_deg: float
    lie_delta_deg: float
    peak_tip_deflection_mm: float
    shot: dict[str, float | None] | None
    bend: BendTrace | None = field(default=None, compare=False, repr=False)
    flight: FlightTrace | None = field(default=None, compare=False, repr=False)

    def record(self) -> dict[str, object]:
        flat = asdict(self)
        flat.pop("bend")
        flat.pop("flight")
        shot = flat.pop("shot") or {}
        return {**flat, **{f"shot_{k}": v for k, v in shot.items()}}


def _row(
    flex: str,
    cpm: float | None,
    run: ShaftRun,
    impact_s: float,
    shot: ShotFn | None,
    head: HeadGeometry | None = None,
) -> FlexRow:
    mph = MPH_PER_MPS
    values = None if shot is None else dict(shot(run))
    flight = None if values is None else values.pop(FLIGHT_KEY, None)
    return FlexRow(
        flex,
        cpm,
        float(run.head_speed_mps.max()) * mph,
        run.at_vector_norm("head_velocity_mps", impact_s) * mph,
        run.at_vector_norm("rigid_velocity_mps", impact_s) * mph,
        run.at("loft_delta_deg", impact_s),
        run.at("face_delta_deg", impact_s),
        run.at("lie_delta_deg", impact_s),
        run.peak_tip_deflection_until_m(impact_s) * 1e3,
        values,
        None if head is None else bend_trace(run, head, impact_s),
        flight,
    )


def run_flex_sweep(
    track: ClubPoseTrack,
    club: cm.ClubSpec,
    head: HeadGeometry,
    impact_time_s: float,
    flexes: Sequence[str] | None = None,
    settings: FittingSettings | None = None,
    shot: ShotFn | None = None,
) -> list[FlexRow]:
    """Run the same grip motion through each flex; one row per flex.

    Args:
        flexes: ratings to run (default: every ``club_shaft_chain`` rating,
            most flexible first).
        shot: maps a run (``None`` for the rigid club) to ball speed, launch
            and carry, and optionally the flight under ``FLIGHT_KEY`` (kept
            on the row, not in its record); omitted without a shot model.

    Raises:
        ValueError: empty or unknown flex list, or any input rejected by
            :func:`simulate_grip_driven_shaft`.
    """
    cfg = settings or load_settings()
    known = cs.flex_ratings()
    chosen = tuple(flexes) if flexes is not None else known
    unknown = [f for f in chosen if f not in known]
    if not chosen or unknown:
        raise ValueError(f"flexes must be a non-empty subset of {known}: {unknown}")
    rows = []
    for flex in chosen:
        chain = cs.build_shaft_chain(club, cfg.chain_links, flex=flex)
        run = simulate_grip_driven_shaft(track, chain, head, impact_time_s, cfg)
        rows.append(
            _row(flex, chain.record["flex_cpm"], run, impact_time_s, shot, head)
        )
        logger.info(
            "%s: impact head speed %.2f mph (rigid %.2f)",
            flex,
            rows[-1].impact_head_speed_mph,
            rows[-1].rigid_impact_head_speed_mph,
        )
    if not all(math.isfinite(r.peak_head_speed_mph) for r in rows):
        raise ValueError("flex sweep produced a non-finite head speed")
    return rows
