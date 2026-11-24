"""Flex sweep: one hand motion through every shaft flex (#15684).

SAME HAND MOTION: each flex gets the identical grip track, so the rows show
what the bend does to the head for that fixed motion. The golfer does not
adapt (no re-timed release, no speed-potential change), so this is a
groundwork answer to "same hands, softer or stiffer shaft", not a model of how
fast a given player would swing each shaft.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import logging
import math

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
ShotFn = Callable[["ShaftRun | None"], dict[str, float | None]]


@dataclass(frozen=True)
class FlexRow:
    """Result of one flex. ``shot`` holds ball speed, launch and carry, if known.

    Deltas are bent minus straight club in the same hand motion, at impact.
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

    def record(self) -> dict[str, object]:
        flat = asdict(self)
        shot = flat.pop("shot") or {}
        return {**flat, **{f"shot_{k}": v for k, v in shot.items()}}


def _row(
    flex: str, cpm: float | None, run: ShaftRun, impact_s: float, shot: ShotFn | None
) -> FlexRow:
    mph = MPH_PER_MPS
    return FlexRow(
        flex,
        cpm,
        float(run.head_speed_mps.max()) * mph,
        run.at_vector_norm("head_velocity_mps", impact_s) * mph,
        run.at_vector_norm("rigid_velocity_mps", impact_s) * mph,
        run.at("loft_delta_deg", impact_s),
        run.at("face_delta_deg", impact_s),
        run.at("lie_delta_deg", impact_s),
        run.peak_tip_deflection_m * 1e3,
        None if shot is None else shot(run),
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
            and carry; omitted when no shot model is available.

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
        rows.append(_row(flex, chain.record["flex_cpm"], run, impact_time_s, shot))
        logger.info(
            "%s: impact head speed %.2f mph (rigid %.2f)",
            flex,
            rows[-1].impact_head_speed_mph,
            rows[-1].rigid_impact_head_speed_mph,
        )
    if not all(math.isfinite(r.peak_head_speed_mph) for r in rows):
        raise ValueError("flex sweep produced a non-finite head speed")
    return rows
