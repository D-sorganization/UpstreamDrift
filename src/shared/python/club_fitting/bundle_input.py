"""Club pose track and shot model of a same-input bundle (#15684).

Reads a matched-swing bundle through the native-viewer exporter's own loaders,
so the rigid reference shot is the exporter's shot. A flexed run changes only
the head centre, velocity and face axes of that rigid series by the bend of the
grip-driven run (same hand motion), then the exporter's impact card and flight
tracer give the shot. Needs a bundle; the synthetic unit tests use
:mod:`flex_sweep` with a stub shot instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.shared.python.club_fitting.flight_trace import (
    FLIGHT_KEY,
    flight_from_playback,
)
from src.shared.python.club_fitting.grip_driven_shaft import HeadGeometry, ShaftRun
from src.shared.python.club_fitting.grip_motion import (
    ClubPoseTrack,
    grip_point_in_club,
)
from src.shared.python.motion_matching import club_models as cm

logger = logging.getLogger(__name__)
MPH_PER_MPS = 2.2369362920544
YD_PER_M = 1.0 / 0.9144
ShotFn = Callable[["ShaftRun | None"], dict[str, Any]]


@dataclass(frozen=True)
class BundleShot:
    """Pose track, head, club and impact of a bundle plus its shot model."""

    track: ClubPoseTrack
    head: HeadGeometry
    club: cm.ClubSpec
    impact_time_s: float
    shot: ShotFn


def _moved_series(series: Any, run: ShaftRun) -> Any:
    """``series`` with the head moved by the bend of ``run`` (rigid outside it)."""
    ts = series.times_s
    inside = (ts >= run.times_s[0]) & (ts <= run.times_s[-1])
    rel = np.einsum("nij,nkj->nik", run.tip_rotation, run.rigid_rotation)
    rot = np.repeat(np.eye(3)[None], len(ts), axis=0)
    rot[inside] = Slerp(run.times_s, Rotation.from_matrix(rel))(ts[inside]).as_matrix()
    d_centre = run.head_centre_m - run.rigid_centre_m
    d_vel = run.head_velocity_mps - run.rigid_velocity_mps
    dc, dv = np.zeros((len(ts), 3)), np.zeros((len(ts), 3))
    for j in range(3):
        dc[inside, j] = np.interp(ts[inside], run.times_s, d_centre[:, j])
        dv[inside, j] = np.interp(ts[inside], run.times_s, d_vel[:, j])
    return replace(
        series,
        face_center_m=series.face_center_m + dc,
        velocity_mps=series.velocity_mps + dv,
        face_normal=np.einsum("nij,nj->ni", rot, series.face_normal),
        toe_axis=np.einsum("nij,nj->ni", rot, series.toe_axis),
        grip_axis=np.einsum("nij,nj->ni", rot, series.grip_axis),
    )


def load_bundle_shot(bundle: Path, club_label: str, out_dir: Path) -> BundleShot:
    """Load ``bundle`` and set up its rigid shot through the exporter.

    Raises:
        ValueError: the bundle has no usable impact time.
    """
    from src.shared.python.impact_parameters.club_impact import club_impact_from_spec
    from src.tools.native_viewer_export import (
        cli,
        club_pose,
        contact_verdict,
        core,
        flight_tracer,
        impact_card,
    )

    args = cli.build_parser().parse_args(
        ["--bundle", str(bundle), "--out", str(out_dir), "--swing", "fit"]
        + ["--club", club_label, "--impact-factors", "--ball-flight"]
    )
    settings = cli.build_settings(args)
    swing = core.load_swing_input(bundle, swing="fit", club=club_label)
    swing, settings, _ = core.prepare_shot(swing, settings)
    if settings.impact_time_s is None:
        raise ValueError("the bundle has no impact time")
    spec = json.loads(swing.bundle.spec_bytes)
    club = cm.recorded_club(spec, club_label.lower())
    model = club_pose.ClubPoseModel(swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    poses = [
        model.body_world(dict(zip(names, map(float, row), strict=True)))
        for row in swing.q
    ]
    rot = np.array([p[:3, :3] for p in poses])
    origin = np.array([p[:3, 3] for p in poses])
    track = ClubPoseTrack(
        np.array([swing.time_s(i) for i in range(swing.n_states)]),
        origin + np.einsum("nij,j->ni", rot, grip_point_in_club(club)),
        rot,
    )
    face = model.face_world(dict(zip(names, map(float, swing.q[0]), strict=True)))
    head = HeadGeometry(rot[0].T @ (face.centre_m - origin[0]), rot[0].T @ face.normal)
    impact = settings.impact_time_s
    if swing.ball is None:
        raise ValueError("the bundle swing places no ball; a flex sweep needs one")
    ball = swing.ball
    club_impact = club_impact_from_spec(flight_tracer._spec_club(swing))
    rigid = flight_tracer.club_series(swing)

    def shot(run: ShaftRun | None) -> dict[str, Any]:
        series = rigid if run is None else _moved_series(rigid, run)
        card = impact_card.card_from_series(series, impact, ball, "mph")
        verdict = contact_verdict.verdict_for(series, impact, ball)
        play = flight_tracer.playback_from_series(
            series, impact, ball, settings.ball_flight, swing.club, club_impact, verdict
        )
        values = {r.key: r.value for r in card.rows}
        carry, side = play.carry_m, play.lateral_m
        return {
            "dynamic_loft_deg": values.get("dynamic_loft_deg"),
            "face_angle_deg": values.get("face_angle_deg"),
            "ball_speed_mph": play.ball_speed_mph,
            "launch_deg": play.launch_deg,
            "spin_rpm": play.spin_rpm,
            "carry_yd": None if carry is None else carry * YD_PER_M,
            "side_yd": None if side is None else side * YD_PER_M,
            FLIGHT_KEY: flight_from_playback(play),
        }

    logger.info("loaded bundle shot, impact at %.4f s", impact)
    return BundleShot(track, head, club, float(impact), shot)
