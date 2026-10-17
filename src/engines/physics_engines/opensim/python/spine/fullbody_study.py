"""Full-body driving of a segmental model from a matched golf swing (SPN-9c, #12758).

Given the trunk coordinates the spine drive already produced (pelvis pose and
intervertebral levels), this stage fills in the rest of the target model from
the golf swing and builds the external loads:

1. Direct coordinates (hip, knee, ankle, elbow) carry the golf value times the
   recorded sign (:func:`fullbody_map.apply_direct_map`).
2. The shoulders are solved from the golf humerus WORLD orientation, neutral
   calibrated, because the golf and target joints use different Euler orders
   (:func:`fullbody_native.solve_orientation_group`); the residual is reported.
3. Held-neutral coordinates are zero; locked coordinates are not written,
   except those the drive unlocked (``unlock_coordinates``): they carry the
   drive's series and are unlocked wherever this stage poses the target, so
   the arms and the ground reaction see the driven trunk (#13056).
4. Loads by policy: ``none`` (gravity only), ``supplied`` (the request's file,
   not generated here), ``grf_only`` (estimated ground reaction on the feet),
   ``club_and_grf`` (the club net wrench applied to the hands plus the ground
   reaction the remaining wrench requires).  See :mod:`fullbody_loads` for why
   the club is applied to the hands and for the Newton-Euler equations.

Frames: ground X anterior, Y up, Z right; radians, metres, newtons, seconds.
Nothing here reads or writes a licensed numeric value; the target model is
only opened by the callers.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.msk_club import (
    CLUB_BODY,
    HAND_BODIES,
    hand_grip_point,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    ContactPoint,
    default_contact_points,
    estimate_ground_reactions,
)
from src.engines.physics_engines.opensim.python.spine.drive import DriveResult
from src.engines.physics_engines.opensim.python.spine.fullbody_loads import (
    POLICIES,
    ClubInertia,
    LoadSet,
    club_net_wrench,
    combine_load_sets,
    hand_reaction_loads,
    loads_record,
    split_between_hands,
    write_load_set,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_map import (
    FullbodyMap,
    apply_direct_map,
    neutral_columns,
)
from src.engines.physics_engines.opensim.python.spine.fullbody_native import (
    body_mass_properties,
    body_poses,
    body_rotations,
    coordinate_names,
    neutral_alignment,
    solve_orientation_group,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    read_golf_coordinates,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    prepare_model_for_id,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import FileRecord
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.kinematic_smoothing import (
    smooth_kinematic_trajectory,
)
from src.shared.python.pose_estimation.model_files import sha256_of

__all__ = [
    "GRF_MIN_FRAMES",
    "FullbodyResult",
    "FullbodySpec",
    "run_fullbody",
    "unrun_record",
    "with_coordinates",
]

#: ``estimate_ground_reactions`` needs more than 12 uniformly sampled frames.
GRF_MIN_FRAMES = 13
_GRF_POLICIES = ("grf_only", "club_and_grf")
_TIME_TOL_S = 1e-6
_ORIGIN = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FullbodySpec:
    """Inputs of :func:`run_fullbody` (frozen).

    ``golf_model_path``/``motion_path`` give the golf coordinate table,
    ``target_model_path`` the model being driven; ``policy`` is one of
    :data:`fullbody_loads.POLICIES`; generated files are ``<stem>_...`` in
    ``output_dir``.  ``filter_hz`` is the optional low-pass of the club wrench
    and ground-reaction kinematics; ``supplied_name`` the basename of a
    supplied ExternalLoads file (policy ``supplied``); ``unlock_coordinates``
    the coordinates locked in the target file that the drive unlocked and
    moves (each needs a trunk series).
    """

    fmap: FullbodyMap
    golf_model_path: Path
    motion_path: Path
    target_model_path: Path
    policy: str
    output_dir: Path
    stem: str
    filter_hz: float | None = None
    supplied_name: str | None = None
    unlock_coordinates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        if not self.stem or "/" in self.stem or "\\" in self.stem:
            raise ValueError(f"stem must be a plain non-empty name, got {self.stem!r}")
        for name in ("golf_model_path", "motion_path", "target_model_path"):
            if not Path(getattr(self, name)).is_file():
                raise FileNotFoundError(
                    f"{name} not found: {Path(getattr(self, name)).name}"
                )
        if self.filter_hz is not None and not self.filter_hz > 0.0:
            raise ValueError(f"filter_hz must be positive, got {self.filter_hz!r}")
        if not isinstance(self.unlock_coordinates, tuple) or not all(
            isinstance(n, str) for n in self.unlock_coordinates
        ):
            raise TypeError("unlock_coordinates must be a tuple of names")


@dataclass(frozen=True)
class FullbodyResult:
    """Outcome of :func:`run_fullbody`.

    ``coordinates`` are the target coordinate series written by this stage
    (read-only copies; trunk and pelvis included), ``load_xml`` the generated
    ExternalLoads file or ``None``, ``artifacts`` its hashed files, ``record``
    the :func:`fullbody_loads.loads_record` and ``metrics`` scalar results.
    """

    coordinates: Mapping[str, np.ndarray]
    load_xml: Path | None
    artifacts: tuple[FileRecord, ...]
    record: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)


def with_coordinates(
    drive: DriveResult, coordinates: Mapping[str, np.ndarray]
) -> DriveResult:
    """Copy of ``drive`` whose coordinate table is extended by ``coordinates``.

    Raises ``ValueError`` (from :class:`DriveResult`) when a series length does
    not match the drive's sample count.
    """
    return replace(drive, coordinates={**drive.coordinates, **coordinates})


def unrun_record(map_id: str, policy: str) -> dict[str, Any]:
    """Receipt record of a full-body option whose stage did not run."""
    return {
        "policy": policy,
        "map_id": map_id,
        "included": [],
        "excluded": [],
        "statement": "full-body stage not run: no external loads were applied",
    }


# ------------------------------------------------------------------ helpers


def _check_grid(time_s: np.ndarray, golf_time: np.ndarray) -> None:
    if time_s.shape != golf_time.shape or not np.allclose(
        time_s, golf_time, atol=_TIME_TOL_S, rtol=0.0
    ):
        raise ValueError(
            "the trunk time grid does not match the golf motion time grid "
            f"({time_s.size} vs {golf_time.size} samples)"
        )


def _trunk_coordinates(
    fmap: FullbodyMap,
    trunk: Mapping[str, np.ndarray],
    n_frames: int,
    unlocked: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Trunk series for the pelvis, spine-drive and unlocked targets.

    Raises ``ValueError`` when a pelvis or unlocked coordinate has no series.
    """
    lacking = [n for n in unlocked if n not in trunk]
    if lacking:
        raise ValueError(f"unlocked coordinates lack a trunk series: {lacking}")
    out: dict[str, np.ndarray] = {}
    wanted = (
        *fmap.targets_with("driven_by_pelvis_pose"),
        *fmap.targets_with("driven_by_spine_drive"),
        *unlocked,
    )
    for name in wanted:
        if name in trunk:
            series = np.asarray(trunk[name], dtype=float)
            if series.shape != (n_frames,):
                raise ValueError(f"trunk[{name!r}] must have shape ({n_frames},)")
            out[name] = series
    missing = [n for n in fmap.targets_with("driven_by_pelvis_pose") if n not in out]
    if missing:
        raise ValueError(f"trunk lacks pelvis coordinates: {missing}")
    return out


def _unlock(model: Any, names: tuple[str, ...]) -> None:
    """Unlock ``names`` in ``model`` before its first ``initSystem``."""
    coords = model.getCoordinateSet()
    unknown = [n for n in names if coords.getIndex(n) < 0]
    if unknown:
        raise ValueError(f"cannot unlock unknown coordinates {unknown}")
    for name in names:
        coords.get(name).set_locked(False)


def _transfer_coordinates(
    fmap: FullbodyMap, golf_q: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Direct and neutral coordinates; every direct golf column must exist."""
    absent = [e.golf for e in fmap.direct if e.golf not in golf_q]
    if absent:
        raise ValueError(f"golf motion lacks mapped coordinates: {absent}")
    out = apply_direct_map(fmap, golf_q)
    n_frames = len(next(iter(golf_q.values())))
    out.update(neutral_columns(fmap, n_frames))
    return out


def _solve_groups(
    spec: FullbodySpec,
    golf: Any,
    target: Any,
    golf_q: Mapping[str, np.ndarray],
    coords: dict[str, np.ndarray],
) -> dict[str, float]:
    """Solve every orientation group into ``coords``.

    Returns the largest residual and the largest frame-to-frame coordinate
    step over the groups [rad]; a large step is a branch switch.
    """
    worst = step = 0.0
    for group in spec.fmap.groups:
        world = body_rotations(golf, golf_q, group.golf_body)
        align = neutral_alignment(golf, group.golf_body, target, group.target_body)
        solution = solve_orientation_group(
            target, group, coords, np.einsum("nij,jk->nik", world, align)
        )
        coords.update(solution.coordinates)
        worst = max(worst, float(np.max(solution.residual_rad)))
        step = max(step, solution.max_step_rad)
    return {"orientation_max_residual_rad": worst, "orientation_max_step_rad": step}


def _smoothed(
    time_s: np.ndarray, table: Mapping[str, np.ndarray], filter_hz: float | None
) -> dict[str, np.ndarray]:
    if filter_hz is None:
        return {k: np.array(v, dtype=float) for k, v in table.items()}
    names = list(table)
    result = smooth_kinematic_trajectory(
        time_s, np.column_stack([table[k] for k in names]), cutoff_hz=filter_hz
    )
    if not result.smoothing_applied:
        raise ValueError(f"low-pass filter at {filter_hz} Hz could not be applied")
    return {name: np.array(result.q[:, j]) for j, name in enumerate(names)}


# ------------------------------------------------------------------- loads


def _golf_hand_sides(fmap: FullbodyMap) -> list[tuple[str, str, str]]:
    """``(golf_hand, target_hand, side)`` in a fixed order."""
    sides = {body: side for side, body in HAND_BODIES.items()}
    rows = []
    for golf_hand, target_hand in sorted(fmap.hand_bodies.items()):
        if golf_hand not in sides:
            raise ValueError(f"golf hand body {golf_hand!r} is not a club hand body")
        rows.append((golf_hand, target_hand, sides[golf_hand]))
    return rows


def _club_loads(
    spec: FullbodySpec,
    golf: Any,
    golf_q: Mapping[str, np.ndarray],
    time_s: np.ndarray,
) -> LoadSet:
    """Hand reaction wrench of the golf club, on the target hand bodies."""
    rows = _golf_hand_sides(spec.fmap)
    poses = body_poses(golf, golf_q, [CLUB_BODY, *(r[0] for r in rows)])
    mass, com_local, tensor = body_mass_properties(golf, CLUB_BODY)
    club = poses[CLUB_BODY]
    com = club[:, :3, :3] @ com_local + club[:, :3, 3]
    net = club_net_wrench(
        time_s,
        com,
        club[:, :3, :3],
        ClubInertia(mass, com_local, tensor),
        filter_hz=spec.filter_hz,
    )
    grips, origins = [], []
    for golf_hand, _target, side in rows:
        pose = poses[golf_hand]
        point = np.append(hand_grip_point(side), 1.0)
        grips.append(pose @ point)
        origins.append(pose[:, :3, 3])
    grip_world = np.stack([g[:, :3] for g in grips], axis=1)
    origin_world = np.stack(origins, axis=1)
    weights = [1.0 / len(rows)] * len(rows)
    force, torque = hand_reaction_loads(
        split_between_hands(net, com, grip_world, weights), grip_world, origin_world
    )
    points = tuple(
        ContactPoint(f"club_{golf_hand}", target_hand, _ORIGIN, "")
        for golf_hand, target_hand, _side in rows
    )
    return LoadSet(points, force, torque, ("club_hand_wrench",) * len(rows))


def _foot_points(fmap: FullbodyMap) -> tuple[ContactPoint, ...]:
    points = tuple(p for p in default_contact_points() if p.body in fmap.foot_bodies)
    if not points:
        raise ValueError("the map's foot_bodies match no ground-contact point")
    return points


def _ground_loads(
    spec: FullbodySpec,
    time_s: np.ndarray,
    coords: Mapping[str, np.ndarray],
    hands: LoadSet | None,
) -> tuple[LoadSet, float]:
    """Estimated ground reaction on the feet and its worst residual norm."""
    with tempfile.TemporaryDirectory() as scratch:
        xml = None
        if hands is not None:
            xml = write_load_set(scratch, time_s, hands, "hands")
        prepared = prepare_model_for_id(
            spec.target_model_path,
            external_loads=xml,
            unlock_coordinates=spec.unlock_coordinates,
        )
        names = coordinate_names(prepared.model)
        zeros = np.zeros(time_s.size)
        table = _smoothed(
            time_s, {n: coords.get(n, zeros) for n in names}, spec.filter_hz
        )
        points = _foot_points(spec.fmap)
        estimate = estimate_ground_reactions(
            prepared.model,
            time_s,
            table,
            points,
            keep_external=hands is not None,
        )
    torques = np.zeros_like(estimate["forces"])
    torques[:, :, 1] = estimate["spins"]
    kinds = ("ground_reaction",) * len(points)
    loads = LoadSet(points, estimate["forces"], torques, kinds)
    return loads, float(np.max(estimate["residual_norm"]))


def _build_loads(
    spec: FullbodySpec,
    golf: Any,
    golf_q: Mapping[str, np.ndarray],
    coords: Mapping[str, np.ndarray],
    time_s: np.ndarray,
) -> tuple[LoadSet | None, dict[str, float]]:
    if spec.policy not in _GRF_POLICIES:
        return None, {}
    hands = None
    if spec.policy == "club_and_grf":
        hands = _club_loads(spec, golf, golf_q, time_s)
    ground, residual = _ground_loads(spec, time_s, coords, hands)
    loads = ground if hands is None else combine_load_sets(hands, ground)
    return loads, {"grf_max_residual_norm": residual}


# ------------------------------------------------------------------- driver


def run_fullbody(
    spec: FullbodySpec, time_s: np.ndarray, trunk: Mapping[str, np.ndarray]
) -> FullbodyResult:
    """Drive the whole target model and build its external loads.

    ``trunk`` holds the pelvis and spine-drive coordinate series on ``time_s``
    (a drive result's ``coordinates``).  Preconditions (``ValueError``): the
    policy needs at least :data:`GRF_MIN_FRAMES` frames when it estimates a
    ground reaction; ``time_s`` equals the golf motion's time grid; every
    mapped golf coordinate and every pelvis target is present.
    Postconditions: every driven coordinate is finite; the orientation residual
    is reported in ``metrics`` and never repaired; ``artifacts`` hash the
    generated files (basenames only).  Pelvis residuals are not computed here;
    they come from the inverse dynamics that consumes the loads.
    """
    time = np.asarray(time_s, dtype=float)
    if spec.policy in _GRF_POLICIES and time.size < GRF_MIN_FRAMES:
        raise ValueError(
            f"policy {spec.policy} needs at least {GRF_MIN_FRAMES} frames, "
            f"got {time.size}"
        )
    golf_time, golf_q = read_golf_coordinates(spec.motion_path, spec.golf_model_path)
    _check_grid(time, golf_time)
    osim = import_opensim()
    golf = osim.Model(str(spec.golf_model_path))
    target = osim.Model(str(spec.target_model_path))
    _unlock(target, spec.unlock_coordinates)
    coords = _trunk_coordinates(spec.fmap, trunk, time.size, spec.unlock_coordinates)
    coords.update(_transfer_coordinates(spec.fmap, golf_q))
    metrics = _solve_groups(spec, golf, target, golf_q, coords)
    for name, series in coords.items():
        ensure(bool(np.all(np.isfinite(series))), f"{name} must be finite")
    loads, extra = _build_loads(spec, golf, golf_q, coords, time)
    metrics.update(extra)
    xml: Path | None = None
    artifacts: tuple[FileRecord, ...] = ()
    if loads is not None:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        xml = write_load_set(
            spec.output_dir, time, loads, f"{spec.stem}_external_loads"
        )
        sto = xml.with_suffix(".sto")
        artifacts = (
            FileRecord("external_loads_xml", xml.name, sha256_of(xml)),
            FileRecord("external_loads_sto", sto.name, sha256_of(sto)),
        )
    record = loads_record(
        spec.policy, loads, map_id=spec.fmap.map_id, supplied_name=spec.supplied_name
    )
    frozen = {k: np.array(v, dtype=float) for k, v in coords.items()}
    for series in frozen.values():
        series.setflags(write=False)
    return FullbodyResult(frozen, xml, artifacts, record, metrics)
