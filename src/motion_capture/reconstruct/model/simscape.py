"""Validate joint-axis conventions against logged GolfSwing3D_Kinetic data (#9714).

The dataset generator writes, per time step, every body's Simscape joint
angles (``<Body>Logs_AngularPosition*``, degrees), its world rotation matrix
(``<Body>Logs_Rotation_Transform_Iij``) and its world position. For a joint
between parent body P and child body C the logged relative rotation must be

    R_P^T R_C = A · R_joint(q) · B

with ``R_joint`` the joint's primitive sequence of its logged angles and
``A``, ``B`` constant frame offsets (the model's fixed rigid transforms). We
try every signed axis order for the joint's primitives, solve ``A`` and
``B`` by alternating rotation Procrustes, and keep the hypothesis whose
residual is smallest. A residual near zero across hundreds of frames is a
proof of the convention; a large one means the hypothesis set does not
contain the model's construction and says so. Segment offsets in the parent
body frame are recovered the same way from the positions.

Nothing here runs MATLAB: the evidence is the model's own logs.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from src.shared.python.core.contracts import require

Array = npt.NDArray[np.float64]

#: Joint -> (parent body, child body, candidate angle-column sets). A logged
#: frame pair may span more than one Simscape joint (the Spine sensor sits
#: before the spine universal; the forearm sensor is past the pronation
#: joint), so every candidate set is tried and the best-fitting one is kept.
SCAP_L = ("LScapLogs_AngularPositionX", "LScapLogs_AngularPositionY")
SCAP_R = ("RScapLogs_AngularPositionX", "RScapLogs_AngularPositionY")
SPINE = ("SpineLogs_AngularPositionX", "SpineLogs_AngularPositionY")
TORSO = "TorsoLogs_AngularPosition"
LE, RE = (
    "AngularKinematicsLogs_LEAngularPosition",
    "AngularKinematicsLogs_REAngularPosition",
)
LF, RF = "LFLogs_AngularPosition", "RFLogs_AngularPosition"
LS = tuple(f"LSLogs_AngularPosition{a}" for a in ("X", "Y", "_Z"))
RS = tuple(f"RSLogs_AngularPosition{a}" for a in ("X", "Y", "_Z"))
HIP = tuple(f"HipLogs_HipAngularPosition{a}" for a in "XYZ")
JOINTS: dict[str, tuple[str, str, tuple[tuple[str, ...], ...]]] = {
    # World-referenced: the lower trunk ("TorsoLogs" sensor) carries the
    # hip's three rotations and the torso revolute; 4-angle sets are tried in
    # the two listed column orders only (see identify_joint).
    "hip_torso": ("World", "Torso", ((*HIP, TORSO), (TORSO, *HIP))),
    # The relative rotation between the two trunk sensors is the spine
    # universal: "SpineLogs" is the upper trunk, rigid with the hub.
    "spine": ("Torso", "Spine", (SPINE,)),
    "left_scapula": ("Spine", "LScap", (SCAP_L,)),
    "right_scapula": ("Spine", "RScap", (SCAP_R,)),
    "left_shoulder": ("LScap", "LS", (LS,)),
    "right_shoulder": ("RScap", "RS", (RS,)),
    # The forearm sensor sits after the elbow hinge and the pro/supination
    # revolute; the two angles together explain LS -> LF.
    "left_elbow": ("LS", "LF", ((LE, LF),)),
    "right_elbow": ("RS", "RF", ((RE, RF),)),
}
BODIES = ("Spine", "Torso", "LScap", "RScap", "LS", "RS", "LF", "RF")
AXES = "xyz"
CONVERGED_RAD = 1e-4  # residual below which a hypothesis is accepted outright


@dataclass(frozen=True)
class Frames:
    """Logged bodies over all frames: rotations ``(T,3,3)`` and positions ``(T,3)``."""

    rotation: Mapping[str, Array]
    position: Mapping[str, Array]
    angles_deg: Mapping[str, Array]  # column name -> (T,)
    frames: int


def _matrix(row: Mapping[str, str], body: str) -> Array:
    m = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            m[i, j] = float(row[f"{body}Logs_Rotation_Transform_I{i + 1}{j + 1}"])
    return m


def load_trials(csv_files: Iterable[Path]) -> Frames:
    """Stack every row of every trial CSV. Precondition: at least one file."""
    files = list(csv_files)
    require(bool(files), "no trial CSV files")
    rot: dict[str, list[Array]] = {b: [] for b in BODIES}
    pos: dict[str, list[Array]] = {b: [] for b in BODIES}
    angles: dict[str, list[float]] = {}
    columns = sorted(
        {c for _, _, cands in JOINTS.values() for cols in cands for c in cols}
    )
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for body in BODIES:
                    rot[body].append(_matrix(row, body))
                    pos[body].append(
                        np.array(
                            [
                                float(row[f"{body}Logs_GlobalPosition_{k}"])
                                for k in (1, 2, 3)
                            ]
                        )
                    )
                for col in columns:
                    angles.setdefault(col, []).append(float(row[col]))
    frames = len(next(iter(rot.values())))
    return Frames(
        rotation={b: np.stack(v) for b, v in rot.items()},
        position={b: np.stack(v) for b, v in pos.items()},
        angles_deg={k: np.asarray(v) for k, v in angles.items()},
        frames=frames,
    )


def _primitive(angles_deg: Array, order: Sequence[str], signs: Sequence[int]) -> Array:
    """Intrinsic rotation ``(T,3,3)`` for the ordered, signed primitives.

    Any number of primitives (a 4-angle hip+torso span is allowed), composed
    as a product of single-axis rotations, first primitive outermost.
    """
    a = np.radians(angles_deg) * np.asarray(signs)
    out = np.broadcast_to(np.eye(3), (a.shape[0], 3, 3)).copy()
    for k, axis in enumerate(order):
        r = Rotation.from_euler(axis.upper(), a[:, k : k + 1]).as_matrix()
        out = np.einsum("tij,tjk->tik", out, r)
    return out


def _axis_orders(k: int) -> list[tuple[str, ...]]:
    """Distinct axes for up to three primitives; for four, hip xyz orders + one."""
    if k <= 3:
        return list(itertools.permutations(AXES, k))
    return [(*hip, last) for hip in itertools.permutations(AXES, 3) for last in AXES]


def _mean_rotation(mats: Array) -> Array:
    u, _, vt = np.linalg.svd(mats.sum(axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _spread_rad(mats: Array, mean: Array) -> float:
    rel = np.einsum("ij,tjk->tik", mean.T, mats)
    return float(np.mean(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1)))


def _als(
    r_rel: Array, r_joint: Array, b: Array, iterations: int
) -> tuple[Array, Array, float]:
    a = np.eye(3)
    for _ in range(iterations):
        a = _mean_rotation(
            np.einsum("tij,tkj->tik", r_rel, np.einsum("tij,jk->tik", r_joint, b))
        )
        b = _mean_rotation(
            np.einsum("tji,tjk->tik", np.einsum("ij,tjk->tik", a, r_joint), r_rel)
        )
    fit = np.einsum("ij,tjk,kl->til", a, r_joint, b)
    return a, b, _spread_rad(np.einsum("tji,tjk->tik", fit, r_rel), np.eye(3))


def _solve_frames(
    r_rel: Array, r_joint: Array, iterations: int = 20
) -> tuple[Array, Array, float]:
    """``A``, ``B`` constant with ``r_rel ~ A r_joint B``; returns residual in radians.

    Alternating Procrustes is not convex in ``(A, B)``; it is restarted from
    the identity and from the 24 axis-aligned rotations as post-frame seeds
    and the best is kept. The sign of the residual decides, not the seed.
    """
    seeds = [np.eye(3), *_axis_aligned_rotations()]
    best: tuple[Array, Array, float] | None = None
    for seed in seeds:
        result = _als(r_rel, r_joint, seed, iterations)
        if best is None or result[2] < best[2]:
            best = result
        if best[2] < CONVERGED_RAD:  # an exact convention needs no more seeds
            break
    assert best is not None
    return best


def _axis_aligned_rotations() -> list[Array]:
    """The 24 proper rotations that permute and sign-flip the axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs, strict=True)):
                m[row, col] = s
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


@dataclass(frozen=True)
class JointConvention:
    joint: str
    order: str  # e.g. "xy", "xyz", "z"
    signs: tuple[int, ...]
    pre_frame_rotvec: tuple[float, ...]  # A
    post_frame_rotvec: tuple[float, ...]  # B
    residual_rad: float
    offset_parent_frame_m: tuple[float, ...]  # child origin in parent frame
    offset_spread_m: float
    frames: int
    reference: str = "parent"  # what the logged rotation was compared against
    angle_columns: tuple[str, ...] = ()

    @property
    def sequence(self) -> str:
        return " ".join(
            f"{a}{'+' if s > 0 else '-'}"
            for a, s in zip(self.order, self.signs, strict=True)
        )


def identify_joint(frames: Frames, joint: str) -> JointConvention:
    """Best signed axis order for ``joint``; precondition: a known joint name."""
    require(joint in JOINTS, "unknown joint", joint)
    parent, child, candidates = JOINTS[joint]
    # Two readings of the sensor: the child's rotation relative to the parent
    # body (world-referenced sensors) or the joint's own follower-to-base
    # rotation (sensors inside the joint subsystem). Both are tried.
    references = {"sensor": frames.rotation[child]}
    if parent != "World":
        references["parent"] = np.einsum(
            "tji,tjk->tik", frames.rotation[parent], frames.rotation[child]
        )
    best: tuple[float, str, tuple[int, ...], Array, Array, str, tuple[str, ...]] | None
    best = None
    ordered = {
        perm
        for cand in candidates
        for perm in (itertools.permutations(cand) if len(cand) <= 3 else (cand,))
    }
    for cols in sorted(ordered):
        k = len(cols)
        angles = np.column_stack([frames.angles_deg[c] for c in cols])
        for reference, r_rel in references.items():
            for order in _axis_orders(k):
                for signs in itertools.product((1, -1), repeat=k):
                    r_joint = _primitive(angles, order, signs)
                    a, b, res = _solve_frames(r_rel, r_joint)
                    if best is None or res < best[0]:
                        best = (res, "".join(order), signs, a, b, reference, cols)
    assert best is not None
    res, order_str, signs, a, b, reference, cols = best
    if parent == "World":
        d = frames.position[child]
    else:
        d = np.einsum(
            "tji,tj->ti",
            frames.rotation[parent],
            frames.position[child] - frames.position[parent],
        )
    return JointConvention(
        joint=joint,
        order=order_str,
        signs=tuple(int(s) for s in signs),
        pre_frame_rotvec=tuple(float(v) for v in Rotation.from_matrix(a).as_rotvec()),
        post_frame_rotvec=tuple(float(v) for v in Rotation.from_matrix(b).as_rotvec()),
        residual_rad=res,
        offset_parent_frame_m=tuple(float(v) for v in d.mean(axis=0)),
        offset_spread_m=float(np.linalg.norm(d - d.mean(axis=0), axis=1).mean()),
        frames=frames.frames,
        reference=reference,
        angle_columns=tuple(cols),
    )


def validate(csv_files: Iterable[Path]) -> dict[str, Any]:
    """Every joint's convention with residuals, as a JSON-ready record."""
    frames = load_trials(csv_files)
    results = {name: identify_joint(frames, name) for name in JOINTS}
    return {
        "schema_version": "simscape-axis-validation/1.0.0",
        "frames": frames.frames,
        "joints": {
            name: {
                "parent": JOINTS[name][0],
                "child": JOINTS[name][1],
                "sequence": c.sequence,
                "reference": c.reference,
                "angle_columns": list(c.angle_columns),
                "order": c.order,
                "signs": list(c.signs),
                "pre_frame_rotvec": list(c.pre_frame_rotvec),
                "post_frame_rotvec": list(c.post_frame_rotvec),
                "residual_rad": c.residual_rad,
                "offset_parent_frame_m": list(c.offset_parent_frame_m),
                "offset_spread_m": c.offset_spread_m,
            }
            for name, c in results.items()
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "| joint | parent → child | reference | primitives | residual rad | pre-frame rotvec | post-frame rotvec | offset in parent frame (m) | offset spread (m) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, j in report["joints"].items():
        pre = ", ".join(f"{v:+.3f}" for v in j["pre_frame_rotvec"])
        post = ", ".join(f"{v:+.3f}" for v in j["post_frame_rotvec"])
        off = ", ".join(f"{v:+.4f}" for v in j["offset_parent_frame_m"])
        lines.append(
            f"| {name} | {j['parent']} → {j['child']} | {j['reference']} | {j['sequence']} | {j['residual_rad']:.2e} | "
            f"{pre} | {post} | {off} | {j['offset_spread_m']:.2e} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", type=Path, required=True, help="Dataset Generator dir"
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON output")
    args = parser.parse_args(argv)
    files = sorted(args.datasets.glob("golf_swing_dataset_*/trial_*.csv"))
    report = validate(files)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
