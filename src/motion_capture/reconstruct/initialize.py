"""Camera placement from the golfer alone: no calibration object (C3, #9623).

The first take of a new setup has no extrinsics. Confident joint detections
seen by two cameras in the same frame are correspondences; with known
intrinsics (#9622) the essential matrix, found under RANSAC, gives each
camera's rotation and the *direction* of its position relative to the first
camera. Scale comes from the subject: the anchor segment (a measured length)
is triangulated with the unit baseline and the baseline is rescaled to match.
Orientation and origin come from the subject too: the average hip-to-neck
direction at the start of the take is "up", and the mid-hip at the first
frame is the origin — the ADR-0041 world frame minus the target direction,
which the ball line (C1) adds later.

Every pairwise solution reports its inlier count; a pair below the minimum is
refused rather than guessed. The result is a *start* for the joint fit, which
then refines everything with the rigid-segment and symmetry constraints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

from .bundle import Observations
from .cameras import PinholeCamera
from .geometry import triangulate
from .skeleton import JOINT_NAMES, PARENTS

Array = npt.NDArray[np.float64]
MIN_INLIERS = 40
UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class PairSolution:
    camera_id: str
    correspondences: int
    inliers: int
    scale_from_anchor: float


@dataclass(frozen=True)
class Initialization:
    cameras: tuple[PinholeCamera, ...]
    pairs: tuple[PairSolution, ...]

    @property
    def ok(self) -> bool:
        return all(p.inliers >= MIN_INLIERS for p in self.pairs)


def _correspondences(
    obs: Observations, a: int, b: int, min_confidence: float
) -> tuple[Array, Array, npt.NDArray[np.intp]]:
    """Pixel pairs seen confidently by cameras ``a`` and ``b`` in the same frame."""
    both = (obs.confidence[a] >= min_confidence) & (obs.confidence[b] >= min_confidence)
    both &= np.isfinite(obs.pixels[a]).all(axis=2) & np.isfinite(obs.pixels[b]).all(
        axis=2
    )
    idx = np.argwhere(both)
    return obs.pixels[a][both], obs.pixels[b][both], idx


def relative_pose(
    px_ref: Array, px_other: Array, k_ref: Array, k_other: Array
) -> tuple[Array, Array, int]:
    """``(R, t_unit, inliers)`` of the other camera relative to the reference.

    Returns ``T_ref_from_other``: rotation and unit translation expressing the
    other camera's frame in the reference camera's frame. Precondition: at
    least eight correspondences.
    """
    import cv2

    require(px_ref.shape[0] >= 8, "need at least eight correspondences")

    # Normalise both sets with their own intrinsics so one K suffices for cv2.
    def normalise(px: Array, k: Array) -> Array:
        hom = np.column_stack([px, np.ones(px.shape[0])])
        n = hom @ np.linalg.inv(k).T
        return np.ascontiguousarray(n[:, :2], dtype=np.float64)

    n_ref, n_other = normalise(px_ref, k_ref), normalise(px_other, k_other)
    e, mask = cv2.findEssentialMat(
        n_other, n_ref, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.5e-3
    )
    require(e is not None and e.shape == (3, 3), "essential matrix not found")
    _, r, t, pose_mask = cv2.recoverPose(e, n_other, n_ref, np.eye(3), mask=mask)
    inliers = int(np.count_nonzero(pose_mask))
    # cv2: x_ref = R x_other + t  ->  T_ref_from_other = (R, t)
    return np.asarray(r, dtype=float), np.asarray(t, dtype=float).reshape(3), inliers


def _anchor_scale(
    cams: Sequence[PinholeCamera],
    obs: Observations,
    pair: tuple[int, int],
    anchor: tuple[str, float],
    joint_names: Sequence[str],
) -> float:
    """Ratio measured-length / triangulated-length of the anchor over the take."""
    child, length = anchor
    parent = PARENTS.get(child)
    require(parent is not None, "anchor must be a segment (a joint with a parent)")
    assert parent is not None
    names = list(joint_names)
    kc, kp = names.index(child), names.index(parent)
    sub = [cams[pair[0]], cams[pair[1]]]
    measured = []
    for t in range(obs.confidence.shape[1]):
        pts = []
        for k in (kc, kp):
            res = triangulate(
                sub, obs.pixels[list(pair), t, k], obs.confidence[list(pair), t, k]
            )
            if not res.ok:
                break
            pts.append(res.point_m)
        if len(pts) == 2:
            measured.append(float(np.linalg.norm(pts[0] - pts[1])))
    require(len(measured) >= 5, "anchor segment seen in too few frames", len(measured))
    return length / float(np.median(measured))


def initialize_cameras(
    obs: Observations,
    intrinsics: Sequence[Array],
    image_sizes: Sequence[tuple[int, int]],
    *,
    anchor: tuple[str, float],
    joint_names: Sequence[str] = JOINT_NAMES,
    min_confidence: float = 0.5,
) -> Initialization:
    """Place every camera relative to the first from the joints alone.

    Preconditions: one K and image size per camera; the anchor names a
    segment. Postcondition: the first camera sits at the origin of an
    intermediate frame; :func:`subject_frame` re-expresses everything in the
    subject-defined world frame.
    """
    require(
        len(intrinsics) == len(obs.camera_ids) == len(image_sizes), "one K per camera"
    )
    ref = PinholeCamera(
        obs.camera_ids[0], intrinsics[0], np.eye(3), np.zeros(3), image_sizes[0]
    )
    cams: list[PinholeCamera] = [ref]
    pairs: list[PairSolution] = []
    for c in range(1, len(obs.camera_ids)):
        px_ref, px_c, _ = _correspondences(obs, 0, c, min_confidence)
        r, t_unit, inliers = relative_pose(px_ref, px_c, intrinsics[0], intrinsics[c])
        unit_cam = PinholeCamera(
            obs.camera_ids[c], intrinsics[c], r, t_unit, image_sizes[c]
        )
        scale = _anchor_scale([ref, unit_cam], obs, (0, 1), anchor, joint_names)
        cams.append(
            PinholeCamera(
                obs.camera_ids[c], intrinsics[c], r, t_unit * scale, image_sizes[c]
            )
        )
        pairs.append(
            PairSolution(obs.camera_ids[c], int(px_ref.shape[0]), inliers, scale)
        )
    return Initialization(cameras=tuple(cams), pairs=tuple(pairs))


def subject_frame(
    cams: Sequence[PinholeCamera],
    obs: Observations,
    *,
    joint_names: Sequence[str] = JOINT_NAMES,
    frames: int = 10,
) -> tuple[PinholeCamera, ...]:
    """Re-express the cameras so hip-to-neck is up and the first mid-hip is the origin.

    Uses the first ``frames`` frames (address). Postcondition: the returned
    cameras keep all relative geometry; only the world frame changes.
    """
    names = list(joint_names)
    hip, neck = names.index("mid_hip"), names.index("neck")
    hips, necks = [], []
    for t in range(min(frames, obs.confidence.shape[1])):
        h = triangulate(cams, obs.pixels[:, t, hip], obs.confidence[:, t, hip])
        n = triangulate(cams, obs.pixels[:, t, neck], obs.confidence[:, t, neck])
        if h.ok and n.ok:
            hips.append(h.point_m)
            necks.append(n.point_m)
    require(bool(hips), "mid_hip and neck must be triangulable at address")
    up = np.mean(np.array(necks) - np.array(hips), axis=0)
    up /= np.linalg.norm(up)
    origin = np.array(hips[0])
    # Rotation taking `up` to +y; the remaining yaw is free until the ball line.
    axis = np.cross(up, UP)
    s, c = float(np.linalg.norm(axis)), float(up @ UP)
    if s < 1e-9:
        r_new_from_old = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = axis / s
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        r_new_from_old = np.eye(3) + s * kx + (1 - c) * (kx @ kx)
    out = []
    for cam in cams:
        out.append(
            PinholeCamera(
                cam.camera_id,
                cam.matrix,
                r_new_from_old @ cam.rotation_world_from_camera,
                r_new_from_old @ (cam.position_m - origin),
                cam.image_size_px,
            )
        )
    return tuple(out)
