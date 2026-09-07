"""Joint fit of camera poses, 3-D joints and bone lengths (C3 + C4 core).

One robust least-squares problem over all frames of a take:

    residuals = [ w * (project(cam_c, X_t,k) - z_c,t,k) / sigma_px ]   (reprojection)
              + [ (|X_t,parent - X_t,k| - L_k) / sigma_bone ]         (rigid segments)
              + [ (L_k - L_k^prior) / sigma_prior ]                    (anthropometric prior)
              + [ (L_left - L_right) / sigma_sym ]                     (symmetry)

Unknowns: every camera's rotation (axis-angle) and position except the first
camera, which fixes the gauge; every joint in every frame; one length per
segment shared by all frames (the "learn the bone lengths" part). Solved with
SciPy's trust-region reflective least squares under a Huber loss so a gross
detection cannot pull the fit, with a sparse Jacobian pattern so takes of
thousands of frames stay tractable.

The fit is evidence: the result carries per-observation residuals, the
observations rejected by the residual gate after convergence (with their
residuals), the recovered lengths, and the camera records. Initialisation
comes from triangulating the cleaned detections with the *previous*
placement (or a rough guess), which is what "learn from the previous take"
means in practice. Ownership of this fitter is decided in #9630.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from src.shared.python.core.contracts import require

from .cameras import PinholeCamera
from .geometry import triangulate
from .skeleton import JOINT_NAMES, PARENTS, SYMMETRIC_PAIRS

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class BundleOptions:
    """Weights in physical units; all validated."""

    sigma_px: float = 1.0
    sigma_bone_m: float = 0.005  # how rigid a segment is allowed to look
    sigma_prior_m: float = 0.05  # spread of the anthropometric prior
    sigma_symmetry_m: float = 0.01
    # Scale gauge: with one camera fixed, a global scale about its centre leaves
    # every reprojection unchanged, so scale must come from a measured length
    # on the subject (segment name, metres) with a tight sigma; without it the
    # anthropometric prior is the only, weak, anchor.
    scale_anchor: tuple[str, float] | None = None
    # A measured length is treated as known: surviving outliers otherwise trade
    # the anchor away (2.7 sigma at 1 mm on the synthetic harness) and scale
    # drifts; 0.4 % of scale is already 1.7 cm at 4 m.
    sigma_anchor_m: float = 0.0002
    huber_delta: float = 3.0  # in sigma_px units
    gate_px: float = 6.0  # rejection gate after convergence
    max_iterations: int = 100  # solver iterations (each costs several evaluations)

    def __post_init__(self) -> None:
        for name in ("sigma_px", "sigma_bone_m", "sigma_prior_m", "sigma_symmetry_m"):
            require(getattr(self, name) > 0, f"{name} must be positive")
        require(self.huber_delta > 0 and self.gate_px > 0, "delta and gate positive")
        require(self.sigma_anchor_m > 0, "sigma_anchor_m must be positive")
        if self.scale_anchor is not None:
            require(self.scale_anchor[1] > 0, "scale anchor length must be positive")
        require(self.max_iterations >= 1, "max_iterations must be at least 1")


@dataclass(frozen=True)
class Observations:
    """Detections of one take: ``pixels[c, t, k]`` and ``confidence[c, t, k]``."""

    camera_ids: tuple[str, ...]
    pixels: Array  # (C, T, K, 2)
    confidence: Array  # (C, T, K)

    def __post_init__(self) -> None:
        c, t, k, two = self.pixels.shape
        require(two == 2 and self.confidence.shape == (c, t, k), "shape mismatch")
        require(len(self.camera_ids) == c, "one id per camera")
        require(c >= 2 and t >= 1 and k >= 1, "need >= 2 cameras, frames, joints")


@dataclass(frozen=True)
class RejectedObservation:
    camera_id: str
    frame: int
    joint: str
    residual_px: float


@dataclass(frozen=True)
class BundleResult:
    cameras: tuple[PinholeCamera, ...]
    joints_3d_m: Array  # (T, K, 3)
    bone_lengths_m: dict[str, float]
    residuals_px: Array  # (C, T, K), NaN where unobserved
    rejected: tuple[RejectedObservation, ...]
    rms_px: float
    iterations: int
    converged: bool
    initial_rms_px: float = field(default=float("nan"))
    unobservable_points: int = 0  # (frame, joint) pairs with < 2 surviving views


def _segments(joint_names: Sequence[str]) -> list[tuple[int, int, str]]:
    """``(child index, parent index, name)`` for every segment of the skeleton."""
    index = {n: i for i, n in enumerate(joint_names)}
    out = []
    for name in joint_names:
        parent = PARENTS.get(name)
        if parent is not None and parent in index:
            out.append((index[name], index[parent], name))
    return out


class _Problem:
    """Packs and unpacks the parameter vector and evaluates residuals."""

    def __init__(
        self,
        cams: Sequence[PinholeCamera],
        obs: Observations,
        joints0: Array,
        lengths0: Mapping[str, float],
        prior: Mapping[str, float],
        options: BundleOptions,
        joint_names: Sequence[str],
    ) -> None:
        self.cams, self.obs, self.options = list(cams), obs, options
        self.joint_names = list(joint_names)
        self.segments = _segments(self.joint_names)
        self.seg_index = {name: i for i, (_, _, name) in enumerate(self.segments)}
        self.prior = np.array([prior[name] for _, _, name in self.segments])
        self.n_cam, self.n_t, self.n_k = obs.confidence.shape
        self.mask: Mask = np.isfinite(obs.pixels).all(axis=3) & (obs.confidence > 0)
        self.weights: Array = np.sqrt(np.clip(obs.confidence, 0.0, 1.0)) * self.mask
        self.obs_index = np.argwhere(self.mask)  # rows of (c, t, k)
        self.sym = [
            (self.seg_index[a], self.seg_index[b])
            for a, b in SYMMETRIC_PAIRS
            if a in self.seg_index and b in self.seg_index
        ]
        self.seg_weights: Array = np.ones((self.n_t, len(self.segments)))
        self.anchor: tuple[int, float] | None = None
        if options.scale_anchor is not None:
            name, value = options.scale_anchor
            require(name in self.seg_index, "scale anchor must name a segment", name)
            self.anchor = (self.seg_index[name], float(value))
        self.x0 = self.pack(
            cams, joints0, np.array([lengths0[name] for _, _, name in self.segments])
        )

    # ----- parameter vector layout: [cam1..camN-1 (6 each)] [joints T*K*3] [lengths S]
    def pack(
        self, cams: Sequence[PinholeCamera], joints: Array, lengths: Array
    ) -> Array:
        parts = []
        for cam in cams[1:]:
            parts.append(
                Rotation.from_matrix(cam.rotation_world_from_camera).as_rotvec()
            )
            parts.append(cam.translation_world_from_camera_m)
        parts.append(np.asarray(joints, dtype=float).ravel())
        parts.append(np.asarray(lengths, dtype=float))
        return np.concatenate(parts)

    def unpack(self, x: Array) -> tuple[list[PinholeCamera], Array, Array]:
        cams = [self.cams[0]]
        pos = 0
        for cam in self.cams[1:]:
            rotvec, trans = x[pos : pos + 3], x[pos + 3 : pos + 6]
            pos += 6
            cams.append(
                PinholeCamera(
                    cam.camera_id,
                    cam.matrix,
                    Rotation.from_rotvec(rotvec).as_matrix(),
                    trans,
                    cam.image_size_px,
                )
            )
        n_j = self.n_t * self.n_k * 3
        joints = x[pos : pos + n_j].reshape(self.n_t, self.n_k, 3)
        lengths = x[pos + n_j :]
        return cams, joints, lengths

    def residuals(self, x: Array) -> Array:
        cams, joints, lengths = self.unpack(x)
        o = self.options
        out = []
        flat = joints.reshape(-1, 3)
        for c, cam in enumerate(cams):
            px, _ = cam.project(flat)
            px = px.reshape(self.n_t, self.n_k, 2)
            diff = (px - self.obs.pixels[c]) * self.weights[c][..., None] / o.sigma_px
            out.append(np.nan_to_num(diff[self.mask[c]]).ravel())
        for s, (child, parent, _) in enumerate(self.segments):
            seg = np.linalg.norm(joints[:, child] - joints[:, parent], axis=1)
            out.append(self.seg_weights[:, s] * (seg - lengths[s]) / o.sigma_bone_m)
        out.append((lengths - self.prior) / o.sigma_prior_m)
        for a, b in self.sym:
            out.append(np.atleast_1d((lengths[a] - lengths[b]) / o.sigma_symmetry_m))
        if self.anchor is not None:
            s, value = self.anchor
            out.append(np.atleast_1d((lengths[s] - value) / o.sigma_anchor_m))
        return np.concatenate(out)

    def _camera_columns(self, c: int) -> tuple[int, int]:
        """Parameter columns of camera ``c``'s rotvec (3) then position (3)."""
        return 6 * (c - 1), 6 * (c - 1) + 6

    def _joint_column(self, t: int, k: int) -> int:
        return 6 * (self.n_cam - 1) + (t * self.n_k + k) * 3

    def _length_column(self, s: int) -> int:
        return 6 * (self.n_cam - 1) + self.n_t * self.n_k * 3 + s

    def _reprojection_blocks(
        self, cams: list[PinholeCamera], joints: Array, x: Array
    ) -> tuple[list[int], list[int], list[float], int]:
        """COO entries of the reprojection rows: analytic in points and camera
        position, finite-difference in the three rotation parameters."""
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        flat = joints.reshape(-1, 3)
        row = 0
        eps = 1e-6
        for c, cam in enumerate(cams):
            idx = np.argwhere(self.mask[c])
            if idx.size == 0:
                continue
            r_cw = cam.rotation_world_from_camera.T
            fx, fy = cam.matrix[0, 0], cam.matrix[1, 1]
            pts = joints[idx[:, 0], idx[:, 1]]  # (M, 3)
            pc = (pts - cam.translation_world_from_camera_m) @ r_cw.T
            z = np.maximum(pc[:, 2], 1e-9)
            w = self.weights[c][idx[:, 0], idx[:, 1]] / self.options.sigma_px  # (M,)
            # d uv / d pc, per observation: (M, 2, 3)
            d_uv = np.zeros((len(idx), 2, 3))
            d_uv[:, 0, 0] = fx / z
            d_uv[:, 0, 2] = -fx * pc[:, 0] / z**2
            d_uv[:, 1, 1] = fy / z
            d_uv[:, 1, 2] = -fy * pc[:, 1] / z**2
            d_x = np.einsum("mij,jk->mik", d_uv, r_cw) * w[:, None, None]  # d/dX
            rot_cols: Array | None = None
            if c > 0:
                base_px, _ = cam.project(pts)
                rot_cols = np.zeros((len(idx), 2, 3))
                c0, _ = self._camera_columns(c)
                for a in range(3):
                    xp = x.copy()
                    xp[c0 + a] += eps
                    cam_p = self.unpack(xp)[0][c]
                    px_p, _ = cam_p.project(pts)
                    rot_cols[:, :, a] = (px_p - base_px) / eps * w[:, None]
            for m, (t_i, k_i) in enumerate(idx):
                j0 = self._joint_column(int(t_i), int(k_i))
                for r in range(2):
                    for a in range(3):
                        rows.append(row + r)
                        cols.append(j0 + a)
                        vals.append(float(d_x[m, r, a]))
                    if c > 0 and rot_cols is not None:
                        c0, _ = self._camera_columns(c)
                        for a in range(3):
                            rows.append(row + r)
                            cols.append(c0 + a)
                            vals.append(float(rot_cols[m, r, a]))
                            rows.append(row + r)
                            cols.append(c0 + 3 + a)
                            vals.append(float(-d_x[m, r, a]))
                row += 2
        return rows, cols, vals, row

    def jacobian(self, x: Array) -> csr_matrix:
        """Sparse Jacobian of :meth:`residuals` at ``x`` (same row order)."""
        cams, joints, lengths = self.unpack(x)
        o = self.options
        rows, cols, vals, row = self._reprojection_blocks(cams, joints, x)
        for s, (child, parent, _) in enumerate(self.segments):
            d = joints[:, child] - joints[:, parent]
            seg = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            unit = d / seg[:, None] / o.sigma_bone_m
            for t_i in range(self.n_t):
                sw = float(self.seg_weights[t_i, s])
                jc, jp = self._joint_column(t_i, child), self._joint_column(t_i, parent)
                for a in range(3):
                    rows += [row, row]
                    cols += [jc + a, jp + a]
                    vals += [sw * float(unit[t_i, a]), -sw * float(unit[t_i, a])]
                rows.append(row)
                cols.append(self._length_column(s))
                vals.append(-sw / o.sigma_bone_m)
                row += 1
        for s in range(len(self.segments)):
            rows.append(row)
            cols.append(self._length_column(s))
            vals.append(1.0 / o.sigma_prior_m)
            row += 1
        for a, b in self.sym:
            rows += [row, row]
            cols += [self._length_column(a), self._length_column(b)]
            vals += [1.0 / o.sigma_symmetry_m, -1.0 / o.sigma_symmetry_m]
            row += 1
        if self.anchor is not None:
            rows.append(row)
            cols.append(self._length_column(self.anchor[0]))
            vals.append(1.0 / o.sigma_anchor_m)
            row += 1
        n_params = self._length_column(len(self.segments))
        return csr_matrix((vals, (rows, cols)), shape=(row, n_params))


def initial_joints(
    cams: Sequence[PinholeCamera], obs: Observations, *, gate_px: float
) -> Array:
    """Triangulate every joint in every frame with the current placement.

    Joints seen by fewer than two views (or rejected everywhere) start at the
    mean of the frame's triangulated joints, so the fit has a finite start.
    """
    c, t, k = obs.confidence.shape
    joints = np.full((t, k, 3), np.nan)
    for ti in range(t):
        for ki in range(k):
            res = triangulate(
                cams, obs.pixels[:, ti, ki], obs.confidence[:, ti, ki], gate_px=gate_px
            )
            if res.ok:
                joints[ti, ki] = res.point_m
        frame_mean = (
            np.nanmean(joints[ti], axis=0) if np.isfinite(joints[ti]).any() else 0.0
        )
        joints[ti] = np.where(np.isfinite(joints[ti]), joints[ti], frame_mean)
    return np.nan_to_num(joints)


def _huber_weight(u: Array, delta: float) -> Array:
    a = np.abs(u)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def _residual_map(
    cams: Sequence[PinholeCamera], joints: Array, obs: Observations, mask: Mask
) -> Array:
    """Reprojection residual per observation ``(C, T, K)``; NaN where unobserved."""
    out = np.full(obs.confidence.shape, np.nan)
    flat = joints.reshape(-1, 3)
    n_t, n_k = joints.shape[:2]
    for c, cam in enumerate(cams):
        px, _ = cam.project(flat)
        d = np.linalg.norm(px.reshape(n_t, n_k, 2) - obs.pixels[c], axis=2)
        out[c] = np.where(mask[c], d, np.nan)
    return out


def _leave_one_out_rejections(
    cams: Sequence[PinholeCamera],
    obs: Observations,
    residual_px: Array,
    options: BundleOptions,
) -> Mask:
    """Mask ``(C, T, K)`` of observations a per-point leave-one-out test rejects."""
    reject = np.zeros(obs.confidence.shape, dtype=bool)
    worst = np.nanmax(np.where(np.isfinite(residual_px), residual_px, -1.0), axis=0)
    for t, k in np.argwhere(worst > options.gate_px):
        point = triangulate(
            cams, obs.pixels[:, t, k], obs.confidence[:, t, k], gate_px=options.gate_px
        )
        for cid in point.rejected:
            c = obs.camera_ids.index(cid)
            if obs.confidence[c, t, k] > 0:
                reject[c, t, k] = True
    return reject


def _solve(
    problem: _Problem, x_start: Array, options: BundleOptions
) -> tuple[Array, int]:
    fit = least_squares(
        problem.residuals,
        x_start,
        jac=problem.jacobian,
        method="trf",
        max_nfev=options.max_iterations,
        ftol=1e-8,
        xtol=1e-8,
        x_scale="jac",
    )
    return np.asarray(fit.x, dtype=float), int(fit.nfev)


def _segment_units(problem: _Problem, x: Array, options: BundleOptions) -> Array:
    """Rigid-segment residuals in sigma_bone units, ``(T, S)``."""
    _, joints, lengths = problem.unpack(x)
    cols = []
    for s, (child, parent, _) in enumerate(problem.segments):
        seg = np.linalg.norm(joints[:, child] - joints[:, parent], axis=1)
        cols.append((seg - lengths[s]) / options.sigma_bone_m)
    return np.column_stack(cols)


def _robust_fit(
    problem: _Problem, obs: Observations, options: BundleOptions
) -> tuple[Array, int]:
    """Iteratively reweighted least squares, robust on reprojection and segments.

    SciPy's ``loss=`` would also flatten the anchor and rigid-segment
    residuals, which must stay exact, so the Huber weights are applied to the
    reprojection and segment terms only. Stage widths shrink to huber_delta.
    """
    x = problem.x0
    total = 0
    base = problem.weights.copy()
    widths = (
        None,
        20.0 * options.huber_delta,
        options.huber_delta,
        options.huber_delta,
    )
    for delta in widths:
        if delta is not None:
            cams, joints, _ = problem.unpack(x)
            res = _residual_map(cams, joints, obs, problem.mask)
            u = np.nan_to_num(res / options.sigma_px, nan=0.0)
            problem.weights = base * np.sqrt(_huber_weight(u, delta))
            problem.seg_weights = np.sqrt(
                _huber_weight(_segment_units(problem, x, options), delta)
            )
        x, nfev = _solve(problem, x, options)
        total += nfev
    problem.weights = base
    return x, total


def _reject_and_refit(
    problem: _Problem, obs: Observations, options: BundleOptions, x: Array
) -> tuple[Array, Mask, int]:
    """Leave-one-out rejection per point, drop unobservable points, refit once.

    A gross detection drags its 3-D point, so the innocent views of the same
    joint exceed the gate too; the triangulation rule assigns blame. A point
    left with a single view is unobservable, and keeping that view would let
    a wrong detection bend the rigid-segment terms, so it is dropped as well.
    """
    cams, joints, _ = problem.unpack(x)
    residual_px = _residual_map(cams, joints, obs, problem.mask)
    reject = _leave_one_out_rejections(cams, obs, residual_px, options)
    survivors = problem.mask & ~reject
    lonely = survivors.sum(axis=0) < 2
    reject |= survivors & lonely[None, :, :]
    if not reject.any():
        return x, reject, 0
    problem.weights = np.where(reject, 0.0, problem.weights)
    problem.mask = problem.mask & ~reject
    x, nfev = _solve(problem, x, options)
    return x, reject, nfev


def bundle_adjust(
    cams: Sequence[PinholeCamera],
    obs: Observations,
    *,
    length_prior_m: Mapping[str, float],
    options: BundleOptions = BundleOptions(),
    joint_names: Sequence[str] = JOINT_NAMES,
    joints0: Array | None = None,
) -> BundleResult:
    """Refine cameras (all but the first), joints and bone lengths jointly.

    Preconditions: the skeleton segments of ``joint_names`` are all in
    ``length_prior_m``; observations align with ``cams``. Postcondition:
    the returned cameras keep the first camera exactly (gauge), every segment
    length is positive, and ``rejected`` lists every observation whose final
    reprojection residual exceeds ``options.gate_px``.
    """
    require(len(cams) == len(obs.camera_ids), "cams must align with observations")
    require(
        all(c.camera_id == i for c, i in zip(cams, obs.camera_ids, strict=True)),
        "camera order",
    )
    names = list(joint_names)
    require(obs.confidence.shape[2] == len(names), "joint count mismatch")
    segments = _segments(names)
    missing = [n for _, _, n in segments if n not in length_prior_m]
    require(not missing, "length prior missing segments", missing)
    start = (
        initial_joints(cams, obs, gate_px=options.gate_px)
        if joints0 is None
        else joints0
    )
    problem = _Problem(cams, obs, start, length_prior_m, length_prior_m, options, names)
    r0 = problem.residuals(problem.x0)
    n_obs = 2 * int(problem.mask.sum())
    initial_rms = float(np.sqrt(np.mean(r0[:n_obs] ** 2))) * options.sigma_px
    x, total_nfev = _robust_fit(problem, obs, options)
    x, reject_mask, refit_nfev = _reject_and_refit(problem, obs, options, x)
    total_nfev += refit_nfev
    cams_out, joints, lengths = problem.unpack(x)
    residual_px = _residual_map(cams_out, joints, obs, problem.mask | reject_mask)
    over_gate = np.nan_to_num(residual_px, nan=0.0) > options.gate_px
    flagged = reject_mask | over_gate
    rejected = tuple(
        RejectedObservation(
            obs.camera_ids[c], int(t), names[k], float(residual_px[c, t, k])
        )
        for c, t, k in np.argwhere(flagged)
    )
    kept = residual_px[np.isfinite(residual_px) & ~flagged]
    return BundleResult(
        cameras=tuple(cams_out),
        joints_3d_m=joints,
        bone_lengths_m={
            name: float(lengths[i]) for i, (_, _, name) in enumerate(segments)
        },
        residuals_px=residual_px,
        rejected=rejected,
        rms_px=float(np.sqrt(np.mean(kept**2))) if kept.size else float("nan"),
        iterations=total_nfev,
        converged=bool(np.isfinite(x).all()),
        initial_rms_px=initial_rms,
        unobservable_points=int(
            ((problem.mask | reject_mask) & ~flagged).sum(axis=0).__lt__(2).sum()
        ),
    )


def observations_from_views(
    views: Mapping[str, Mapping[str, Any]], camera_ids: Sequence[str]
) -> Observations:
    """Stack ``view-observations`` payloads (same layout, same frame count)."""
    require(len(camera_ids) >= 2, "need at least two views")
    first = views[camera_ids[0]]
    names = list(first["detector_layout"]["keypoint_names"])
    total = int(first["frames_total"])
    px = np.full((len(camera_ids), total, len(names), 2), np.nan)
    conf = np.zeros((len(camera_ids), total, len(names)))
    for c, cid in enumerate(camera_ids):
        payload = views[cid]
        require(
            list(payload["detector_layout"]["keypoint_names"]) == names,
            "views must share one detector layout",
        )
        fps = float(payload["fps"])
        for row in payload["frames"]:
            t = int(round(float(row["time_s"]) * fps))
            if 0 <= t < total:
                px[c, t] = np.asarray(row["keypoints_px"], dtype=float)
                conf[c, t] = np.asarray(row["confidence"], dtype=float)
    return Observations(tuple(camera_ids), px, conf)
