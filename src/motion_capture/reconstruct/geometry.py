"""Multi-view geometry for the consumer-side fitter: triangulation with residuals.

Direct linear transform (DLT) triangulation of one point from N calibrated
views, weighted by detector confidence, followed by the reprojection residual
in every contributing view and a first-order covariance of the point. Every
result says which views contributed and which were rejected by the residual
gate, in line with ADR-0041's "record contributing and rejected cameras".

This module exists so the fitter can be tested; whether the reference
implementation ultimately ships from Tools (#9630) does not change the
contract expressed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import PreconditionError, require

from .cameras import PinholeCamera

Array: TypeAlias = npt.NDArray[np.float64]
MIN_VIEWS = 2


@dataclass(frozen=True)
class TriangulatedPoint:
    """One 3-D point with its evidence."""

    point_m: Array  # (3,)
    residuals_px: dict[str, float]  # camera_id -> reprojection residual
    contributing: tuple[str, ...]
    rejected: tuple[str, ...]
    covariance_m2: Array | None  # (3, 3) first-order, None when < 2 views

    @property
    def ok(self) -> bool:
        return len(self.contributing) >= MIN_VIEWS

    @property
    def rms_px(self) -> float | None:
        if not self.contributing:
            return None
        r = np.array([self.residuals_px[c] for c in self.contributing])
        return float(np.sqrt(np.mean(r**2)))


def projection_matrix(cam: PinholeCamera) -> Array:
    """``3x4`` ``P = K [R_cw | t_cw]`` mapping world points to pixels."""
    r_cw = cam.rotation_world_from_camera.T
    t_cw = -r_cw @ cam.translation_world_from_camera_m
    return cam.matrix @ np.hstack([r_cw, t_cw[:, None]])


def dlt(
    cams: Sequence[PinholeCamera], pixels: Array, weights: Array | None = None
) -> Array:
    """Weighted DLT solution for one point seen at ``pixels`` ``(N, 2)``."""
    require(len(cams) >= MIN_VIEWS, "triangulation needs at least two views")
    px = np.asarray(pixels, dtype=float).reshape(-1, 2)
    require(px.shape[0] == len(cams), "one pixel per camera", px.shape)
    w = np.ones(len(cams)) if weights is None else np.asarray(weights, dtype=float)
    rows = []
    for cam, (u, v), wi in zip(cams, px, w, strict=True):
        p = projection_matrix(cam)
        rows.append(np.sqrt(max(wi, 1e-9)) * (u * p[2] - p[0]))
        rows.append(np.sqrt(max(wi, 1e-9)) * (v * p[2] - p[1]))
    a = np.stack(rows)
    _, _, vt = np.linalg.svd(a)
    x = vt[-1]
    require(abs(x[3]) > 1e-12, "point at infinity")
    return x[:3] / x[3]


def reprojection_residuals(
    cams: Sequence[PinholeCamera], point_m: Array, pixels: Array
) -> Array:
    """Euclidean pixel residual of ``point_m`` in every camera (NaN if behind)."""
    out = np.full(len(cams), np.nan)
    for i, cam in enumerate(cams):
        proj, in_front = cam.project(point_m[None, :])
        if in_front[0]:
            out[i] = float(np.linalg.norm(proj[0] - pixels[i]))
    return out


def _covariance(
    cams: Sequence[PinholeCamera], point_m: Array, sigma_px: float
) -> Array:
    """First-order covariance from the stacked projection Jacobians."""
    jac = []
    for cam in cams:
        pc = cam.camera_from_world(point_m[None, :])[0]
        z = max(pc[2], 1e-9)
        fx, fy = cam.matrix[0, 0], cam.matrix[1, 1]
        d_uv_d_pc = np.array(
            [[fx / z, 0.0, -fx * pc[0] / z**2], [0.0, fy / z, -fy * pc[1] / z**2]]
        )
        jac.append(d_uv_d_pc @ cam.rotation_world_from_camera.T)
    j = np.vstack(jac)
    jtj = j.T @ j
    return np.asarray(sigma_px**2 * np.linalg.pinv(jtj), dtype=float)


def _is_degenerate_geometry(cams: Sequence[PinholeCamera], pixels: Array) -> bool:
    """Detect coincident cameras or collinear viewing rays before DLT."""
    n = len(cams)
    if n < MIN_VIEWS:
        return True
    c0 = cams[0].position_m
    if all(float(np.linalg.norm(c.position_m - c0)) < 1e-5 for c in cams[1:]):
        return True
    if n == 2:
        c1, c2 = cams[0], cams[1]
        b = c2.position_m - c1.position_m
        dist = float(np.linalg.norm(b))
        if dist < 1e-5:
            return True
        b_hat = b / dist
        px = np.asarray(pixels, dtype=float).reshape(-1, 2)
        rays = []
        for cam, (u, v) in zip(cams, px, strict=True):
            fx, fy = cam.matrix[0, 0], cam.matrix[1, 1]
            cx, cy = cam.matrix[0, 2], cam.matrix[1, 2]
            v_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            r_world = cam.rotation_world_from_camera @ v_cam
            r_norm = float(np.linalg.norm(r_world))
            if r_norm < 1e-9:
                return True
            rays.append(r_world / r_norm)
        cos_rays = abs(float(np.dot(rays[0], rays[1])))
        if cos_rays > 1.0 - 1e-5:
            return True
        cos_b0 = abs(float(np.dot(rays[0], b_hat)))
        cos_b1 = abs(float(np.dot(rays[1], b_hat)))
        if cos_b0 > 1.0 - 1e-5 and cos_b1 > 1.0 - 1e-5:
            return True
    return False


def triangulate(
    cams: Sequence[PinholeCamera],
    pixels: Array,
    confidence: Array | None = None,
    *,
    gate_px: float = 8.0,
    sigma_px: float = 1.0,
) -> TriangulatedPoint:
    """Triangulate one point, reject views beyond ``gate_px``, and re-solve.

    Preconditions: at least two cameras, one pixel per camera; confidences in
    [0, 1] when given (a view with confidence 0 is not used). Postcondition:
    ``contributing`` and ``rejected`` partition the views that were used, and
    ``ok`` is false when fewer than two views survive.
    """
    require(len(cams) >= MIN_VIEWS, "triangulation needs at least two views")
    require(gate_px > 0 and sigma_px > 0, "gate_px and sigma_px must be positive")
    px = np.asarray(pixels, dtype=float).reshape(-1, 2)
    conf = np.ones(len(cams)) if confidence is None else np.asarray(confidence, float)
    require(len(cams) == px.shape[0] == conf.shape[0], "cams, pixels, confidence align")
    require(bool(np.all((conf >= 0) & (conf <= 1))), "confidence within [0, 1]")
    ids = [c.camera_id for c in cams]
    usable = [i for i in range(len(cams)) if conf[i] > 0 and np.all(np.isfinite(px[i]))]
    rejected: list[str] = [ids[i] for i in range(len(cams)) if i not in usable]
    residuals: dict[str, float] = {}
    point: Array = np.full(3, np.nan)

    def solve(subset: list[int]) -> tuple[Array, Array]:
        sub = [cams[i] for i in subset]
        if _is_degenerate_geometry(sub, px[subset]):
            return np.full(3, np.nan), np.full(len(subset), np.nan)
        try:
            pt = dlt(sub, px[subset], conf[subset])
            return pt, reprojection_residuals(sub, pt, px[subset])
        except (
            PreconditionError,
            ZeroDivisionError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            return np.full(3, np.nan), np.full(len(subset), np.nan)

    # A wrong view contaminates the joint solution, so its residual is not
    # reliably the largest: judge by leaving each view out in turn and keeping
    # the subset whose worst residual is smallest, while three or more remain.
    while len(usable) >= MIN_VIEWS:
        point, res = solve(usable)
        for i, r in zip(usable, res, strict=True):
            residuals[ids[i]] = float(r)
        if np.isfinite(res).all() and res.max() <= gate_px:
            break
        if len(usable) == MIN_VIEWS:
            break  # two views cannot assign blame (see tests)
        best_k, best_worst = -1, np.inf
        for k in range(len(usable)):
            trial = usable[:k] + usable[k + 1 :]
            _, trial_res = solve(trial)
            worst = (
                float(np.nanmax(trial_res)) if np.isfinite(trial_res).any() else np.inf
            )
            if worst < best_worst:
                best_k, best_worst = k, worst
        if best_k == -1 or not np.isfinite(best_worst):
            break
        rejected.append(ids[usable[best_k]])
        usable = usable[:best_k] + usable[best_k + 1 :]

    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(res)):
        rejected.extend([ids[i] for i in usable if ids[i] not in rejected])
        usable = []
        point = np.full(3, np.nan)

    contributing = tuple(ids[i] for i in usable) if len(usable) >= MIN_VIEWS else ()
    cov = (
        _covariance([cams[i] for i in usable], point, sigma_px)
        if len(usable) >= MIN_VIEWS
        else None
    )
    return TriangulatedPoint(
        point_m=point,
        residuals_px=residuals,
        contributing=contributing,
        rejected=tuple(rejected),
        covariance_m2=cov,
    )
