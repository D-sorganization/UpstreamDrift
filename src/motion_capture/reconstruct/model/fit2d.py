"""Image-space fit: the articulated model against 2-D keypoints (#9794).

The same whole-trajectory solver as :mod:`.fit`, with the data term in
pixels::

    (project_v(landmark_model) - keypoint_px) / sigma_px     per view v

so the model can be matched to any number of camera views, one included,
without triangulating first. The cameras are known (from a previous
multi-camera match, recorded in the provenance) and the segment lengths are
the model's (learned or measured): with a single view the in-plane motion
is observed directly, depth comes from perspective and the lengths, and the
continuity and rest priors settle what one view cannot see.

Observations: ``keypoints_px (T, V, L, 2)`` and ``confidence (T, V, L)`` in
the model's landmark order; confidence 0 is "unobserved".
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from src.shared.python.core.contracts import require

from ..cameras import PinholeCamera
from .fit import Array, FitOptions, ModelFit, RejectedLandmark, _Problem, solve_problem
from .kinematics import ArticulatedModel
from .session import LandmarkMap

Mask = npt.NDArray[np.bool_]
DEFAULT_SIGMA_PX = 4.0
DEFAULT_ROOT_DEPTH_M = 3.5


@dataclass(frozen=True)
class ImageSpaceSource:
    """Which views to fit in image space and where their cameras come from."""

    views: tuple[str, ...]
    cameras_from: str = ""
    observation_set: str = "observations"
    variant: str = ""

    def __post_init__(self) -> None:
        require(len(self.views) >= 1, "at least one view")

    def as_provenance(self) -> dict[str, Any]:
        return {
            "kind": "image_space",
            "views": list(self.views),
            "cameras_from": self.cameras_from,
            "observation_set": self.observation_set,
        }


MIN_DEPTH_M = 0.05


def _projection_jacobians(
    camera: PinholeCamera, points_world: Array
) -> tuple[Array, Array, Mask]:
    """``px (N, 2)``, ``d px / d world (N, 2, 3)`` and an in-front mask."""
    pc = camera.camera_from_world(points_world)
    k = camera.matrix
    z = pc[:, 2]
    in_front = z > MIN_DEPTH_M
    zs = np.where(in_front, z, 1.0)
    uv = np.column_stack(
        [
            (k[0, 0] * pc[:, 0] + k[0, 1] * pc[:, 1]) / zs + k[0, 2],
            k[1, 1] * pc[:, 1] / zs + k[1, 2],
        ]
    )
    d_cam = np.zeros((pc.shape[0], 2, 3))
    d_cam[:, 0, 0] = k[0, 0] / zs
    d_cam[:, 0, 1] = k[0, 1] / zs
    d_cam[:, 0, 2] = -(k[0, 0] * pc[:, 0] + k[0, 1] * pc[:, 1]) / zs**2
    d_cam[:, 1, 1] = k[1, 1] / zs
    d_cam[:, 1, 2] = -k[1, 1] * pc[:, 1] / zs**2
    r_cw = camera.rotation_world_from_camera.T  # d pc / d world
    return uv, d_cam @ r_cw, in_front


class _Problem2D(_Problem):
    """Landmark term in pixels over ``V`` views; priors inherited."""

    def __init__(
        self,
        model: ArticulatedModel,
        keypoints_px: Array,
        confidence: Array,
        cameras: Sequence[PinholeCamera],
        fps: float,
        options: FitOptions,
        lengths: Mapping[str, float],
        sigma_px: float,
    ) -> None:
        self._init_common(model, keypoints_px.shape[0], fps, options, lengths)
        self.v, self.l = keypoints_px.shape[1:3]
        self.cameras = tuple(cameras)
        self.sigma_px = sigma_px
        self.obs = np.nan_to_num(keypoints_px)
        finite = np.isfinite(keypoints_px).all(axis=3)
        self.mask = finite & (confidence > 0)
        self.base_w = np.sqrt(np.clip(np.nan_to_num(confidence), 0, 1)) * self.mask
        self.w = self.base_w.copy()

    # -- data term --------------------------------------------------------------
    def _project_all(
        self, q: Array, lengths: Mapping[str, float]
    ) -> tuple[Array, Array, Array]:
        """``px (V, T, L, 2)``, ``jac (V, T, L, 2, 3)``, ``valid (V, T, L)``."""
        lm = self.model.landmarks(q, lengths).reshape(-1, 3)
        px = np.zeros((self.v, self.t, self.l, 2))
        jac = np.zeros((self.v, self.t, self.l, 2, 3))
        valid = np.zeros((self.v, self.t, self.l), dtype=bool)
        for c, camera in enumerate(self.cameras):
            uv, d_world, in_front = _projection_jacobians(camera, lm)
            px[c] = uv.reshape(self.t, self.l, 2)
            jac[c] = d_world.reshape(self.t, self.l, 2, 3)
            valid[c] = in_front.reshape(self.t, self.l) & self.mask[:, c]
        return px, jac, valid

    def _weights_vtl(self) -> Array:
        return np.transpose(self.w, (1, 0, 2))  # (V, T, L)

    def landmark_residuals(self, q: Array, lengths: Mapping[str, float]) -> Array:
        px, _, valid = self._project_all(q, lengths)
        obs = np.transpose(self.obs, (1, 0, 2, 3))
        w = self._weights_vtl() * valid
        return ((px - obs) * w[..., None] / self.sigma_px).ravel()

    def _landmark_block(self, q: Array, lengths: Mapping[str, float]) -> csr_matrix:
        _, d_world, valid = self._project_all(q, lengths)
        w = self._weights_vtl() * valid / self.sigma_px  # (V, T, L)
        jac_lm = self.model.jacobian(q, lengths).reshape(self.t, self.l, 3, self.n)
        # (V, T, L, 2, 3) @ (T, L, 3, n) -> (V, T, L, 2, n)
        d_q = np.einsum("vtlcx,tlxn->vtlcn", d_world, jac_lm) * w[..., None, None]
        rows_per_vt = self.l * 2
        n_rows = self.v * self.t * rows_per_vt
        row_ids = np.arange(n_rows).reshape(self.v, self.t, self.l, 2)
        rows = np.repeat(row_ids, self.n).astype(np.int64)
        frame = np.broadcast_to(np.arange(self.t)[None, :, None, None], row_ids.shape)
        cols = (frame[..., None] * self.n + np.arange(self.n)).ravel()
        vals = d_q.ravel()
        if self.length_names:
            k = len(self.length_names)
            lj = self.model.length_jacobian(q, lengths, self.length_names)
            lj = lj.reshape(self.t, self.l, 3, k)
            d_len = np.einsum("vtlcx,tlxk->vtlck", d_world, lj) * w[..., None, None]
            rows = np.concatenate([rows, np.repeat(row_ids, k).astype(np.int64)])
            cols = np.concatenate(
                [cols, np.tile(self.t * self.n + np.arange(k), n_rows)]
            )
            vals = np.concatenate([vals, d_len.ravel()])
        return csr_matrix((vals, (rows, cols)), shape=(n_rows, self.n_params))

    # -- hooks --------------------------------------------------------------------
    def residual_px_tvl(self, q: Array, lengths: Mapping[str, float]) -> Array:
        """Unweighted pixel distance per (frame, view, landmark), NaN unobserved."""
        px, _, valid = self._project_all(q, lengths)
        d = np.linalg.norm(px - np.transpose(self.obs, (1, 0, 2, 3)), axis=3)
        return np.where(
            np.transpose(valid, (1, 0, 2)), np.transpose(d, (1, 0, 2)), np.nan
        )

    def landmark_residual_m(self, x: Array) -> Array:
        return np.full((self.t, self.l), np.nan)

    def residual_sigma(self, x: Array) -> Array:
        q, lengths = self.unpack(x)
        return self.residual_px_tvl(q, lengths) / self.sigma_px

    def reject(self, index: tuple[int, ...]) -> RejectedLandmark:
        t, v, k = index
        self.base_w[t, v, k] = 0.0
        self.mask[t, v, k] = False
        self.w = self.base_w.copy()
        name = f"{self.model.landmark_names[k]}@{self.cameras[v].camera_id}"
        return RejectedLandmark(int(t), name, 0.0)

    def report(self, q: Array, lengths: Mapping[str, float]) -> dict[str, Any]:
        residual_px = self.residual_px_tvl(q, lengths)
        finite = residual_px[np.isfinite(residual_px)]
        return {
            "residual_m": np.full((self.t, self.l), np.nan),
            "weights": (self.base_w**2).max(axis=1),
            "rms_m": float("nan"),
            "residual_px": residual_px,
            "rms_px": float(np.sqrt(np.mean(finite**2))) if finite.size else None,
        }


def back_project(camera: PinholeCamera, px: Array, depth_m: float) -> Array:
    """World point ``depth_m`` along the ray through pixel ``px``."""
    k = camera.matrix
    x = (px[0] - k[0, 2] - k[0, 1] * (px[1] - k[1, 2]) / k[1, 1]) / k[0, 0]
    y = (px[1] - k[1, 2]) / k[1, 1]
    direction = camera.rotation_world_from_camera @ np.array([x, y, 1.0])
    return camera.position_m + depth_m * direction


def initial_root_2d(
    keypoints_px: Array,
    confidence: Array,
    cameras: Sequence[PinholeCamera],
    *,
    depth_m: float = DEFAULT_ROOT_DEPTH_M,
    root_index: int = 0,
) -> Array:
    """``(T, 3)`` starting root positions from the root landmark's pixels.

    Two or more views: least-squares ray intersection; one view: the ray at
    ``depth_m``. Frames without any confident root keep the previous frame
    (or the first later one). Precondition: at least one confident root.
    """
    t = keypoints_px.shape[0]
    out = np.full((t, 3), np.nan)
    for f in range(t):
        rays = []
        for c, camera in enumerate(cameras):
            if confidence[f, c, root_index] > 0:
                px = keypoints_px[f, c, root_index]
                far = back_project(camera, px, 10.0)
                rays.append((camera.position_m, far - camera.position_m))
        if len(rays) >= 2:
            out[f] = _intersect(rays)
        elif len(rays) == 1:
            origin, direction = rays[0]
            out[f] = origin + depth_m * direction / np.linalg.norm(direction)
    finite = np.isfinite(out).all(axis=1)
    require(bool(finite.any()), "no confident root landmark in any frame")
    idx = np.arange(t)
    for c in range(3):
        out[:, c] = np.interp(idx, idx[finite], out[finite, c])
    return out


def _intersect(rays: Sequence[tuple[Array, Array]]) -> Array:
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for origin, direction in rays:
        d = direction / np.linalg.norm(direction)
        m = np.eye(3) - np.outer(d, d)
        a += m
        b += m @ origin
    return np.asarray(np.linalg.lstsq(a, b, rcond=None)[0], dtype=float)


def map_landmarks_2d(
    landmark_map: LandmarkMap,
    model: ArticulatedModel,
    keypoints_px: Array,
    confidence: Array,
    joint_names: Sequence[str],
) -> tuple[Array, Array]:
    """Reconstruct-joint pixels ``(T, V, K, 2)`` -> model landmarks ``(T, V, L, 2)``.

    Tuple sources are averaged (confidence: the minimum); missing sources
    give confidence 0.
    """
    t, v = keypoints_px.shape[:2]
    names = list(joint_names)
    out = np.zeros((t, v, len(model.landmark_names), 2))
    conf = np.zeros((t, v, len(model.landmark_names)))
    for k, landmark in enumerate(model.landmark_names):
        source = landmark_map.to_reconstruct.get(landmark)
        if source is None:
            continue
        sources = (source,) if isinstance(source, str) else tuple(source)
        cols = [names.index(s) for s in sources]
        out[:, :, k] = keypoints_px[:, :, cols].mean(axis=2)
        conf[:, :, k] = confidence[:, :, cols].min(axis=2)
    return out, conf


def fit_trajectory_2d(
    model: ArticulatedModel,
    keypoints_px: Array,
    confidence: Array,
    cameras: Sequence[PinholeCamera],
    fps: float,
    *,
    lengths_m: Mapping[str, float] | None = None,
    options: FitOptions | None = None,
    q0: Array | None = None,
) -> ModelFit:
    """Continuous joint-angle trajectory matching the keypoints of ``V`` views.

    Preconditions: ``keypoints_px`` is ``(T, V, L, 2)`` and ``confidence``
    ``(T, V, L)`` in the model's landmark order; one camera per view;
    positive fps and sigma; at least one confident observation.
    Postcondition: ``rms_px`` and ``residual_px`` are set; ``rms_m`` is NaN.
    """
    o = options or FitOptions()
    kp = np.asarray(keypoints_px, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    n_l = len(model.landmark_names)
    require(
        kp.ndim == 4 and kp.shape[2:] == (n_l, 2), "keypoints (T, V, L, 2)", kp.shape
    )
    require(conf.shape == kp.shape[:3], "confidence (T, V, L)", conf.shape)
    require(len(cameras) == kp.shape[1], "one camera per view", len(cameras))
    require(fps > 0, "positive fps")
    require(bool((conf > 0).any()), "at least one confident observation")
    lengths = dict(model.spec.lengths_m if lengths_m is None else lengths_m)
    problem = _Problem2D(model, kp, conf, cameras, fps, o, lengths, o.sigma_px)
    if q0 is None:
        q_start = np.zeros((kp.shape[0], model.n_dof))
        q_start[:, :3] = initial_root_2d(kp, conf, cameras, depth_m=o.root_depth_m)
    else:
        q_start = np.asarray(q0, dtype=float).reshape(kp.shape[0], -1)
        require(q_start.shape == (kp.shape[0], model.n_dof), "q0 shape")
    return solve_problem(problem, q_start, lengths, fps, o)


# -- session entry -------------------------------------------------------------


def frame_index(row: Mapping[str, Any], fps: float) -> int:
    return int(round(float(row["time_s"]) * fps))


def load_view_keypoints(
    session_root: Path,
    observation_set: str,
    views: Sequence[str],
) -> tuple[Array, Array, float, list[Path]]:
    """``(keypoints (T, V, K, 2), confidence (T, V, K), fps, files)`` in JOINT_NAMES.

    Frames are placed by ``round(time_s * fps)``; frames a view did not
    detect get confidence 0. Precondition: every view file exists and all
    views share one fps.
    """
    from ..layouts import to_reconstruct_layout
    from ..skeleton import JOINT_NAMES

    require(len(views) >= 1, "at least one view")
    files = [session_root / observation_set / f"{v}.json" for v in views]
    for path in files:
        require(path.is_file(), "observation file missing", str(path))
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    payloads = [
        p
        if tuple(p["detector_layout"]["keypoint_names"]) == JOINT_NAMES
        else to_reconstruct_layout(p)
        for p in payloads
    ]
    fps = float(payloads[0]["fps"])
    require(fps > 0, "views must state a positive fps")
    require(all(abs(float(p["fps"]) - fps) < 1e-6 for p in payloads), "fps differ")
    frames = max(
        (frame_index(row, fps) for p in payloads for row in p["frames"]), default=-1
    )
    require(frames >= 0, "no frames in the observation set")
    k = len(JOINT_NAMES)
    kp = np.zeros((frames + 1, len(views), k, 2))
    conf = np.zeros((frames + 1, len(views), k))
    for c, payload in enumerate(payloads):
        for row in payload["frames"]:
            i = frame_index(row, fps)
            kp[i, c] = np.asarray(row["keypoints_px"], dtype=float)
            conf[i, c] = np.asarray(row["confidence"], dtype=float)
    return kp, conf, fps, files


def cameras_for_views(
    session_root: Path, cameras_from: str, views: Sequence[str]
) -> tuple[list[PinholeCamera], Path]:
    """Cameras of a previous variant's reconstruction, in ``views`` order."""
    from ...variants import variant_dir
    from ..fit import RECONSTRUCTION_FILE, cameras_from_records

    path = variant_dir(session_root, cameras_from) / "reconstruct" / RECONSTRUCTION_FILE
    require(path.is_file(), "cameras-from variant has no reconstruction", str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c.camera_id: c for c in cameras_from_records(payload["cameras"])}
    missing = [v for v in views if v not in by_id]
    require(not missing, "cameras-from variant lacks cameras for views", missing)
    return [by_id[v] for v in views], path


def fit_session_model_2d(
    session_root: Path,
    spec: Any,
    landmark_map: LandmarkMap,
    source: ImageSpaceSource,
    *,
    options: FitOptions | None = None,
    out_subdir: str | None = None,
) -> tuple[ModelFit, Path]:
    """Image-space fit of ``spec`` to ``views``; writes ``variants/<variant>/model/``.

    The cameras come from ``cameras_from``'s reconstruction and the measured
    lengths from its session summary; both are recorded in the provenance.
    Precondition: the views have observation files and cameras.
    """
    from ...variants import ensure_variant, register_variant, variant_dir
    from .session import Stamp, options_dict, write_fit

    views, cameras_from = source.views, source.cameras_from
    kp, conf, fps, files = load_view_keypoints(
        session_root, source.observation_set, views
    )
    cameras, cameras_file = cameras_for_views(session_root, cameras_from, views)
    summary_file = (
        variant_dir(session_root, cameras_from)
        / "reconstruct"
        / ("session_reconstruction.json")
    )
    measured: dict[str, float] = {}
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        measured = dict(summary.get("measured_lengths_m") or {})
    model = ArticulatedModel(spec)
    from ..skeleton import JOINT_NAMES

    lm_px, lm_conf = map_landmarks_2d(landmark_map, model, kp, conf, JOINT_NAMES)
    lengths = landmark_map.lengths(measured, spec)
    fit = fit_trajectory_2d(
        model, lm_px, lm_conf, cameras, fps, lengths_m=lengths, options=options
    )
    root = ensure_variant(session_root, source.variant)
    parameters = {
        "model": spec.name,
        "source": source.as_provenance(),
        **options_dict(options),
    }
    out_dir = write_fit(
        root,
        out_subdir,
        fit,
        fps,
        spec,
        landmark_map,
        model,
        Stamp(
            inputs=[*files, cameras_file],
            parameters=parameters,
            derived_from=[cameras_file, *files],
            base=session_root,
        ),
    )
    register_variant(
        session_root,
        source.variant,
        views=views,
        observation_set=source.observation_set,
        source={"kind": "image_space", "cameras_from": cameras_from},
        module=__name__,
    )
    return fit, out_dir
