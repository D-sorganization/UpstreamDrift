"""MotionBERT monocular 3-D lifting evaluation harness (UpstreamDrift #9683).

Runs the released MB_ft_h36m_global_lite checkpoint (Apache-2.0 code; weights
fine-tuned on Human3.6M) on (a) the reconstruct synthetic bundle, where 3-D
truth exists, and (b) the real single-camera take. Self-contained: it reads
the observation JSON files directly and maps our 15-joint or MediaPipe-33
layouts onto H36M-17 the way MotionBERT's halpe2h36m does.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Layout expected beside this script's working directory (see the evaluation
# doc): MotionBERT/ (clone), MB_ft_h36m_global_lite.bin (verified download),
# synth/ (reconstruct synth bundle). Override with MB_ROOT.
import os

HERE = Path(os.environ.get("MB_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(HERE / "MotionBERT"))
from lib.model.DSTformer import DSTformer  # noqa: E402
from lib.utils.utils_data import flip_data  # noqa: E402

CKPT = HERE / "MB_ft_h36m_global_lite.bin"
CKPT_SHA256 = "9811155371db4ca5d20f31a36a232d41012e12e1333882888a564d741861148f"
H36M = [
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "Spine",
    "Thorax",
    "Nose",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]
# H36M joint -> source joint name(s) in our 15-joint reconstruct layout.
FROM_OURS = {
    "Hip": "mid_hip",
    "RHip": "right_hip",
    "RKnee": "right_knee",
    "RAnkle": "right_ankle",
    "LHip": "left_hip",
    "LKnee": "left_knee",
    "LAnkle": "left_ankle",
    "Thorax": "neck",
    "Nose": "nose",
    "LShoulder": "left_shoulder",
    "LElbow": "left_elbow",
    "LWrist": "left_wrist",
    "RShoulder": "right_shoulder",
    "RElbow": "right_elbow",
    "RWrist": "right_wrist",
}
MEDIAPIPE = {  # MediaPipe-33 indices for our joint names
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}
OUR_ORDER = [  # names shared between our layout and H36M for error reporting
    "mid_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "neck",
    "nose",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]
H36M_IDX_FOR_OURS = [
    H36M.index({v: k for k, v in FROM_OURS.items()}[n]) for n in OUR_ORDER
]
SEGMENTS = [
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("mid_hip", "neck"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]


def load_model() -> torch.nn.Module:
    import hashlib

    digest = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        raise SystemExit(f"checkpoint digest {digest} != pinned {CKPT_SHA256}")
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=256,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=lambda d: torch.nn.LayerNorm(d, eps=1e-6),
        maxlen=243,
        num_joints=17,
    )
    ckpt = torch.load(
        CKPT, map_location="cpu", weights_only=True
    )  # tensors only; never unpickle code from a downloaded file
    state = {k.replace("module.", "", 1): v for k, v in ckpt["model_pos"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def ours_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """(T,15,2) px, (T,15) conf in OUR_ORDER for every frame (NaN where absent)."""
    names = list(payload["detector_layout"]["keypoint_names"])
    fps, total = float(payload["fps"]), int(payload["frames_total"])
    w, h = int(payload.get("width") or 1920), int(payload.get("height") or 1200)
    px = np.full((total, 15, 2), np.nan)
    conf = np.zeros((total, 15))
    mediapipe = "left_shoulder" in names and "mid_hip" not in names and len(names) >= 29
    for row in payload["frames"]:
        t = int(round(float(row["time_s"]) * fps))
        if not 0 <= t < total:
            continue
        kp = np.asarray(row["keypoints_px"], float)
        c = np.asarray(row["confidence"], float)
        for j, name in enumerate(OUR_ORDER):
            if mediapipe:
                if name == "mid_hip":
                    idx = (MEDIAPIPE["left_hip"], MEDIAPIPE["right_hip"])
                elif name == "neck":
                    idx = (MEDIAPIPE["left_shoulder"], MEDIAPIPE["right_shoulder"])
                else:
                    idx = (MEDIAPIPE[name],)
                px[t, j] = kp[list(idx)].mean(axis=0)
                conf[t, j] = c[list(idx)].min()
            else:
                k = names.index(name)
                px[t, j] = kp[k]
                conf[t, j] = c[k]
    return px, conf, fps, w, h


def h36m_input(px: np.ndarray, conf: np.ndarray, w: int, h: int) -> np.ndarray:
    """(T,17,3) normalised like MotionBERT's read_input; Spine/Head synthesised."""
    T = px.shape[0]
    out = np.zeros((T, 17, 3))
    ours = {n: i for i, n in enumerate(OUR_ORDER)}
    for j, name in enumerate(H36M):
        if name in FROM_OURS:
            k = ours[FROM_OURS[name]]
            out[:, j, :2], out[:, j, 2] = px[:, k], conf[:, k]
    hip, neck, nose = ours["mid_hip"], ours["neck"], ours["nose"]
    out[:, 7, :2] = 0.5 * (px[:, hip] + px[:, neck])
    out[:, 7, 2] = np.minimum(conf[:, hip], conf[:, neck])
    out[:, 10, :2] = px[:, nose] + 0.5 * (px[:, nose] - px[:, neck])
    out[:, 10, 2] = np.minimum(conf[:, nose], conf[:, neck])
    bad = ~np.isfinite(out[:, :, :2]).all(axis=2)
    out[bad] = 0.0  # missing joint: centre of frame after normalisation, confidence 0
    out[:, :, :2] = (out[:, :, :2] - np.array([w, h]) / 2.0) / (min(w, h) / 2.0)
    out[bad, :2] = 0.0
    return out.astype(np.float32)


def lift(model: torch.nn.Module, inp: np.ndarray, clip_len: int = 243) -> np.ndarray:
    """(T,17,3) MotionBERT output (normalised units, root depth zeroed at clip start)."""
    outs = []
    with torch.no_grad():
        for s in range(0, inp.shape[0], clip_len):
            x = torch.from_numpy(inp[s : s + clip_len][None])
            y = (model(x) + flip_data(model(flip_data(x)))) / 2.0
            y[:, 0, 0, 2] = 0
            outs.append(y[0].numpy())
    return np.concatenate(outs)


def similarity_align(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Procrustes: rotate/scale/translate pred onto truth (per frame). Returns aligned, scale."""
    mp, mt = pred.mean(0), truth.mean(0)
    p, t = pred - mp, truth - mt
    u, s, vt = np.linalg.svd(p.T @ t)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    dmat = np.diag([1, 1, d])
    r = vt.T @ dmat @ u.T
    scale = (s * np.diag(dmat)).sum() / (p**2).sum()
    return (scale * (r @ p.T)).T + mt, scale


def bone_cv(joints: np.ndarray) -> dict[str, float]:
    ours = {n: i for i, n in enumerate(OUR_ORDER)}
    out = {}
    for a, b in SEGMENTS:
        L = np.linalg.norm(joints[:, ours[a]] - joints[:, ours[b]], axis=1)
        L = L[np.isfinite(L) & (L > 0)]
        out[f"{a}->{b}"] = float(L.std() / L.mean()) if L.size else float("nan")
    return out


def eval_synthetic(model: torch.nn.Module, bundle: Path) -> dict:
    truth = json.loads(Path(bundle / "truth.json").read_text(encoding="utf-8"))
    tj = np.asarray(truth["joints_3d_m"])  # (T,15,3) in truth joint order
    tnames = list(truth["joint_names"])
    t_ours = tj[:, [tnames.index(n) for n in OUR_ORDER]]
    report = {}
    for path in sorted((bundle / "observations").glob("*.json")):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        px, conf, fps, w, h = ours_from_payload(payload)
        t0 = time.perf_counter()
        pred17 = lift(model, h36m_input(px, conf, w, h))
        dt = time.perf_counter() - t0
        pred = pred17[:, H36M_IDX_FOR_OURS]  # (T,15,3) in OUR_ORDER
        mpjpe_sim, scales, mpjpe_rigid = [], [], []
        for k in range(pred.shape[0]):
            aligned, s = similarity_align(pred[k], t_ours[k])
            mpjpe_sim.append(np.linalg.norm(aligned - t_ours[k], axis=1).mean())
            scales.append(s)
        # rigid (no per-frame scale): one global scale from the median anchor, then per-frame rotation only
        g = float(np.median(scales))
        for k in range(pred.shape[0]):
            p = pred[k] * g
            mp, mt = p.mean(0), t_ours[k].mean(0)
            u, _, vt = np.linalg.svd((p - mp).T @ (t_ours[k] - mt))
            d = np.sign(np.linalg.det(vt.T @ u.T))
            r = vt.T @ np.diag([1, 1, d]) @ u.T
            mpjpe_rigid.append(
                np.linalg.norm((r @ (p - mp).T).T + mt - t_ours[k], axis=1).mean()
            )
        # bone-length error after global scale vs truth bone lengths
        ours = {n: i for i, n in enumerate(OUR_ORDER)}
        bl_err = {}
        for a, b in SEGMENTS:
            lp = np.linalg.norm(pred[:, ours[a]] - pred[:, ours[b]], axis=1) * g
            lt = np.linalg.norm(t_ours[:, ours[a]] - t_ours[:, ours[b]], axis=1)
            bl_err[f"{a}->{b}"] = float(np.median(np.abs(lp - lt) / lt))
        report[payload["view"]] = {
            "frames": int(pred.shape[0]),
            "ms_per_frame": 1000 * dt / pred.shape[0],
            "mpjpe_similarity_mm": 1000 * float(np.mean(mpjpe_sim)),
            "mpjpe_similarity_p95_mm": 1000 * float(np.percentile(mpjpe_sim, 95)),
            "mpjpe_rigid_globalscale_mm": 1000 * float(np.mean(mpjpe_rigid)),
            "scale_cv": float(np.std(scales) / np.mean(scales)),
            "bone_length_rel_error_median": bl_err,
            "bone_cv_pred": bone_cv(pred),
        }
    return report


def eval_real(model: torch.nn.Module, obs: Path, stride: int) -> dict:
    payload = json.loads(Path(obs).read_text(encoding="utf-8"))
    px, conf, fps, w, h = ours_from_payload(payload)
    px, conf = px[::stride], conf[::stride]
    covered = np.isfinite(px).all(axis=(1, 2)) & (conf.min(axis=1) >= 0.5)
    t0 = time.perf_counter()
    pred17 = lift(model, h36m_input(px, conf, w, h))
    dt = time.perf_counter() - t0
    pred = pred17[:, H36M_IDX_FOR_OURS]
    cov = pred[covered]
    ours = {n: i for i, n in enumerate(OUR_ORDER)}

    def flex(a, b, c):
        v1, v2 = cov[:, ours[a]] - cov[:, ours[b]], cov[:, ours[c]] - cov[:, ours[b]]
        cos = (v1 * v2).sum(1) / np.maximum(
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-9
        )
        return 180 - np.degrees(np.arccos(np.clip(cos, -1, 1)))

    sym = {}
    for left_name, right_name in (
        ("left_hip->left_knee", "right_hip->right_knee"),
        ("left_shoulder->left_elbow", "right_shoulder->right_elbow"),
    ):
        la, lb = left_name.split("->")
        ra, rb = right_name.split("->")
        L = np.linalg.norm(cov[:, ours[la]] - cov[:, ours[lb]], axis=1)
        R = np.linalg.norm(cov[:, ours[ra]] - cov[:, ours[rb]], axis=1)
        sym[left_name + " vs " + right_name] = float(
            np.median(np.abs(L - R) / (0.5 * (L + R)))
        )
    return {
        "frames_in": int(px.shape[0]),
        "frames_covered": int(covered.sum()),
        "fps_effective": fps / stride,
        "ms_per_frame": 1000 * dt / px.shape[0],
        "bone_cv_covered": bone_cv(cov),
        "left_right_asymmetry_median": sym,
        "knee_flexion_range_deg": [
            float(np.percentile(flex("right_hip", "right_knee", "right_ankle"), 5)),
            float(np.percentile(flex("right_hip", "right_knee", "right_ankle"), 95)),
        ],
        "elbow_flexion_range_deg": [
            float(np.percentile(flex("left_shoulder", "left_elbow", "left_wrist"), 5)),
            float(np.percentile(flex("left_shoulder", "left_elbow", "left_wrist"), 95)),
        ],
        "depth_range_normalised": [
            float(cov[:, :, 2].min()),
            float(cov[:, :, 2].max()),
        ],
    }


if __name__ == "__main__":
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "checkpoint": CKPT.name,
        "params_M": n_params / 1e6,
        "torch": torch.__version__,
    }
    synth = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "synth"
    result["synthetic"] = eval_synthetic(model, synth)
    real = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if real is not None and real.is_file():
        result["real_take"] = eval_real(model, real, stride=2)
    Path("motionbert_eval_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))
