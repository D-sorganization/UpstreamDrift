"""Diagnose Candidate 100 terminal marker plateau and active bounds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_100"
)


import logging

logger = logging.getLogger(__name__)


def diagnose() -> None:
    cand_path = EVIDENCE_DIR / "returned-candidate.json"
    pino_path = EVIDENCE_DIR / "pinocchio_replay.mat"
    ret_path = EVIDENCE_DIR / "returned.json"

    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    pino = sio.loadmat(str(pino_path))

    labels = cand["marker_labels"]
    pred = pino["markers_m"][-1]  # terminal frame (25, 3)
    target = pino["target_m"][-1]  # terminal frame (25, 3)
    valid = pino["valid"][-1].astype(bool)

    diffs = np.linalg.norm(pred - target, axis=-1) * 1000.0  # in mm

    logger.info("=== Terminal Marker Errors at t = 0.85 s (Candidate 100) ===")
    ranked = sorted(
        zip(labels, diffs, valid, strict=True), key=lambda x: x[1], reverse=True
    )
    for lbl, err, val in ranked:
        v_str = "VALID" if val else "MISSING"
        logger.info("  %-20s: %8.2f mm (%s)", lbl, err, v_str)

    valid_errs = diffs[valid]
    logger.info("Terminal RMS: %.3f mm", np.sqrt(np.mean(valid_errs**2)))
    logger.info("Max Terminal Marker: %.3f mm (%s)", np.max(valid_errs), ranked[0][0])

    # Pelvis yaw check
    wl_i = labels.index("WaistLeft")
    wr_i = labels.index("WaistRight")
    vp = pred[wr_i, :2] - pred[wl_i, :2]
    vt = target[wr_i, :2] - target[wl_i, :2]
    yp = np.degrees(np.arctan2(vp[1], vp[0]))
    yt = np.degrees(np.arctan2(vt[1], vt[0]))
    ydiff = (yp - yt + 180) % 360 - 180
    yerr_pct = abs(ydiff) / max(abs(yt), 1.0) * 100
    logger.info(
        "Pelvis Yaw Pred: %.2f deg, Target: %.2f deg, Diff: %.2f deg, Error: %.2f%%",
        yp,
        yt,
        ydiff,
        yerr_pct,
    )

    if ret_path.is_file():
        ret = json.loads(ret_path.read_text(encoding="utf-8"))
        logger.info("Optimizer message: %s", ret.get("message"))
        logger.info("Active bounds count: %s", ret.get("active_bounds_count"))
        logger.info(
            "Iterations: %s, Evaluations: %s",
            ret.get("iterations"),
            ret.get("evaluations"),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    diagnose()
