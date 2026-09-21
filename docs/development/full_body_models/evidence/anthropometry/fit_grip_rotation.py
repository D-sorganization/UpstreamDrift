"""Fit the constant hand-to-club rotation of each wrist from a matched run
(MM-2, #10104).

Reads one or more unbounded ground-support runs
(``full_body_spec_hipcal_scaled.json`` and ``ik_trajectory.npz``),
reconstructs per frame the rotation from each proximal forearm to its hand
body (club for the lead hand, right-hand standoff for the trail hand), checks
that ``grip_fit.wrist_angles`` reproduces the matched pronation, cock and
flexion exactly, then fits one follower-side rotation per hand over all the
runs together that minimises the excursions beyond the human ranges, and
reports the residual per run. Writes ``fit_grip_rotation_receipt.json`` next
to this script. The fitted angles are the builder's ``--lead-grip-rotation``
and ``--trail-grip-rotation`` (defaults ``GRIP_ROTATION_DEG``). Runs must be
matched with the same grip rotation, since the fit is relative to it; the
receipt records the document's value.

    python fit_grip_rotation.py --run <run_a> --run <run_b>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.engines.physics_engines.mujoco.python.full_body_ik import (  # noqa: E402
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (  # noqa: E402
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching import grip_fit  # noqa: E402

HERE = Path(__file__).resolve().parent
SIDES = {
    "lead": ("Spherical Solid", "Clubface Vector", ("LFInput", "LWInputX", "LWInputY")),
    "trail": ("Spherical Solid1", "RHandStandoff", ("RFInput", "RWInputX", "RWInputY")),
}


def _body(document: dict, suffix: str) -> str:
    return next(b["name"] for b in document["bodies"] if b["name"].endswith(suffix))


def _wrist_joint(document: dict, child: str) -> dict:
    return next(j for j in document["joints"] if j["child"] == child)


def relative_rotations(
    document: dict, q: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]]:
    """Per side: (relative rotations (frames, 3, 3), wrist base, coordinates)."""
    attachments = {
        label: (a["body"], a["offset_m"])
        for label, a in document["marker_attachments"].items()
        if a.get("offset_m") is not None and np.isfinite(a["offset_m"]).all()
    }
    adapter = NativeMujocoFullBodyModel(json.dumps(document).encode())
    kin = FullBodyMarkerKinematics(adapter, attachments)
    out = {}
    for side, (forearm_suffix, hand_suffix, coords) in SIDES.items():
        forearm, hand = _body(document, forearm_suffix), _body(document, hand_suffix)
        joint = _wrist_joint(document, hand)
        c2f = np.asarray(joint["child_to_follower"], float)[:3, :3]
        base = np.asarray(joint["parent_to_base"], float)[:3, :3]
        stack = np.empty((len(q), 3, 3))
        for k, qk in enumerate(q):
            poses = kin.body_poses(qk, [forearm, hand])
            stack[k] = poses[forearm][0].T @ poses[hand][0] @ c2f
        out[side] = (stack, base, coords)
    return out


def _per_run(
    stack: np.ndarray, base: np.ndarray, ranges: list, rotation_deg: tuple
) -> dict:
    angles = np.degrees(
        grip_fit.wrist_angles(stack @ grip_fit.grip_rotation(rotation_deg), base)
    )
    ex = grip_fit.excess_deg(angles, ranges)
    return {
        "rms_excess_deg": round(float(np.sqrt(np.mean(ex**2))), 2),
        "max_excess_deg": dict(
            zip(
                grip_fit.ANGLE_NAMES,
                [round(float(v), 1) for v in ex.max(axis=0)],
                strict=True,
            )
        ),
        "angle_range_deg": {
            n: [round(float(c.min()), 1), round(float(c.max()), 1)]
            for n, c in zip(grip_fit.ANGLE_NAMES, angles.T, strict=True)
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("grip_fit")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument(
        "--receipt", type=Path, default=HERE / "fit_grip_rotation_receipt.json"
    )
    args = parser.parse_args()
    runs: dict[str, dict] = {}
    for run in args.run:
        document = json.loads((run / "full_body_spec_hipcal_scaled.json").read_text())
        q = np.load(run / "ik_trajectory.npz")["q"]
        runs[run.name] = {
            "document": document,
            "q": q,
            "sides": relative_rotations(document, q),
        }
    first = next(iter(runs.values()))["document"]
    ranges = first["coordinate_ranges_deg"]
    receipt: dict = {
        "runs": {
            run.name: {"path": str(run), "frames": int(len(runs[run.name]["q"]))}
            for run in args.run
        },
        "document_grip_rotation_deg": first.get("subject", {}).get("grip_rotation_deg"),
        "metric": "rms of the wrist and forearm excursions beyond coordinate_ranges_deg over all frames of all runs",
        "sides": {},
    }
    for side, (_, base, coords) in next(iter(runs.values()))["sides"].items():
        stacks = []
        for name, r in runs.items():
            stack = r["sides"][side][0]
            order = list(r["document"]["coordinate_order"])
            matched = r["q"][:, [order.index(c) for c in coords]]
            # The match may sit on the far Euler branch (cock beyond 90 deg);
            # the chain rebuilt from its angles must equal the reconstruction.
            check = float(np.abs(grip_fit.wrist_rotation(matched, base) - stack).max())
            if check > 1e-6:
                raise RuntimeError(f"{name} {side}: rebuilt chain differs by {check}")
            receipt["runs"][name][f"{side}_frames_on_far_euler_branch"] = int(
                (np.abs(matched[:, 1]) > np.pi / 2).sum()
            )
            stacks.append(stack)
        fit = grip_fit.fit_grip_rotation(
            np.concatenate(stacks), base, [ranges[c] for c in coords]
        )
        receipt["sides"][side] = {
            "coordinates": list(coords),
            **fit.as_document(),
            "per_run_after": {
                name: _per_run(
                    r["sides"][side][0],
                    base,
                    [ranges[c] for c in coords],
                    fit.rotation_deg,
                )
                for name, r in runs.items()
            },
        }
        log.info(
            "%s: rotation %s deg, rms excess %.1f -> %.2f deg, max excess after %s",
            side, [round(v, 1) for v in fit.rotation_deg], fit.cost_before, fit.cost_after,
            receipt["sides"][side]["max_excess_after_deg"],
        )  # fmt: skip
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
