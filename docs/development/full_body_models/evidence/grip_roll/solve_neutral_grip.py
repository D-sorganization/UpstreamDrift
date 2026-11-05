"""SHOW-A2 (#14551): neutral-grip totals and the re-solved address pose.

Sets each hand to the HX-canonical neutral grip (psi 40, gamma 0) with alpha solved until ``motion_matching.grip_roll`` reads
``--target`` deg face-on, writes the totals as a ``fit_grip_rotation``-style
receipt (apply with ``scripts.set_spec_grip_rotation``), then re-solves the arm
coordinates at address so the club pose and the grip closure are unchanged.
The solve is not bounded by ``coordinate_ranges_deg``: violations are listed.
Run from the repo root with ``--spec`` the pre-change spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
    export_full_body_mjcf,
)  # noqa: E402
from measure_address_grip_roll import Q, SPEC, measure  # noqa: E402

from src.shared.python.motion_matching.execution.grip_rotation import (  # noqa: E402
    apply_grip_rotation,
)
from src.shared.python.model_appearance.grip_pose import (  # noqa: E402
    ALPHA_RANGE_DEG,
    PALM_DIAGONAL_DEG,
)
from src.shared.python.motion_matching.grip_contract import (  # noqa: E402
    HandAngles,
    hand_to_euler,
)
from src.shared.python.motion_matching.grip_roll import (  # noqa: E402
    address_grip_roll_deg,
    grip_pose_state_from_spec,
)

ARM = ("SInput", "EInput", "FInput", "WInput", "Scap")


def _neutral_total(
    spec: dict, side: str, club_rot: np.ndarray, target: float
) -> list[float]:
    """HX-canonical neutral grip (psi 40, gamma 0) whose face-on roll is ``target``.

    Solves ``alpha`` so ``grip_roll`` reads ``target`` for ``side`` at the club
    orientation ``club_rot`` (the dorsal axis depends only on the hand offset).
    """

    def triple(alpha: float) -> list[float]:
        return list(hand_to_euler(HandAngles(alpha, PALM_DIAGONAL_DEG, 0.0)))

    def error(alpha: float) -> float:
        doc, _ = apply_grip_rotation(spec, {side: triple(alpha)})
        roll = address_grip_roll_deg(grip_pose_state_from_spec(doc, club_rot))[
            0 if side == "L" else 1
        ]
        return (roll - target + 180.0) % 360.0 - 180.0

    return triple(brentq(error, *ALPHA_RANGE_DEG))


class Rig:
    """A MuJoCo model of one spec: poses ``q`` and reads club and closure sites."""

    def __init__(self, spec_bytes: bytes) -> None:
        xml, self.meta = export_full_body_mjcf(spec_bytes, visual=True)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.order = self.meta["coordinate_order"]
        self.joint = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in self.order
        ]

    def sites(self, q: np.ndarray):
        self.data.qpos[self.joint] = q
        mujoco.mj_forward(self.model, self.data)

        def get(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            return self.data.site_xpos[i].copy(), self.data.site_xmat[i].reshape(
                3, 3
            ).copy()

        return (
            get(self.meta["frame_sites"]["LW"]),
            get("native_closure_a"),
            get("native_closure_b"),
        )


def resolve_swing(old: Rig, new: Rig, q: np.ndarray) -> np.ndarray:
    """Arm coordinates of ``new`` that keep every frame's club pose and closure.

    ``q`` is in ``old.order``; the result has the same shape. Frame ``t`` starts
    from the solution of frame ``t-1`` so the solution stays continuous.
    """
    idx = [
        i for i, n in enumerate(new.order) if n[0] in "LR" and any(k in n for k in ARM)
    ]
    out = np.array(q, dtype=float)
    guess = out[0, idx]
    for t in range(len(out)):
        (rp, rR), _, _ = old.sites(q[t])

        def resid(x, t=t, rp=rp, rR=rR):
            row = out[t].copy()
            row[idx] = x
            (p, R), a, b = new.sites(row)
            return np.concatenate(
                [
                    (p - rp) * 100,
                    Rot.from_matrix(rR.T @ R).as_rotvec() * 17.2,
                    (a[0] - b[0]) * 100,
                    Rot.from_matrix(a[1].T @ b[1]).as_rotvec() * 17.2,
                    1e-3 * (x - q[t, idx]) * 57.3,
                ]
            )

        sol = least_squares(resid, guess, xtol=1e-12, ftol=1e-12, max_nfev=400)
        out[t, idx] = guess = sol.x
    return out


def violations(spec: dict, order: list[str], q: np.ndarray) -> dict[str, dict]:
    """Per coordinate: the frames outside ``coordinate_ranges_deg`` and the extremes."""
    found = {}
    for n, (lo, hi) in spec["coordinate_ranges_deg"].items():
        if n not in order:
            continue
        deg = np.degrees(q[:, order.index(n)])
        bad = int(np.sum((deg < lo - 0.5) | (deg > hi + 0.5)))
        if bad:
            found[n] = {
                "frames": bad,
                "min": round(float(deg.min()), 1),
                "max": round(float(deg.max()), 1),
                "range": [lo, hi],
            }
    return found


def record_fixture_provenance(club: str) -> None:
    """Re-pin ``provenance.json`` to the in-place re-solved fixture of ``club``.

    The pipeline-origin fields keep describing the capture-chain run; the new
    ``neutral_grip_resolve`` block records what A2 changed and under which spec.
    """
    path = ROOT / "tests/fixtures/club_face/provenance.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    record = doc["clubs"][club]
    npz = ROOT / f"tests/fixtures/club_face/swing_q_{club}.npz"
    record["neutral_grip_resolve"] = {
        "issue": 14551,
        "script": "docs/development/full_body_models/evidence/grip_roll/"
        "solve_neutral_grip.py",
        "previous_fixture_sha256": record["fixture_sha256"],
        "showpiece_spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "note": "swing and address pose re-solved in place for the 15 deg neutral "
        "grip at the unchanged club path; the fields above describe the "
        "capture-chain run the fixture was first generated from",
    }
    record["fixture_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=SPEC)
    ap.add_argument("--target", type=float, default=15.0)
    ap.add_argument(
        "--q",
        type=Path,
        default=Q,
        help="pre-change ik_trajectory.npz (frame 0 is the address)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("neutral_grip.json")
    )
    ap.add_argument(
        "--resolve",
        nargs=2,
        metavar=("NPZ", "ORDER_JSON_CLUB"),
        action="append",
        default=[],
        help="re-solve a swing npz in place (club key from address_poses.json)",
    )
    ap.add_argument(
        "--address-pose", help="club key in tests/fixtures/club_face/address_poses.json"
    )
    ap.add_argument(
        "--record-provenance",
        metavar="CLUB",
        help="re-pin club_face/provenance.json to the re-solved fixture and exit",
    )
    args = ap.parse_args()
    if args.record_provenance:
        record_fixture_provenance(args.record_provenance)
        return 0
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    q0 = np.load(args.q)["q"][0]
    before = measure(args.spec.read_bytes(), q0)
    old = Rig(args.spec.read_bytes())
    club_rot = old.sites(q0)[0][1]
    totals = {s_: _neutral_total(spec, s_, club_rot, args.target) for s_ in ("L", "R")}
    new_spec, _ = apply_grip_rotation(spec, totals)
    new = Rig(json.dumps(new_spec, sort_keys=True).encode())
    q1 = resolve_swing(old, new, q0[None, :])[0]
    final = measure(json.dumps(new_spec, sort_keys=True).encode(), q1)
    (rp, rR), _, _ = old.sites(q0)
    (p, R), a, b = new.sites(q1)
    receipt = {
        "issue": 14551,
        "convention": "face-on grip roll, motion_matching/grip_roll.py",
        "target_roll_deg": args.target,
        "before_roll_deg": {
            "lead": round(before["lead"], 2),
            "trail": round(before["trail"], 2),
        },
        "after_roll_deg": {
            "lead": round(final["lead"], 2),
            "trail": round(final["trail"], 2),
        },
        "club_position_error_mm": round(float(np.linalg.norm(p - rp) * 1000), 3),
        "club_rotation_error_deg": round(
            float(np.degrees(np.linalg.norm(Rot.from_matrix(rR.T @ R).as_rotvec()))), 3
        ),
        "closure_error_mm": round(float(np.linalg.norm(a[0] - b[0]) * 1000), 3),
        "sides": {
            "lead": {"total_deg": [round(v, 4) for v in totals["L"]]},
            "trail": {"total_deg": [round(v, 4) for v in totals["R"]]},
        },
        "swings_resolved": {},
    }
    poses = json.loads(
        (ROOT / "tests/fixtures/club_face/address_poses.json").read_text()
    )["poses"]
    for path, club in args.resolve:
        order = poses[club]["coordinate_order"] if club in poses else new.order
        src = np.load(path)
        q = src["q"]
        perm = [
            order.index(n) for n in new.order
        ]  # new-order column i is source column perm[i]
        q_new = np.array(q)
        q_new[:, perm] = resolve_swing(old, new, q[:, perm])
        extra = {k: src[k] for k in src.files if k != "q"}
        np.savez(path, q=q_new, **extra)
        receipt["swings_resolved"][Path(path).name] = {
            "frames": int(len(q)),
            "outside_coordinate_ranges": violations(
                new_spec, new.order, q_new[:, perm]
            ),
        }
    if args.address_pose:
        path = ROOT / "tests/fixtures/club_face/address_poses.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        entry = doc["poses"][args.address_pose]
        perm = [entry["coordinate_order"].index(n) for n in new.order]
        q = np.asarray(entry["q_rad"], dtype=float)
        solved = np.array(q)
        solved[perm] = resolve_swing(old, new, q[perm][None, :])[0]
        entry["q_rad"] = [float(v) for v in solved]
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
