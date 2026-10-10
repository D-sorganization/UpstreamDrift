"""MuJoCo adapter: loads the pack MJCF and reports canonical-frame FK."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from ..model import CoordinateInfo, EngineAdapter, PoseEval
from ._common import PELVIS_ORIGIN, build_model_text, native_angles

_BAR_PARTS = ("barbell_shaft", "barbell_left_sleeve", "barbell_right_sleeve")


class MujocoAdapter(EngineAdapter):
    """MJCF model loaded with ``mujoco.MjModel``; z-up, ``qpos0 == ref``."""

    engine = "mujoco"
    frame = "z_up"

    def __init__(self, pack: Any, lift: str, anthro: Any) -> None:
        super().__init__(pack, lift, anthro)
        self._xml = build_model_text(pack, lift, anthro)
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)

    def model_text(self) -> str:
        return self._xml

    def _body(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

    def _bodies(self) -> list[str]:
        return [self.model.body(i).name for i in range(1, self.model.nbody)]

    def coordinates(self) -> list[CoordinateInfo]:
        m, out = self.model, []
        kinds = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        for j in range(m.njnt):
            lim = bool(m.jnt_limited[j])
            adr = int(m.jnt_qposadr[j])
            out.append(
                CoordinateInfo(
                    name=m.joint(j).name,
                    lower=float(m.jnt_range[j, 0]) if lim else None,
                    upper=float(m.jnt_range[j, 1]) if lim else None,
                    default=float(self._start_qpos()[adr]),
                    kind=kinds.get(int(m.jnt_type[j]), "other"),
                    zero_offset=float(m.qpos0[adr]) if int(m.jnt_type[j]) == 3 else 0.0,
                )
            )
        return out

    def _start_qpos(self) -> np.ndarray:
        m = self.model
        return np.array(m.key_qpos[0] if m.nkey else m.qpos0, dtype=float)

    def segment_masses(self) -> dict[str, float]:
        return {n: float(self.model.body_mass[self._body(n)]) for n in self._bodies()}

    def _welds(self) -> list[dict[str, Any]]:
        m, out = self.model, []
        for e in range(m.neq):
            if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD):
                continue
            out.append(
                {
                    "name": m.eq(e).name,
                    "body1": m.body(int(m.eq_obj1id[e])).name,
                    "body2": m.body(int(m.eq_obj2id[e])).name,
                    "relpose": np.array(m.eq_data[e][3:10], dtype=float),
                    "index": e,
                }
            )
        return out

    @staticmethod
    def _bar_root(bar_free: list[str], welds: list[dict[str, Any]]) -> str:
        if not bar_free:
            return "attached"
        others = sorted(
            {
                b
                for w in welds
                for b in (w["body1"], w["body2"])
                if not b.startswith("barbell")
            }
        )
        return "free body, welded to " + ", ".join(others)

    def structure(self) -> dict[str, Any]:
        m = self.model
        welds = self._welds()
        hands = {
            w["body1"]: w["name"] for w in welds if w["body1"].startswith("hand_")
        } | {w["body2"]: w["name"] for w in welds if w["body2"].startswith("hand_")}
        bar_free = [
            m.joint(j).name
            for j in range(m.njnt)
            if int(m.jnt_type[j]) == 0 and m.joint(j).name.startswith("barbell")
        ]
        feet = [
            m.geom(g).name or f"geom{g}"
            for g in range(m.ngeom)
            if m.body(int(m.geom_bodyid[g])).name.startswith("foot")
            and int(m.geom_contype[g]) != 0
        ]
        return {
            "n_bodies": m.nbody - 1,
            "nq": int(m.nq),
            "nv": int(m.nv),
            "root_joint": "free" if int(m.jnt_type[0]) == 0 else "fixed",
            "joint_names": [m.joint(j).name for j in range(m.njnt)],
            "bar_root": self._bar_root(bar_free, welds),
            "bar_hand_attachment": {
                side: ("weld constraint" if f"hand_{side}" in hands else "none")
                for side in ("l", "r")
            },
            "weld_constraints": [w["name"] for w in welds],
            "n_keyframes": int(m.nkey),
            "contact": {
                "model": "engine contact (penalty, soft)",
                "ground_friction": [float(x) for x in self._ground_friction()],
                "foot_contact_geoms": feet,
                "ground_plane": any(
                    int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                    for g in range(m.ngeom)
                ),
            },
            "inertia_diag": {
                n: [float(x) for x in m.body_inertia[self._body(n)]]
                for n in self._bodies()
            },
            "total_mass": float(m.body_mass.sum()),
        }

    def _ground_friction(self) -> np.ndarray:
        m = self.model
        for g in range(m.ngeom):
            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                return np.asarray(m.geom_friction[g])
        return np.full(3, np.nan)

    def _place_bar(self, qpos: np.ndarray) -> None:
        """Satisfy the weld chain from the left hand outward (FK-only helper)."""
        m, d = self.model, self.data
        placed = {n for n in self._bodies() if n not in _BAR_PARTS}
        # Right-hand welds go last so the left hand defines the bar and the
        # right hand is left as a measurable closure residual.
        welds = sorted(self._welds(), key=lambda w: w["body1"] == "hand_r")
        progress = True
        while progress:
            progress = False
            for w in welds:
                if w["body1"] in placed and w["body2"] not in placed:
                    body1, body2 = self._body(w["body1"]), self._body(w["body2"])
                    mujoco.mj_kinematics(m, d)
                    pos = d.xpos[body1] + d.xmat[body1].reshape(3, 3) @ w["relpose"][:3]
                    quat = np.zeros(4)
                    mujoco.mju_mulQuat(quat, d.xquat[body1], w["relpose"][3:7])
                    adr = int(m.jnt_qposadr[int(m.body_jntadr[body2])])
                    qpos[adr : adr + 3] = pos
                    qpos[adr + 3 : adr + 7] = quat
                    d.qpos[:] = qpos
                    placed.add(w["body2"])
                    progress = True

    def _closure(self) -> dict[str, float]:
        m, d = self.model, self.data
        out: dict[str, float] = {}
        rows = [
            (int(d.efc_id[i]), d.efc_pos[i])
            for i in range(d.nefc)
            if int(d.efc_type[i]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        ]
        for w in self._welds():
            if not w["body1"].startswith("hand_"):
                continue
            err = [p for idx, p in rows if idx == w["index"]][:3]
            out[w["body1"][-1]] = float(np.linalg.norm(err)) if err else float("nan")
        return out

    def _sole_z(self) -> float | None:
        """Lowest point of the foot contact boxes (world z)."""
        m, d = self.model, self.data
        lows = []
        for g in range(m.ngeom):
            body = m.body(int(m.geom_bodyid[g])).name
            if not body.startswith("foot") or not (
                int(m.geom_contype[g]) and int(m.geom_conaffinity[g])
            ):
                continue
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            half = float(np.abs(rot[2]) @ m.geom_size[g])
            lows.append(float(d.geom_xpos[g][2]) - half)
        return min(lows) if lows else None

    def _system_com(self) -> np.ndarray:
        """Mass-weighted CoM over every body (the bar is a child of the world)."""
        m, d = self.model, self.data
        mass = m.body_mass[1:]
        return np.asarray(mass @ d.xipos[1:] / mass.sum())

    def evaluate(self, q: Mapping[str, float] | None) -> PoseEval:
        m, d = self.model, self.data
        d.qpos[:] = self._start_qpos()
        notes: list[str] = []
        if q is not None:
            qpos = np.array(m.qpos0, dtype=float)
            root = int(m.jnt_qposadr[0])
            start = self._start_qpos()
            # Keep the pack's own root orientation (the bench press is made
            # supine by the keyframe quaternion, not by body geometry); the
            # root is only re-placed where the pack leaves the lifter free.
            qpos[root + 3 : root + 7] = start[root + 3 : root + 7]
            if self.lift == "bench_press":
                qpos[root : root + 3] = start[root : root + 3]
            else:
                qpos[root : root + 3] = PELVIS_ORIGIN
            for name, angle in native_angles(self.engine, q).items():
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"MuJoCo model has no joint {name!r}")
                qpos[int(m.jnt_qposadr[jid])] += angle
            d.qpos[:] = qpos
            self._place_bar(qpos)
            notes.append(
                "bar placed from its first weld; right-hand weld left as residual"
            )
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        positions = {n: np.array(d.xpos[self._body(n)]) for n in self._bodies()}
        return PoseEval(
            positions=positions,
            com=self._system_com(),
            total_mass=float(m.body_mass.sum()),
            closure=self._closure(),
            notes=notes,
            sole_z=self._sole_z(),
        )

    def start_contact_force(self) -> dict[str, Any]:
        m, d = self.model, self.data
        self.evaluate(None)
        ground = {
            g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == "world"
        }
        wrench, total, n_ground, n_self = np.zeros(6), 0.0, 0, 0
        other = 0.0
        pairs: set[str] = set()
        for c in range(d.ncon):
            contact = d.contact[c]
            if int(contact.geom1) in ground or int(contact.geom2) in ground:
                mujoco.mj_contactForce(m, d, c, wrench)
                total += float(wrench[0])
                n_ground += 1
            else:
                mujoco.mj_contactForce(m, d, c, wrench)
                other += float(wrench[0])
                n_self += 1
                pairs.add(
                    "|".join(
                        sorted(
                            m.body(int(m.geom_bodyid[int(g)])).name
                            for g in (contact.geom1, contact.geom2)
                        )
                    )
                )
        return {
            "non_ground_normal_force_n": other,
            "non_ground_body_pairs": sorted(pairs),
            "value_n": total,
            "n_ground_contacts": n_ground,
            "n_non_ground_contacts": n_self,
            "reason": None,
        }

    def bar_hold_wrench(self) -> dict[str, Any]:
        from .mujoco_bar_hold import bar_hold_wrench as _bar_hold_wrench

        return _bar_hold_wrench(self._xml, self._welds())

    def smoke_step(self) -> dict[str, Any]:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        steps = 200
        for _ in range(steps):
            mujoco.mj_step(m, d)
        finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
        return {
            "loaded": True,
            "stepped": finite,
            "steps": steps,
            "dt": float(m.opt.timestep),
            "pelvis_z_end": float(d.xpos[self._body("pelvis")][2]),
            "bar_z_end": float(d.xpos[self._body("barbell_shaft")][2]),
            "max_abs_qvel": float(np.max(np.abs(d.qvel))),
        }
