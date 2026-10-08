"""Pinocchio adapter: loads the pack URDF (with a free-flyer root)."""

from __future__ import annotations

import importlib
from defusedxml import ElementTree as ET
from collections.abc import Mapping
from typing import Any

import numpy as np

pin: Any = importlib.import_module("pinocchio")

from ..model import CoordinateInfo, EngineAdapter, PoseEval
from ._common import PELVIS_ORIGIN, build_model_text, native_angles


class PinocchioAdapter(EngineAdapter):
    """URDF loaded with ``pin.buildModelFromXML``; z-up.

    A URDF cannot express a floating base, so the audit adds
    ``JointModelFreeFlyer`` exactly as the pack's own Pink/Crocoddyl addons do.
    """

    engine = "pinocchio"
    frame = "z_up"

    def __init__(self, pack: Any, lift: str, anthro: Any) -> None:
        super().__init__(pack, lift, anthro)
        self._xml = build_model_text(pack, lift, anthro)
        self._root = ET.fromstring(self._xml)
        self.model = pin.buildModelFromXML(self._xml, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

    def model_text(self) -> str:
        return self._xml

    def _links(self) -> dict[str, ET.Element]:
        return {e.get("name", ""): e for e in self._root.findall("link")}

    def _bodies(self) -> list[str]:
        return [
            n
            for n in self._links()
            if not n.endswith(("_1", "_2")) or n.startswith("barbell")
        ]

    def _initial_q(self) -> np.ndarray:
        helpers = importlib.import_module(
            f"{self.pack.package}.shared.utils.urdf_helpers"
        )
        return np.array(helpers.get_initial_configuration(self.model, self._xml))

    def coordinates(self) -> list[CoordinateInfo]:
        m, out = self.model, []
        q0 = self._initial_q()
        for jid in range(1, m.njoints):
            name = m.names[jid]
            if name == "root_joint":
                out.append(CoordinateInfo("root_joint", None, None, None, "free"))
                continue
            idx = int(m.joints[jid].idx_q)
            out.append(
                CoordinateInfo(
                    name=name,
                    lower=float(m.lowerPositionLimit[idx]),
                    upper=float(m.upperPositionLimit[idx]),
                    default=float(q0[idx]),
                    kind="hinge",
                )
            )
        return out

    def segment_masses(self) -> dict[str, float]:
        out = {}
        for name, link in self._links().items():
            mass = link.find("inertial/mass")
            out[name] = float(mass.get("value", "nan")) if mass is not None else 0.0
        return out

    def _hand_attachment(self) -> dict[str, str]:
        out = {"l": "none", "r": "none"}
        for joint in self._root.findall("joint"):
            parent = joint.find("parent").get("link", "")  # type: ignore[union-attr]
            child = joint.find("child").get("link", "")  # type: ignore[union-attr]
            for side in out:
                if parent == f"hand_{side}" and child.startswith("barbell"):
                    out[side] = f"{joint.get('type')} joint (bar is child)"
            for side in out:
                if (
                    out[side] == "none"
                    and f"hand_{side}" in joint.get("name", "")
                    and parent.startswith("barbell")
                ):
                    out[side] = f"none (only dummy frame {child})"
        return out

    def _bar_parent(self) -> str:
        for joint in self._root.findall("joint"):
            if joint.find("child").get("link", "") == "barbell_shaft":  # type: ignore[union-attr]
                parent = joint.find("parent").get("link", "")  # type: ignore[union-attr]
                return f"{joint.get('type')} joint, child of {parent}"
        return "unattached"

    def structure(self) -> dict[str, Any]:
        links = self._links()
        xml_joints = {
            j.get("name", ""): j.get("type", "") for j in self._root.findall("joint")
        }
        feet = [
            n for n in ("foot_l", "foot_r") if links[n].find("collision") is not None
        ]
        masses = self.segment_masses()
        inertia = {}
        for name, link in links.items():
            node = link.find("inertial/inertia")
            if node is not None:
                inertia[name] = [
                    float(node.get(k, "nan")) for k in ("ixx", "iyy", "izz")
                ]
        return {
            "n_bodies": len(links),
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "root_joint": "none in URDF (consumer adds free-flyer; audit used JointModelFreeFlyer)",
            "joint_names": list(xml_joints),
            "bar_root": self._bar_parent(),
            "bar_hand_attachment": self._hand_attachment(),
            "weld_constraints": [
                n for n, t in xml_joints.items() if t == "fixed" and "hand" in n
            ],
            "n_keyframes": 0,
            "contact": {
                "model": "none (URDF has collision boxes only)",
                "foot_collision_links": feet,
            },
            "inertia_diag": inertia,
            "total_mass": float(sum(masses.values())),
            "engine_total_mass": float(pin.computeTotalMass(self.model)),
        }

    def _sole_z(self) -> float | None:
        """Lowest corner of the foot collision boxes (world z)."""
        m, d = self.model, self.data
        lows = []
        for name in ("foot_l", "foot_r"):
            link = self._links()[name]
            for col in link.findall("collision"):
                box = col.find("geometry/box")
                origin = col.find("origin")
                if box is None:
                    continue
                size = [float(x) for x in box.get("size", "").split()]
                xyz = [
                    float(x)
                    for x in (
                        origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                    ).split()
                ]
                placement = d.oMf[m.getFrameId(name, pin.BODY)]
                for sx in (-0.5, 0.5):
                    for sy in (-0.5, 0.5):
                        for sz in (-0.5, 0.5):
                            local = np.array(xyz) + np.array([sx, sy, sz]) * size
                            lows.append(
                                float(
                                    (
                                        placement.rotation @ local
                                        + placement.translation
                                    )[2]
                                )
                            )
        return min(lows) if lows else None

    def evaluate(self, q: Mapping[str, float] | None) -> PoseEval:
        m, d = self.model, self.data
        notes: list[str] = []
        if q is None:
            qv = self._initial_q()
            notes.append("pack start = URDF initial_position overlay; pelvis at origin")
        else:
            qv = pin.neutral(m)
            qv[0:3] = PELVIS_ORIGIN
            for name, angle in native_angles(self.engine, q).items():
                if not m.existJointName(name):
                    raise ValueError(f"Pinocchio model has no joint {name!r}")
                qv[int(m.joints[m.getJointId(name)].idx_q)] = angle
        pin.forwardKinematics(m, d, qv)
        pin.updateFramePlacements(m, d)
        positions = {
            n: np.array(d.oMf[m.getFrameId(n, pin.BODY)].translation)
            for n in self._bodies()
        }
        com = np.array(pin.centerOfMass(m, d, qv))
        return PoseEval(
            positions=positions,
            com=com,
            total_mass=float(pin.computeTotalMass(m)),
            closure=None,
            notes=notes,
            sole_z=self._sole_z(),
        )

    def start_contact_force(self) -> dict[str, Any]:
        return {
            "value_n": None,
            "reason": "URDF carries no contact model (Pinocchio_Models#432)",
        }

    def smoke_step(self) -> dict[str, Any]:
        m, d = self.model, self.data
        qv, v = self._initial_q(), np.zeros(m.nv)
        steps, dt = 100, 1e-3
        tau = np.zeros(m.nv)
        for _ in range(steps):
            a = pin.aba(m, d, qv, v, tau)
            v = v + a * dt
            qv = pin.integrate(m, qv, v * dt)
        finite = bool(np.all(np.isfinite(qv)) and np.all(np.isfinite(v)))
        return {
            "loaded": True,
            "stepped": finite,
            "steps": steps,
            "dt": dt,
            "note": "unactuated free-fall under ABA; no contact model exists in URDF",
            "max_abs_v": float(np.max(np.abs(v))),
        }
