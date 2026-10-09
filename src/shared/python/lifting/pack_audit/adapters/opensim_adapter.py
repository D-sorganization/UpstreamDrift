"""OpenSim adapter: loads the pack ``.osim`` and reports canonical-frame FK."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import opensim as osim

from ..frames import from_canonical, to_canonical
from ..model import CoordinateInfo, EngineAdapter, PoseEval
from ._common import PELVIS_ORIGIN, build_model_text, native_angles

_KINDS = {1: "rotational", 2: "translational", 3: "coupled"}


def _vec3(v: Any) -> list[float]:
    return [float(v.get(i)) for i in range(3)]


class OpensimAdapter(EngineAdapter):
    """OSIM model loaded with ``opensim.Model``; Y-up, rotated to canonical."""

    engine = "opensim"
    frame = "y_up"

    def __init__(self, pack: Any, lift: str, anthro: Any) -> None:
        super().__init__(pack, lift, anthro)
        self._xml = build_model_text(pack, lift, anthro)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{lift}.osim"
            path.write_text(self._xml, encoding="utf-8")
            old = osim.Logger.getLevelString()
            osim.Logger.setLevelString("Error")
            try:
                self.model = osim.Model(os.fspath(path))
            finally:
                osim.Logger.setLevelString(old)
        self.state = self.model.initSystem()

    def model_text(self) -> str:
        return self._xml

    def _bodies(self) -> list[str]:
        bs = self.model.getBodySet()
        return [bs.get(i).getName() for i in range(bs.getSize())]

    def coordinates(self) -> list[CoordinateInfo]:
        cs = self.model.getCoordinateSet()
        out = []
        for i in range(cs.getSize()):
            c = cs.get(i)
            out.append(
                CoordinateInfo(
                    name=c.getName(),
                    lower=float(c.getRangeMin()),
                    upper=float(c.getRangeMax()),
                    default=float(c.getDefaultValue()),
                    kind=_KINDS.get(int(c.getMotionType()), "other"),
                )
            )
        return out

    def segment_masses(self) -> dict[str, float]:
        bs = self.model.getBodySet()
        return {n: float(bs.get(n).getMass()) for n in self._bodies()}

    def structure(self) -> dict[str, Any]:
        m = self.model
        js = m.getJointSet()
        joints = {
            js.get(i).getName(): js.get(i).getConcreteClassName()
            for i in range(js.getSize())
        }
        cons = m.getConstraintSet()
        constraints = {
            cons.get(i).getName(): cons.get(i).getConcreteClassName()
            for i in range(cons.getSize())
        }
        hands = self._hand_attachment()
        fs = m.getForceSet()
        forces = [fs.get(i).getConcreteClassName() for i in range(fs.getSize())]
        bs = m.getBodySet()
        return {
            "n_bodies": int(m.getNumBodies()),
            "nq": int(m.getNumCoordinates()),
            "nv": int(m.getNumSpeeds()),
            "root_joint": self._root_joint_class(),
            "joint_names": list(joints),
            "bar_root": self._bar_parent(),
            "bar_hand_attachment": hands,
            "weld_constraints": list(constraints),
            "constraints": constraints,
            "n_keyframes": 0,
            "contact": {
                "model": "SmoothSphereHalfSpaceForce"
                if "SmoothSphereHalfSpaceForce" in forces
                else "none",
                "n_contact_forces": len(forces),
            },
            "n_muscles": int(m.getMuscles().getSize()),
            "inertia_diag": {
                n: _vec3(bs.get(n).getInertia().getMoments()) for n in self._bodies()
            },
            "total_mass": float(m.getTotalMass(self.state)),
        }

    def _bar_parent(self) -> str:
        js = self.model.getJointSet()
        for i in range(js.getSize()):
            j = js.get(i)
            if j.getChildFrame().findBaseFrame().getName() == "barbell_shaft":
                parent = j.getParentFrame().findBaseFrame().getName()
                return f"{j.getConcreteClassName()}, child of {parent}"
        return "unattached"

    def _root_joint_class(self) -> str:
        js = self.model.getJointSet()
        for i in range(js.getSize()):
            j = js.get(i)
            if j.getChildFrame().findBaseFrame().getName() == "pelvis":
                return str(j.getConcreteClassName())
        return "unknown"

    def _hand_attachment(self) -> dict[str, str]:
        """Per hand: how the bar is tied to it (joint parent, constraint or none)."""
        m = self.model
        out = {"l": "none", "r": "none"}
        js = m.getJointSet()
        for i in range(js.getSize()):
            j = js.get(i)
            parent = j.getParentFrame().findBaseFrame().getName()
            child = j.getChildFrame().findBaseFrame().getName()
            for side in out:
                if parent == f"hand_{side}" and child.startswith("barbell"):
                    out[side] = f"{j.getConcreteClassName()} (bar is child)"
        cons = m.getConstraintSet()
        for i in range(cons.getSize()):
            c = cons.get(i)
            if c.getConcreteClassName() != "WeldConstraint":
                continue
            weld = osim.WeldConstraint.safeDownCast(c)
            ends = {
                weld.getConnectee(k).findBaseFrame().getName()
                for k in ("frame1", "frame2")
            }
            for side in out:
                if f"hand_{side}" in ends and any(
                    e.startswith("barbell") for e in ends
                ):
                    out[side] = "WeldConstraint (closure)"
        return out

    def _sole_z(self, state: Any) -> float | None:
        """Lowest foot contact-sphere surface point (canonical z)."""
        cg = self.model.getContactGeometrySet()
        lows = []
        for i in range(cg.getSize()):
            sph = osim.ContactSphere.safeDownCast(cg.get(i))
            if sph is None:
                continue
            frame = sph.getFrame()
            if not frame.findBaseFrame().getName().startswith("foot"):
                continue
            centre = frame.findStationLocationInGround(state, sph.get_location())
            lows.append(float(self._canon(centre)[2]) - float(sph.getRadius()))
        return min(lows) if lows else None

    def _canon(self, v: Any) -> np.ndarray:
        return to_canonical(self.frame, _vec3(v))

    def _closure(self, state: Any) -> dict[str, float] | None:
        cons = self.model.getConstraintSet()
        if cons.getSize() == 0:
            return None
        self.model.realizePosition(state)
        err = state.getQErr()
        n = err.size()
        vals = np.array([err.get(i) for i in range(n)])
        # Weld QErr is [rotation(3), translation(3)] per constraint; the
        # translation block is reported as the closure residual in metres.
        return {
            "weld_translation_norm": float(np.linalg.norm(vals[-3:])),
            "n_qerr": float(n),
        }

    def _assembly_shift(self, state: Any) -> list[str]:
        """Coordinates the engine moved off their default while assembling."""
        cs = self.model.getCoordinateSet()
        moved = []
        for i in range(cs.getSize()):
            c = cs.get(i)
            delta = float(c.getValue(state) - c.getDefaultValue())
            if abs(delta) > 1e-6:
                moved.append(f"{c.getName()} moved {delta:+.4f} by assembly")
        return moved

    def evaluate(self, q: Mapping[str, float] | None) -> PoseEval:
        m = self.model
        state = m.initializeState()
        notes: list[str] = []
        if q is not None:
            cs = m.getCoordinateSet()
            for i in range(cs.getSize()):
                cs.get(i).setValue(state, 0.0, False)
            up = from_canonical(self.frame, PELVIS_ORIGIN)
            if cs.contains("pelvis_tx"):
                for name, value in zip(
                    ("pelvis_tx", "pelvis_ty", "pelvis_tz"), up, strict=True
                ):
                    cs.get(name).setValue(state, float(value), False)
            else:
                notes.append("pelvis has no free joint; root left at the model pose")
            for name, angle in native_angles(self.engine, q).items():
                if not cs.contains(name):
                    raise ValueError(f"OpenSim model has no coordinate {name!r}")
                cs.get(name).setValue(state, angle, False)
            notes.append("constraints not enforced; weld residual reported")
        else:
            notes.extend(self._assembly_shift(state))
        m.realizePosition(state)
        bs = m.getBodySet()
        positions = {
            n: self._canon(bs.get(n).getTransformInGround(state).p())
            for n in self._bodies()
        }
        com = self._canon(m.calcMassCenterPosition(state))
        return PoseEval(
            positions=positions,
            com=com,
            total_mass=float(m.getTotalMass(state)),
            closure=self._closure(state),
            notes=notes,
            sole_z=self._sole_z(state),
        )

    def start_contact_force(self) -> dict[str, Any]:
        m = self.model
        state = m.initializeState()
        m.realizeDynamics(state)
        forces = m.getForceSet()
        total = np.zeros(3)
        for i in range(forces.getSize()):
            rec = forces.get(i).getRecordValues(state)
            total += np.array([rec.get(k) for k in range(3)])
        vertical = float(to_canonical(self.frame, total.tolist())[2])
        return {"value_n": vertical, "n_forces": int(forces.getSize()), "reason": None}

    def smoke_step(self) -> dict[str, Any]:
        m = self.model
        state = m.initializeState()
        horizon = 0.05
        try:
            manager = osim.Manager(m)
            state.setTime(0.0)
            manager.initialize(state)
            final = manager.integrate(horizon)
            m.realizePosition(final)
            ok = bool(
                np.all(np.isfinite([final.getY().get(i) for i in range(final.getNY())]))
            )
            bar_z = self._canon(
                m.getBodySet().get("barbell_shaft").getTransformInGround(final).p()
            )[2]
            return {
                "loaded": True,
                "stepped": ok,
                "horizon_s": horizon,
                "bar_z_end": float(bar_z),
            }
        except RuntimeError as exc:
            return {
                "loaded": True,
                "stepped": False,
                "horizon_s": horizon,
                "error": str(exc)[:300],
            }
