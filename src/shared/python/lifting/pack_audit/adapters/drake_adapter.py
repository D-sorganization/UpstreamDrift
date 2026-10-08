"""Drake adapter: loads the pack SDF through the pack's own ``load_sdf``."""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..model import CoordinateInfo, EngineAdapter, PoseEval
from ._common import PELVIS_ORIGIN, build_model_text, native_angles


class DrakeAdapter(EngineAdapter):
    """SDFormat loaded in a ``MultibodyPlant``; z-up."""

    engine = "drake"
    frame = "z_up"

    def __init__(self, pack: Any, lift: str, anthro: Any) -> None:
        super().__init__(pack, lift, anthro)
        self._xml = build_model_text(pack, lift, anthro)
        loader = importlib.import_module(f"{pack.package}.loader")
        self._loader = loader
        self.loaded = loader.load_sdf(self._xml, lift, time_step=0.0)
        self.plant = self.loaded.plant

    def model_text(self) -> str:
        return self._xml

    def _bodies(self) -> list[str]:
        p = self.plant
        names = [
            p.get_body(i).name() for i in p.GetBodyIndices(self.loaded.model_instance)
        ]
        return [n for n in names if n != "ground_plane"]

    def _revolute_joints(self) -> list[Any]:
        p = self.plant
        return [
            p.get_joint(i)
            for i in p.GetJointIndices(self.loaded.model_instance)
            if p.get_joint(i).type_name() == "revolute"
        ]

    def coordinates(self) -> list[CoordinateInfo]:
        out = []
        for j in self._revolute_joints():
            out.append(
                CoordinateInfo(
                    name=j.name(),
                    lower=float(j.position_lower_limits()[0]),
                    upper=float(j.position_upper_limits()[0]),
                    default=float(j.default_positions()[0]),
                    kind="hinge",
                )
            )
        if self.plant.GetBodyByName("pelvis").is_floating_base_body():
            out.insert(
                0, CoordinateInfo("pelvis_floating_base", None, None, None, "free")
            )
        return out

    def segment_masses(self) -> dict[str, float]:
        p = self.plant
        return {n: float(p.GetBodyByName(n).default_mass()) for n in self._bodies()}

    def _hand_attachment(self) -> dict[str, str]:
        out = {"l": "none", "r": "none"}
        p = self.plant
        for i in p.GetJointIndices(self.loaded.model_instance):
            j = p.get_joint(i)
            parent, child = j.parent_body().name(), j.child_body().name()
            for side in out:
                if parent == f"hand_{side}" and child.startswith("barbell"):
                    out[side] = f"{j.type_name()} joint (bar is child)"
        return out

    def _bar_parent(self) -> str:
        p = self.plant
        for i in p.GetJointIndices(self.loaded.model_instance):
            j = p.get_joint(i)
            if j.child_body().name() == "barbell_shaft":
                return f"{j.type_name()} joint, child of {j.parent_body().name()}"
        return "unattached"

    def structure(self) -> dict[str, Any]:
        p = self.plant
        root = ET.fromstring(self._xml)  # noqa: S314 - own generated SDF
        contacts = [
            c.get("name", "")
            for c in root.iter("collision")
            if c.get("name", "").endswith("_contact")
        ]
        inertia = {}
        for n in self._bodies():
            moments = p.GetBodyByName(n).default_rotational_inertia().get_moments()
            inertia[n] = [float(x) for x in moments]
        joints = [
            p.get_joint(i).name() for i in p.GetJointIndices(self.loaded.model_instance)
        ]
        return {
            "n_bodies": len(self._bodies()),
            "nq": int(p.num_positions()),
            "nv": int(p.num_velocities()),
            "root_joint": "free (6 dof, implicit)"
            if p.GetBodyByName("pelvis").is_floating_base_body()
            else "welded",
            "joint_names": joints,
            "bar_root": self._bar_parent(),
            "bar_hand_attachment": self._hand_attachment(),
            "weld_constraints": [j for j in joints if "barbell" in j and "hand" in j],
            "n_keyframes": 0,
            "initial_pose": (
                self.loaded.initial_pose.name if self.loaded.initial_pose else None
            ),
            "contact": {
                "model": "hydroelastic (compliant) foot boxes",
                "foot_contact_geoms": contacts,
            },
            "inertia_diag": inertia,
            "total_mass": float(sum(self.segment_masses().values())),
        }

    def evaluate(self, q: Mapping[str, float] | None) -> PoseEval:
        p = self.plant
        ctx = p.CreateDefaultContext()
        notes: list[str] = []
        if q is not None:
            transforms = importlib.import_module("pydrake.math")
            angles = native_angles(self.engine, q)
            for j in self._revolute_joints():
                j.set_angle(ctx, float(angles.pop(j.name(), 0.0)))
            if angles:
                raise ValueError(f"Drake model has no joint(s) {sorted(angles)}")
            pelvis = p.GetBodyByName("pelvis")
            if pelvis.is_floating_base_body():
                p.SetFreeBodyPose(
                    ctx, pelvis, transforms.RigidTransform(list(PELVIS_ORIGIN))
                )
            else:
                notes.append("pelvis is welded; root left at the model pose")
        positions = {
            n: np.array(p.EvalBodyPoseInWorld(ctx, p.GetBodyByName(n)).translation())
            for n in self._bodies()
        }
        com = np.array(p.CalcCenterOfMassPositionInWorld(ctx))
        heights = self._loader._sole_corner_heights(p, self.loaded.scene_graph, ctx)
        return PoseEval(
            positions=positions,
            com=com,
            total_mass=float(sum(self.segment_masses().values())),
            closure=None,
            notes=notes,
            sole_z=min(heights) if heights else None,
        )

    def start_contact_force(self) -> dict[str, Any]:
        return {
            "value_n": None,
            "reason": "pack exposes no contact-results API (Drake_Models#371)",
        }

    def smoke_step(self) -> dict[str, Any]:
        mod_plant = importlib.import_module("pydrake.multibody.plant")
        analysis = importlib.import_module("pydrake.systems.analysis")
        systems = importlib.import_module("pydrake.systems.framework")
        builder = systems.DiagramBuilder()
        plant, sg = mod_plant.AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
        parsing = importlib.import_module("pydrake.multibody.parsing")
        parsing.Parser(plant).AddModelsFromString(self._xml, "sdf")
        plant.Finalize()
        loader = self._loader
        loader.apply_initial_pose(
            plant, loader.parse_initial_pose(self._xml), scene_graph=sg
        )
        diagram = builder.Build()
        sim = analysis.Simulator(diagram)
        horizon = 0.05
        try:
            sim.AdvanceTo(horizon)
        except RuntimeError as exc:
            return {
                "loaded": True,
                "stepped": False,
                "horizon_s": horizon,
                "error": str(exc)[:300],
            }
        ctx = diagram.GetSubsystemContext(plant, sim.get_context())
        q = plant.GetPositions(ctx)
        bar = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName("barbell_shaft"))
        return {
            "loaded": True,
            "stepped": bool(np.all(np.isfinite(q))),
            "horizon_s": horizon,
            "dt": 1e-3,
            "bar_z_end": float(bar.translation()[2]),
        }
