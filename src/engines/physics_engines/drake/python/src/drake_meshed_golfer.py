"""The v2 full-body golfer in Drake: the same model and meshes MuJoCo renders.

The Drake GUI used to open the primitive-box ``GolfURDFGenerator`` model. This
module builds the specification robot (the model the best fits use) and
dresses it with the engine-neutral body meshes of
``model_appearance.body_meshes`` so MeshCat shows the same golfer as MuJoCo.
Visual geometry only; the plant's dynamics are those of the specification.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.model_appearance.schema import AppearanceDocument
from src.tools.native_viewer_export.backends._body import scene_appearance
from src.tools.native_viewer_export.backends._meshcat_scene import (
    SceneVisuals,
    build_scene_visuals,
    check_bodies,
)
from src.tools.native_viewer_export.backends._subprocess import export_urdf


@dataclass
class MeshedGolfer:
    """A finalized Drake plant of the specification golfer and its diagram."""

    plant: Any
    diagram: Any
    scene_graph: Any
    instance: Any
    body_links: dict[str, str]
    solid_links: dict[str, str]
    coordinate_order: tuple[str, ...]
    scene: SceneVisuals
    mesh_dir: Any  # TemporaryDirectory: MeshCat reads the OBJ files lazily


@dataclass(frozen=True)
class ParityReport:
    """Drake versus MuJoCo built from one specification."""

    missing_in_drake: tuple[str, ...]
    missing_in_mujoco: tuple[str, ...]
    max_body_mass_error_kg: float
    total_mass_error_kg: float


def register_scene_visuals(
    plant: Any, inst: Any, links: dict[str, str], scene: SceneVisuals
) -> None:
    """Register the shared scene items as Drake visual geometry (spec frames)."""
    from pydrake.geometry import Box, Capsule, Ellipsoid, Mesh
    from pydrake.math import RigidTransform, RotationMatrix

    for item in scene.items:
        if item.kind == "mesh":
            shape: Any = Mesh(item.mesh_path)
        elif item.kind == "capsule":
            shape = Capsule(*item.size)
        elif item.kind == "ellipsoid":
            shape = Ellipsoid(*item.size)
        else:
            shape = Box(*item.size)
        plant.RegisterVisualGeometry(
            plant.GetBodyByName(links[item.body], inst),
            RigidTransform(RotationMatrix(item.rotation), item.translation),
            shape,
            item.name,
            np.array(item.rgba),
        )


def build_meshed_golfer(
    spec_bytes: bytes,
    meshcat: Any | None = None,
    appearance: AppearanceDocument | None = None,
) -> MeshedGolfer:
    """Build the specification golfer in Drake with shared body meshes.

    Preconditions: ``spec_bytes`` is a full-body specification document.
    Postconditions: the plant is finalized and welded to the world; every
    visual sits on a plant body; a ``MeshcatVisualizer`` is wired when
    ``meshcat`` is given. ``appearance`` defaults to the library appearance.
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be bytes")
    from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, Role
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    xml, meta = export_urdf(bytes(spec_bytes))
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    inst = Parser(plant).AddModelsFromString(xml, "urdf")[0]
    links = dict(meta["body_links"])
    plant.WeldFrames(
        plant.world_frame(), plant.GetBodyByName(links["world"], inst).body_frame()
    )
    mesh_dir = tempfile.TemporaryDirectory(prefix="ud_visuals_")
    scene = build_scene_visuals(
        json.loads(spec_bytes),
        Path(mesh_dir.name),
        appearance or scene_appearance(False, None),
    )
    check_bodies(scene, links)
    register_scene_visuals(plant, inst, links, scene)
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            meshcat,
            MeshcatVisualizerParams(role=Role.kIllustration),
        )
    return MeshedGolfer(
        plant,
        builder.Build(),
        scene_graph,
        inst,
        links,
        dict(meta["solid_links"]),
        tuple(meta["coordinate_order"]),
        scene,
        mesh_dir,
    )


def drake_mujoco_parity(spec_bytes: bytes) -> ParityReport:
    """Compare Drake and MuJoCo built from the same specification.

    Checks that every spec body and every coordinate joint exists in both
    engines and that per-body and total masses agree.
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be bytes")
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    golfer = build_meshed_golfer(bytes(spec_bytes))
    plant = golfer.plant
    ctx = plant.CreateDefaultContext()
    xml, _ = export_full_body_mjcf(bytes(spec_bytes))
    model = mujoco.MjModel.from_xml_string(xml)

    mj_body = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): float(model.body_mass[i])
        for i in range(model.nbody)
    }
    mj_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    }
    # The URDF carries inertia on rigidly attached solid links; credit each to
    # the body link it is welded to.
    carried_mass = {link: 0.0 for link in golfer.body_links.values()}
    for index in plant.GetJointIndices(golfer.instance):
        joint = plant.get_joint(index)
        parent, child = joint.parent_body().name(), joint.child_body().name()
        if parent in carried_mass and child not in carried_mass:
            carried_mass[parent] += plant.get_body(joint.child_body().index()).get_mass(
                ctx
            )
    for link in carried_mass:
        carried_mass[link] += plant.GetBodyByName(link, golfer.instance).get_mass(ctx)
    missing_drake: list[str] = []
    missing_mujoco: list[str] = []
    worst = 0.0
    for body, link in golfer.body_links.items():
        if body == "world":
            continue
        if body not in mj_body:
            missing_mujoco.append(body)
            continue
        mass = carried_mass[link]
        worst = max(worst, abs(mass - mj_body[body]))
    for name in golfer.coordinate_order:
        if not plant.HasJointNamed(name, golfer.instance):
            missing_drake.append(name)
        if name not in mj_joints:
            missing_mujoco.append(name)
    total = abs(float(plant.CalcTotalMass(ctx)) - float(model.body_mass.sum()))
    return ParityReport(tuple(missing_drake), tuple(missing_mujoco), worst, total)
