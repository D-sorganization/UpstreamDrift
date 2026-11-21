"""MuJoCo provider for per-segment angular momentum about the whole-body COM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .segment_angular_momentum import (
    REFERENCE_POINT,
    SegmentAngularMomentumFrame,
    SegmentGroupCatalog,
    body_angular_momentum_about_point,
    build_segment_angular_momentum_frame,
)

if TYPE_CHECKING:
    import mujoco


def default_root_body_id(model: mujoco.MjModel) -> int:
    """Return the lightest non-world body that roots the mechanism (default 1)."""
    for body_id in range(1, model.nbody):
        if float(model.body(body_id).mass[0]) > 0.0:
            return body_id
    return 1


def whole_body_com_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qvel: np.ndarray,
    *,
    root_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-body COM position and velocity from the MuJoCo subtree field."""
    import mujoco

    mujoco.mj_subtreeVel(model, data)
    com = np.asarray(data.subtree_com[root_body_id], dtype=float)
    linvel = np.asarray(data.subtree_linvel[root_body_id], dtype=float)
    return com, linvel


def mujoco_subtree_angular_momentum(
    model: mujoco.MjModel, data: mujoco.MjData, *, root_body_id: int
) -> np.ndarray:
    """Independent whole-body angular momentum reference (subtree about subtree COM)."""
    import mujoco

    mujoco.mj_subtreeVel(model, data)
    return np.asarray(data.subtree_angmom[root_body_id], dtype=float).copy()


def _body_spin_and_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    import mujoco

    body = model.body(body_id)
    mass = float(body.mass[0])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    com_vel = jacp @ qvel
    ang_vel_world = jacr @ qvel
    com_pos = np.asarray(data.xipos[body_id], dtype=float)
    inertia = np.asarray(body.inertia, dtype=float)
    rot = np.asarray(data.ximat[body_id], dtype=float).reshape(3, 3)
    return com_pos, com_vel, ang_vel_world, inertia, mass, rot


class MujocoSegmentAngularMomentumProvider:
    """``SegmentAngularMomentumProvider`` backed by native MuJoCo body kinematics."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        catalog: SegmentGroupCatalog,
        time_s: float = 0.0,
        root_body_id: int | None = None,
    ) -> None:
        if model is None or data is None:
            raise ValueError("model and data must be provided")
        if not isinstance(catalog, SegmentGroupCatalog):
            raise TypeError("catalog must be a SegmentGroupCatalog")
        if root_body_id is None:
            root_body_id = default_root_body_id(model)
        if root_body_id < 1 or root_body_id >= model.nbody:
            raise ValueError("root_body_id must name a non-world body")
        self.model = model
        self.data = data
        self.catalog = catalog
        self.time_s = float(time_s)
        self.root_body_id = int(root_body_id)

    def get_segment_angular_momentum(self) -> SegmentAngularMomentumFrame | None:
        contributions: dict[str, np.ndarray | None] = {}
        spin_only: dict[str, np.ndarray | None] = {}
        qvel = np.asarray(self.data.qvel, dtype=float)
        ref_com, ref_vel = whole_body_com_kinematics(
            self.model, self.data, qvel, root_body_id=self.root_body_id
        )
        qualified = False
        for body_id in range(1, self.model.nbody):
            name = self.model.body(body_id).name
            if not name:
                continue
            com_pos, com_vel, ang_vel, inertia, mass, rot = _body_spin_and_motion(
                self.model, self.data, body_id, qvel
            )
            if mass <= 0.0 or not np.isfinite(inertia).all():
                contributions[name] = None
                spin_only[name] = None
                continue
            contrib, spin = body_angular_momentum_about_point(
                com_world=com_pos,
                com_vel_world=com_vel,
                ang_vel_world=ang_vel,
                inertia_principal=inertia,
                rot_body_to_world=rot,
                mass=mass,
                reference_com_world=ref_com,
                reference_com_vel_world=ref_vel,
            )
            contributions[name] = contrib
            spin_only[name] = spin
            qualified = True
        if not qualified:
            return None
        source = (
            f"mujoco per-body H about {REFERENCE_POINT} "
            f"(mj_jacBodyCom, xipos, ximat, body.inertia)"
        )
        return build_segment_angular_momentum_frame(
            time_s=self.time_s,
            source=source,
            contributions=contributions,
            spin_only=spin_only,
            catalog=self.catalog,
        )
