"""MuJoCo section reactions for unambiguous rod geometry, using scratch mjData.

Native force layout and spatial-tendon limitation:
https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rnepostconstraint
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .axial_loads import AxialLoadFrame, axial_force_from_proximal_reaction


class MujocoAxialLoadSource:
    """Extract proximal reactions for named bodies with one capsule/cylinder.

    A rod endpoint must coincide with the proximal joint(s), or the origin for
    a fixed body. This defines the axis without guessing from body names or mesh
    extents. Free bodies, ambiguous geometry, tendons and callbacks are unavailable.
    Rebuild this adapter whenever the model is replaced or edited.
    """

    def __init__(self, model: Any) -> None:
        import mujoco

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be MjModel")
        self.model = model
        self._native = mujoco
        self._scratch = mujoco.MjData(model)
        self._axes = self._discover_axes()

    def _discover_axes(self) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
        model, native = self.model, self._native
        axes = {}
        rods = (native.mjtGeom.mjGEOM_CAPSULE, native.mjtGeom.mjGEOM_CYLINDER)
        for body in range(1, model.nbody):
            name = native.mj_id2name(model, native.mjtObj.mjOBJ_BODY, body)
            if not name or model.body_geomnum[body] != 1:
                continue
            geom = model.body_geomadr[body]
            if model.geom_type[geom] not in rods:
                continue
            start = model.body_jntadr[body]
            count = model.body_jntnum[body]
            joints = list(range(start, start + count))
            if any(model.jnt_type[j] == native.mjtJoint.mjJNT_FREE for j in joints):
                continue
            anchor = model.jnt_pos[joints[0]] if joints else np.zeros(3)
            if any(
                not np.allclose(model.jnt_pos[j], anchor, atol=1e-9, rtol=0)
                for j in joints
            ):
                continue
            rotation = np.empty(9)
            native.mju_quat2Mat(rotation, model.geom_quat[geom])
            half_axis = rotation.reshape(3, 3)[:, 2] * model.geom_size[geom, 1]
            first, second = (
                model.geom_pos[geom] - half_axis,
                model.geom_pos[geom] + half_axis,
            )
            if np.allclose(first, anchor, atol=1e-9, rtol=0):
                axes[body] = (name, first.copy(), second.copy())
            elif np.allclose(second, anchor, atol=1e-9, rtol=0):
                axes[body] = (name, second.copy(), first.copy())
        return axes

    def sample(self, data: Any) -> AxialLoadFrame | None:
        """Compute current forces on scratch data; never modify live state/model."""
        native, model = self._native, self.model
        if not isinstance(data, native.MjData) or data.model is not model:
            raise TypeError("data must belong to this source's model")
        if not self._axes or model.ntendon or model.nplugin:
            return None
        callbacks = (
            native.get_mjcb_control,
            native.get_mjcb_passive,
            native.get_mjcb_sensor,
            native.get_mjcb_contactfilter,
            native.get_mjcb_act_bias,
            native.get_mjcb_act_dyn,
            native.get_mjcb_act_gain,
        )
        if any(get_callback() is not None for get_callback in callbacks):
            return None
        scratch = self._scratch
        native.mj_copyData(scratch, model, data)
        native.mj_forward(model, scratch)
        native.mj_rnePostConstraint(model, scratch)
        values = {}
        for body, (name, proximal, distal) in self._axes.items():
            rotation = scratch.xmat[body].reshape(3, 3)
            origin = scratch.xpos[body]
            values[name] = axial_force_from_proximal_reaction(
                scratch.cfrc_int[body, 3:],
                origin + rotation @ proximal,
                origin + rotation @ distal,
            )
        return AxialLoadFrame(
            float(data.time),
            values,
            "MuJoCo cfrc_int parent-on-body reaction; proximal rod endpoint section; world XYZ",
        )
