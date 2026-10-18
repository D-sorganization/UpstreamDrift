"""MuJoCo section reactions for unambiguous segment axes, using scratch mjData.

Native force layout and spatial-tendon limitation:
https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rnepostconstraint
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from .axial_loads import AxialLoadFrame, axial_force_from_proximal_reaction


AXIS_POLICIES = ("rod", "kinematic")
_ANCHOR_TOL = 1e-9
_MIN_AXIS_LENGTH_M = 1e-6


class MujocoAxialLoadSource:
    """Extract proximal reactions for named bodies with an unambiguous axis.

    ``axis_policy="rod"`` (default): the body has one capsule/cylinder whose
    endpoint coincides with the proximal joint(s), or the origin for a fixed
    body. This defines the axis without guessing from body names or mesh
    extents.

    ``axis_policy="kinematic"``: the rod rule first; bodies it does not qualify
    use the kinematic tree instead (a topological rule; names and meshes are
    never consulted). The proximal point is the anchor of the body's FIRST joint,
    the one nearest the parent, where the parent transmits load; later joints in
    the same body only add articulation inside the body, so offsets between joint
    anchors do not move the load-carrying section. The distal point is the
    body-frame origin of the single child ("kinematic"), the centroid of the child
    body origins when there are 2+ children ("kinematic_centroid"), or the body COM
    when there are none ("kinematic"). Axes shorter than 1e-6 m are skipped.
    Bodies with a free joint, and root bodies parented to the world with three or
    more slide joints (a translational root), are always skipped: their proximal
    reaction is the residual "hand-of-god" force that drives the root, not a
    section load. The rod rule is unchanged and still requires a common anchor.

    Free bodies, ambiguous geometry, tendons and callbacks are unavailable.
    ``axis_sources`` maps each qualified body name to the rule ("rod" or
    "kinematic") that defined its axis, so an overlay can be audited.
    Rebuild this adapter whenever the model is replaced or edited.
    """

    def __init__(self, model: Any, *, axis_policy: str = "rod") -> None:
        import mujoco

        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be MjModel")
        if axis_policy not in AXIS_POLICIES:
            raise ValueError(
                f"axis_policy must be one of {AXIS_POLICIES}, got {axis_policy!r}"
            )
        self.model = model
        self.axis_policy = axis_policy
        self._native = mujoco
        self._scratch = mujoco.MjData(model)
        self._axes, sources = self._discover_axes()
        self._axis_sources = MappingProxyType(sources)

    @property
    def axis_sources(self) -> Mapping[str, str]:
        """Read-only body name -> "rod" | "kinematic" for every qualified body.

        Postcondition: keys equal the body names that appear in sampled frames.
        """
        return self._axis_sources

    def _proximal_anchor(self, body: int, *, first_joint: bool) -> np.ndarray | None:
        """Joint anchor of a body (origin if jointless); None if unusable.

        None means a free joint is present, or (unless ``first_joint``) the joints
        do not share one anchor. With ``first_joint`` the first joint's anchor wins.
        """
        model = self.model
        free = int(self._native.mjtJoint.mjJNT_FREE)
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        joints = list(range(start, start + count))
        if any(int(model.jnt_type[j]) == free for j in joints):
            return None
        anchor = model.jnt_pos[joints[0]] if joints else np.zeros(3)
        if not first_joint and any(
            not np.allclose(model.jnt_pos[j], anchor, atol=_ANCHOR_TOL, rtol=0)
            for j in joints
        ):
            return None
        return np.array(anchor, dtype=float)

    def _rod_axis(self, body: int, anchor: np.ndarray) -> tuple | None:
        model, native = self.model, self._native
        # int() both sides: numpy ints vs pybind enums compare differently by version.
        rods = {int(native.mjtGeom.mjGEOM_CAPSULE), int(native.mjtGeom.mjGEOM_CYLINDER)}
        if model.body_geomnum[body] != 1:
            return None
        geom = model.body_geomadr[body]
        if int(model.geom_type[geom]) not in rods:
            return None
        rotation = np.empty(9)
        native.mju_quat2Mat(rotation, model.geom_quat[geom])
        half_axis = rotation.reshape(3, 3)[:, 2] * model.geom_size[geom, 1]
        first, second = (
            model.geom_pos[geom] - half_axis,
            model.geom_pos[geom] + half_axis,
        )
        if np.allclose(first, anchor, atol=_ANCHOR_TOL, rtol=0):
            return first.copy(), second.copy()
        if np.allclose(second, anchor, atol=_ANCHOR_TOL, rtol=0):
            return second.copy(), first.copy()
        return None

    def _is_translational_root(self, body: int) -> bool:
        model = self.model
        if model.body_parentid[body] != 0:
            return False
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        slide = int(self._native.mjtJoint.mjJNT_SLIDE)
        kinds = [int(model.jnt_type[j]) for j in range(start, start + count)]
        return kinds.count(slide) >= 3

    def _kinematic_axis(self, body: int, anchor: np.ndarray) -> tuple | None:
        model = self.model
        children = np.flatnonzero(model.body_parentid[1:] == body) + 1
        rule = "kinematic"
        if len(children) > 1:
            distal = np.mean(model.body_pos[children], axis=0)
            rule = "kinematic_centroid"
        elif len(children) == 1:
            distal = np.array(model.body_pos[children[0]], dtype=float)
        else:
            distal = np.array(model.body_ipos[body], dtype=float)
        if float(np.linalg.norm(distal - anchor)) < _MIN_AXIS_LENGTH_M:
            return None
        return anchor.copy(), distal, rule

    def _discover_axes(
        self,
    ) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[str, str]]:
        model, native = self.model, self._native
        axes: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        sources: dict[str, str] = {}
        for body in range(1, model.nbody):
            name = native.mj_id2name(model, native.mjtObj.mjOBJ_BODY, body)
            if not name:
                continue
            found, rule = None, "rod"
            strict = self._proximal_anchor(body, first_joint=False)
            if strict is not None:
                found = self._rod_axis(body, strict)
            if found is None and self.axis_policy == "kinematic":
                anchor = self._proximal_anchor(body, first_joint=True)
                if anchor is not None and not self._is_translational_root(body):
                    found = self._kinematic_axis(body, anchor)
                    rule = found[2] if found else rule
            if found is None:
                continue
            axes[body] = (name, found[0], found[1])
            sources[name] = rule
        return axes, sources

    def _available(self) -> bool:
        native, model = self._native, self.model
        if not self._axes or model.ntendon or model.nplugin:
            return False
        callbacks = (
            native.get_mjcb_control,
            native.get_mjcb_passive,
            native.get_mjcb_sensor,
            native.get_mjcb_contactfilter,
            native.get_mjcb_act_bias,
            native.get_mjcb_act_dyn,
            native.get_mjcb_act_gain,
        )
        return not any(get_callback() is not None for get_callback in callbacks)

    def _project(self, time_s: float) -> AxialLoadFrame:
        """Project scratch cfrc_int onto each qualified axis (scratch is solved)."""
        scratch = self._scratch
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
            float(time_s),
            values,
            "MuJoCo cfrc_int parent-on-body reaction; proximal section of "
            f"{self.axis_policy}-policy segment axis; world XYZ",
        )

    def sample(self, data: Any) -> AxialLoadFrame | None:
        """Compute current forces on scratch data; never modify live state/model."""
        native, model = self._native, self.model
        if not isinstance(data, native.MjData) or data.model is not model:
            raise TypeError("data must belong to this source's model")
        if not self._available():
            return None
        from src.shared.python.engine_core.mujoco_compat import copy_mjdata_state

        copy_mjdata_state(self._scratch, data)
        native.mj_forward(model, self._scratch)
        native.mj_rnePostConstraint(model, self._scratch)
        return self._project(data.time)

    def _checked(self, name: str, value: Any, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def sample_prescribed(
        self, qpos: Any, qvel: Any, qacc: Any, time_s: float
    ) -> AxialLoadFrame | None:
        """Axial loads for a prescribed motion via inverse dynamics.

        Use when qpos/qvel/qacc come from a fit rather than forward simulation.
        Inputs are copied (never modified); ``mj_inverse`` then
        ``mj_rnePostConstraint`` run on scratch data with the prescribed qacc.
        Preconditions: shapes (nq,), (nv,), (nv,); all finite; finite ``time_s``
        (ValueError otherwise). Postcondition: same projection and sign as
        ``sample`` (tension positive); returns None when the model is unsupported.
        """
        model, native, scratch = self.model, self._native, self._scratch
        q = self._checked("qpos", qpos, model.nq)
        v = self._checked("qvel", qvel, model.nv)
        a = self._checked("qacc", qacc, model.nv)
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if not self._available():
            return None
        native.mj_resetData(model, scratch)
        scratch.qpos[:], scratch.qvel[:], scratch.qacc[:] = q, v, a
        scratch.time = float(time_s)
        native.mj_inverse(model, scratch)
        scratch.qacc[:] = a  # mj_inverse consumes qacc; make the invariant explicit
        native.mj_rnePostConstraint(model, scratch)
        return self._project(time_s)
