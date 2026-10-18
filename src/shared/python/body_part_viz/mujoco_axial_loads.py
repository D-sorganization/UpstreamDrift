"""MuJoCo section reactions for unambiguous segment axes, using scratch mjData.

Native force layout and spatial-tendon limitation:
https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rnepostconstraint
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .axial_loads import AxialLoadFrame, axial_force_from_proximal_reaction


AXIS_POLICIES = ("rod", "kinematic")
_ANCHOR_TOL = 1e-9
_MIN_AXIS_LENGTH_M = 1e-6


class ExternalWrench(Protocol):
    """An external wrench on a named body, applied at a world-frame point.

    Structurally satisfied by ``force_overlay.contracts.OverlayWrench``.
    ``force_n``/``torque_nm`` are world-frame vectors or ``None`` (absent half).
    """

    body: str
    point_m: Sequence[float]
    force_n: Sequence[float] | None
    torque_nm: Sequence[float] | None


@dataclass(frozen=True)
class BodyWrench:
    """World-frame wrench on a named body at a point (``ExternalWrench``)."""

    body: str
    point_m: tuple[float, float, float]
    force_n: tuple[float, float, float] | None
    torque_nm: tuple[float, float, float] | None


def _vec3(value: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class GripClosure:
    """Physical closure wrench of a rigid weld loop, or why it is unavailable.

    ``multiplier_world`` is the 6-D weld multiplier ``[force; torque]`` (world
    frame, N and N m) of the explicit KKT solve: ``+lambda`` acts on body A at
    ``point_a`` and ``-lambda`` on body B at ``point_b``. ``wrenches`` is that
    pair as ``ExternalWrench`` objects. An unavailable closure has no wrenches,
    no multiplier and a non-empty ``reason``; it is never a zero wrench.
    """

    wrenches: tuple[BodyWrench, ...] = ()
    multiplier_world: tuple[float, ...] | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        """True when the closure wrench pair is present."""
        return bool(self.wrenches)

    @classmethod
    def unavailable(cls, reason: str) -> GripClosure:
        """A closure that could not be resolved; ``reason`` must be non-empty."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("an unavailable closure needs a non-empty reason")
        return cls(reason=reason)

    @classmethod
    def from_multiplier(
        cls,
        *,
        body_a: str,
        body_b: str,
        point_a_m: Sequence[float] | np.ndarray,
        point_b_m: Sequence[float] | np.ndarray,
        multiplier: Sequence[float] | np.ndarray,
    ) -> GripClosure:
        """Build the +/- pair from a 6-D multiplier ``[force; torque]``.

        Postcondition: the two forces and the two torques sum to zero exactly
        (Newton's third law); the pair's moment about any point equals
        ``(point_a - point_b) x force``, i.e. zero when the weld is closed.
        Raises ValueError for a multiplier that is not a finite 6-vector.
        """
        lam = np.asarray(multiplier, dtype=float)
        if lam.shape != (6,):
            raise ValueError(f"multiplier must have shape (6,), got {lam.shape}")
        if not np.isfinite(lam).all():
            raise ValueError("multiplier must be finite")
        pair = (
            BodyWrench(body_a, _vec3(point_a_m), _vec3(lam[:3]), _vec3(lam[3:])),
            BodyWrench(body_b, _vec3(point_b_m), _vec3(-lam[:3]), _vec3(-lam[3:])),
        )
        return cls(pair, tuple(float(x) for x in lam))


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
        self._private: dict[tuple[bool, bool], tuple[Any, Any]] = {}
        self._loop_bodies = self._weld_loop_bodies()
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

    def _project(
        self,
        time_s: float,
        scratch: Any,
        note: str = "",
        unavailable: frozenset[int] = frozenset(),
    ) -> AxialLoadFrame:
        """Project scratch cfrc_int onto each qualified axis (scratch is solved).

        Bodies in ``unavailable`` report ``None``, never a computed value.
        """
        values: dict[str, float | None] = {}
        for body, (name, proximal, distal) in self._axes.items():
            if body in unavailable:
                values[name] = None
                continue
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
            f"{self.axis_policy}-policy segment axis; world XYZ{note}",
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
        return self._project(data.time, self._scratch)

    def _checked(self, name: str, value: Any, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def _private_pair(self, *, contact_off: bool, equality_off: bool) -> tuple:
        """Private model copy with native contact and/or equality disabled.

        The live model is never touched: only this copy carries the flags.
        """
        key = (contact_off, equality_off)
        if key not in self._private:
            bits = self._native.mjtDisableBit
            model = copy.copy(self.model)
            if contact_off:
                model.opt.disableflags |= int(bits.mjDSBL_CONTACT)
            if equality_off:
                model.opt.disableflags |= int(bits.mjDSBL_EQUALITY)
            self._private[key] = (model, self._native.MjData(model))
        return self._private[key]

    def _contactless_pair(self) -> tuple[Any, Any]:
        """Private model copy with native contact disabled, plus its scratch data."""
        return self._private_pair(contact_off=True, equality_off=False)

    def _eq_body(self, objtype: int, obj: int) -> int:
        native = self._native
        if objtype == int(native.mjtObj.mjOBJ_SITE):
            return int(self.model.site_bodyid[obj])
        return int(obj)

    def _weld_loop_bodies(self) -> frozenset[int]:
        """Bodies whose proximal section carries a weld/connect loop reaction.

        For each active weld or connect equality between two bodies, the bodies
        on the tree path between them excluding their lowest common ancestor
        (that body's own parent joint sees only the internal loop force).
        """
        model, native = self.model, self._native
        kinds = {int(native.mjtEq.mjEQ_WELD), int(native.mjtEq.mjEQ_CONNECT)}
        out: set[int] = set()
        for eq in range(model.neq):
            if int(model.eq_type[eq]) not in kinds or not model.eq_active0[eq]:
                continue
            pair = [
                self._eq_body(int(model.eq_objtype[eq]), int(obj))
                for obj in (model.eq_obj1id[eq], model.eq_obj2id[eq])
            ]
            chains = []
            for body in pair:
                chain = [body]
                while chain[-1] != 0:
                    chain.append(int(model.body_parentid[chain[-1]]))
                chains.append(chain)
            common = next(b for b in chains[0] if b in chains[1])
            for chain in chains:
                out.update(b for b in chain[: chain.index(common)] if b != 0)
        return frozenset(out)

    def _apply_external(
        self, model: Any, data: Any, wrenches: Sequence[ExternalWrench]
    ) -> None:
        """Write ``wrenches`` into ``data.xfrc_applied`` (MuJoCo COM convention).

        xfrc_applied is a world-frame force and torque ABOUT the body COM
        (``xipos``); a force ``F`` at point ``p`` therefore adds ``(p - xipos) x F``
        to the torque. Needs ``data`` kinematics (``xipos``) to be current.
        """
        native = self._native
        native.mj_kinematics(model, data)
        data.xfrc_applied[:] = 0.0
        for wrench in wrenches:
            body = native.mj_name2id(model, native.mjtObj.mjOBJ_BODY, wrench.body)
            if body < 0:
                raise ValueError(f"external wrench on unknown body {wrench.body!r}")
            point = self._checked("external wrench point_m", wrench.point_m, 3)
            if wrench.force_n is not None:
                force = self._checked("external wrench force_n", wrench.force_n, 3)
                data.xfrc_applied[body, :3] += force
                data.xfrc_applied[body, 3:] += np.cross(point - data.xipos[body], force)
            if wrench.torque_nm is not None:
                data.xfrc_applied[body, 3:] += self._checked(
                    "external wrench torque_nm", wrench.torque_nm, 3
                )

    def _run_inverse(
        self,
        qpos: Any,
        qvel: Any,
        qacc: Any,
        time_s: float,
        external_wrenches: Sequence[ExternalWrench] | None,
        closure: GripClosure | None,
    ) -> tuple[Any, Any, str, frozenset[int]] | None:
        """Validate, run ``mj_inverse`` + ``mj_rnePostConstraint`` on scratch data.

        Returns ``(model, scratch, note, unavailable_bodies)`` or None when the
        model is unsupported. With an active weld loop the weld equality is
        disabled on the private copy; its physical closure wrench (``closure``)
        is injected instead, so the loop is not closed by solref/solimp
        impedance. Without one, the loop bodies are reported unavailable.
        """
        model, native, scratch = self.model, self._native, self._scratch
        q = self._checked("qpos", qpos, model.nq)
        v = self._checked("qvel", qvel, model.nv)
        a = self._checked("qacc", qacc, model.nv)
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if closure is not None and not isinstance(closure, GripClosure):
            raise TypeError("closure must be a GripClosure or None")
        if not self._available():
            return None
        loop = bool(self._loop_bodies)
        note = ""
        wrenches: list[Any] | None = None
        blank: frozenset[int] = frozenset()
        if external_wrenches is not None:
            wrenches = list(external_wrenches)
            note = "; external contact wrenches, native contacts off"
        if loop:
            closure = closure or GripClosure.unavailable(
                "no closure wrench supplied for the active grip weld"
            )
            if closure.available:
                wrenches = [*(wrenches or []), *closure.wrenches]
                note += "; grip closure wrench applied, weld equality off"
            else:
                blank = self._loop_bodies
                note += f"; grip closure unavailable ({closure.reason})"
        model, scratch = (
            self._private_pair(
                contact_off=external_wrenches is not None, equality_off=loop
            )
            if (external_wrenches is not None or loop)
            else (model, scratch)
        )
        native.mj_resetData(model, scratch)
        scratch.qpos[:], scratch.qvel[:], scratch.qacc[:] = q, v, a
        scratch.time = float(time_s)
        if wrenches is not None:
            self._apply_external(model, scratch, wrenches)
        native.mj_inverse(model, scratch)
        scratch.qacc[:] = a  # mj_inverse consumes qacc; make the invariant explicit
        native.mj_rnePostConstraint(model, scratch)
        return model, scratch, note, blank

    def required_generalized_force(
        self,
        qpos: Any,
        qvel: Any,
        qacc: Any,
        external_wrenches: Sequence[ExternalWrench] | None = None,
        closure: GripClosure | None = None,
    ) -> np.ndarray | None:
        """Joint efforts a prescribed motion requires beyond the external wrenches.

        ``qfrc_inverse`` minus the generalized force of ``xfrc_applied`` (MuJoCo's
        inverse leaves the latter out), i.e. what the actuators must supply.

        Same inputs, private copies and closure handling as
        :meth:`sample_prescribed`. With the physical closure wrench injected this
        equals the applied efforts of the forward solve that produced the
        motion; ``None`` when the model is unsupported.
        """
        run = self._run_inverse(qpos, qvel, qacc, 0.0, external_wrenches, closure)
        if run is None:
            return None
        model, scratch = run[0], run[1]
        external = np.zeros(model.nv)
        for body in range(1, model.nbody):
            wrench = scratch.xfrc_applied[body]
            if wrench.any():
                self._native.mj_applyFT(
                    model,
                    scratch,
                    wrench[:3],
                    wrench[3:],
                    scratch.xipos[body],
                    body,
                    external,
                )
        return np.array(scratch.qfrc_inverse) - external

    def sample_prescribed(
        self,
        qpos: Any,
        qvel: Any,
        qacc: Any,
        time_s: float,
        external_wrenches: Sequence[ExternalWrench] | None = None,
        closure: GripClosure | None = None,
    ) -> AxialLoadFrame | None:
        """Axial loads for a prescribed motion via inverse dynamics.

        Use when qpos/qvel/qacc come from a fit rather than forward simulation.
        Inputs are copied (never modified); ``mj_inverse`` then
        ``mj_rnePostConstraint`` run on scratch data with the prescribed qacc.
        Preconditions: shapes (nq,), (nv,), (nv,); all finite; finite ``time_s``
        (ValueError otherwise). Postcondition: same projection and sign as
        ``sample`` (tension positive); returns None when the model is unsupported.

        ``external_wrenches`` (default ``None``: MuJoCo's own contacts decide) is
        the single source of truth for ground reaction when the model's support
        comes from an external contact law instead of native geoms: a prescribed
        pose has no penetration-dependent native contact, so without it the root
        residual carries the body and the legs read tension. Passing a sequence
        (possibly empty, e.g. no foot loaded) disables native contacts on a
        private model copy, so nothing is counted twice, and applies each wrench
        through ``xfrc_applied``, which ``mj_inverse`` and ``mj_rnePostConstraint``
        include in the section loads. Raises ValueError for an unknown body or a
        non-finite wrench.

        ``closure`` is the physical wrench of an active ``weld``/``connect``
        equality (a rigid grip). The prescribed-motion inverse has a nullspace
        over the loop, so MuJoCo's soft weld would allocate it by impedance, not
        by physics. The weld is therefore disabled on the private copy and
        ``closure``'s +/- pair is applied instead. If the model has an active
        weld and ``closure`` is omitted or unavailable, every body on the loop
        path reports ``None`` (never zero, never the impedance value) and the
        frame's source says why. Models without such an equality are unchanged.
        """
        run = self._run_inverse(qpos, qvel, qacc, time_s, external_wrenches, closure)
        if run is None:
            return None
        _, scratch, note, blank = run
        return self._project(time_s, scratch, note, blank)
