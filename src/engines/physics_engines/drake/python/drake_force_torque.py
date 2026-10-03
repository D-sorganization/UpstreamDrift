"""Drake force/torque overlay source (FTO-11, #11296, ADR-0052).

Reads Drake's own output ports and emits an engine-agnostic
``ForceTorqueFrame``:

* ``JOINT_REACTION`` from ``plant.get_reaction_forces_output_port()``. Drake
  returns each entry as the spatial force applied *on the child body* by the
  joint, at the child frame ``Jc`` origin, expressed in the **Jc frame** (not
  world, contrary to the issue text; verified empirically with a tilted
  inverted pendulum). We rotate by ``R_WJc`` into world and use
  ``translational()`` as force and ``rotational()`` as torque. A hanging
  pendulum at rest reports ``+m*g`` along world +z on the link (pinned by the
  unit tests).
* ``JOINT_ACTUATOR`` from ``plant.get_net_actuation_output_port()`` for
  revolute actuated joints only: torque ``u * R_WJc @ axis`` at ``Jc``.
* ``CONTACT`` from ``plant.get_contact_results_output_port()``: point pairs
  (force on body B at ``contact_point()``, negative on A) and hydroelastic
  surfaces (``F_Ac_W()`` on body A at the surface centroid, negative on B).
* ``GRAVITY`` (opt-in): ``m*g`` at each body's centre of mass.

Note: for discrete-time plants Drake's reaction-force port reads zero until the
simulator has advanced at least one step; sample after stepping.

Unavailable data is omitted and listed in ``unavailable_labels``; it is never
reported as zero. The shared ``ForceTorqueFrame`` has no legend field, so the
list is exposed on the source.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydrake.geometry import SceneGraph
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import BodyIndex, RevoluteJoint
from pydrake.systems.framework import Context

from src.shared.python.body_part_viz.axial_loads import (
    AxialLoadFrame,
    axial_force_from_proximal_reaction,
)
from src.shared.python.core.process_safety import narrow_catch
from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.segment_axes import (
    segment_axes_from_joint_tree,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["DrakeForceTorqueSource"]

_BAD_LABEL_CHARS = re.compile(r"[^A-Za-z0-9_.:-]")


def _vec(v: Any) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=float).reshape(3)
    return (float(a[0]), float(a[1]), float(a[2]))


def _unique_names(items: Sequence[tuple[Any, str, str]]) -> dict[Any, str]:
    """Map key -> label-safe name, prefixing the model instance on collisions."""
    counts = Counter(name for _, name, _ in items)
    out: dict[Any, str] = {}
    for key, name, instance in items:
        full = name if counts[name] == 1 else f"{instance}.{name}"
        out[key] = _BAD_LABEL_CHARS.sub("_", full)
    return out


def _joint_torque_wrench(
    label: str, body: str, point: Any, axis_world: Any, torque_nm: float, source: str
) -> OverlayWrench:
    """Pure torque wrench about ``axis_world`` (force half stays unavailable)."""
    axis = np.asarray(axis_world, dtype=float)
    return OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label=label,
        body=body,
        point_m=_vec(point),
        torque_nm=_vec(axis * float(torque_nm)),
        source=source,
    )


class DrakeForceTorqueSource:
    """Samples a finalized Drake plant into a ``ForceTorqueFrame``.

    Preconditions: ``plant`` is a finalized ``MultibodyPlant``; ``diagram`` (if
    given) contains the plant and its ``SceneGraph`` and supplies the contexts.
    """

    ENGINE = "drake"

    def __init__(self, plant: MultibodyPlant, diagram: Any = None) -> None:
        if not isinstance(plant, MultibodyPlant):
            raise TypeError("plant must be a pydrake MultibodyPlant")
        if not plant.is_finalized():
            raise ValueError("plant must be finalized before sampling")
        self._plant = plant
        self._scene_graph = self._find_scene_graph(diagram)
        self._joints = [plant.get_joint(i) for i in plant.GetJointIndices()]
        self._bodies = [plant.get_body(BodyIndex(i)) for i in range(plant.num_bodies())]
        self._body_names = _unique_names(
            [
                (b.index(), b.name(), plant.GetModelInstanceName(b.model_instance()))
                for b in self._bodies
            ]
        )
        self._joint_names = _unique_names(
            [
                (j.index(), j.name(), plant.GetModelInstanceName(j.model_instance()))
                for j in self._joints
            ]
        )
        self._unavailable: tuple[str, ...] = ()

    @staticmethod
    def _find_scene_graph(diagram: Any) -> SceneGraph | None:
        if diagram is None:
            return None
        for system in diagram.GetSystems():
            if isinstance(system, SceneGraph):
                return system
        return None

    @property
    def unavailable_labels(self) -> tuple[str, ...]:
        """Labels omitted by the most recent ``sample`` because unavailable."""
        return self._unavailable

    # -- helpers -----------------------------------------------------------
    def _check_context(self, ctx: Any) -> None:
        if not isinstance(ctx, Context):
            raise TypeError("plant_context must be a pydrake Context")
        try:
            self._plant.GetPositions(ctx)
        except RuntimeError as exc:
            raise ValueError("plant_context does not belong to this plant") from exc

    def _body_name(self, body_index: Any) -> str:
        return self._body_names[body_index]

    def _reaction_wrenches(self, ctx: Context) -> list[OverlayWrench]:
        plant = self._plant
        reactions = plant.get_reaction_forces_output_port().Eval(ctx)
        out: list[OverlayWrench] = []
        for joint in self._joints:
            sf = reactions[int(joint.index())]
            pose = plant.CalcRelativeTransform(
                ctx, plant.world_frame(), joint.frame_on_child()
            )
            rot = pose.rotation().matrix()
            point = pose.translation()
            out.append(
                OverlayWrench(
                    kind=WrenchKind.JOINT_REACTION,
                    label=f"joint_reaction:{self._joint_names[joint.index()]}",
                    body=self._body_name(joint.child_body().index()),
                    point_m=_vec(point),
                    force_n=_vec(rot @ np.asarray(sf.translational())),
                    torque_nm=_vec(rot @ np.asarray(sf.rotational())),
                    source="drake:reaction_forces_port",
                )
            )
        return out

    def _actuator_wrenches(self, ctx: Context) -> list[OverlayWrench]:
        plant = self._plant
        actuated = [
            plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()
        ]
        actuated = [
            a
            for a in actuated
            if isinstance(a.joint(), RevoluteJoint) and a.num_inputs() == 1
        ]
        if not actuated:
            return []
        labels = [
            f"joint_actuator:{self._joint_names[a.joint().index()]}" for a in actuated
        ]
        if not plant.get_actuation_input_port().HasValue(ctx):
            self._unavailable += tuple(labels)
            return []
        with narrow_catch(RuntimeError, log_message="drake net actuation port"):
            net = np.asarray(plant.get_net_actuation_output_port().Eval(ctx))
            out: list[OverlayWrench] = []
            for actuator, label in zip(actuated, labels, strict=True):
                joint = actuator.joint()
                frame = joint.frame_on_child()
                pose = plant.CalcRelativeTransform(ctx, plant.world_frame(), frame)
                out.append(
                    _joint_torque_wrench(
                        label,
                        self._body_name(joint.child_body().index()),
                        pose.translation(),
                        pose.rotation().matrix() @ joint.revolute_axis(),
                        float(net[actuator.input_start()]),
                        "drake:net_actuation_port",
                    )
                )
            return out
        self._unavailable += tuple(labels)
        return []

    def _read_contact_results(self, ctx: Context) -> Any:
        """Evaluate the contact-results port (may raise ``RuntimeError``)."""
        return self._plant.get_contact_results_output_port().Eval(ctx)

    def _contact_wrenches(self, ctx: Context) -> list[OverlayWrench]:
        plant = self._plant
        world = plant.world_body().index()
        entries: list[tuple[Any, Any, Any, Any]] = []  # (body, point, force, torque)
        try:
            results = self._read_contact_results(ctx)
            for i in range(results.num_point_pair_contacts()):
                info = results.point_pair_contact_info(i)
                force = np.asarray(info.contact_force())
                point = info.contact_point()
                entries.append((info.bodyB_index(), point, force, None))
                entries.append((info.bodyA_index(), point, -force, None))
            n_hydro = results.num_hydroelastic_contacts()
            inspector = (
                self._scene_graph.model_inspector() if self._scene_graph else None
            )
            if n_hydro and inspector is None:
                self._unavailable += ("contact:hydroelastic",)
                n_hydro = 0
            for i in range(n_hydro):
                info = results.hydroelastic_contact_info(i)
                surface = info.contact_surface()
                body_a = plant.GetBodyFromFrameId(inspector.GetFrameId(surface.id_M()))
                body_b = plant.GetBodyFromFrameId(inspector.GetFrameId(surface.id_N()))
                sf = info.F_Ac_W()
                f, t = np.asarray(sf.translational()), np.asarray(sf.rotational())
                centroid = surface.centroid()
                entries.append((body_a.index(), centroid, f, t))
                entries.append((body_b.index(), centroid, -f, -t))
        except RuntimeError:
            logger.warning(
                "Drake contact results unavailable (SceneGraph query port not "
                "connected or not evaluable); contact channel omitted"
            )
            self._unavailable += ("contact:results",)
            return []
        out: list[OverlayWrench] = []
        for body, point, force, torque in entries:
            if body == world:
                continue
            out.append(
                OverlayWrench(
                    kind=WrenchKind.CONTACT,
                    label=f"contact:{self._body_name(body)}:{len(out)}",
                    body=self._body_name(body),
                    point_m=_vec(point),
                    force_n=_vec(force),
                    torque_nm=None if torque is None else _vec(torque),
                    source="drake:contact_results_port",
                )
            )
        return out

    def _gravity_wrenches(self, ctx: Context) -> list[OverlayWrench]:
        plant = self._plant
        g = np.asarray(plant.gravity_field().gravity_vector())
        out: list[OverlayWrench] = []
        for body in self._bodies:
            index = body.index()
            if index == plant.world_body().index() or body.get_mass(ctx) <= 0.0:
                continue
            com = plant.EvalBodyPoseInWorld(
                ctx, body
            ) @ body.CalcCenterOfMassInBodyFrame(ctx)
            out.append(
                OverlayWrench(
                    kind=WrenchKind.GRAVITY,
                    label=f"gravity:{self._body_name(index)}",
                    body=self._body_name(index),
                    point_m=_vec(com),
                    force_n=_vec(body.get_mass(ctx) * g),
                    source="drake:gravity_field",
                )
            )
        return out

    def _axial_loads(
        self, ctx: Context, reactions: list[OverlayWrench], time_s: float
    ) -> AxialLoadFrame | None:
        plant = self._plant
        origins: dict[str, np.ndarray] = {}
        body_of_joint: dict[str, str] = {}
        child_joint_of: dict[str, list[str]] = {}
        parent_label: dict[str, str] = {}
        for joint in self._joints:
            name = self._joint_names[joint.index()]
            origins[name] = np.asarray(
                plant.CalcRelativeTransform(
                    ctx, plant.world_frame(), joint.frame_on_child()
                ).translation()
            )
            body_of_joint[name] = self._body_name(joint.child_body().index())
            parent_label[body_of_joint[name]] = f"joint_reaction:{name}"
            parent = self._body_name(joint.parent_body().index())
            child_joint_of.setdefault(parent, []).append(name)
        axes, _ = segment_axes_from_joint_tree(origins, child_joint_of, body_of_joint)
        force_on = {w.label: w.force_n for w in reactions}
        values = {
            axis.body: axial_force_from_proximal_reaction(
                force_on[parent_label[axis.body]], axis.proximal_m, axis.distal_m
            )
            for axis in axes
        }
        if not values:
            return None
        return AxialLoadFrame(time_s, values, "drake:joint_reaction_proximal")

    # -- public API --------------------------------------------------------
    def sample(
        self, plant_context: Context, *, include_gravity: bool = False
    ) -> ForceTorqueFrame:
        """Return the overlay frame for ``plant_context``.

        Raises:
            TypeError: ``plant_context`` is not a pydrake ``Context``.
            ValueError: the context does not belong to this plant.

        Postconditions: every wrench is in world coordinates and applied *to*
        its body; actuator wrenches carry torque only; unavailable channels are
        omitted and named in ``unavailable_labels`` (never zero-filled); axial
        loads are tension-positive and only cover bodies with a single child
        joint.
        """
        self._check_context(plant_context)
        self._unavailable = ()
        time_s = float(plant_context.get_time())
        reactions = self._reaction_wrenches(plant_context)
        wrenches = list(reactions)
        wrenches += self._actuator_wrenches(plant_context)
        wrenches += self._contact_wrenches(plant_context)
        if include_gravity:
            wrenches += self._gravity_wrenches(plant_context)
        return ForceTorqueFrame(
            time_s=time_s,
            engine=self.ENGINE,
            wrenches=tuple(wrenches),
            axial_loads=self._axial_loads(plant_context, reactions, time_s),
        )
