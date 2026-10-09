"""Driven-swing simulation of the OpenSim contact grip (issue #11739, OSV-7).

The same input and topology as the bushing runner
(:class:`~...grip_bushing_sim.BushingGripSimulator`): the 44 body coordinates
are prescribed from a matched trajectory and the club is a free body.  Here
the hand-club interface is the distributed contact of the exporter's
``contact`` grip model: spherical pads on the hands pressed against a closed
grip mesh on the club through one ``ElasticFoundationForce`` per pad.  The
per-pad record values give the per-pad force on the club; summed per hand they
give the per-hand wrench.  Sign convention (binding, as for the bushing): every
wrench is the loading exerted BY THE HAND ON THE CLUB, world axes; the torque
is the free torque at the hand grip point on the club.

The right hand is tied to the left hand body exactly as in the bushing runner
(the matched IK does not close the two-hand loop), so the right pads are
re-attached to the left hand body at the tied-frame offset.

OpenSim is imported lazily; callers must skip when it is unavailable.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.full_body_grip_contact import (
    CONTACT_FORCE_PREFIX,
    ContactGripConfig,
    pad_name,
)
from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.engines.physics_engines.opensim.python.grip_bushing_sim import (
    CLUB_BODY,
    BushingGripSimulator,
    BushingRun,
    _inv,
    _osim,
    _transform_to_rt,
)
from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.elastic_foundation import (
    ElasticFoundationParameters,
    elastic_foundation_parameters,
)
from src.shared.python.grip_contact.grip_mesh import capped_cylinder_mesh, write_obj
from src.shared.python.grip_contact.pad_contact import PadContactModel
from src.shared.python.grip_contact.parity import GripKineticsSeries


ENGINE = "opensim_contact"
_SIDES = (("L", "left"), ("R", "right"))
_CALIBRATION_HALF_LENGTH_M = 0.02
_GRAVITY_M_S2 = 9.80665


def single_pad_model(
    ef: ElasticFoundationParameters,
    mesh_path: Path,
    pad_mass_kg: float = 0.5,
    gravity_m_s2: float = _GRAVITY_M_S2,
) -> tuple[Any, Any, Any]:
    """One pad sphere on a slider above a fixed grip mesh: ``(model, force, coord)``.

    The mesh axis is the ground z axis through the origin; the pad slides
    along the ground y axis (the radial direction of the mesh vertex ring at
    azimuth 0), so the coordinate is the pad-centre distance from the axis and
    the penetration is ``grip_radius + pad_radius - coordinate``.  The force is
    the same ``ElasticFoundationForce`` the grip model uses.
    """
    osim = _osim()
    osim.Logger.setLevelString("Error")
    model = osim.Model()
    model.setGravity(osim.Vec3(0.0, -gravity_m_s2, 0.0))
    body = osim.Body("pad", pad_mass_kg, osim.Vec3(0), osim.Inertia(1e-3))
    model.addBody(body)
    quarter = osim.Vec3(0.0, 0.0, np.pi / 2)
    joint = osim.SliderJoint(
        "slide", model.getGround(), osim.Vec3(0), quarter, body, osim.Vec3(0), quarter
    )
    model.addJoint(joint)
    layout = ef.layout
    joint.updCoordinate().setDefaultValue(
        layout.grip_radius_m + layout.pad_radius_m - ef.preload_penetration_m
    )
    model.addContactGeometry(
        osim.ContactSphere(layout.pad_radius_m, osim.Vec3(0), body, "pad")
    )
    model.addContactGeometry(
        osim.ContactMesh(
            str(mesh_path), osim.Vec3(0), osim.Vec3(0), model.getGround(), "grip"
        )
    )
    force = osim.ElasticFoundationForce()
    force.setName("pad_force")
    force.addGeometry("pad")
    force.addGeometry("grip")
    force.setStiffness(ef.stiffness_n_m3)
    force.setDissipation(ef.dissipation_s_m)
    force.setStaticFriction(ef.static_friction)
    force.setDynamicFriction(ef.dynamic_friction)
    force.setViscousFriction(ef.viscous_friction)
    force.setTransitionVelocity(ef.transition_velocity_m_s)
    model.addForce(force)
    return model, force, joint.updCoordinate()


def write_calibration_mesh(ef: ElasticFoundationParameters, directory: Path) -> Path:
    """Short closed grip mesh (axis z, ring vertex 0 on +y: the slider ray)."""
    layout = ef.layout
    verts, faces = capped_cylinder_mesh(
        layout.grip_radius_m,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        (-_CALIBRATION_HALF_LENGTH_M, _CALIBRATION_HALF_LENGTH_M),
    )
    path = directory / "calibration_mesh.obj"
    write_obj(path, verts, faces)
    return path


def pad_normal_force_n(
    ef: ElasticFoundationParameters, mesh_path: Path, penetration_m: float
) -> float:
    """Static normal force of one pad pressed ``penetration_m`` into the mesh."""
    model, force, coord = single_pad_model(ef, mesh_path)
    state = model.initSystem()
    layout = ef.layout
    coord.setValue(state, layout.grip_radius_m + layout.pad_radius_m - penetration_m)
    model.realizeDynamics(state)
    return float(force.getRecordValues(state).get(1))


def calibrated_foundation(
    ef: ElasticFoundationParameters, directory: Path
) -> ElasticFoundationParameters:
    """Scale the foundation stiffness so one pad carries ``force_per_pad_n`` at the preload.

    The Winkler estimate is within about 15 % of the real mesh contact; the
    force is exactly proportional to the stiffness, so one static evaluation of
    the real mesh contact at the preload fixes it (no integration, no fitting
    to a swing).  Raises ``RuntimeError`` if the pad does not touch the mesh.
    """
    mesh = write_calibration_mesh(ef, directory)
    measured = pad_normal_force_n(ef, mesh, ef.preload_penetration_m)
    if measured <= 0.0:
        raise RuntimeError("calibration pad does not touch the grip mesh")
    return ef.with_stiffness(ef.stiffness_n_m3 * ef.force_per_pad_n / measured)


@dataclass(frozen=True)
class OpenSimContactRun(BushingRun):
    """A :class:`BushingRun` plus the outputs only a distributed contact has.

    ``normal_force_n[side]`` is ``(n, pads)`` (pressure along the grip),
    ``roll_slip_rad`` / ``axial_slip_m`` the hand-to-club slip.
    """

    normal_force_n: dict[str, np.ndarray]
    roll_slip_rad: dict[str, np.ndarray]
    axial_slip_m: dict[str, np.ndarray]

    def to_contact_run(self, metadata: dict[str, Any] | None = None) -> ContactRun:
        """The engine-agnostic :class:`ContactRun` (series plus contact outputs)."""
        series = GripKineticsSeries(
            engine=ENGINE,
            time_s=self.time_s,
            force_on_club_n=self.force_on_club_n,
            torque_on_club_nm=self.torque_on_club_nm,
            grip_point_m=self.grip_point_m,
            deflection_m=self.deflection_m,
            rotation_deflection_rad=self.rotation_deflection_rad,
            club_rotation=self.club_rotation,
            metadata=dict(metadata or {}),
        )
        return ContactRun(
            series, self.normal_force_n, self.roll_slip_rad, self.axial_slip_m
        )


class ContactGripSimulator(BushingGripSimulator):
    """Prescribed-arm, free-club simulation of the contact grip model.

    Args:
        spec_bytes, names, time_s, q: as :class:`BushingGripSimulator`.
        pads: shared pad model (layout and law) the foundation is matched to.
        interface: grip interface (default: built from the spec).
        foundation: foundation parameters (default: matched to ``pads`` and
            calibrated by one static evaluation, :func:`calibrated_foundation`).
        mesh_dir: directory for the grip OBJ files (default: a temporary
            directory that lives as long as the simulator).
    """

    def __init__(
        self,
        spec_bytes: bytes,
        names: Sequence[str],
        time_s: np.ndarray,
        q: np.ndarray,
        pads: PadContactModel,
        interface: GripInterface | None = None,
        foundation: ElasticFoundationParameters | None = None,
        mesh_dir: Path | None = None,
    ) -> None:
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if mesh_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="grip_meshes_")
            mesh_dir = Path(self._tmp.name)
        interface = interface or GripInterface.from_spec(json.loads(spec_bytes))
        if foundation is None:
            foundation = calibrated_foundation(
                elastic_foundation_parameters(pads), mesh_dir
            )
        self.contact_config = ContactGripConfig(pads, mesh_dir, foundation)
        self.export_receipt: dict[str, Any] = {}
        self._frames: dict[tuple[str, str], Any] = {}
        self._pad_forces: dict[str, list[tuple[Any, int, int]]] = {}
        super().__init__(spec_bytes, names, time_s, q, interface)
        self._index_records()

    # ------------------------------------------------------------ build hooks
    def _export_xml(self, spec_bytes: bytes, interface: GripInterface | None) -> str:
        xml, receipt = export_full_body_osim(
            spec_bytes,
            grip_model="contact",
            grip_interface=interface,
            grip_contact=self.contact_config,
        )
        self.export_receipt = receipt
        return xml

    def _strip_forces(self) -> None:
        forces = self._model.updForceSet()
        for i in range(forces.getSize() - 1, -1, -1):
            if not forces.get(i).getName().startswith(CONTACT_FORCE_PREFIX):
                forces.remove(i)

    def _tie_right_hand_to_left(self, spec: dict[str, Any]) -> None:
        """Hand and club grip frames, and the right pads moved to the left hand."""
        osim = self._osim
        gi = self.interface or GripInterface.from_spec(spec)
        tie = self._tied_right_matrix(spec)
        left_hand = self._left_hand_matrix(spec, gi)
        self._frames[("L", "hand")] = self._add_offset_frame(
            "grip_hand_frame_left", "LGrip", left_hand
        )
        self._frames[("R", "hand")] = self._add_offset_frame(
            "grip_hand_frame_right", "LGrip", tie
        )
        for side, _ in _SIDES:
            self._frames[(side, "club")] = self._add_offset_frame(
                f"grip_club_frame_{side}", CLUB_BODY, gi.frame(side).matrix()
            )
        layout = self.contact_config.ef.layout
        hand = self._model.getBodySet().get("LGrip")
        for k, local in enumerate(layout.positions_grip_frame("R")):
            geom = osim.ContactSphere.safeDownCast(
                self._model.getContactGeometrySet().get(f"grip_pad_{pad_name('R', k)}")
            )
            geom.connectSocket_frame(hand)
            at = tie[:3, :3] @ local + tie[:3, 3]
            geom.set_location(osim.Vec3(*(float(x) for x in at)))
        self._model.finalizeConnections()

    def _left_hand_matrix(self, spec: dict[str, Any], gi: GripInterface) -> np.ndarray:
        """Left grip frame on the left hand body (which sits at the wrist frame)."""
        club = spec["closure"]["body_b"]
        wrist = next(j for j in spec["joints"] if j["child"] == club)
        return _inv(np.asarray(wrist["child_to_follower"], float)) @ (gi.left.matrix())

    def _frame(self, side_name: str, kind: str) -> Any:
        return self._frames[("L" if side_name == "left" else "R", kind)]

    def _index_records(self) -> None:
        """Locate the club-body force and torque entries of every pad record."""
        osim = self._osim
        layout = self.contact_config.ef.layout
        for side, _ in _SIDES:
            rows = []
            for k in range(layout.pad_count):
                name = f"{CONTACT_FORCE_PREFIX}_{pad_name(side, k)}"
                force = osim.ElasticFoundationForce.safeDownCast(
                    self._model.getForceSet().get(name)
                )
                labels = force.getRecordLabels()
                names = [labels.get(i) for i in range(labels.size())]
                rows.append(
                    (
                        force,
                        names.index(f"{name}.{CLUB_BODY}.force.X"),
                        names.index(f"{name}.{CLUB_BODY}.torque.X"),
                    )
                )
            self._pad_forces[side] = rows

    # ----------------------------------------------------------------- sample
    def _side_sample(self, state: Any, name: str) -> dict[str, Any]:
        side = "L" if name == "left" else "R"
        layout = self.contact_config.ef.layout
        r1, p1 = _transform_to_rt(self._frame(name, "hand").getTransformInGround(state))
        r2, p2 = _transform_to_rt(self._frame(name, "club").getTransformInGround(state))
        axis = r2[:, 0]
        axis_point = p2 + r2 @ layout.axis_offset_grip_frame(side)
        centres = p1 + layout.positions_grip_frame(side) @ r1.T
        total = np.zeros(6)
        normal = np.zeros(len(centres))
        for k, (force, i_f, i_t) in enumerate(self._pad_forces[side]):
            rec = force.getRecordValues(state)
            f = np.array([rec.get(i_f + j) for j in range(3)])
            total[:3] += f
            total[3:] += np.array([rec.get(i_t + j) for j in range(3)])
            rel = centres[k] - axis_point
            radial = rel - (axis @ rel) * axis
            normal[k] = -float(f @ radial) / float(np.linalg.norm(radial))
        rel_rot = r1.T @ r2
        return {
            "record": total,
            "rot1": r1,
            "point": p2,
            "delta": r1.T @ (p2 - p1),
            "angle": float(np.linalg.norm(Rotation.from_matrix(rel_rot).as_rotvec())),
            "normal": normal,
            "roll": float(
                np.arctan2(rel_rot[2, 1] - rel_rot[1, 2], rel_rot[1, 1] + rel_rot[2, 2])
            ),
        }

    @staticmethod
    def _split_record(rec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rec[:, 0:3], rec[:, 3:6]

    def _assemble(self, samples: np.ndarray, rows: list[dict[str, Any]]) -> Any:
        base = super()._assemble(samples, rows)
        return OpenSimContactRun(
            **vars(base),
            normal_force_n={
                s: np.array([r[s]["normal"] for r in rows]) for s, _ in _SIDES
            },
            roll_slip_rad={
                s: np.array([r[s]["roll"] for r in rows]) for s, _ in _SIDES
            },
            axial_slip_m={
                s: np.array([r[s]["delta"][0] for r in rows]) for s, _ in _SIDES
            },
        )
