"""Marker kinematics and pose inverse kinematics on the MuJoCo full-body model.

Markers attach to spec frames (upper body, via the exported frame sites) or to
spec bodies (lower limbs) with offsets in the specification's body frame; the
MJCF places each body at its joint follower frame, so those offsets are mapped
through the adapter's ``body_frames`` before use. The solver is a damped
Gauss-Newton on the 41 scalar coordinates with analytic MuJoCo Jacobians and
three soft terms: marker residuals, dual-grip weld closure, and a one-sided
penalty that keeps every contact sphere above the ground plane. It is a
kinematic tool: no dynamics, no contact forces, no scaling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.contact_law import GroundPlane

Array = NDArray[np.float64]
Attachment = tuple[str, tuple[float, float, float]]


@dataclass(frozen=True)
class PoseFit:
    """Result of one pose solve."""

    q: Array
    marker_rms_m: float
    per_marker_m: dict[str, float]
    closure_error_m: float
    closure_error_rad: float
    lowest_sphere_height_m: float
    iterations: int


def _rotation_error(r_a: Array, r_b: Array) -> Array:
    """Small-angle rotation vector taking frame b onto frame a (world axes)."""
    r = r_a @ r_b.T
    return 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


class FullBodyMarkerKinematics:
    """Forward kinematics of attached markers and the pose solver."""

    def __init__(
        self,
        adapter: NativeMujocoFullBodyModel,
        attachments: Mapping[str, Attachment],
    ) -> None:
        if not attachments:
            raise ValueError("At least one marker attachment is required")
        mj: Any = adapter._mj
        self._mj = mj
        self.adapter = adapter
        self.model = adapter.model
        self.data = adapter.data
        self.labels = tuple(attachments)
        self.coordinate_order = tuple(adapter.coordinate_order)
        self._qpos = np.array(
            [self.model.joint(name).qposadr[0] for name in self.coordinate_order]
        )
        # MuJoCo Jacobian columns follow tree (DOF) order, not the spec order.
        self._dof = np.array(
            [self.model.joint(name).dofadr[0] for name in self.coordinate_order]
        )
        sites = adapter.metadata["frame_sites"]
        self._body_ids: list[int] = []
        self._local: list[Array] = []
        for label, (body, offset) in attachments.items():
            local = np.asarray(offset, dtype=float)
            if local.shape != (3,) or not np.isfinite(local).all():
                raise ValueError(f"Marker {label} needs a finite 3-vector offset")
            if body in sites:
                site = self.model.site(sites[body]).id
                rot = np.zeros(9)
                mj.mju_quat2Mat(rot, self.model.site_quat[site])
                local = self.model.site_pos[site] + rot.reshape(3, 3) @ local
                body_id = int(self.model.site_bodyid[site])
            else:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body)
                if body_id < 0:
                    raise ValueError(f"Marker {label} references unknown body {body}")
                frame = adapter.body_frames.get(body)
                if frame is None:
                    raise ValueError(f"Marker {label} body {body} has no joint frame")
                local = frame[:3, :3] @ local + frame[:3, 3]  # spec body -> MJCF body
            self._body_ids.append(body_id)
            self._local.append(local)
        self._closure = list(adapter._closure)
        self._spheres = {
            name: (int(info["site_id"]), float(info["radius"]))
            for name, info in adapter._spheres.items()
        }

    # -- forward kinematics -------------------------------------------------
    def _set(self, q: Array) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self._qpos),) or not np.isfinite(q).all():
            raise ValueError("Coordinates must be a finite vector of model size")
        self.data.qpos[self._qpos] = q
        self._mj.mj_fwdPosition(self.model, self.data)

    def marker_positions(self, q: Array) -> Array:
        """World positions (markers, 3) of every attached marker at ``q``."""
        self._set(q)
        return self._positions()

    def _positions(self) -> Array:
        out = np.empty((len(self.labels), 3))
        for k, (body, local) in enumerate(
            zip(self._body_ids, self._local, strict=True)
        ):
            out[k] = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ local
        return out

    def _marker_jacobian(self, positions: Array) -> Array:
        nv = self.model.nv
        jac = np.empty((len(self.labels), 3, nv))
        buffer = np.zeros((3, nv))
        for k, body in enumerate(self._body_ids):
            self._mj.mj_jac(self.model, self.data, buffer, None, positions[k], body)
            jac[k] = buffer[:, self._dof]
        return jac

    def body_poses(
        self, q: Array, bodies: Sequence[str]
    ) -> dict[str, tuple[Array, Array]]:
        """World pose ``(R, t)`` of each named spec frame or body at ``q``.

        The convention matches the shared marker calibration: a body point
        ``p`` sits at ``R @ p + t`` in the world.
        """
        self._set(q)
        sites = self.adapter.metadata["frame_sites"]
        poses: dict[str, tuple[Array, Array]] = {}
        for body in bodies:
            if body in sites:
                site = self.model.site(sites[body]).id
                rotation = self.data.site_xmat[site].reshape(3, 3).copy()
                translation = self.data.site_xpos[site].copy()
            else:
                body_id = self._mj.mj_name2id(
                    self.model, self._mj.mjtObj.mjOBJ_BODY, body
                )
                if body_id < 0:
                    raise ValueError(f"Unknown body {body}")
                frame = self.adapter.body_frames.get(body)
                if frame is None:
                    raise ValueError(f"Body {body} has no joint frame")
                r_mj = self.data.xmat[body_id].reshape(3, 3)
                rotation = r_mj @ frame[:3, :3]  # spec body axes in the world
                translation = self.data.xpos[body_id] + r_mj @ frame[:3, 3]
            poses[body] = (rotation, translation)
        return poses

    def sphere_heights(self, q: Array, ground: GroundPlane) -> dict[str, float]:
        """Signed height of each contact sphere's lowest point above the plane."""
        self._set(q)
        return self._sphere_heights(ground)

    def _sphere_heights(self, ground: GroundPlane) -> dict[str, float]:
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        return {
            name: float(self.data.site_xpos[site] @ n - ground.height_m - radius)
            for name, (site, radius) in self._spheres.items()
        }

    def closure_error(self, q: Array) -> tuple[float, float]:
        """Position and orientation mismatch of the dual-grip weld at ``q``."""
        self._set(q)
        a, b = self._closure
        pos = np.linalg.norm(self.data.site_xpos[a] - self.data.site_xpos[b])
        rot = _rotation_error(
            self.data.site_xmat[a].reshape(3, 3), self.data.site_xmat[b].reshape(3, 3)
        )
        return float(pos), float(np.linalg.norm(rot))

    def _pinned_spheres(self, flat_feet: bool | Sequence[str]) -> frozenset[str]:
        if isinstance(flat_feet, bool):
            return frozenset(self._spheres) if flat_feet else frozenset()
        names = frozenset(flat_feet)
        unknown = names - set(self._spheres)
        if unknown:
            raise ValueError(f"Unknown contact spheres: {sorted(unknown)}")
        return names

    def _bounds(
        self, bounds: Mapping[str, tuple[float, float]] | None
    ) -> tuple[Array, Array]:
        low = np.full(len(self.coordinate_order), -np.inf)
        high = np.full(len(self.coordinate_order), np.inf)
        for name, (lo, hi) in (bounds or {}).items():
            if name not in self.coordinate_order:
                raise ValueError(f"Unknown bounded coordinate {name}")
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
                raise ValueError(f"Bounds for {name} must be finite with low < high")
            index = self.coordinate_order.index(name)
            low[index], high[index] = float(lo), float(hi)
        return low, high

    # -- inverse kinematics -------------------------------------------------
    def solve_pose(
        self,
        targets: Array,
        valid: Array,
        q_init: Array,
        *,
        ground: GroundPlane,
        iterations: int = 60,
        prior_weight: float = 1e-2,
        closure_weight: float = 1e2,
        ground_weight: float = 1e2,
        flat_feet: bool | Sequence[str] = False,
        balance_weight: float = 0.0,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        anchors: Mapping[str, Sequence[float]] | None = None,
        damping: float = 1e-4,
        locked: Mapping[str, float] | None = None,
        tolerance_m: float = 1e-7,
    ) -> PoseFit:
        """Least-squares pose for one frame of marker targets.

        Minimises marker error plus ``prior_weight`` times the distance from
        ``q_init``, ``closure_weight`` times the weld closure error and
        ``ground_weight`` times the depth of any sphere below the ground.
        ``locked`` pins named coordinates. ``flat_feet`` (True for every contact
        sphere, or the names of the spheres in stance) pulls those spheres onto
        the plane two-sided, which also allows a solve with no markers at all
        (a standing pose from a prior). ``balance_weight``
        pulls the centre of mass over the centroid of the contact spheres in
        the ground plane (a statically balanced stance). ``bounds`` clamps
        named coordinates to ``(low, high)`` after every step (projected
        Levenberg-Marquardt). ``anchors`` maps sphere names to ground points:
        a planted sphere is held with its bottom on that point (three rows,
        weight ``ground_weight``), which is what a foot that does not slide
        means; anchored spheres are not also pinned flat. Precondition:
        targets (markers, 3), valid (markers,), at least three valid markers
        unless ``flat_feet``, finite q_init of model size. Postcondition: the
        returned q equals q_init on locked names.
        """
        targets = np.asarray(targets, dtype=float)
        mask = np.asarray(valid, dtype=bool)
        if targets.shape != (len(self.labels), 3) or mask.shape != (len(self.labels),):
            raise ValueError("Targets must be (markers, 3) with a validity vector")
        mask = mask & np.isfinite(targets).all(axis=1)
        pinned = self._pinned_spheres(flat_feet)
        planted = self._anchor_targets(anchors, ground)
        pinned = pinned - frozenset(planted)
        if mask.sum() < 3 and not pinned:
            raise ValueError("At least three valid markers are required")
        if iterations < 1 or min(prior_weight, closure_weight, ground_weight) < 0:
            raise ValueError("Iterations must be positive and weights nonnegative")
        if balance_weight < 0:
            raise ValueError("Iterations must be positive and weights nonnegative")
        q = np.asarray(q_init, dtype=float).copy()
        low, high = self._bounds(bounds)
        q = np.clip(q, low, high)
        free = np.ones(len(q), dtype=bool)
        for name, value in (locked or {}).items():
            if name not in self.coordinate_order:
                raise ValueError(f"Unknown locked coordinate {name}")
            index = self.coordinate_order.index(name)
            q[index] = float(value)
            free[index] = False
        nv = self.model.nv
        sqrt_prior = np.sqrt(prior_weight)

        def residuals(q_k: Array) -> tuple[Array, Array]:
            self._set(q_k)
            positions = self._positions()
            jac = self._marker_jacobian(positions)
            rows = [(positions[mask] - targets[mask]).reshape(-1)]
            jacs = [jac[mask].reshape(-1, nv)]
            rows.append(sqrt_prior * (q_k - q_init))
            jacs.append(sqrt_prior * np.eye(nv))
            self._append_closure(rows, jacs, closure_weight)
            self._append_ground(rows, jacs, ground, ground_weight, pinned)
            self._append_anchors(rows, jacs, planted, ground_weight)
            self._append_balance(rows, jacs, ground, balance_weight)
            return np.concatenate(rows), np.concatenate(jacs)[:, free]

        # Levenberg-Marquardt: a step is kept only when it lowers the cost.
        residual, jacobian = residuals(q)
        cost = float(residual @ residual)
        lam = max(damping, 1e-12)
        done = 0
        while done < iterations:
            done += 1
            gram = jacobian.T @ jacobian
            step = np.linalg.solve(
                gram + lam * np.diag(np.diag(gram) + 1e-9), -jacobian.T @ residual
            )
            trial = q.copy()
            trial[free] += step
            trial = np.clip(trial, low, high)
            trial_residual, trial_jacobian = residuals(trial)
            trial_cost = float(trial_residual @ trial_residual)
            if trial_cost < cost:
                improvement = cost - trial_cost
                q, residual, jacobian, cost = (
                    trial,
                    trial_residual,
                    trial_jacobian,
                    trial_cost,
                )
                lam = max(lam / 3.0, 1e-12)
                if improvement < tolerance_m**2 and np.linalg.norm(step) < 1e-9:
                    break
            else:
                lam *= 10.0
                if lam > 1e12:
                    break
        self._set(q)
        positions = self._positions()
        errors = np.linalg.norm(positions - targets, axis=1)
        rms = float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else 0.0
        per_marker = {
            label: float(errors[k]) for k, label in enumerate(self.labels) if mask[k]
        }
        heights = self._sphere_heights(ground)
        pos_err, rot_err = self.closure_error(q)
        return PoseFit(
            q=q,
            marker_rms_m=rms,
            per_marker_m=per_marker,
            closure_error_m=pos_err,
            closure_error_rad=rot_err,
            lowest_sphere_height_m=min(heights.values()),
            iterations=done,
        )

    def _append_closure(
        self, rows: list[Array], jacs: list[Array], weight: float
    ) -> None:
        if weight <= 0:
            return
        a, b = self._closure
        nv = self.model.nv
        jp_a, jr_a = np.zeros((3, nv)), np.zeros((3, nv))
        jp_b, jr_b = np.zeros((3, nv)), np.zeros((3, nv))
        self._mj.mj_jacSite(self.model, self.data, jp_a, jr_a, a)
        self._mj.mj_jacSite(self.model, self.data, jp_b, jr_b, b)
        w = np.sqrt(weight)
        rows.append(w * (self.data.site_xpos[a] - self.data.site_xpos[b]))
        jacs.append(w * (jp_a - jp_b)[:, self._dof])
        rot = _rotation_error(
            self.data.site_xmat[a].reshape(3, 3), self.data.site_xmat[b].reshape(3, 3)
        )
        rows.append(w * rot)
        jacs.append(w * (jr_a - jr_b)[:, self._dof])

    def _append_ground(
        self,
        rows: list[Array],
        jacs: list[Array],
        ground: GroundPlane,
        weight: float,
        pinned: frozenset[str],
    ) -> None:
        if weight <= 0:
            return
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        nv = self.model.nv
        w = np.sqrt(weight)
        for name, (site, radius) in self._spheres.items():
            depth = float(self.data.site_xpos[site] @ n - ground.height_m - radius)
            if depth >= 0.0 and name not in pinned:
                continue  # one-sided: only spheres below the plane are pushed up
            jp = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, jp, None, site)
            rows.append(np.array([w * depth]))
            jacs.append(w * (n @ jp)[None, self._dof])

    def _anchor_targets(
        self, anchors: Mapping[str, Sequence[float]] | None, ground: GroundPlane
    ) -> dict[str, Array]:
        if not anchors:
            return {}
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        targets: dict[str, Array] = {}
        for name, point in anchors.items():
            if name not in self._spheres:
                raise ValueError(f"Unknown contact sphere to anchor: {name}")
            p = np.asarray(point, dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError(f"Anchor for {name} must be a finite 3-vector")
            radius = self._spheres[name][1]
            targets[name] = p - (p @ n - ground.height_m) * n + radius * n
        return targets

    def _append_anchors(
        self,
        rows: list[Array],
        jacs: list[Array],
        targets: Mapping[str, Array],
        weight: float,
    ) -> None:
        if weight <= 0 or not targets:
            return
        nv = self.model.nv
        w = np.sqrt(weight)
        for name, target in targets.items():
            site = self._spheres[name][0]
            jp = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, jp, None, site)
            rows.append(w * (self.data.site_xpos[site] - target))
            jacs.append(w * jp[:, self._dof])

    def sphere_ground_points(self, q: Array, ground: GroundPlane) -> dict[str, Array]:
        """Projection of each contact sphere centre onto the ground plane at ``q``."""
        self._set(q)
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        out = {}
        for name, (site, _) in self._spheres.items():
            c = self.data.site_xpos[site].copy()
            out[name] = c - (c @ n - ground.height_m) * n
        return out

    def _append_balance(
        self, rows: list[Array], jacs: list[Array], ground: GroundPlane, weight: float
    ) -> None:
        if weight <= 0:
            return
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        basis = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2].T  # plane axes
        nv = self.model.nv
        self._mj.mj_comPos(self.model, self.data)
        jac_com = np.zeros((3, nv))
        self._mj.mj_jacSubtreeCom(self.model, self.data, jac_com, 1)
        com = self.data.subtree_com[1].copy()
        centres = np.zeros(3)
        jac_centres = np.zeros((3, nv))
        for site, _ in self._spheres.values():
            jp = np.zeros((3, nv))
            self._mj.mj_jacSite(self.model, self.data, jp, None, site)
            centres += self.data.site_xpos[site]
            jac_centres += jp
        centres /= len(self._spheres)
        jac_centres /= len(self._spheres)
        w = np.sqrt(weight)
        rows.append(w * basis @ (com - centres))
        jacs.append(w * (basis @ (jac_com - jac_centres))[:, self._dof])

    def support_offset(self, q: Array, ground: GroundPlane) -> float:
        """Distance in the ground plane from the CoM to the sphere centroid at ``q``."""
        self._set(q)
        self._mj.mj_comPos(self.model, self.data)
        n = np.asarray(ground.normal, dtype=float)
        n = n / np.linalg.norm(n)
        centres = np.mean(
            [self.data.site_xpos[site] for site, _ in self._spheres.values()], axis=0
        )
        offset = self.data.subtree_com[1] - centres
        offset = offset - (offset @ n) * n
        return float(np.linalg.norm(offset))

    def solve_trajectory(
        self,
        targets: Array,
        valid: Array,
        q_init: Array,
        *,
        ground: GroundPlane,
        frames: Sequence[int] | None = None,
        flat_feet_per_frame: Sequence[Sequence[str]] | None = None,
        plant_stance: bool = False,
        prior_trajectory: Array | None = None,
        **options: Any,
    ) -> tuple[Array, list[PoseFit]]:
        """Solve consecutive frames, each warm-started from the previous one.

        ``flat_feet_per_frame`` names, per capture frame, the contact spheres
        in stance (pinned to the plane). With ``plant_stance`` a sphere that
        enters stance is anchored to its ground point at that frame for as
        long as it stays in stance, so planted feet do not slide.
        ``prior_trajectory`` (frames, nv) replaces the warm start: frame ``k``
        starts from and is pulled toward its row. Precondition: targets
        (frames, markers, 3), valid (frames, markers). Postcondition: one row
        of q and one PoseFit per requested frame.
        """
        targets = np.asarray(targets, dtype=float)
        mask = np.asarray(valid, dtype=bool)
        if targets.ndim != 3 or mask.shape != targets.shape[:2]:
            raise ValueError("Trajectory targets must be (frames, markers, 3)")
        indices = list(range(targets.shape[0])) if frames is None else list(frames)
        if (
            flat_feet_per_frame is not None
            and len(flat_feet_per_frame) != targets.shape[0]
        ):
            raise ValueError("flat_feet_per_frame needs one entry per capture frame")
        if plant_stance and flat_feet_per_frame is None:
            raise ValueError("plant_stance needs flat_feet_per_frame")
        if prior_trajectory is not None:
            prior = np.asarray(prior_trajectory, dtype=float)
            if prior.shape != (targets.shape[0], len(self.coordinate_order)):
                raise ValueError("prior_trajectory must be (frames, coordinates)")
        q = np.asarray(q_init, dtype=float)
        fits: list[PoseFit] = []
        anchors: dict[str, Array] = {}
        for k in indices:
            stance = (
                () if flat_feet_per_frame is None else tuple(flat_feet_per_frame[k])
            )
            if plant_stance:
                anchors = {
                    name: point for name, point in anchors.items() if name in stance
                }
            start = q if prior_trajectory is None else prior[k]
            fit = self.solve_pose(
                targets[k],
                mask[k],
                start,
                ground=ground,
                flat_feet=stance if flat_feet_per_frame is not None else False,
                anchors=anchors if plant_stance else None,
                **options,
            )
            if plant_stance:
                points = self.sphere_ground_points(fit.q, ground)
                for name in stance:
                    anchors.setdefault(name, points[name])
            fits.append(fit)
            q = fit.q
        return np.array([fit.q for fit in fits]), fits
