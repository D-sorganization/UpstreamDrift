"""Articulated kinematics: a tree of segments with typed joints (#9711).

The model is data: :class:`Joint` records name the parent, the rest offset
of the joint from its parent frame (a direction scaled by a named segment
length), which rotation axes the joint has (a hinge has one, a universal two,
a ball joint three) and their limits. The root carries three translations
as well. State ``q`` is one number per degree of freedom; forward kinematics
maps ``q`` for every frame at once to joint positions, and a finite-difference
Jacobian gives the sensitivity of the observable landmarks to every DOF.

Rotations are intrinsic and applied in the joint's axis order (``"xyz"`` is
Rx(a) Ry(b) Rz(c)); an axis-subset joint simply omits factors. This keeps
hinge joints exactly one-dimensional, which is what makes joint limits and
"continuous motion" meaningful constraints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

Array = npt.NDArray[np.float64]
AXES = "xyz"
FULL_RANGE = (-np.pi, np.pi)


@dataclass(frozen=True)
class Joint:
    """One joint of the tree; it moves the segment that ends at ``name``.

    ``direction`` is the rest-pose unit offset of this joint from the parent
    joint, expressed in the parent joint's frame; ``length`` names the
    segment-length parameter that scales it (``None`` for the root).
    """

    name: str
    parent: str | None
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    length: str | None = None
    axes: str = "xyz"
    limits_rad: tuple[tuple[float, float], ...] = ()
    landmark: bool = True
    #: Constant frames around the moving primitives: the child frame is
    #: ``parent . R(pre) . R_axes(q) . R(post)``. They let a joint carry a
    #: source model's fixed transforms (Simscape's rigid frames, #9714) so
    #: its angles are ours without conversion. Rotation vectors, radians.
    pre_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    post_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        require(self.name.strip() != "", "joint needs a name")
        require(
            len(self.pre_rotvec) == 3 and len(self.post_rotvec) == 3,
            "pre/post rotation vectors need three components",
        )
        require(all(a in AXES for a in self.axes), "axes must be x, y or z", self.axes)
        require(len(set(self.axes)) == len(self.axes), "axes must be distinct")
        d = np.asarray(self.direction, dtype=float)
        if self.parent is None:
            require(self.length is None, "root has no segment length")
        elif self.length is None:
            # Co-located joint: shares its parent's position (a scapula joint
            # sitting at the hub, a pronation joint at the elbow).
            require(bool(np.all(d == 0)), "co-located joint has a zero direction")
        else:
            require(bool(abs(np.linalg.norm(d) - 1.0) < 1e-6), "direction must be unit")
        if self.limits_rad:
            require(len(self.limits_rad) == len(self.axes), "one limit pair per axis")
            for lo, hi in self.limits_rad:
                require(lo < hi, "limit must be lo < hi", (lo, hi))

    def limits(self) -> tuple[tuple[float, float], ...]:
        return self.limits_rad or tuple(FULL_RANGE for _ in self.axes)

    def constant_frames(self) -> tuple[Array, Array]:
        """``(R(pre), R(post))`` as 3x3 matrices."""
        return _rotvec_matrix(self.pre_rotvec), _rotvec_matrix(self.post_rotvec)


@dataclass(frozen=True)
class ModelSpec:
    """Joints in an order where every parent precedes its children."""

    name: str
    joints: tuple[Joint, ...]
    lengths_m: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [j.name for j in self.joints]
        require(len(set(names)) == len(names), "joint names must be unique")
        require(
            bool(self.joints) and self.joints[0].parent is None, "first joint is root"
        )
        seen: set[str] = set()
        for j in self.joints:
            if j.parent is not None:
                require(j.parent in seen, "parent must precede child", j.name)
            if j.length is not None:
                require(j.length in self.lengths_m, "length parameter missing", j.name)
                require(self.lengths_m[j.length] > 0, "length must be > 0", j.name)
            seen.add(j.name)


def _rotvec_matrix(rotvec: Sequence[float]) -> Array:
    """Rodrigues: rotation matrix of a rotation vector (identity for zero)."""
    v = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def _axis_rotations(angles: Array, axes: str) -> Array:
    """``(T, 3, 3)`` intrinsic rotation for ``angles`` ``(T, len(axes))``."""
    t = angles.shape[0]
    out = np.broadcast_to(np.eye(3), (t, 3, 3)).copy()
    for k, axis in enumerate(axes):
        c, s = np.cos(angles[:, k]), np.sin(angles[:, k])
        r = np.zeros((t, 3, 3))
        i = AXES.index(axis)
        j, m = (i + 1) % 3, (i + 2) % 3
        r[:, i, i] = 1.0
        r[:, j, j], r[:, j, m] = c, -s
        r[:, m, j], r[:, m, m] = s, c
        out = np.einsum("tij,tjk->tik", out, r)
    return out


def decompose_primitives(
    rotations: Array, axes: str, signs: Sequence[int] | None = None
) -> tuple[Array, Array]:
    """Angles ``(T, k)`` of the intrinsic primitives ``axes`` that best give
    ``rotations`` ``(T, 3, 3)``, and the unrepresentable remainder ``(T,)``.

    Fewer than three primitives are completed with the remaining axes and
    the extra angles are reported as the remainder (radians), so a caller
    exporting a hinge from a general rotation can see what was dropped.
    ``signs`` (+1/-1 per primitive) flips the reported angle, for source
    models whose positive sense differs (#9714). Precondition: distinct axes.
    """
    from scipy.spatial.transform import Rotation

    require(0 < len(axes) <= 3 and len(set(axes)) == len(axes), "distinct axes")
    full = list(axes)
    for a in AXES:
        if len(full) == 3:
            break
        if a != full[-1] and a not in full:
            full.append(a)
    k = len(axes)
    sequence: Any = "".join(full).upper()
    euler = Rotation.from_matrix(np.asarray(rotations, dtype=float)).as_euler(sequence)
    s = np.ones(k) if signs is None else np.asarray(signs, dtype=float)
    remainder = np.abs(euler[:, k:]).max(axis=1) if k < 3 else np.zeros(len(euler))
    return euler[:, :k] * s, remainder


class ArticulatedModel:
    """Forward kinematics over frames for a :class:`ModelSpec`."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.joints = spec.joints
        self.index = {j.name: i for i, j in enumerate(self.joints)}
        self.dof_names: list[str] = [f"{self.joints[0].name}.t{a}" for a in AXES]
        self._slices: list[tuple[int, int]] = []
        col = 3
        for j in self.joints:
            self._slices.append((col, col + len(j.axes)))
            self.dof_names += [f"{j.name}.r{a}" for a in j.axes]
            col += len(j.axes)
        self.n_dof = col
        self.landmark_names: tuple[str, ...] = tuple(
            j.name for j in self.joints if j.landmark
        )
        self._landmark_rows = [self.index[n] for n in self.landmark_names]
        self._constant = [j.constant_frames() for j in self.joints]

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    def limits(self) -> tuple[Array, Array]:
        """``(lo, hi)`` per DOF; translations unbounded."""
        lo = [-np.inf] * 3
        hi = [np.inf] * 3
        for j in self.joints:
            for a, b in j.limits():
                lo.append(a)
                hi.append(b)
        return np.array(lo), np.array(hi)

    def dof_slice(self, joint: str) -> slice:
        a, b = self._slices[self.index[joint]]
        return slice(a, b)

    def forward(self, q: Array, lengths: Mapping[str, float] | None = None) -> Array:
        """Joint positions ``(T, J, 3)`` for states ``q`` ``(T, n_dof)``.

        Precondition: ``q`` has ``n_dof`` columns. Postcondition: every
        segment has exactly its length in every frame.
        """
        return self.forward_frames(q, lengths)[0]

    def frames(self, q: Array, lengths: Mapping[str, float] | None = None) -> Array:
        """Body orientations ``(T, J, 3, 3)``, world from body, per joint."""
        return self.forward_frames(q, lengths)[1]

    def forward_frames(
        self, q: Array, lengths: Mapping[str, float] | None = None
    ) -> tuple[Array, Array]:
        """``(positions (T, J, 3), frames (T, J, 3, 3))`` in one pass."""
        q = np.asarray(q, dtype=float)
        require(q.ndim == 2 and q.shape[1] == self.n_dof, "q must be (T, n_dof)")
        lengths = dict(self.spec.lengths_m if lengths is None else lengths)
        t = q.shape[0]
        positions = np.zeros((t, self.n_joints, 3))
        frames = np.zeros((t, self.n_joints, 3, 3))
        for i, j in enumerate(self.joints):
            a, b = self._slices[i]
            pre, post = self._constant[i]
            moving = (
                _axis_rotations(q[:, a:b], j.axes)
                if b > a
                else np.broadcast_to(np.eye(3), (t, 3, 3))
            )
            local = np.einsum("ij,tjk,kl->til", pre, moving, post)
            if j.parent is None:
                positions[:, i] = q[:, :3]
                frames[:, i] = local
                continue
            p = self.index[j.parent]
            if j.length is None:
                positions[:, i] = positions[:, p]
            else:
                offset = float(lengths[j.length]) * np.asarray(j.direction)
                positions[:, i] = positions[:, p] + np.einsum(
                    "tij,j->ti", frames[:, p], offset
                )
            frames[:, i] = np.einsum("tij,tjk->tik", frames[:, p], local)
        return positions, frames

    def landmarks(self, q: Array, lengths: Mapping[str, float] | None = None) -> Array:
        """Observable joint positions ``(T, L, 3)`` in :attr:`landmark_names` order."""
        return self.forward(q, lengths)[:, self._landmark_rows]

    def jacobian(
        self,
        q: Array,
        lengths: Mapping[str, float] | None = None,
        *,
        eps: float = 1e-6,
    ) -> Array:
        """``d landmarks / d q`` as ``(T, L*3, n_dof)`` by central differences.

        Each DOF is perturbed for all frames at once, so the cost is
        ``2 * n_dof`` forward passes regardless of the number of frames.
        """
        q = np.asarray(q, dtype=float)
        base_shape = (q.shape[0], len(self._landmark_rows) * 3)
        out = np.zeros((*base_shape, self.n_dof))
        for d in range(self.n_dof):
            plus = q.copy()
            minus = q.copy()
            plus[:, d] += eps
            minus[:, d] -= eps
            diff = self.landmarks(plus, lengths) - self.landmarks(minus, lengths)
            out[:, :, d] = diff.reshape(base_shape) / (2 * eps)
        return out

    def length_jacobian(
        self,
        q: Array,
        lengths: Mapping[str, float],
        names: Sequence[str],
        eps: float = 1e-6,
    ) -> Array:
        """``d landmarks / d length`` for the named lengths, ``(T, L*3, len(names))``."""
        q = np.asarray(q, dtype=float)
        base = self.landmarks(q, lengths).reshape(q.shape[0], -1)
        out = np.zeros((*base.shape, len(names)))
        for k, name in enumerate(names):
            bumped = {**lengths, name: lengths[name] + eps}
            out[:, :, k] = (self.landmarks(q, bumped).reshape(base.shape) - base) / eps
        return out


def wrap_angles(q: Array, model: ArticulatedModel) -> Array:
    """Rotation DOFs wrapped into (-pi, pi]; translations untouched."""
    out = np.asarray(q, dtype=float).copy()
    out[:, 3:] = (out[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    return out
