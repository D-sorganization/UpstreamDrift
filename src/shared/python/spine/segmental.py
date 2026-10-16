"""Segmental spine kinematics: per-level transforms and ordered composition (SPN-1b).

Conventions follow ADR-0055 items 1-3. Each :class:`LevelTransform` maps
coordinates in the SUPERIOR vertebra frame into the INFERIOR vertebra frame::

    T_i = T_i0 @ T(R(theta_i), d_i)

``theta_i = (FE, LB, AR)`` are ISB ZXY intrinsic angles (profile ``isb_spine``
of :mod:`src.shared.python.biomechanics.joint_conventions`; radians). The
neutral offset ``T_i0`` is excluded from the reported angles and included in
every composition. A chain composes caudal factor first::

    T_caudal->cranial = T_a @ T_b @ T_c ...

Per-level angles never add; :meth:`SegmentalPose.angle_sum_diagnostic` is a
labelled diagnostic only. Units: radians, metres. Frames: X anterior,
Y cranial, Z right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.biomechanics.joint_conventions import (
    OrientationResult,
    RotationConvention,
    anatomical_joint_angles,
    orientations_to_matrix,
)
from src.shared.python.contracts import ensure
from src.shared.python.spine.levels import SpinalLevel

__all__ = ["LevelTransform", "SegmentalKinematicsSeries", "SegmentalPose"]

_ISB_SPINE = "isb_spine"
_ISB_SPINE_CONVENTION = RotationConvention("ZXY", degrees=False)
_ROT_TOL = 1e-9


def _frozen(value: np.ndarray) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _as_float(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array: {exc}") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_proper_rotation(rot: np.ndarray, name: str) -> None:
    if not np.allclose(rot.T @ rot, np.eye(3), atol=_ROT_TOL, rtol=0):
        raise ValueError(f"{name} must be a proper rotation (orthonormal)")
    if abs(float(np.linalg.det(rot)) - 1.0) > _ROT_TOL:
        raise ValueError(f"{name} must be a proper rotation (det = +1)")


def _homogeneous(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


@dataclass(frozen=True)
class LevelTransform:
    """Transform of one intervertebral level (superior frame into inferior).

    Preconditions (explicit ``TypeError``/``ValueError``): ``level`` is a
    :class:`SpinalLevel`; ``neutral`` is a finite (4, 4) rigid transform with a
    proper rotation block and bottom row ``[0, 0, 0, 1]``; ``rotation`` is a
    finite proper (3, 3) rotation (orthonormal, det +1, to 1e-9);
    ``translation`` is a finite (3,) vector (zeros for rotational models).

    Postconditions: stored arrays are copies and read-only.
    """

    level: SpinalLevel
    neutral: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.level, SpinalLevel):
            raise TypeError(
                f"level must be a SpinalLevel, got {type(self.level).__name__}"
            )
        neutral = _as_float(self.neutral, (4, 4), "neutral")
        if not np.allclose(neutral[3], (0.0, 0.0, 0.0, 1.0), atol=_ROT_TOL, rtol=0):
            raise ValueError("neutral must have bottom row [0, 0, 0, 1]")
        _check_proper_rotation(neutral[:3, :3], "neutral rotation block (proper)")
        rotation = _as_float(self.rotation, (3, 3), "rotation")
        _check_proper_rotation(rotation, "rotation (proper)")
        translation = _as_float(self.translation, (3,), "translation")
        object.__setattr__(self, "neutral", _frozen(neutral))
        object.__setattr__(self, "rotation", _frozen(rotation))
        object.__setattr__(self, "translation", _frozen(translation))

    def matrix(self) -> np.ndarray:
        """Return the (4, 4) transform ``T_i0 @ T(R, d)`` (superior into inferior)."""
        out = self.neutral @ _homogeneous(self.rotation, self.translation)
        ensure(out.shape == (4, 4), "level matrix must be 4x4", out.shape)
        return out

    @classmethod
    def from_isb_angles(
        cls,
        level: SpinalLevel,
        angles_rad: np.ndarray,
        neutral: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> LevelTransform:
        """Build a level from ISB ZXY angles ``(FE, LB, AR)`` in radians.

        Uses the same intrinsic ZXY convention as :meth:`isb_angles`, so the
        angles round-trip away from gimbal lock. ``neutral`` defaults to the
        identity and ``translation`` to zeros.

        Raises:
            ValueError: if ``angles_rad`` is not a finite length-3 vector.
        """
        angles = _as_float(angles_rad, (3,), "angles_rad")
        rot = orientations_to_matrix(angles, "euler", _ISB_SPINE_CONVENTION)
        return cls(
            level,
            np.eye(4) if neutral is None else neutral,
            rot,
            np.zeros(3) if translation is None else translation,
        )

    def isb_angles_result(self) -> OrientationResult:
        """Full ISB ``isb_spine`` extraction (radians) incl. the ``.singular`` flag.

        Excludes the neutral offset. At gimbal lock (``|LB| = 90 deg``) the
        result follows ``joint_conventions``: ``.singular`` is True and FE/AR
        are not individually identifiable, though the triple still
        reconstructs the rotation.
        """
        return anatomical_joint_angles(self.rotation, _ISB_SPINE)

    def isb_angles(self) -> np.ndarray:
        """Return ``(FE, LB, AR)`` in radians; the neutral offset is excluded."""
        return self.isb_angles_result().values


def _validate_contiguous(transforms: tuple[LevelTransform, ...]) -> None:
    if not transforms:
        raise ValueError("SegmentalPose requires at least one LevelTransform")
    for item in transforms:
        if not isinstance(item, LevelTransform):
            raise TypeError(
                f"transforms must hold LevelTransform, got {type(item).__name__}"
            )
    ordinals = [item.level.ordinal for item in transforms]
    if any(b - a != 1 for a, b in zip(ordinals, ordinals[1:], strict=False)):
        raise ValueError(
            "levels must be unique and contiguous in caudal-to-cranial order, "
            f"got {[item.level.name for item in transforms]}"
        )


@dataclass(frozen=True)
class SegmentalPose:
    """Contiguous caudal-to-cranial run of :class:`LevelTransform`.

    Raises ``TypeError``/``ValueError`` if empty, non-contiguous, duplicated or
    not caudal-first.
    """

    transforms: tuple[LevelTransform, ...]

    def __post_init__(self) -> None:
        items = tuple(self.transforms)
        _validate_contiguous(items)
        object.__setattr__(self, "transforms", items)

    @property
    def levels(self) -> tuple[SpinalLevel, ...]:
        """Levels of the pose, caudal to cranial."""
        return tuple(item.level for item in self.transforms)

    def _select(
        self, caudal: SpinalLevel | None, cranial: SpinalLevel | None
    ) -> tuple[LevelTransform, ...]:
        levels = self.levels
        lo = levels[0] if caudal is None else caudal
        hi = levels[-1] if cranial is None else cranial
        for name, lv in (("caudal", lo), ("cranial", hi)):
            if not isinstance(lv, SpinalLevel):
                raise TypeError(
                    f"{name} must be a SpinalLevel, got {type(lv).__name__}"
                )
            if lv not in levels:
                raise ValueError(f"{name} level {lv.name} not in pose {levels}")
        if lo.ordinal > hi.ordinal:
            raise ValueError(
                f"caudal level {lo.name} lies cranial to {hi.name}; "
                "give the caudal level first"
            )
        return self.transforms[levels.index(lo) : levels.index(hi) + 1]

    def compose(
        self, caudal: SpinalLevel | None = None, cranial: SpinalLevel | None = None
    ) -> np.ndarray:
        """Ordered (4, 4) product, caudal factor first, over ``caudal..cranial``.

        Defaults span the whole pose. Postcondition: the result is the explicit
        product ``T_a @ T_b @ ...`` of the selected :meth:`LevelTransform.matrix`.
        """
        out = np.eye(4)
        for item in self._select(caudal, cranial):
            out = out @ item.matrix()
        return out

    def relative_rotation(
        self, caudal: SpinalLevel, cranial: SpinalLevel
    ) -> np.ndarray:
        """(3, 3) rotation part of :meth:`compose` over ``caudal..cranial``."""
        return self.compose(caudal, cranial)[:3, :3]

    def angle_sum_diagnostic(self) -> np.ndarray:
        """Sum of per-level ISB angles (radians). NOT a definition of global motion.

        Cardan angles do not add (ADR-0055 item 3); this is a labelled
        diagnostic for comparison with the composed rotation only.
        """
        return np.sum([item.isb_angles() for item in self.transforms], axis=0)


@dataclass(frozen=True)
class SegmentalKinematicsSeries:
    """Time series of :class:`SegmentalPose` with provenance.

    Preconditions: ``time_s`` finite and strictly increasing; ``poses`` has
    one pose per time sample, all with the same level set; ``valid`` is a bool
    mask of the same length; ``distribution_source`` is a non-empty string
    (ADR-0055 item 7). Arrays are stored read-only.
    """

    time_s: np.ndarray
    poses: tuple[SegmentalPose, ...]
    distribution_source: str
    valid: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.distribution_source, str):
            raise TypeError("distribution_source must be a str")
        if not self.distribution_source.strip():
            raise ValueError("distribution_source must be non-empty")
        time_s = np.asarray(self.time_s, dtype=float)
        if time_s.ndim != 1 or time_s.size == 0:
            raise ValueError("time_s must be a non-empty 1-D array")
        if not np.all(np.isfinite(time_s)):
            raise ValueError("time_s must be finite")
        if np.any(np.diff(time_s) <= 0):
            raise ValueError("time_s must be strictly increasing")
        poses = tuple(self.poses)
        valid = np.asarray(self.valid)
        if valid.dtype != bool:
            raise ValueError("valid must be a bool array")
        if len(poses) != time_s.size or valid.shape != time_s.shape:
            raise ValueError(
                "length mismatch: time_s, poses and valid must have equal length"
            )
        for pose in poses:
            if not isinstance(pose, SegmentalPose):
                raise TypeError("poses must hold SegmentalPose")
            if pose.levels != poses[0].levels:
                raise ValueError("every frame must have the same level set")
        object.__setattr__(self, "time_s", _frozen(time_s))
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "valid", valid.copy())
        self.valid.setflags(write=False)

    def relative_rotations(
        self, caudal: SpinalLevel, cranial: SpinalLevel
    ) -> np.ndarray:
        """Read-only (n, 3, 3) relative rotations ``caudal..cranial`` per frame."""
        out = np.stack([p.relative_rotation(caudal, cranial) for p in self.poses])
        out.setflags(write=False)
        return out
