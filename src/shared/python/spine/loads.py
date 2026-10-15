"""Spinal intervertebral load series and the tension-positive bridge (SPN-5).

Convention (ADR-0055 item 5, binding). For every observable the wrench is the
force and moment exerted ON the SUPERIOR vertebra BY the INFERIOR vertebra,
expressed in the INFERIOR vertebra ISB frame (X anterior, Y cranial, Z right)
about the mid-disc joint centre:

* compression positive = ``+F_y`` (a compressed disc pushes the superior body
  cranially);
* anterior-posterior (AP) shear = ``F_x``, positive anterior;
* medio-lateral (ML) shear = ``F_z``, positive right;
* tension-positive consumers (``body_part_viz/axial_loads.py``) receive
  ``tension = -compression``.

Moving the reference point of a wrench uses
:func:`src.shared.python.force_overlay.conversions.move_wrench_point`, which
transports the moment as ``M' = M + r x F``.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.contracts import ensure
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind
from src.shared.python.force_overlay.conversions import move_wrench_point
from src.shared.python.spine.levels import SpinalLevel

__all__ = [
    "REQUIRED_PROVENANCE_KEYS",
    "LoadObservable",
    "SpinalAxialLoadProvider",
    "SpinalLoadSeries",
    "from_wrench_in_frame",
]

#: Provenance keys every series must carry, each with a non-empty string value.
#: ``model_id`` / ``model_sha256`` identify the musculoskeletal model;
#: ``solver`` names the analysis (ID, static optimisation, JointReaction);
#: ``cost_exponent`` is the static-optimisation exponent p (use ``"n/a"`` for
#: observables that do not use one); ``distribution_source`` records how the
#: level kinematics were distributed; ``external_loads`` states which external
#: loads were included.
REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = (
    "model_id",
    "model_sha256",
    "solver",
    "cost_exponent",
    "distribution_source",
    "external_loads",
)

_NORMALISED_KEY = "normalised_by_body_mass_kg"
_QUANTITIES = ("compression", "ap_shear", "ml_shear")
_ROTATION_TOL = 1e-6
_TIME_MATCH_TOL_S = 1e-9


class LoadObservable(enum.Enum):
    """The three load observables, never conflated (ADR-0055 item 5)."""

    GENERALIZED_EFFORT = "generalized_effort"
    JOINT_REACTION = "joint_reaction"
    PASSIVE_ELEMENT = "passive_element"


def _frozen(name: str, value: ArrayLike) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    array.setflags(write=False)
    return array


def _check_provenance(provenance: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping of str to str")
    for key in REQUIRED_PROVENANCE_KEYS:
        value = provenance.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"provenance key {key!r} is required and must be a non-empty string"
            )
    return MappingProxyType(dict(provenance))


@dataclass(frozen=True)
class SpinalLoadSeries:
    """Per-level wrench time series on the superior body, inferior ISB frame.

    Preconditions (raise ``TypeError``/``ValueError``): ``time_s`` is a finite,
    strictly increasing 1-D array; ``levels`` is a non-empty tuple of distinct
    :class:`SpinalLevel`; ``force_n`` and ``moment_nm`` are finite arrays of
    shape ``(n, L, 3)``; ``provenance`` holds every key of
    :data:`REQUIRED_PROVENANCE_KEYS` with a non-empty string.

    Postconditions: all arrays are read-only copies and ``provenance`` is an
    immutable mapping. ``force_unit`` is ``"N"`` (moment ``"N*m"``) or, after
    :meth:`in_body_weights`, ``"BW"`` (moment ``"BW*m"``).
    """

    time_s: np.ndarray
    levels: tuple[SpinalLevel, ...]
    observable: LoadObservable
    force_n: np.ndarray
    moment_nm: np.ndarray
    provenance: Mapping[str, str]
    force_unit: str = "N"

    def __post_init__(self) -> None:
        if not isinstance(self.observable, LoadObservable):
            raise TypeError("observable must be a LoadObservable")
        if not isinstance(self.levels, tuple) or not all(
            isinstance(level, SpinalLevel) for level in self.levels
        ):
            raise TypeError("levels must be a tuple of SpinalLevel")
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be non-empty and contain distinct levels")
        time_s = _frozen("time_s", self.time_s)
        if time_s.ndim != 1 or time_s.size == 0:
            raise ValueError("time_s must be a non-empty 1-D array")
        if np.any(np.diff(time_s) <= 0):
            raise ValueError("time_s must be strictly increasing")
        shape = (time_s.size, len(self.levels), 3)
        force = _frozen("force_n", self.force_n)
        moment = _frozen("moment_nm", self.moment_nm)
        for name, array in (("force_n", force), ("moment_nm", moment)):
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "force_n", force)
        object.__setattr__(self, "moment_nm", moment)
        object.__setattr__(self, "provenance", _check_provenance(self.provenance))

    def compression_n(self) -> np.ndarray:
        """Compression, shape ``(n, L)``; positive = compression (``+F_y``)."""
        return self.force_n[:, :, 1]

    def ap_shear_n(self) -> np.ndarray:
        """AP shear, shape ``(n, L)``; positive anterior (``+F_x``)."""
        return self.force_n[:, :, 0]

    def ml_shear_n(self) -> np.ndarray:
        """ML shear, shape ``(n, L)``; positive right (``+F_z``)."""
        return self.force_n[:, :, 2]

    def peak(self, quantity: str) -> dict[SpinalLevel, tuple[float, float]]:
        """Return ``(signed value, time_s)`` of the largest magnitude per level.

        ``quantity`` is one of ``compression``, ``ap_shear``, ``ml_shear``.
        Ties resolve to the earliest sample.
        """
        if quantity not in _QUANTITIES:
            raise ValueError(f"quantity must be one of {_QUANTITIES}, got {quantity!r}")
        values = {
            "compression": self.compression_n,
            "ap_shear": self.ap_shear_n,
            "ml_shear": self.ml_shear_n,
        }[quantity]()
        index = np.argmax(np.abs(values), axis=0)
        return {
            level: (float(values[i, k]), float(self.time_s[i]))
            for k, (level, i) in enumerate(zip(self.levels, index, strict=True))
        }

    def in_body_weights(self, body_mass_kg: float) -> SpinalLoadSeries:
        """Return a copy normalised by body weight ``m * g`` (g = 9.80665).

        Forces become multiples of body weight (``force_unit == "BW"``) and
        moments body weight times metres. The mass is recorded in provenance.
        A series can be normalised only once.
        """
        if isinstance(body_mass_kg, bool) or not isinstance(
            body_mass_kg, int | float | np.floating | np.integer
        ):
            raise TypeError("body_mass_kg must be a real number")
        if not math.isfinite(body_mass_kg) or body_mass_kg <= 0:
            raise ValueError("body_mass_kg must be finite and positive")
        if self.force_unit != "N":
            raise ValueError("series is already normalised to body weight")
        weight_n = float(body_mass_kg) * GRAVITY_FLOAT
        provenance = {**self.provenance, _NORMALISED_KEY: str(float(body_mass_kg))}
        result = replace(
            self,
            force_n=self.force_n / weight_n,
            moment_nm=self.moment_nm / weight_n,
            provenance=provenance,
            force_unit="BW",
        )
        ensure(result.force_unit == "BW", "normalised series must be in BW", result)
        return result


def _rotation_stack(rotation: ArrayLike, n: int, n_levels: int) -> np.ndarray:
    rot = np.asarray(rotation, dtype=float)
    if rot.shape == (3, 3):
        rot = np.broadcast_to(rot, (n, n_levels, 3, 3))
    elif rot.shape == (n_levels, 3, 3):
        rot = np.broadcast_to(rot[None], (n, n_levels, 3, 3))
    elif rot.shape != (n, n_levels, 3, 3):
        raise ValueError(
            "rotation_frame_to_isb must have shape (3,3), (L,3,3) or (n,L,3,3); "
            f"got {rot.shape}"
        )
    if not np.isfinite(rot).all():
        raise ValueError("rotation_frame_to_isb must be finite")
    gram = np.einsum("...ji,...jk->...ik", rot, rot)
    if not np.allclose(gram, np.eye(3), atol=_ROTATION_TOL) or not np.allclose(
        np.linalg.det(rot), 1.0, atol=_ROTATION_TOL
    ):
        raise ValueError("rotation_frame_to_isb must be proper rotations (R^T R = I)")
    return rot


def _offset_stack(offset: ArrayLike, n_levels: int) -> np.ndarray:
    arr = np.asarray(offset, dtype=float)
    if arr.shape == (3,):
        arr = np.broadcast_to(arr, (n_levels, 3))
    if arr.shape != (n_levels, 3) or not np.isfinite(arr).all():
        raise ValueError("point_offset_m must be a finite (3,) or (L,3) array")
    return arr


def _transport(
    level: SpinalLevel, force: np.ndarray, moment: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """Move one wrench from ``offset`` to the joint centre; return the moment."""
    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label=f"joint_reaction:{level.name}",
        body=level.superior,
        point_m=tuple(offset),
        force_n=tuple(force),
        torque_nm=tuple(moment),
        source="spine.loads",
    )
    moved = move_wrench_point(wrench, (0.0, 0.0, 0.0))
    return np.asarray(moved.torque_nm, dtype=float)


def from_wrench_in_frame(
    *,
    time_s: ArrayLike,
    levels: tuple[SpinalLevel, ...],
    observable: LoadObservable,
    force: ArrayLike,
    moment: ArrayLike,
    rotation_frame_to_isb: ArrayLike,
    point_offset_m: ArrayLike,
    provenance: Mapping[str, str],
) -> SpinalLoadSeries:
    """Build a series from a wrench given in an arbitrary source frame.

    ``force`` and ``moment`` have shape ``(n, L, 3)`` and are the wrench on the
    superior body by the inferior body, applied at a reference point located
    ``point_offset_m`` from the mid-disc joint centre, with that offset
    expressed in the inferior ISB frame (shape ``(3,)`` or ``(L, 3)``).
    ``rotation_frame_to_isb`` maps source-frame vectors into the inferior ISB
    frame (``v_isb = R @ v_src``); shape ``(3,3)``, ``(L,3,3)`` or
    ``(n,L,3,3)``.

    The force is rotated, then the wrench is moved to the joint centre with
    ``move_wrench_point`` (``M' = M + r x F`` with ``r`` the offset).

    Raises:
        TypeError, ValueError: on any violated precondition (see
            :class:`SpinalLoadSeries` plus the shape and rotation checks here).
    """
    time_arr = np.asarray(time_s, dtype=float)
    force_arr = _frozen("force", force)
    moment_arr = _frozen("moment", moment)
    if not isinstance(levels, tuple):
        raise TypeError("levels must be a tuple of SpinalLevel")
    shape = (time_arr.size, len(levels), 3)
    for name, array in (("force", force_arr), ("moment", moment_arr)):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    rot = _rotation_stack(rotation_frame_to_isb, shape[0], shape[1])
    offsets = _offset_stack(point_offset_m, shape[1])
    force_isb = np.einsum("nlij,nlj->nli", rot, force_arr)
    moment_isb = np.einsum("nlij,nlj->nli", rot, moment_arr)
    moved = np.empty_like(moment_isb)
    for i in range(shape[0]):
        for k, level in enumerate(levels):
            moved[i, k] = _transport(
                level, force_isb[i, k], moment_isb[i, k], offsets[k]
            )
    return SpinalLoadSeries(
        time_s=time_arr,
        levels=levels,
        observable=observable,
        force_n=force_isb,
        moment_nm=moved,
        provenance=provenance,
    )


class SpinalAxialLoadProvider:
    """``AxialLoadProvider`` view of one sample of a :class:`SpinalLoadSeries`.

    Values are keyed by level name and are tension-positive:
    ``tension = -compression``. The sample time must coincide with a series
    sample (no interpolation); the series must be in newtons.
    """

    def __init__(self, series: SpinalLoadSeries, time_s: float) -> None:
        if not isinstance(series, SpinalLoadSeries):
            raise TypeError("series must be a SpinalLoadSeries")
        if series.force_unit != "N":
            raise ValueError("axial bridge requires a series in newtons, not BW")
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        index = int(np.argmin(np.abs(series.time_s - time_s)))
        if abs(float(series.time_s[index]) - time_s) > _TIME_MATCH_TOL_S:
            raise ValueError(f"time_s={time_s} does not match a series sample")
        self._series = series
        self._index = index

    def get_segment_axial_loads(self) -> AxialLoadFrame | None:
        """Return the tension-positive axial force at each level (N)."""
        series = self._series
        compression = series.compression_n()[self._index]
        values: dict[str, Any] = {
            level.name: -float(c)
            for level, c in zip(series.levels, compression, strict=True)
        }
        source = (
            f"spine.loads:{series.observable.value}:"
            f"{series.provenance['distribution_source']}"
        )
        return AxialLoadFrame(float(series.time_s[self._index]), values, source)
