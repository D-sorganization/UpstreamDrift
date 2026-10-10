"""Same-input grip-kinetics parity across engines (issue #11739, OSV-7 phase 2).

:class:`GripKineticsSeries` is the engine-agnostic record of one bushing-grip
run (OpenSim reference, MuJoCo, Drake, Pinocchio): per-hand wrench ON THE
CLUB in the world frame with the free torque at the grip point, the grip
points, and the bushing deflections.  :func:`parity_errors` compares a
candidate with the reference on the shared 2 ms sample times.

Metric (binding, fixed before any comparison was run).  For a quantity
``x(t)`` (vector or scalar) with reference ``x_ref(t)``:

* peak error ``|max_t |x| - max_t |x_ref|| / max_t |x_ref|`` and
* RMS error ``sqrt(mean_t |x - x_ref|^2) / max_t |x_ref|`` (normalised by the
  reference peak, so near-zero samples do not inflate it).

Acceptance: peak error at most :data:`PEAK_TOLERANCE` (5 %) and RMS error at
most :data:`RMS_TOLERANCE` (2 %) for every quantity of
:data:`PARITY_QUANTITIES`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    GripSeries,
    HandWrench,
    analyze_grip,
)
from src.shared.python.grip_contact.force_decomposition import decompose_hand_forces

#: Relative error of the peak magnitude (fraction of the reference peak).
PEAK_TOLERANCE = 0.05
#: RMS of the difference, normalised by the reference peak.
RMS_TOLERANCE = 0.02
SIDES = ("L", "R")
PARITY_QUANTITIES = (
    "force_L",
    "force_R",
    "net_force",
    "internal_force",
    "squeeze",
    "couple",
    "deflection_L",
    "deflection_R",
    "rotation_L",
    "rotation_R",
)

__all__ = [
    "PARITY_QUANTITIES",
    "PEAK_TOLERANCE",
    "RMS_TOLERANCE",
    "BushingProbe",
    "GripKineticsSeries",
    "QuantityError",
    "parity_errors",
]


def _per_side(name: str, value: Mapping[str, Any], shape: tuple[int, ...]) -> dict:
    if set(value) != set(SIDES):
        raise ValueError(f"{name} must have exactly the sides {SIDES}")
    out = {}
    for side in SIDES:
        arr = np.asarray(value[side], dtype=float)
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}[{side}] must be finite with shape {shape}")
        out[side] = arr
    return out


@dataclass(frozen=True)
class GripKineticsSeries:
    """One engine's bushing-grip run on the shared sample times (world, SI).

    ``torque_on_club_nm`` is the free torque at the grip point (club-side
    bushing frame origin); ``deflection_m`` is ``R_hand^T (p_club - p_hand)``
    and ``rotation_deflection_rad`` the angle of ``R_hand^T R_club``.
    """

    engine: str
    time_s: np.ndarray
    force_on_club_n: Mapping[str, np.ndarray]
    torque_on_club_nm: Mapping[str, np.ndarray]
    grip_point_m: Mapping[str, np.ndarray]
    deflection_m: Mapping[str, np.ndarray]
    rotation_deflection_rad: Mapping[str, np.ndarray]
    club_rotation: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float)
        if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0.0):
            raise ValueError("time_s must be strictly increasing with >= 2 samples")
        n = t.size
        object.__setattr__(self, "time_s", t)
        for name in ("force_on_club_n", "torque_on_club_nm", "grip_point_m"):
            object.__setattr__(self, name, _per_side(name, getattr(self, name), (n, 3)))
        object.__setattr__(
            self, "deflection_m", _per_side("deflection_m", self.deflection_m, (n, 3))
        )
        object.__setattr__(
            self,
            "rotation_deflection_rad",
            _per_side("rotation_deflection_rad", self.rotation_deflection_rad, (n,)),
        )
        rot = np.asarray(self.club_rotation, dtype=float)
        if rot.shape != (n, 3, 3) or not np.all(np.isfinite(rot)):
            raise ValueError("club_rotation must be a finite (n, 3, 3) array")
        object.__setattr__(self, "club_rotation", rot)

    @classmethod
    def from_frames(
        cls,
        engine: str,
        time_s: np.ndarray,
        wrench: Mapping[str, tuple[np.ndarray, np.ndarray]],
        hand_pose: Mapping[str, tuple[np.ndarray, np.ndarray]],
        club_frame_pose: Mapping[str, tuple[np.ndarray, np.ndarray]],
        club_rotation: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> GripKineticsSeries:
        """Build from per-side ``(force, torque)`` and frame ``(R, p)`` series.

        Deflections are computed here, identically for every engine.
        """
        defl, angle = {}, {}
        for side in SIDES:
            r1, p1 = (np.asarray(a, float) for a in hand_pose[side])
            r2, p2 = (np.asarray(a, float) for a in club_frame_pose[side])
            defl[side] = np.einsum("nji,nj->ni", r1, p2 - p1)
            rel = np.einsum("nji,njk->nik", r1, r2)
            angle[side] = np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1)
        return cls(
            engine=engine,
            time_s=time_s,
            force_on_club_n={s: wrench[s][0] for s in SIDES},
            torque_on_club_nm={s: wrench[s][1] for s in SIDES},
            grip_point_m={s: club_frame_pose[s][1] for s in SIDES},
            deflection_m=defl,
            rotation_deflection_rad=angle,
            club_rotation=club_rotation,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_bushing_run(cls, engine: str, run: Any) -> GripKineticsSeries:
        """Adopt an OpenSim :class:`BushingRun` (same field names)."""
        return cls(
            engine=engine,
            time_s=run.time_s,
            force_on_club_n=run.force_on_club_n,
            torque_on_club_nm=run.torque_on_club_nm,
            grip_point_m=run.grip_point_m,
            deflection_m=run.deflection_m,
            rotation_deflection_rad=run.rotation_deflection_rad,
            club_rotation=run.club_rotation,
        )

    # ------------------------------------------------------------ analysis
    def analyses(self) -> list[GripAnalysis]:
        """Per-sample :func:`analyze_grip` (``split_method="bushing"``)."""
        out = []
        for i in range(self.time_s.size):
            hands = {
                s: HandWrench(
                    s,
                    tuple(self.grip_point_m[s][i]),
                    tuple(self.force_on_club_n[s][i]),
                    tuple(self.torque_on_club_nm[s][i]),
                )
                for s in SIDES
            }
            out.append(
                analyze_grip(
                    hands["L"],
                    hands["R"],
                    split_method="bushing",
                    club_rotation=self.club_rotation[i],
                    metadata={"solver": self.engine},
                )
            )
        return out

    def grip_series(self) -> GripSeries:
        """The GCV-10 :class:`GripSeries` of this run (plots and overlays)."""
        return GripSeries.from_analyses(self.time_s.tolist(), self.analyses())

    def quantities(self) -> dict[str, np.ndarray]:
        """Every quantity of :data:`PARITY_QUANTITIES` as an ``(n, k)`` array."""
        series = self.grip_series()
        dec = decompose_hand_forces(
            self.force_on_club_n["L"],
            self.force_on_club_n["R"],
            self.grip_point_m["L"],
            self.grip_point_m["R"],
        )
        out = {
            "force_L": series.left_force_n,
            "force_R": series.right_force_n,
            "net_force": series.net_force_n,
            "internal_force": dec.internal_n,
            "squeeze": np.atleast_1d(dec.internal_axial_n)[:, None],
            "couple": series.couple_nm,
        }
        for side in SIDES:
            out[f"deflection_{side}"] = self.deflection_m[side]
            out[f"rotation_{side}"] = self.rotation_deflection_rad[side][:, None]
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}

    # ----------------------------------------------------------- storage
    def save_npz(self, path: Path) -> None:
        """Write the series to ``path`` (overwritten by name)."""
        arrays: dict[str, Any] = {
            "engine": np.array(self.engine),
            "time_s": self.time_s,
            "club_rotation": self.club_rotation,
        }
        for side in SIDES:
            arrays[f"force_on_club_{side}_n"] = self.force_on_club_n[side]
            arrays[f"torque_on_club_{side}_nm"] = self.torque_on_club_nm[side]
            arrays[f"grip_point_{side}_m"] = self.grip_point_m[side]
            arrays[f"deflection_{side}_m"] = self.deflection_m[side]
            arrays[f"rotation_deflection_{side}_rad"] = self.rotation_deflection_rad[
                side
            ]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: Path) -> GripKineticsSeries:
        """Read a series written by :meth:`save_npz`."""
        with np.load(path, allow_pickle=False) as data:

            def side(key: str) -> dict[str, np.ndarray]:
                return {s: np.asarray(data[key.format(s=s)]) for s in SIDES}

            return cls(
                engine=str(data["engine"]),
                time_s=np.asarray(data["time_s"]),
                force_on_club_n=side("force_on_club_{s}_n"),
                torque_on_club_nm=side("torque_on_club_{s}_nm"),
                grip_point_m=side("grip_point_{s}_m"),
                deflection_m=side("deflection_{s}_m"),
                rotation_deflection_rad=side("rotation_deflection_{s}_rad"),
                club_rotation=np.asarray(data["club_rotation"]),
            )


@dataclass(frozen=True)
class BushingProbe:
    """Bushing forces at a static club displacement (``F = K delta`` checks).

    ``force_n`` is the per-hand force on the club, ``hand_rotation`` the
    (shared) world rotation of the hand frames, and ``engine_total_force_n``
    the total force read back from the engine's own generalised force on the
    club.
    """

    force_n: Mapping[str, np.ndarray]
    hand_rotation: np.ndarray
    engine_total_force_n: np.ndarray


@dataclass(frozen=True)
class QuantityError:
    """Peak and RMS error of one quantity against the reference."""

    peak_reference: float
    peak_candidate: float
    peak_error: float
    rms_error: float

    def passes(self) -> bool:
        """True when both errors are inside the parity tolerances."""
        return self.peak_error <= PEAK_TOLERANCE and self.rms_error <= RMS_TOLERANCE

    def to_dict(self) -> dict[str, float | bool]:
        """JSON-ready form."""
        return {
            "peak_reference": self.peak_reference,
            "peak_candidate": self.peak_candidate,
            "peak_error": self.peak_error,
            "rms_error": self.rms_error,
            "passes": self.passes(),
        }


def parity_errors(
    candidate: GripKineticsSeries,
    reference: GripKineticsSeries,
    t_end_s: float | None = None,
) -> dict[str, QuantityError]:
    """Peak and RMS errors of every parity quantity (see module docstring).

    Raises:
        ValueError: if the two time bases differ or a reference peak is zero.
    """
    t = reference.time_s
    if candidate.time_s.shape != t.shape or not np.allclose(
        candidate.time_s, t, atol=1e-9
    ):
        raise ValueError("candidate and reference must share their sample times")
    keep = np.ones(t.size, bool) if t_end_s is None else t <= t_end_s + 1e-12
    cand, ref = candidate.quantities(), reference.quantities()
    out = {}
    for name in PARITY_QUANTITIES:
        x, x_ref = cand[name][keep], ref[name][keep]
        mag, mag_ref = np.linalg.norm(x, axis=1), np.linalg.norm(x_ref, axis=1)
        peak_ref = float(mag_ref.max())
        if peak_ref <= 0.0:
            raise ValueError(f"reference peak of {name} is zero")
        rms = float(np.sqrt(np.mean(np.sum((x - x_ref) ** 2, axis=1))))
        out[name] = QuantityError(
            peak_reference=peak_ref,
            peak_candidate=float(mag.max()),
            peak_error=abs(float(mag.max()) - peak_ref) / peak_ref,
            rms_error=rms / peak_ref,
        )
    return out
