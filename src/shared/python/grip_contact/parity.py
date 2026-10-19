"""Same-input grip-kinetics parity across engines (issue #11739, OSV-7 phase 2; #13308, HRG-GS-G10).

:class:`GripKineticsSeries` is the engine-agnostic record of one bushing-grip or
contact-grip run (OpenSim, MuJoCo, Drake, Pinocchio): per-hand wrench ON THE
CLUB in the world frame with the free torque at the grip point, the grip
points, and the hand-to-club deflections. :func:`parity_errors` compares a
candidate with the reference on the shared 2 ms sample times.

Metric (binding, fixed before any comparison was run). For a quantity
``x(t)`` (vector or scalar) with reference ``x_ref(t)``:

* peak error ``|max_t |x| - max_t |x_ref|| / max_t |x_ref|`` and
* RMS error ``sqrt(mean_t |x - x_ref|^2) / max_t |x_ref|`` (normalised by the
  reference peak, so near-zero samples do not inflate it).

Acceptance: peak error at most :data:`PEAK_TOLERANCE` (5 %) and RMS error at
most :data:`RMS_TOLERANCE` (2 %).

For two-hand contact grips (HRG-GS-G10, #13308), quantities are explicitly
classified into:

* :data:`DETERMINATE_QUANTITIES` (``net_force`` and ``net_moment_com``, which
  club kinematics fix) — must pass :data:`PEAK_TOLERANCE` and
  :data:`RMS_TOLERANCE`;
* :data:`INDETERMINATE_QUANTITIES` (per-hand split, internal force, squeeze,
  couple, deflections, rotations, and per-pair normal force, tangential force,
  and contact-point drift of the hyperstatic two-hand grip) — reported with
  measured error and labelled :data:`INDETERMINATE_LABEL`, citing
  ``docs/development/full_body_models/GRIP_PARITY_DECISIONS.md`` section 19.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    GripSeries,
    HandWrench,
    analyze_grip,
)
from src.shared.python.grip_contact.force_decomposition import decompose_hand_forces

if TYPE_CHECKING:
    from src.shared.python.grip_contact.club_dynamics import ClubDynamics
    from src.shared.python.grip_contact.contact_run import ContactRun
    from src.shared.python.grip_contact.contact_topology import GripStyle
    from src.shared.python.grip_contact.interface import GripInterface

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
#: Per-pair contact quantities added by HRG-GS-G10 (#13308).
PAIR_PARITY_QUANTITIES = (
    "pair_normal_force",
    "pair_tangential_force",
    "contact_point_drift",
)
#: Quantities dynamically determined by the rigid club's 6-DOF equation of motion.
DETERMINATE_QUANTITIES = (
    "net_force",
    "net_moment_com",
)
#: Quantities belonging to the hyperstatic two-hand internal force/pair partitioning.
INDETERMINATE_QUANTITIES = (
    "force_L",
    "force_R",
    "internal_force",
    "squeeze",
    "couple",
    "deflection_L",
    "deflection_R",
    "rotation_L",
    "rotation_R",
    *PAIR_PARITY_QUANTITIES,
)
#: Complete table of quantities reported in a grip-style contact parity report.
STYLE_PARITY_QUANTITIES = (
    *DETERMINATE_QUANTITIES,
    *INDETERMINATE_QUANTITIES,
)
#: Canonical engines compared in grip-style parity reports.
PARITY_ENGINES = ("pinocchio", "opensim", "mujoco", "drake")
#: Required label for indeterminate quantities in the hyperstatic two-hand grip.
INDETERMINATE_LABEL = (
    "indeterminate (hyperstatic two-hand grip; not identified by club kinematics)"
)
#: Citation to the indeterminacy analysis.
INDETERMINACY_CITATION = (
    "docs/development/full_body_models/GRIP_PARITY_DECISIONS.md "
    "section 19 (Indeterminacy Report)"
)


class QuantityClassification(str, Enum):
    """Observability classification of a grip-parity quantity (#13308)."""

    DETERMINATE = "determinate"
    INDETERMINATE = "indeterminate"


class ParityState(str, Enum):
    """Evaluation state of a quantity in a cross-engine parity report (#13308)."""

    PASS = "pass"  # nosec B105 - status enum label, not a password
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    UNAVAILABLE = "unavailable"


__all__ = [
    "DETERMINATE_QUANTITIES",
    "INDETERMINACY_CITATION",
    "INDETERMINATE_LABEL",
    "INDETERMINATE_QUANTITIES",
    "PAIR_PARITY_QUANTITIES",
    "PARITY_ENGINES",
    "PARITY_QUANTITIES",
    "PEAK_TOLERANCE",
    "RMS_TOLERANCE",
    "STYLE_PARITY_QUANTITIES",
    "BushingProbe",
    "ClassifiedQuantityResult",
    "EngineParityRow",
    "GripKineticsSeries",
    "ParityState",
    "QuantityClassification",
    "QuantityError",
    "StyleParityReport",
    "build_style_parity_report",
    "classify_parity_quantity",
    "parity_errors",
]


def classify_parity_quantity(name: str) -> QuantityClassification:
    """Return the explicit observability classification of ``name``.

    Preconditions:
        ``name`` is a known quantity in :data:`STYLE_PARITY_QUANTITIES` or a
        per-pair key prefixed with ``<pair_metric>:``.
    Postconditions:
        Returns :attr:`QuantityClassification.DETERMINATE` only for
        ``net_force`` and ``net_moment_com``, and
        :attr:`QuantityClassification.INDETERMINATE` for all hyperstatic
        split/pair quantities.
    """
    if not isinstance(name, str):
        raise TypeError(f"quantity name must be a str, got {type(name).__name__}")
    if name in DETERMINATE_QUANTITIES:
        return QuantityClassification.DETERMINATE
    if name in INDETERMINATE_QUANTITIES:
        return QuantityClassification.INDETERMINATE
    if ":" in name:
        prefix, pair_name = name.split(":", 1)
        if prefix in PAIR_PARITY_QUANTITIES and bool(pair_name.strip()):
            return QuantityClassification.INDETERMINATE
    raise ValueError(f"unknown parity quantity: {name!r}")


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
    """One engine's bushing-grip or contact-grip run on shared sample times (world, SI).

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

    def net_moment_com_nm(
        self,
        club: ClubDynamics | None = None,
        interface: GripInterface | None = None,
    ) -> np.ndarray:
        """Net moment on the club about the club centre of mass as an ``(n, 3)`` array."""
        if club is not None and interface is not None:
            dp_l = np.asarray(club.com_m, dtype=float) - np.asarray(
                interface.left.position_m, dtype=float
            )
        elif "club_com_offset_l_m" in self.metadata:
            dp_l = np.asarray(self.metadata["club_com_offset_l_m"], dtype=float)
        else:
            # Canonical driver bare-club COM offset from lead grip point in club frame
            dp_l = np.array(
                (0.02501411692651008, 0.8679856230031948, 0.004910663712740029),
                dtype=float,
            )
        p_com = self.grip_point_m["L"] + np.einsum(
            "nij,j->ni", self.club_rotation, dp_l
        )
        return (
            self.torque_on_club_nm["L"]
            + self.torque_on_club_nm["R"]
            + np.cross(self.grip_point_m["L"] - p_com, self.force_on_club_n["L"])
            + np.cross(self.grip_point_m["R"] - p_com, self.force_on_club_n["R"])
        )

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


def _compute_quantity_error(
    x: np.ndarray, x_ref: np.ndarray, name: str
) -> QuantityError:
    """Compute peak and RMS error of ``x`` against ``x_ref`` on matching samples."""
    x_arr = np.asarray(x, dtype=float)
    x_ref_arr = np.asarray(x_ref, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_ref_arr.ndim == 1:
        x_ref_arr = x_ref_arr[:, None]
    mag = np.linalg.norm(x_arr, axis=1)
    mag_ref = np.linalg.norm(x_ref_arr, axis=1)
    peak_ref = float(mag_ref.max())
    if peak_ref <= 0.0:
        raise ValueError(f"reference peak of {name} is zero")
    rms = float(np.sqrt(np.mean(np.sum((x_arr - x_ref_arr) ** 2, axis=1))))
    return QuantityError(
        peak_reference=peak_ref,
        peak_candidate=float(mag.max()),
        peak_error=abs(float(mag.max()) - peak_ref) / peak_ref,
        rms_error=rms / peak_ref,
    )


def _unwrap_series_and_run(
    item: GripKineticsSeries | ContactRun, label: str
) -> tuple[GripKineticsSeries, Any]:
    """Return ``(series, contact_run_or_none)`` from a series or ContactRun."""
    if isinstance(item, GripKineticsSeries):
        return item, None
    series = getattr(item, "series", None)
    if isinstance(series, GripKineticsSeries):
        return series, item
    raise TypeError(
        f"{label} must be GripKineticsSeries or ContactRun, got {type(item).__name__}"
    )


_PAIR_METRIC_ATTRS: tuple[tuple[str, str], ...] = (
    ("pair_normal_force", "pair_normal_force_n"),
    ("pair_tangential_force", "pair_tangential_force_n"),
    ("contact_point_drift", "pair_contact_point_drift_m"),
)


def _extract_physical_pair_arrays(
    cand_run: Any, ref_run: Any, keep: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Stack finite physical hand-grip pair traces into ``(n, n_phys)`` arrays."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric, attr in _PAIR_METRIC_ATTRS:
        c_map = getattr(cand_run, attr, None)
        r_map = getattr(ref_run, attr, None)
        if c_map is None or r_map is None:
            raise ValueError(
                f"{attr} must be populated on both candidate and reference ContactRun "
                f"to evaluate {metric}"
            )
        c_cols: list[np.ndarray] = []
        r_cols: list[np.ndarray] = []
        for pname, r_vals in r_map.items():
            r_arr = np.asarray(r_vals, dtype=float)[keep]
            if np.all(np.isnan(r_arr)):
                continue
            if pname not in c_map:
                raise ValueError(f"candidate ContactRun missing pair {pname!r}")
            c_arr = np.asarray(c_map[pname], dtype=float)[keep]
            c_cols.append(c_arr)
            r_cols.append(r_arr)
        if not r_cols:
            raise ValueError(f"no physical pairs found for {metric}")
        out[metric] = (np.column_stack(c_cols), np.column_stack(r_cols))
    return out


def parity_errors(
    candidate: GripKineticsSeries | ContactRun,
    reference: GripKineticsSeries | ContactRun,
    t_end_s: float | None = None,
    *,
    club: ClubDynamics | None = None,
    interface: GripInterface | None = None,
) -> dict[str, QuantityError]:
    """Peak and RMS errors of every parity quantity (see module docstring).

    When ``candidate`` and ``reference`` are :class:`GripKineticsSeries`
    instances, compares :data:`PARITY_QUANTITIES`. When both are
    :class:`ContactRun` instances, also compares ``net_moment_com`` and
    :data:`PAIR_PARITY_QUANTITIES` using the same :class:`QuantityError` metric
    and tolerances.

    Raises:
        ValueError: if the two time bases differ or a reference peak is zero.
    """
    cand_series, cand_run = _unwrap_series_and_run(candidate, "candidate")
    ref_series, ref_run = _unwrap_series_and_run(reference, "reference")
    t = ref_series.time_s
    if cand_series.time_s.shape != t.shape or not np.allclose(
        cand_series.time_s, t, atol=1e-9
    ):
        raise ValueError("candidate and reference must share their sample times")
    keep = np.ones(t.size, bool) if t_end_s is None else t <= t_end_s + 1e-12
    cand_q, ref_q = cand_series.quantities(), ref_series.quantities()
    out: dict[str, QuantityError] = {}
    for name in PARITY_QUANTITIES:
        out[name] = _compute_quantity_error(cand_q[name][keep], ref_q[name][keep], name)
    if cand_run is not None and ref_run is not None:
        m_cand = cand_series.net_moment_com_nm(club, interface)[keep]
        m_ref = ref_series.net_moment_com_nm(club, interface)[keep]
        out["net_moment_com"] = _compute_quantity_error(m_cand, m_ref, "net_moment_com")
        pair_arrays = _extract_physical_pair_arrays(cand_run, ref_run, keep)
        for metric, (x_c, x_r) in pair_arrays.items():
            out[metric] = _compute_quantity_error(x_c, x_r, metric)
    return out


@dataclass(frozen=True)
class ClassifiedQuantityResult:
    """Classified parity result for one quantity in an engine row (#13308).

    Invariants (enforced in ``__post_init__``):
    - Unavailable quantities store ``NaN`` for ``peak_candidate``,
      ``peak_error``, and ``rms_error``, and require a non-empty ``reason``.
    - Indeterminate quantities have ``state == ParityState.INDETERMINATE`` (when
      available), are labelled with :data:`INDETERMINATE_LABEL`, and are never
      counted as a pass (``passes()`` is ``False``).
    - Determinate quantities pass if and only if available and within
      :data:`PEAK_TOLERANCE` and :data:`RMS_TOLERANCE`.
    """

    quantity: str
    classification: QuantityClassification
    state: ParityState
    peak_reference: float
    peak_candidate: float
    peak_error: float
    rms_error: float
    label: str
    reason: str = ""
    citation: str = ""

    def __post_init__(self) -> None:
        expected_cls = classify_parity_quantity(self.quantity)
        if self.classification != expected_cls:
            raise ValueError(
                f"quantity {self.quantity!r} must have classification "
                f"{expected_cls.value!r}, got {self.classification!r}"
            )
        if self.state == ParityState.UNAVAILABLE:
            if not (
                math.isnan(self.peak_candidate)
                and math.isnan(self.peak_error)
                and math.isnan(self.rms_error)
            ):
                raise ValueError(
                    f"unavailable quantity {self.quantity!r} must have NaN errors, "
                    f"never zero"
                )
            if not self.reason.strip():
                raise ValueError(
                    f"unavailable quantity {self.quantity!r} requires a non-empty reason"
                )
            return

        if not (
            math.isfinite(self.peak_reference)
            and math.isfinite(self.peak_candidate)
            and math.isfinite(self.peak_error)
            and math.isfinite(self.rms_error)
        ):
            raise ValueError(
                f"available quantity {self.quantity!r} must have finite values"
            )
        if self.classification == QuantityClassification.INDETERMINATE:
            if self.state != ParityState.INDETERMINATE:
                raise ValueError(
                    f"indeterminate quantity {self.quantity!r} must have state "
                    f"'indeterminate', got {self.state.value!r}"
                )
            if self.label != INDETERMINATE_LABEL:
                raise ValueError(
                    f"indeterminate quantity {self.quantity!r} must carry "
                    f"INDETERMINATE_LABEL"
                )
            if self.citation != INDETERMINACY_CITATION:
                raise ValueError(
                    f"indeterminate quantity {self.quantity!r} must carry "
                    f"INDETERMINACY_CITATION"
                )
        else:
            within = (
                self.peak_error <= PEAK_TOLERANCE and self.rms_error <= RMS_TOLERANCE
            )
            expected_state = ParityState.PASS if within else ParityState.FAIL
            if self.state != expected_state:
                raise ValueError(
                    f"determinate quantity {self.quantity!r} state mismatch: "
                    f"expected {expected_state.value!r}, got {self.state.value!r}"
                )

    def passes(self) -> bool:
        """True only for available DETERMINATE quantities within tolerances."""
        return (
            self.classification == QuantityClassification.DETERMINATE
            and self.state == ParityState.PASS
        )

    @classmethod
    def from_quantity_error(
        cls, quantity: str, qe: QuantityError
    ) -> ClassifiedQuantityResult:
        """Classify a measured :class:`QuantityError`."""
        classification = classify_parity_quantity(quantity)
        if classification == QuantityClassification.INDETERMINATE:
            return cls(
                quantity=quantity,
                classification=classification,
                state=ParityState.INDETERMINATE,
                peak_reference=qe.peak_reference,
                peak_candidate=qe.peak_candidate,
                peak_error=qe.peak_error,
                rms_error=qe.rms_error,
                label=INDETERMINATE_LABEL,
                citation=INDETERMINACY_CITATION,
            )
        state = ParityState.PASS if qe.passes() else ParityState.FAIL
        return cls(
            quantity=quantity,
            classification=classification,
            state=state,
            peak_reference=qe.peak_reference,
            peak_candidate=qe.peak_candidate,
            peak_error=qe.peak_error,
            rms_error=qe.rms_error,
            label=state.value,
        )

    @classmethod
    def unavailable(
        cls,
        quantity: str,
        reason: str,
        peak_reference: float = float("nan"),
    ) -> ClassifiedQuantityResult:
        """Construct an unavailable quantity result with NaN values and ``reason``."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        classification = classify_parity_quantity(quantity)
        nan = float("nan")
        return cls(
            quantity=quantity,
            classification=classification,
            state=ParityState.UNAVAILABLE,
            peak_reference=peak_reference,
            peak_candidate=nan,
            peak_error=nan,
            rms_error=nan,
            label=f"unavailable ({reason})",
            reason=reason,
            citation=(
                INDETERMINACY_CITATION
                if classification == QuantityClassification.INDETERMINATE
                else ""
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "quantity": self.quantity,
            "classification": self.classification.value,
            "state": self.state.value,
            "peak_reference": self.peak_reference,
            "peak_candidate": self.peak_candidate,
            "peak_error": self.peak_error,
            "rms_error": self.rms_error,
            "passes": self.passes(),
            "label": self.label,
            "reason": self.reason,
            "citation": self.citation,
        }


@dataclass(frozen=True)
class EngineParityRow:
    """Parity evaluation of one engine in a :class:`StyleParityReport` (#13308)."""

    engine: str
    available: bool
    quantities: Mapping[str, ClassifiedQuantityResult]
    pair_quantities: Mapping[str, Mapping[str, ClassifiedQuantityResult]] = field(
        default_factory=dict
    )
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("engine must be a non-empty string")
        if not self.available and not self.unavailable_reason.strip():
            raise ValueError("unavailable EngineParityRow requires unavailable_reason")

    def quantity(self, name: str) -> ClassifiedQuantityResult:
        """Return the :class:`ClassifiedQuantityResult` for ``name``."""
        if name not in self.quantities:
            raise KeyError(f"quantity {name!r} not found in engine {self.engine!r}")
        return self.quantities[name]

    def pair_quantity(self, pair_name: str, metric: str) -> ClassifiedQuantityResult:
        """Return the per-pair :class:`ClassifiedQuantityResult` for ``(pair_name, metric)``."""
        if pair_name not in self.pair_quantities:
            raise KeyError(f"pair {pair_name!r} not found in engine {self.engine!r}")
        if metric not in self.pair_quantities[pair_name]:
            raise KeyError(
                f"metric {metric!r} not found for pair {pair_name!r} in {self.engine!r}"
            )
        return self.pair_quantities[pair_name][metric]

    def determinate_passes(self) -> bool:
        """True if the engine is available and every DETERMINATE quantity passes."""
        if not self.available:
            return False
        return all(
            self.quantities[q].passes()
            for q in DETERMINATE_QUANTITIES
            if q in self.quantities
        )

    def failing_determinate(self) -> dict[str, ClassifiedQuantityResult]:
        """Return any DETERMINATE quantities that did not pass."""
        return {
            q: self.quantities[q]
            for q in DETERMINATE_QUANTITIES
            if q in self.quantities and not self.quantities[q].passes()
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "engine": self.engine,
            "available": self.available,
            "determinate_passes": self.determinate_passes(),
            "unavailable_reason": self.unavailable_reason,
            "quantities": {k: v.to_dict() for k, v in self.quantities.items()},
            "pair_quantities": {
                pname: {m: res.to_dict() for m, res in metrics.items()}
                for pname, metrics in self.pair_quantities.items()
            },
        }


@dataclass(frozen=True)
class StyleParityReport:
    """Cross-engine parity report for one grip style and motion mode (#13308)."""

    style: str
    motion: str
    duration_s: float
    reference_engine: str
    rows: Mapping[str, EngineParityRow]
    citation: str = INDETERMINACY_CITATION

    def row(self, engine: str) -> EngineParityRow:
        """Return the :class:`EngineParityRow` for ``engine``."""
        if engine not in self.rows:
            raise KeyError(f"engine {engine!r} not in report rows {tuple(self.rows)}")
        return self.rows[engine]

    def engine_determinate_passes(self, engine: str) -> bool:
        """True when ``engine`` is available and its DETERMINATE quantities pass."""
        return self.row(engine).determinate_passes()

    def all_available_determinate_pass(self) -> bool:
        """True when at least one engine is available and all available engines pass."""
        avail = [r for r in self.rows.values() if r.available]
        return bool(avail) and all(r.determinate_passes() for r in avail)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "style": self.style,
            "motion": self.motion,
            "duration_s": self.duration_s,
            "reference_engine": self.reference_engine,
            "tolerances": {
                "peak_tolerance": PEAK_TOLERANCE,
                "rms_tolerance": RMS_TOLERANCE,
            },
            "indeterminate_label": INDETERMINATE_LABEL,
            "citation": self.citation,
            "engines": {eng: r.to_dict() for eng, r in self.rows.items()},
        }

    def to_markdown(self) -> str:
        """Render a Markdown summary table for this style and motion mode."""
        lines = [
            f"### Style: `{self.style}` ({self.motion}, {self.duration_s:.3f} s)",
            "",
            f"- Reference engine: `{self.reference_engine}`",
            f"- Determinate tolerances: Peak <= {PEAK_TOLERANCE * 100:.1f} %, "
            f"RMS <= {RMS_TOLERANCE * 100:.1f} %",
            f"- Indeterminate citation: `{self.citation}`",
            "",
            "| Engine | Quantity | Classification | Peak Error (%) | RMS Error (%) | State / Label |",
            "| :--- | :--- | :--- | ---: | ---: | :--- |",
        ]
        for eng, row in self.rows.items():
            for q_name in STYLE_PARITY_QUANTITIES:
                q = row.quantity(q_name)
                peak_str = (
                    "NaN" if math.isnan(q.peak_error) else f"{q.peak_error * 100:.3f}"
                )
                rms_str = (
                    "NaN" if math.isnan(q.rms_error) else f"{q.rms_error * 100:.3f}"
                )
                lines.append(
                    f"| `{eng}` | `{q_name}` | `{q.classification.value}` | "
                    f"{peak_str} | {rms_str} | {q.label} |"
                )
        lines.append("")
        return "\n".join(lines)


def _build_pair_breakdown(
    cand_run: Any, ref_run: Any, keep: np.ndarray
) -> dict[str, dict[str, ClassifiedQuantityResult]]:
    """Build per-pair classified results (physical pairs and NaN hand-hand pairs)."""
    out: dict[str, dict[str, ClassifiedQuantityResult]] = {}
    ref_normal = getattr(ref_run, "pair_normal_force_n", None) or {}
    c_reasons = getattr(cand_run, "channel_reasons", None) or {}
    r_reasons = getattr(ref_run, "channel_reasons", None) or {}

    for pname, r_norm_vals in ref_normal.items():
        r_norm_arr = np.asarray(r_norm_vals, dtype=float)[keep]
        pair_res: dict[str, ClassifiedQuantityResult] = {}
        if np.all(np.isnan(r_norm_arr)):
            reason = (
                c_reasons.get(pname)
                or r_reasons.get(pname)
                or "hand-hand pairs are allocated, not physical forces under prescribed hand motion"
            )
            for metric, _ in _PAIR_METRIC_ATTRS:
                q_key = f"{metric}:{pname}"
                pair_res[metric] = ClassifiedQuantityResult.unavailable(q_key, reason)
        else:
            for metric, attr in _PAIR_METRIC_ATTRS:
                q_key = f"{metric}:{pname}"
                c_arr = np.asarray(getattr(cand_run, attr)[pname], dtype=float)[keep]
                r_arr = np.asarray(getattr(ref_run, attr)[pname], dtype=float)[keep]
                qe = _compute_quantity_error(c_arr, r_arr, q_key)
                pair_res[metric] = ClassifiedQuantityResult.from_quantity_error(
                    q_key, qe
                )
        out[pname] = pair_res
    return out


def _build_available_engine_row(
    engine: str,
    cand_run: ContactRun,
    ref_run: ContactRun,
    t_end_s: float | None,
    club: ClubDynamics | None,
    interface: GripInterface | None,
) -> EngineParityRow:
    """Build an :class:`EngineParityRow` for an available engine."""
    raw_errors = parity_errors(
        cand_run, ref_run, t_end_s=t_end_s, club=club, interface=interface
    )
    t = ref_run.series.time_s
    keep = np.ones(t.size, bool) if t_end_s is None else t <= t_end_s + 1e-12
    quantities = {
        q_name: ClassifiedQuantityResult.from_quantity_error(q_name, raw_errors[q_name])
        for q_name in STYLE_PARITY_QUANTITIES
    }
    pair_quantities = _build_pair_breakdown(cand_run, ref_run, keep)
    return EngineParityRow(
        engine=engine,
        available=True,
        quantities=quantities,
        pair_quantities=pair_quantities,
    )


def _build_unavailable_engine_row(
    engine: str,
    reason: str,
    ref_errors: Mapping[str, QuantityError],
    ref_run: ContactRun,
) -> EngineParityRow:
    """Build an :class:`EngineParityRow` with NaN values and ``reason``."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"unavailable engine {engine!r} requires a non-empty reason")
    quantities = {
        q_name: ClassifiedQuantityResult.unavailable(
            q_name,
            reason,
            peak_reference=(
                ref_errors[q_name].peak_reference
                if q_name in ref_errors
                else float("nan")
            ),
        )
        for q_name in STYLE_PARITY_QUANTITIES
    }
    ref_normal = getattr(ref_run, "pair_normal_force_n", None) or {}
    pair_quantities: dict[str, dict[str, ClassifiedQuantityResult]] = {}
    for pname in ref_normal:
        pair_quantities[pname] = {
            metric: ClassifiedQuantityResult.unavailable(f"{metric}:{pname}", reason)
            for metric in PAIR_PARITY_QUANTITIES
        }
    return EngineParityRow(
        engine=engine,
        available=False,
        quantities=quantities,
        pair_quantities=pair_quantities,
        unavailable_reason=reason,
    )


def build_style_parity_report(
    style: GripStyle | str,
    motion: str,
    duration_s: float,
    reference_run: ContactRun,
    candidate_runs: Mapping[str, ContactRun] | None = None,
    unavailable_engines: Mapping[str, str] | None = None,
    *,
    club: ClubDynamics | None = None,
    interface: GripInterface | None = None,
) -> StyleParityReport:
    """Build a cross-engine parity report for one style across Pinocchio, OpenSim, MuJoCo, and Drake.

    Preconditions:
        ``style`` is a :class:`GripStyle` or non-empty string;
        ``motion`` in ``('hold', 'swing')``;
        ``duration_s > 0`` and finite;
        ``reference_run`` is a valid :class:`ContactRun` (Pinocchio reference).
    Postconditions:
        Returns a :class:`StyleParityReport` with one :class:`EngineParityRow`
        per engine in :data:`PARITY_ENGINES`. Unavailable engines report ``NaN``
        plus their reason and never pass.
    """
    style_name = style.value if hasattr(style, "value") else str(style)
    if not style_name.strip():
        raise ValueError("style must be non-empty")
    if motion not in ("hold", "swing"):
        raise ValueError(f"motion must be 'hold' or 'swing', got {motion!r}")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive and finite, got {duration_s}")
    _, unwrapped_ref = _unwrap_series_and_run(reference_run, "reference_run")
    if unwrapped_ref is None:
        raise TypeError("reference_run must be a ContactRun")

    cand_map = dict(candidate_runs or {})
    unavail_map = dict(unavailable_engines or {})
    overlap_engines = set(cand_map) & set(unavail_map)
    if overlap_engines:
        raise ValueError(
            f"engines cannot be both available and unavailable: {sorted(overlap_engines)}"
        )

    ref_errors = parity_errors(
        reference_run,
        reference_run,
        t_end_s=duration_s,
        club=club,
        interface=interface,
    )
    rows: dict[str, EngineParityRow] = {
        "pinocchio": _build_available_engine_row(
            "pinocchio",
            reference_run,
            reference_run,
            duration_s,
            club,
            interface,
        )
    }

    ordered_engines = list(PARITY_ENGINES)
    for extra_eng in (*cand_map, *unavail_map):
        if extra_eng not in ordered_engines:
            ordered_engines.append(extra_eng)

    for eng in ordered_engines:
        if eng == "pinocchio":
            continue
        if eng in cand_map:
            rows[eng] = _build_available_engine_row(
                eng,
                cand_map[eng],
                reference_run,
                duration_s,
                club,
                interface,
            )
        elif eng in unavail_map:
            rows[eng] = _build_unavailable_engine_row(
                eng, unavail_map[eng], ref_errors, reference_run
            )
        else:
            rows[eng] = _build_unavailable_engine_row(
                eng,
                f"engine {eng!r} was not executed",
                ref_errors,
                reference_run,
            )

    return StyleParityReport(
        style=style_name,
        motion=motion,
        duration_s=float(duration_s),
        reference_engine="pinocchio",
        rows=rows,
    )
