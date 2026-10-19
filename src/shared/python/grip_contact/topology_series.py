"""Cross-model topology grip kinetics series on the shared 2 ms grid (HRG-GS-M3, #13325).

Provides :class:`TopologyKineticsSeries` (frozen), resampling onto the shared
2 ms fixture cadence (:data:`~src.shared.python.grip_contact.swing_input.FIXTURE_DT_S`),
construction from :class:`~src.shared.python.grip_contact.contact_run.ContactRun`
or :class:`~src.shared.python.grip_contact.binding.GripStepReport` sequences,
and :func:`law_level_series` for reference qualification on prescribed site
trajectories.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from src.shared.python.grip_contact.binding import (
    PAIR_LABELS,
    GripStepReport,
    HandClubBinding,
    PairLabel,
    _require_non_empty_str,
    _spin_magnitude_nm,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.swing_input import FIXTURE_DT_S
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    topology_side_wrench,
)

__all__ = [
    "CHANNEL_KINDS",
    "SAMPLE_DT_S",
    "TopologyKineticsSeries",
    "law_level_series",
    "resample_to_2ms_grid",
    "topology_mapping_hash",
    "total_applied_allocated_force_n",
]

SAMPLE_DT_S: Final[float] = FIXTURE_DT_S
SIDES: Final[tuple[str, str]] = ("L", "R")
CHANNEL_KINDS: Final[tuple[str, ...]] = (
    "pair_normal",
    "pair_tangential",
    "pair_spin",
    "hand_force",
    "hand_moment",
)
_ALLOCATED_PRESCRIBED_REASON: Final[str] = (
    "hand-hand pairs are allocated, not physical forces under prescribed hand motion"
)


def topology_mapping_hash(topology_model: TopologyContactModel) -> str:
    """Compute a deterministic SHA-256 mapping hash for ``topology_model``."""
    if not isinstance(topology_model, TopologyContactModel):
        raise TypeError(
            f"topology_model must be TopologyContactModel, got {type(topology_model).__name__}"
        )
    topo = topology_model.topology
    parts = [
        f"style={topo.style.value}",
        f"trigger={topo.trigger_finger}",
        f"grip_radius={topo.grip_radius_m:.9f}",
        f"patch_radius={topology_model.patch_radius_m:.9f}",
    ]
    for pair in topo.pairs:
        parts.append(
            f"pair:{pair.name}:{pair.site_a}:{pair.site_b}:{pair.physical_modes}"
        )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def _build_2ms_grid(time_s: np.ndarray, dt_s: float = SAMPLE_DT_S) -> np.ndarray:
    """Construct or preserve the 2 ms sample grid spanning ``time_s``."""
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or t.size < 2 or not bool(np.isfinite(t).all()):
        raise ValueError("time_s must be a 1-D finite array with >= 2 samples")
    diffs = np.diff(t)
    if not bool((diffs > 0.0).all()):
        raise ValueError("time_s must be strictly increasing (monotonic)")
    if not (math.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be finite and positive")
    if bool(np.allclose(diffs, dt_s, atol=1e-9)):
        return t.copy()
    span = float(t[-1] - t[0])
    if span < dt_s - 1e-12:
        raise ValueError(
            f"time_s span ({span:.6g} s) must be >= {dt_s:.6g} s to resample"
        )
    n_steps = int(math.floor((span + 1e-12) / dt_s))
    return float(t[0]) + np.arange(n_steps + 1, dtype=float) * dt_s


def _interp_1d(t_grid: np.ndarray, t_src: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate a 1-D array onto ``t_grid``, preserving NaN."""
    src = np.asarray(arr, dtype=float)
    if bool(np.isnan(src).any()):
        return np.full(t_grid.size, np.nan, dtype=float)
    if t_grid.shape == t_src.shape and bool(np.allclose(t_grid, t_src, atol=1e-12)):
        return src.copy()
    return np.asarray(np.interp(t_grid, t_src, src), dtype=float)


def _interp_2d(t_grid: np.ndarray, t_src: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate an ``(n, 3)`` array onto ``t_grid``, preserving NaN."""
    src = np.asarray(arr, dtype=float)
    if bool(np.isnan(src).any()):
        return np.full((t_grid.size, src.shape[1]), np.nan, dtype=float)
    if t_grid.shape == t_src.shape and bool(np.allclose(t_grid, t_src, atol=1e-12)):
        return src.copy()
    cols = [np.interp(t_grid, t_src, src[:, c]) for c in range(src.shape[1])]
    return np.column_stack(cols)


def resample_to_2ms_grid(
    time_s: np.ndarray,
    channels_1d: Mapping[str, np.ndarray],
    channels_2d: Mapping[str, np.ndarray] | None = None,
    *,
    dt_s: float = SAMPLE_DT_S,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resample 1-D and 2-D channel mappings onto the shared 2 ms grid."""
    t_src = np.asarray(time_s, dtype=float)
    t_grid = _build_2ms_grid(t_src, dt_s=dt_s)
    out_1d = {k: _interp_1d(t_grid, t_src, v) for k, v in channels_1d.items()}
    out_2d = (
        {k: _interp_2d(t_grid, t_src, v) for k, v in channels_2d.items()}
        if channels_2d is not None
        else {}
    )
    return t_grid, out_1d, out_2d


def _candidate_reason_keys(kind: str, name: str) -> tuple[str, ...]:
    """All acceptable keys in ``channel_reasons`` for ``(kind, name)``."""
    if kind in ("normal", "tangential", "spin"):
        field_prefix = (
            f"pair_{kind}_force_n" if kind != "spin" else "pair_spin_moment_nm"
        )
        return (
            f"{kind}:{name}",
            f"{name}:{kind}",
            f"{field_prefix}:{name}",
            name,
        )
    if kind == "force":
        return (
            f"force_{name}",
            f"force:{name}",
            f"force_on_club_n:{name}",
            name,
        )
    return (
        f"moment_{name}",
        f"moment:{name}",
        f"torque_{name}",
        f"torque:{name}",
        f"moment_on_club_nm:{name}",
        f"torque_on_club_nm:{name}",
        name,
    )


def _validate_1d_channels(
    mapping: Mapping[str, np.ndarray],
    field_name: str,
    n_samples: int,
    known_pairs: set[str],
) -> dict[str, np.ndarray]:
    """Validate per-pair 1-D arrays of length ``n_samples``."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{field_name} must be a Mapping[str, np.ndarray]")
    cleaned: dict[str, np.ndarray] = {}
    for pair_name, raw in mapping.items():
        if pair_name not in known_pairs:
            raise ValueError(f"{field_name} key {pair_name!r} is not present in labels")
        arr = np.asarray(raw, dtype=float)
        if arr.shape != (n_samples,):
            raise ValueError(
                f"{field_name}[{pair_name!r}] must have shape ({n_samples},), "
                f"got {arr.shape}"
            )
        if bool(np.isinf(arr).any()):
            raise ValueError(f"{field_name}[{pair_name!r}] must not contain +/-inf")
        copy_arr = arr.copy()
        copy_arr.setflags(write=False)
        cleaned[pair_name] = copy_arr
    return cleaned


def _validate_2d_channels(
    mapping: Mapping[str, np.ndarray],
    field_name: str,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Validate per-hand ``(n_samples, 3)`` arrays for sides ``('L', 'R')``."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{field_name} must be a Mapping[str, np.ndarray]")
    cleaned: dict[str, np.ndarray] = {}
    for side, raw in mapping.items():
        if side not in SIDES:
            raise ValueError(f"{field_name} side must be in {SIDES}, got {side!r}")
        arr = np.asarray(raw, dtype=float)
        if arr.shape != (n_samples, 3):
            raise ValueError(
                f"{field_name}[{side!r}] must have shape ({n_samples}, 3), "
                f"got {arr.shape}"
            )
        if bool(np.isinf(arr).any()):
            raise ValueError(f"{field_name}[{side!r}] must not contain +/-inf")
        copy_arr = arr.copy()
        copy_arr.setflags(write=False)
        cleaned[side] = copy_arr
    return cleaned


def _validate_reasons_and_nans(
    normal: Mapping[str, np.ndarray],
    tangential: Mapping[str, np.ndarray],
    spin: Mapping[str, np.ndarray],
    force: Mapping[str, np.ndarray],
    moment: Mapping[str, np.ndarray],
    reasons: Mapping[str, str],
) -> dict[str, str]:
    """Enforce that every NaN channel has a reason and every reason targets a NaN channel."""
    if not isinstance(reasons, Mapping):
        raise TypeError("channel_reasons must be a Mapping[str, str]")
    clean_reasons: dict[str, str] = {}
    for k, v in reasons.items():
        key = _require_non_empty_str(k, "channel_reasons key")
        val = _require_non_empty_str(v, f"channel_reasons[{k!r}]")
        clean_reasons[key] = val

    used_reason_keys: set[str] = set()
    channel_groups: list[tuple[str, Mapping[str, np.ndarray]]] = [
        ("normal", normal),
        ("tangential", tangential),
        ("spin", spin),
        ("force", force),
        ("moment", moment),
    ]
    for kind, group in channel_groups:
        for item_name, arr in group.items():
            has_nan = bool(np.isnan(arr).any())
            candidates = _candidate_reason_keys(kind, item_name)
            matched = [ck for ck in candidates if ck in clean_reasons]
            if has_nan and not matched:
                raise ValueError(
                    f"NaN channel '{kind}:{item_name}' must have a reason in channel_reasons"
                )
            if has_nan:
                used_reason_keys.update(matched)

    unused = set(clean_reasons) - used_reason_keys
    if unused:
        raise ValueError(
            "channel_reasons may only be attached to channels that are NaN somewhere; "
            f"finite or unknown channels with reasons: {sorted(unused)}"
        )
    return clean_reasons


@dataclass(frozen=True)
class TopologyKineticsSeries:
    """Engine-agnostic topology grip kinetics series on the shared 2 ms grid.

    Preconditions:
        - ``time_s`` is strictly increasing (monotonic) with >= 2 samples.
        - All per-pair and per-hand arrays have length matching ``time_s.size``.
        - Every channel containing ``NaN`` has a non-empty reason in
          ``channel_reasons``, and reasons are only attached to channels that
          are ``NaN`` somewhere.
        - Every label in ``labels`` is ``"physical"`` or ``"allocated"``.
    """

    engine: str
    model_id: str
    mapping_hash: str
    time_s: np.ndarray
    pair_normal_force_n: Mapping[str, np.ndarray]
    pair_tangential_force_n: Mapping[str, np.ndarray]
    pair_spin_moment_nm: Mapping[str, np.ndarray]
    force_on_club_n: Mapping[str, np.ndarray]
    moment_on_club_nm: Mapping[str, np.ndarray]
    labels: Mapping[str, PairLabel]
    channel_reasons: Mapping[str, str] = field(default_factory=dict)
    applied_allocated_force_n: float = 0.0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.engine, "engine")
        _require_non_empty_str(self.model_id, "model_id")
        _require_non_empty_str(self.mapping_hash, "mapping_hash")

        t = np.asarray(self.time_s, dtype=float)
        if t.ndim != 1 or t.size < 2 or not bool(np.isfinite(t).all()):
            raise ValueError("time_s must be a 1-D finite array with >= 2 samples")
        if not bool((np.diff(t) > 0.0).all()):
            raise ValueError("time_s must be strictly increasing (monotonic)")
        t_clean = t.copy()
        t_clean.setflags(write=False)
        object.__setattr__(self, "time_s", t_clean)
        n = t_clean.size

        if not isinstance(self.labels, Mapping) or not self.labels:
            raise ValueError("labels must be a non-empty Mapping[str, PairLabel]")
        clean_labels: dict[str, PairLabel] = {}
        for p_name, lbl in self.labels.items():
            key = _require_non_empty_str(p_name, "labels key")
            if lbl not in PAIR_LABELS:
                raise ValueError(
                    f"label for {key!r} must be one of {PAIR_LABELS}, got {lbl!r}"
                )
            clean_labels[key] = lbl
        known_pairs = set(clean_labels)
        object.__setattr__(self, "labels", MappingProxyType(clean_labels))

        norm_c = _validate_1d_channels(
            self.pair_normal_force_n, "pair_normal_force_n", n, known_pairs
        )
        tang_c = _validate_1d_channels(
            self.pair_tangential_force_n, "pair_tangential_force_n", n, known_pairs
        )
        spin_c = _validate_1d_channels(
            self.pair_spin_moment_nm, "pair_spin_moment_nm", n, known_pairs
        )
        force_c = _validate_2d_channels(self.force_on_club_n, "force_on_club_n", n)
        moment_c = _validate_2d_channels(self.moment_on_club_nm, "moment_on_club_nm", n)

        clean_reasons = _validate_reasons_and_nans(
            norm_c, tang_c, spin_c, force_c, moment_c, self.channel_reasons
        )

        if (
            not math.isfinite(self.applied_allocated_force_n)
            or self.applied_allocated_force_n != 0.0
        ):
            raise ValueError(
                "applied_allocated_force_n must be 0.0 in any physical series, "
                f"got {self.applied_allocated_force_n}"
            )

        object.__setattr__(self, "pair_normal_force_n", MappingProxyType(norm_c))
        object.__setattr__(self, "pair_tangential_force_n", MappingProxyType(tang_c))
        object.__setattr__(self, "pair_spin_moment_nm", MappingProxyType(spin_c))
        object.__setattr__(self, "force_on_club_n", MappingProxyType(force_c))
        object.__setattr__(self, "moment_on_club_nm", MappingProxyType(moment_c))
        object.__setattr__(self, "channel_reasons", MappingProxyType(clean_reasons))

    @property
    def normal_force_n(self) -> Mapping[str, np.ndarray]:
        """Alias for :attr:`pair_normal_force_n`."""
        return self.pair_normal_force_n

    @property
    def tangential_force_n(self) -> Mapping[str, np.ndarray]:
        """Alias for :attr:`pair_tangential_force_n`."""
        return self.pair_tangential_force_n

    @property
    def spin_moment_nm(self) -> Mapping[str, np.ndarray]:
        """Alias for :attr:`pair_spin_moment_nm`."""
        return self.pair_spin_moment_nm

    @property
    def torque_on_club_nm(self) -> Mapping[str, np.ndarray]:
        """Alias for :attr:`moment_on_club_nm`."""
        return self.moment_on_club_nm

    @property
    def pair_labels(self) -> Mapping[str, PairLabel]:
        """Alias for :attr:`labels`."""
        return self.labels

    def reason_for(self, kind: str, name: str) -> str | None:
        """Return the unavailability reason for ``(kind, name)``, if any."""
        for key in _candidate_reason_keys(kind, name):
            if key in self.channel_reasons:
                return self.channel_reasons[key]
        return None

    @classmethod
    def from_contact_run(
        cls,
        run: ContactRun,
        topology_model: TopologyContactModel,
        *,
        engine: str | None = None,
        mapping_hash: str | None = None,
        model_id: str | None = None,
    ) -> TopologyKineticsSeries:
        """Convert a :class:`ContactRun` into a 2 ms :class:`TopologyKineticsSeries`."""
        if not isinstance(run, ContactRun):
            raise TypeError(f"run must be ContactRun, got {type(run).__name__}")
        if not isinstance(topology_model, TopologyContactModel):
            raise TypeError(
                f"topology_model must be TopologyContactModel, got {type(topology_model).__name__}"
            )
        if run.pair_normal_force_n is None or run.pair_tangential_force_n is None:
            raise ValueError(
                "ContactRun must have pair_normal_force_n and pair_tangential_force_n"
            )

        base_series = run.series
        t_src = np.asarray(base_series.time_s, dtype=float)
        resolved_engine = engine if engine is not None else base_series.engine
        resolved_hash = (
            mapping_hash
            if mapping_hash is not None
            else topology_mapping_hash(topology_model)
        )
        resolved_model_id = model_id if model_id is not None else resolved_engine

        topo = topology_model.topology
        labels: dict[str, PairLabel] = {
            p.name: ("physical" if p.is_hand_grip else "allocated") for p in topo.pairs
        }
        reasons = dict(run.channel_reasons or {})
        spin_src = _extract_spin_from_run(run, topology_model, t_src.size, reasons)

        t_grid = _build_2ms_grid(t_src, dt_s=SAMPLE_DT_S)
        norm_res = {
            k: _interp_1d(t_grid, t_src, v) for k, v in run.pair_normal_force_n.items()
        }
        tang_res = {
            k: _interp_1d(t_grid, t_src, v)
            for k, v in run.pair_tangential_force_n.items()
        }
        spin_res = {k: _interp_1d(t_grid, t_src, v) for k, v in spin_src.items()}
        force_res = {
            s: _interp_2d(t_grid, t_src, base_series.force_on_club_n[s]) for s in SIDES
        }
        moment_res = {
            s: _interp_2d(t_grid, t_src, base_series.torque_on_club_nm[s])
            for s in SIDES
        }

        return cls(
            engine=resolved_engine,
            model_id=resolved_model_id,
            mapping_hash=resolved_hash,
            time_s=t_grid,
            pair_normal_force_n=norm_res,
            pair_tangential_force_n=tang_res,
            pair_spin_moment_nm=spin_res,
            force_on_club_n=force_res,
            moment_on_club_nm=moment_res,
            labels=labels,
            channel_reasons=reasons,
        )

    @classmethod
    def from_step_reports(
        cls,
        times: Sequence[float] | np.ndarray,
        reports: Sequence[GripStepReport],
        *,
        engine: str = "step_reports",
        mapping_hash: str | None = None,
        model_id: str | None = None,
        binding: HandClubBinding | None = None,
    ) -> TopologyKineticsSeries:
        """Assemble a 2 ms :class:`TopologyKineticsSeries` from step reports."""
        t_src = np.asarray(times, dtype=float)
        if not isinstance(reports, Sequence) or len(reports) != t_src.size:
            raise ValueError("reports must be a Sequence with the same length as times")
        if len(reports) < 2:
            raise ValueError("at least 2 GripStepReport items are required")
        for r in reports:
            if not isinstance(r, GripStepReport):
                raise TypeError(
                    f"reports must contain GripStepReport, got {type(r).__name__}"
                )

        alloc_applied = total_applied_allocated_force_n(reports)
        if alloc_applied != 0.0:
            raise ValueError(
                f"reports contain non-zero applied allocated force: {alloc_applied} N"
            )

        resolved_hash = (
            mapping_hash
            if mapping_hash is not None
            else (binding.model_hash if binding is not None else "step_reports")
        )
        resolved_model = (
            model_id
            if model_id is not None
            else (binding.model_id if binding is not None else engine)
        )

        first = reports[0]
        labels: dict[str, PairLabel] = dict(first.pair_labels)
        pair_names = list(labels)

        norm_src = {
            p: np.array([r.normal_force_n[p] for r in reports], dtype=float)
            for p in pair_names
        }
        tang_src = {
            p: np.array([r.tangential_force_n[p] for r in reports], dtype=float)
            for p in pair_names
        }
        spin_src = {
            p: np.array([r.spin_moment_nm[p] for r in reports], dtype=float)
            for p in pair_names
        }
        force_src, moment_src = _extract_side_wrenches_from_reports(reports)
        reasons = _collect_report_reasons(reports)

        t_grid = _build_2ms_grid(t_src, dt_s=SAMPLE_DT_S)
        return cls(
            engine=engine,
            model_id=resolved_model,
            mapping_hash=resolved_hash,
            time_s=t_grid,
            pair_normal_force_n={
                p: _interp_1d(t_grid, t_src, norm_src[p]) for p in pair_names
            },
            pair_tangential_force_n={
                p: _interp_1d(t_grid, t_src, tang_src[p]) for p in pair_names
            },
            pair_spin_moment_nm={
                p: _interp_1d(t_grid, t_src, spin_src[p]) for p in pair_names
            },
            force_on_club_n={s: _interp_2d(t_grid, t_src, force_src[s]) for s in SIDES},
            moment_on_club_nm={
                s: _interp_2d(t_grid, t_src, moment_src[s]) for s in SIDES
            },
            labels=labels,
            channel_reasons=reasons,
            applied_allocated_force_n=alloc_applied,
        )


def _extract_spin_from_run(
    run: ContactRun,
    topology_model: TopologyContactModel,
    n_samples: int,
    reasons: dict[str, str],
) -> dict[str, np.ndarray]:
    """Compute or extract per-pair spin moment arrays from ``run``."""
    topo = topology_model.topology
    spin_src: dict[str, np.ndarray] = {
        p.name: np.full(n_samples, np.nan, dtype=float) for p in topo.pairs
    }
    if run.hand_states is not None and run.club_states is not None:
        for p in topo.hand_grip_pairs():
            spin_src[p.name] = np.zeros(n_samples, dtype=float)
        for k in range(n_samples):
            for s in SIDES:
                h_state = run.hand_states[s][k]
                c_state = run.club_states[s][k]
                sw = topology_side_wrench(topology_model, s, h_state, c_state)
                for pname, pw in sw.pair_wrenches.items():
                    spin_src[pname][k] = _spin_magnitude_nm(pw)
        for p in topo.hand_hand_pairs():
            if p.name not in reasons:
                reasons[p.name] = _ALLOCATED_PRESCRIBED_REASON
    else:
        for p in topo.pairs:
            if p.name not in reasons and f"spin:{p.name}" not in reasons:
                reasons[f"spin:{p.name}"] = (
                    "ContactRun does not include frame states to evaluate spin moment"
                )
    return spin_src


def _extract_side_wrenches_from_reports(
    reports: Sequence[GripStepReport],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract per-hand resultant force and moment arrays from ``reports``."""
    n = len(reports)
    force_src = {s: np.zeros((n, 3), dtype=float) for s in SIDES}
    moment_src = {s: np.zeros((n, 3), dtype=float) for s in SIDES}
    for k, r in enumerate(reports):
        sw_map = r.side_wrenches
        for s in SIDES:
            if s in sw_map:
                force_src[s][k] = sw_map[s].force_n
                moment_src[s][k] = sw_map[s].moment_nm
    return force_src, moment_src


def _collect_report_reasons(reports: Sequence[GripStepReport]) -> dict[str, str]:
    """Merge ``unavailable_reasons`` across a sequence of ``GripStepReport``."""
    reasons: dict[str, str] = {}
    for r in reports:
        for k, v in r.unavailable_reasons.items():
            if k not in reasons:
                reasons[k] = v
    return reasons


def total_applied_allocated_force_n(reports: Sequence[GripStepReport]) -> float:
    """Sum of :meth:`GripStepReport.applied_allocated_force_n` over ``reports``."""
    if not isinstance(reports, Sequence) or not reports:
        raise ValueError("reports must be a non-empty Sequence of GripStepReport")
    total = 0.0
    for r in reports:
        if not isinstance(r, GripStepReport):
            raise TypeError(
                f"reports must contain GripStepReport, got {type(r).__name__}"
            )
        total += r.applied_allocated_force_n()
    return total


def _normalize_frame_sequence(
    seq: Mapping[str, Sequence[BushingState]] | Sequence[Mapping[str, BushingState]],
    n_samples: int,
    name: str,
) -> dict[str, list[BushingState]]:
    """Normalize either ``{side: [BushingState, ...]}`` or ``[{side: BushingState}, ...]``."""
    if isinstance(seq, Mapping):
        if set(seq) != set(SIDES):
            raise ValueError(f"{name} must have keys {SIDES}")
        out: dict[str, list[BushingState]] = {}
        for s in SIDES:
            items = list(seq[s])
            if len(items) != n_samples:
                raise ValueError(
                    f"{name}[{s!r}] length {len(items)} != times length {n_samples}"
                )
            out[s] = items
        return out
    if isinstance(seq, Sequence):
        if len(seq) != n_samples:
            raise ValueError(f"{name} length {len(seq)} != times length {n_samples}")
        return {s: [step[s] for step in seq] for s in SIDES}
    raise TypeError(f"{name} must be a Mapping or Sequence")


def law_level_series(
    topology_model: TopologyContactModel,
    hand_frames_seq: (
        Mapping[str, Sequence[BushingState]] | Sequence[Mapping[str, BushingState]]
    ),
    club_frames_seq: (
        Mapping[str, Sequence[BushingState]] | Sequence[Mapping[str, BushingState]]
    ),
    times: Sequence[float] | np.ndarray,
    *,
    engine: str = "law_reference",
    mapping_hash: str | None = None,
    model_id: str = "topology_law",
    mode: str = "contact",
) -> TopologyKineticsSeries:
    """Evaluate the shared law (:func:`topology_side_wrench`) on prescribed site trajectories."""
    if not isinstance(topology_model, TopologyContactModel):
        raise TypeError(
            f"topology_model must be TopologyContactModel, got {type(topology_model).__name__}"
        )
    t_src = np.asarray(times, dtype=float)
    t_grid = _build_2ms_grid(t_src, dt_s=SAMPLE_DT_S)
    n = t_src.size

    hands = _normalize_frame_sequence(hand_frames_seq, n, "hand_frames_seq")
    clubs = _normalize_frame_sequence(club_frames_seq, n, "club_frames_seq")

    topo = topology_model.topology
    labels: dict[str, PairLabel] = {}
    pair_normal: dict[str, np.ndarray] = {}
    pair_tangential: dict[str, np.ndarray] = {}
    pair_spin: dict[str, np.ndarray] = {}
    reasons: dict[str, str] = {}

    for p in topo.pairs:
        is_phys = p.is_hand_grip and mode in p.physical_modes
        labels[p.name] = "physical" if is_phys else "allocated"
        if is_phys:
            pair_normal[p.name] = np.zeros(n, dtype=float)
            pair_tangential[p.name] = np.zeros(n, dtype=float)
            pair_spin[p.name] = np.zeros(n, dtype=float)
        else:
            pair_normal[p.name] = np.full(n, np.nan, dtype=float)
            pair_tangential[p.name] = np.full(n, np.nan, dtype=float)
            pair_spin[p.name] = np.full(n, np.nan, dtype=float)
            reasons[p.name] = _ALLOCATED_PRESCRIBED_REASON

    force_src = {s: np.zeros((n, 3), dtype=float) for s in SIDES}
    moment_src = {s: np.zeros((n, 3), dtype=float) for s in SIDES}

    for k in range(n):
        for s in SIDES:
            sw = topology_side_wrench(
                topology_model, s, hands[s][k], clubs[s][k], mode=mode
            )
            force_src[s][k] = sw.force_n
            moment_src[s][k] = sw.moment_nm
            for pname, pw in sw.pair_wrenches.items():
                pair_normal[pname][k] = float(pw.normal_force_mag_n)
                pair_tangential[pname][k] = float(pw.tangential_force_mag_n)
                pair_spin[pname][k] = _spin_magnitude_nm(pw)

    resolved_hash = (
        mapping_hash
        if mapping_hash is not None
        else topology_mapping_hash(topology_model)
    )
    return TopologyKineticsSeries(
        engine=engine,
        model_id=model_id,
        mapping_hash=resolved_hash,
        time_s=t_grid,
        pair_normal_force_n={
            p: _interp_1d(t_grid, t_src, arr) for p, arr in pair_normal.items()
        },
        pair_tangential_force_n={
            p: _interp_1d(t_grid, t_src, arr) for p, arr in pair_tangential.items()
        },
        pair_spin_moment_nm={
            p: _interp_1d(t_grid, t_src, arr) for p, arr in pair_spin.items()
        },
        force_on_club_n={s: _interp_2d(t_grid, t_src, force_src[s]) for s in SIDES},
        moment_on_club_nm={s: _interp_2d(t_grid, t_src, moment_src[s]) for s in SIDES},
        labels=labels,
        channel_reasons=reasons,
    )
