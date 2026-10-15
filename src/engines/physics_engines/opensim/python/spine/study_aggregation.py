"""Aggregate spine study receipts into per-level load tables (SPN-13c, #12768).

Input is a set of SPN-13a receipts and their hashed ``<id>_loads.npz``
artifacts.  Output is a deterministic table of per-level peaks (signed value of
largest magnitude, with time), trapezoidal impulse over a window, an optional
body-weight-normalised peak compression and the context-band annotations the
receipt carries, plus group statistics across runs.

Fail-closed rules:

* an artifact is read only after its SHA-256 matches the receipt;
* only runs with receipt status ``pass`` enter the qualified aggregate;
  ``unqualified`` and ``failed`` runs are listed in ``excluded`` with their
  status and reasons, never averaged in;
* observables (``generalized_effort``, ``joint_reaction``) are never merged,
  and groups are keyed by ``distribution_source`` and law id as well;
* context comparisons must carry ``is_acceptance`` false; ``scientific_status``
  of every table is ``unqualified``.

Load sign contract (ADR-0055): compression = ``+F_y`` of the force on the
superior vertebra by the inferior one, in the inferior ISB frame; AP shear
``F_x`` (anterior +), ML shear ``F_z`` (right +).  Moments are ``M_x`` (lateral
bending), ``M_y`` (axial) and ``M_z`` (flexion/extension).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    SpineStudyReceipt,
    load_receipt,
    round_sig,
)
from src.shared.python.contracts import ensure
from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.spine.levels import SpinalLevel

__all__ = [
    "AggregationIntegrityError",
    "ExcludedRun",
    "GroupStat",
    "LevelRow",
    "LevelSummaryTable",
    "QuantityStat",
    "VariantComparison",
    "VariantDelta",
    "VerifiedRun",
    "aggregate_receipts",
    "compare",
    "load_verified_run",
    "summarize_run",
    "write_csv",
    "write_json",
]

GENERALIZED_EFFORT = "generalized_effort"
JOINT_REACTION = "joint_reaction"
_LOADS_ROLE = "loads_npz"
_NPZ_KEYS = (
    "time_s",
    "levels",
    "id_time_s",
    "id_moment_nm",
    "joint_reaction_force_n",
    "joint_reaction_moment_nm",
)
_CHUNK = 1 << 20
_LEVEL_ORDER = {level.name: i for i, level in enumerate(SpinalLevel)}
_MOMENT_AXES = (("moment_x", 0), ("moment_y", 1), ("moment_z", 2))
_FORCE_AXES = (("ap_shear", 0), ("compression", 1), ("ml_shear", 2))
#: Fixed CSV quantity order with units of the value and of its impulse.
_QUANTITY_UNITS = (
    ("compression", "N", "N*s"),
    ("ap_shear", "N", "N*s"),
    ("ml_shear", "N", "N*s"),
    ("moment_x", "N*m", "N*m*s"),
    ("moment_y", "N*m", "N*m*s"),
    ("moment_z", "N*m", "N*m*s"),
)


class AggregationIntegrityError(ValueError):
    """An artifact or record failed an integrity check (fail closed)."""


@dataclass(frozen=True)
class VerifiedRun:
    """A receipt with its hash-verified load arrays (read-only)."""

    receipt: SpineStudyReceipt
    levels: tuple[SpinalLevel, ...]
    time_s: np.ndarray
    id_time_s: np.ndarray
    id_moment_nm: np.ndarray
    joint_reaction_force_n: np.ndarray
    joint_reaction_moment_nm: np.ndarray


@dataclass(frozen=True)
class QuantityStat:
    """Peak (signed, largest magnitude), its time and the window impulse."""

    name: str
    units: str
    peak: float
    t_peak_s: float
    impulse: float


@dataclass(frozen=True)
class LevelRow:
    """One run, observable and level; see :func:`summarize_run`."""

    study_id: str
    observable: str
    level: str
    distribution_source: str
    law: str
    window_s: tuple[float, float]
    stats: tuple[QuantityStat, ...]
    peak_compression_bw: float | None
    context: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExcludedRun:
    """A run kept out of the qualified aggregate, with the reason."""

    study_id: str
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GroupStat:
    """Across-run statistics of one (observable, level, quantity, source, law)."""

    observable: str
    level: str
    quantity: str
    units: str
    distribution_source: str
    law: str
    n_runs: int
    min_peak: float
    median_peak: float
    max_peak: float
    mean_impulse: float


@dataclass(frozen=True)
class LevelSummaryTable:
    """Rows, group statistics and the explicit exclusions of one aggregate."""

    rows: tuple[LevelRow, ...]
    groups: tuple[GroupStat, ...]
    excluded: tuple[ExcludedRun, ...]
    scientific_status: str = "unqualified"

    @property
    def n_included(self) -> int:
        """Number of distinct qualified runs."""
        return len({row.study_id for row in self.rows})

    @property
    def n_excluded(self) -> int:
        """Number of excluded runs."""
        return len(self.excluded)


@dataclass(frozen=True)
class VariantDelta:
    """Median-peak difference of a variant against the baseline."""

    variant: str
    observable: str
    level: str
    quantity: str
    units: str
    baseline_source: str
    variant_source: str
    baseline_law: str
    variant_law: str
    baseline_peak: float
    variant_peak: float
    delta: float
    ratio: float | None


@dataclass(frozen=True)
class VariantComparison:
    """Deltas plus the keys that could not be compared, with reasons."""

    deltas: tuple[VariantDelta, ...]
    unmatched: tuple[tuple[str, str, str, str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loads_record(receipt: SpineStudyReceipt) -> Any:
    found = [a for a in receipt.artifacts if a.role == _LOADS_ROLE]
    if len(found) != 1:
        raise AggregationIntegrityError(
            f"{receipt.study_id}: expected one {_LOADS_ROLE!r} artifact, "
            f"found {len(found)}"
        )
    return found[0]


def _read_arrays(path: Path, study_id: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in _NPZ_KEYS if k not in data.files]
            if missing:
                raise AggregationIntegrityError(
                    f"{study_id}: artifact lacks arrays {missing}"
                )
            arrays = {k: np.array(data[k]) for k in _NPZ_KEYS}
    except (OSError, ValueError, EOFError) as exc:
        if isinstance(exc, AggregationIntegrityError):
            raise
        raise AggregationIntegrityError(
            f"{study_id}: cannot read artifact: {exc}"
        ) from exc
    return arrays


def _check_shapes(arrays: dict[str, np.ndarray], n_levels: int, study_id: str) -> None:
    n, m = arrays["time_s"].size, arrays["id_time_s"].size
    expect = {
        "joint_reaction_force_n": (n, n_levels, 3),
        "joint_reaction_moment_nm": (n, n_levels, 3),
        "id_moment_nm": (m, n_levels, 3),
    }
    for name, shape in expect.items():
        if arrays[name].shape != shape:
            raise AggregationIntegrityError(
                f"{study_id}: {name} has shape {arrays[name].shape}, expected {shape}"
            )
    for name in ("time_s", "id_time_s"):
        series = arrays[name]
        if series.ndim != 1 or not np.all(np.isfinite(series)):
            raise AggregationIntegrityError(f"{study_id}: {name} must be finite 1-D")
        if series.size < 2 or np.any(np.diff(series) <= 0):
            raise AggregationIntegrityError(
                f"{study_id}: {name} must be strictly increasing, >= 2 samples"
            )
    for name in ("joint_reaction_force_n", "joint_reaction_moment_nm", "id_moment_nm"):
        if not np.all(np.isfinite(arrays[name])):
            raise AggregationIntegrityError(f"{study_id}: {name} is not finite")


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def load_verified_run(
    receipt_path: str | Path, artifact_dir: str | Path | None = None
) -> VerifiedRun:
    """Load a receipt and its loads artifact after verifying the SHA-256.

    ``artifact_dir`` defaults to the receipt's folder (receipts hold basenames).

    Raises:
        AggregationIntegrityError: artifact missing, hash mismatch, unreadable
            or inconsistent arrays.
        ValueError: the receipt itself is malformed.

    Postcondition: arrays are finite and read-only.
    """
    receipt_file = Path(receipt_path)
    receipt = load_receipt(receipt_file)
    folder = Path(artifact_dir) if artifact_dir is not None else receipt_file.parent
    record = _loads_record(receipt)
    path = folder / record.name
    if not path.is_file():
        raise AggregationIntegrityError(
            f"{receipt.study_id}: artifact {record.name!r} is missing"
        )
    actual = _sha256(path)
    if actual != record.sha256:
        raise AggregationIntegrityError(
            f"{receipt.study_id}: sha256 of {record.name!r} does not match the "
            f"receipt ({actual} != {record.sha256})"
        )
    arrays = _read_arrays(path, receipt.study_id)
    try:
        levels = tuple(SpinalLevel[str(name)] for name in arrays["levels"])
    except KeyError as exc:
        raise AggregationIntegrityError(
            f"{receipt.study_id}: unknown level {exc.args[0]!r}"
        ) from exc
    _check_shapes(arrays, len(levels), receipt.study_id)
    run = VerifiedRun(
        receipt=receipt,
        levels=levels,
        time_s=_readonly(arrays["time_s"]),
        id_time_s=_readonly(arrays["id_time_s"]),
        id_moment_nm=_readonly(arrays["id_moment_nm"]),
        joint_reaction_force_n=_readonly(arrays["joint_reaction_force_n"]),
        joint_reaction_moment_nm=_readonly(arrays["joint_reaction_moment_nm"]),
    )
    ensure(not run.time_s.flags.writeable, "run arrays must be read-only", run)
    return run


def _check_window(
    window: tuple[float, float] | None, time_s: np.ndarray
) -> tuple[float, float]:
    if window is None:
        return float(time_s[0]), float(time_s[-1])
    lo, hi = float(window[0]), float(window[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ValueError(f"window_s must be finite with start < end, got {window!r}")
    if lo < time_s[0] or hi > time_s[-1]:
        raise ValueError(
            f"window_s {window!r} lies outside the record [{time_s[0]}, {time_s[-1]}] s"
        )
    return lo, hi


def _impulse(
    time_s: np.ndarray, values: np.ndarray, window: tuple[float, float]
) -> float:
    inside = (time_s > window[0]) & (time_s < window[1])
    grid = np.concatenate(([window[0]], time_s[inside], [window[1]]))
    edge = np.interp(grid, time_s, values)
    return float(np.sum(0.5 * (edge[1:] + edge[:-1]) * np.diff(grid)))


def _stat(
    name: str,
    units: str,
    time_s: np.ndarray,
    values: np.ndarray,
    window: tuple[float, float],
) -> QuantityStat:
    index = int(np.argmax(np.abs(values)))
    return QuantityStat(
        name=name,
        units=units,
        peak=float(values[index]),
        t_peak_s=float(time_s[index]),
        impulse=_impulse(time_s, values, window),
    )


def _context_for(run: VerifiedRun, level: str) -> tuple[dict[str, Any], ...]:
    items = []
    for item in run.receipt.context_comparisons:
        if item.get("is_acceptance") is not False:
            raise AggregationIntegrityError(
                f"{run.receipt.study_id}: context comparison "
                f"{item.get('band_id')!r} does not carry is_acceptance false"
            )
        if item.get("level") == level:
            items.append(dict(item))
    return tuple(sorted(items, key=lambda d: (str(d.get("band_id")), str(d))))


def _row(
    run: VerifiedRun,
    observable: str,
    level: str,
    window: tuple[float, float],
    stats: list[QuantityStat],
    bw: float | None,
) -> LevelRow:
    receipt = run.receipt
    return LevelRow(
        study_id=receipt.study_id,
        observable=observable,
        level=level,
        distribution_source=receipt.distribution_source,
        law=receipt.law,
        window_s=window,
        stats=tuple(sorted(stats, key=lambda s: s.name)),
        peak_compression_bw=bw,
        context=_context_for(run, level) if observable == JOINT_REACTION else (),
    )


def _check_body_mass(body_mass_kg: float | None) -> float | None:
    if body_mass_kg is None:
        return None
    if isinstance(body_mass_kg, bool) or not isinstance(body_mass_kg, int | float):
        raise TypeError("body_mass_kg must be a real number or None")
    if not math.isfinite(body_mass_kg) or body_mass_kg <= 0:
        raise ValueError("body_mass_kg must be finite and positive")
    return float(body_mass_kg) * GRAVITY_FLOAT


def summarize_run(
    run: VerifiedRun,
    window_s: tuple[float, float] | None = None,
    body_mass_kg: float | None = None,
) -> list[LevelRow]:
    """Per-level rows for the two observables stored in the artifact.

    ``window_s`` (for example the downswing) bounds the impulse only; peaks
    use the whole record.  ``None`` means the full record.  A passive-element
    observable is not stored by SPN-13a and is therefore not reported.

    Raises:
        TypeError / ValueError: bad window or body mass.
        AggregationIntegrityError: a context comparison claims acceptance.

    Postcondition: rows are ordered observable, then caudal-to-cranial level.
    """
    weight_n = _check_body_mass(body_mass_kg)
    rows: list[LevelRow] = []
    for k, level in enumerate(run.levels):
        name = level.name
        window = _check_window(window_s, run.id_time_s)
        moments = run.id_moment_nm[:, k, :]
        stats = [
            _stat(q, "N*m", run.id_time_s, moments[:, i], window)
            for q, i in _MOMENT_AXES
        ]
        rows.append(_row(run, GENERALIZED_EFFORT, name, window, stats, None))
        window = _check_window(window_s, run.time_s)
        force = run.joint_reaction_force_n[:, k, :]
        moment = run.joint_reaction_moment_nm[:, k, :]
        stats = [_stat(q, "N", run.time_s, force[:, i], window) for q, i in _FORCE_AXES]
        stats += [
            _stat(q, "N*m", run.time_s, moment[:, i], window) for q, i in _MOMENT_AXES
        ]
        bw = None
        if weight_n is not None:
            comp = next(s for s in stats if s.name == "compression")
            bw = comp.peak / weight_n
        rows.append(_row(run, JOINT_REACTION, name, window, stats, bw))
    rows.sort(key=lambda r: (r.observable, _LEVEL_ORDER[r.level]))
    return rows


def _group_stats(rows: Iterable[LevelRow]) -> tuple[GroupStat, ...]:
    buckets: dict[tuple[str, str, str, str, str, str], list[QuantityStat]] = {}
    for row in rows:
        for stat in row.stats:
            key = (
                row.observable,
                row.level,
                stat.name,
                stat.units,
                row.distribution_source,
                row.law,
            )
            buckets.setdefault(key, []).append(stat)
    groups = []
    for key, items in buckets.items():
        peaks = np.sort([s.peak for s in items])
        groups.append(
            GroupStat(
                observable=key[0],
                level=key[1],
                quantity=key[2],
                units=key[3],
                distribution_source=key[4],
                law=key[5],
                n_runs=len(items),
                min_peak=float(peaks[0]),
                median_peak=float(np.median(peaks)),
                max_peak=float(peaks[-1]),
                mean_impulse=float(np.mean([s.impulse for s in items])),
            )
        )
    groups.sort(
        key=lambda g: (
            g.observable,
            _LEVEL_ORDER[g.level],
            g.quantity,
            g.distribution_source,
            g.law,
        )
    )
    return tuple(groups)


def aggregate_receipts(
    receipt_paths: Sequence[str | Path],
    window_s: tuple[float, float] | None = None,
    body_mass_kg: float | None = None,
) -> LevelSummaryTable:
    """Aggregate the ``pass`` runs among ``receipt_paths``.

    Each receipt's artifacts are looked up beside it and hash-verified; any
    integrity failure of a ``pass`` run aborts the whole aggregate.
    ``unqualified`` and ``failed`` runs are reported in ``excluded`` and never
    enter rows or group statistics.  Input order is irrelevant.

    Raises:
        ValueError: no receipts, or a duplicated ``study_id``.
        AggregationIntegrityError: see :func:`load_verified_run`.
    """
    paths = [Path(p) for p in receipt_paths]
    if not paths:
        raise ValueError("aggregate_receipts needs at least one receipt")
    receipts = sorted(
        ((load_receipt(p), p) for p in paths), key=lambda t: t[0].study_id
    )
    ids = [r.study_id for r, _ in receipts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate study_id among receipts: {sorted(ids)}")
    rows: list[LevelRow] = []
    excluded: list[ExcludedRun] = []
    for receipt, path in receipts:
        if receipt.status != "pass":
            excluded.append(
                ExcludedRun(receipt.study_id, receipt.status, receipt.status_reasons)
            )
            continue
        run = load_verified_run(path)
        rows.extend(summarize_run(run, window_s, body_mass_kg))
    table = LevelSummaryTable(
        rows=tuple(rows), groups=_group_stats(rows), excluded=tuple(excluded)
    )
    ensure(
        table.n_included + table.n_excluded == len(paths),
        "every receipt is either included or excluded",
        table,
    )
    return table


def _by_key(
    table: LevelSummaryTable,
) -> dict[tuple[str, str, str], list[GroupStat]]:
    keyed: dict[tuple[str, str, str], list[GroupStat]] = {}
    for group in table.groups:
        keyed.setdefault((group.observable, group.level, group.quantity), []).append(
            group
        )
    return keyed


def compare(
    baseline: LevelSummaryTable, variants: Mapping[str, LevelSummaryTable]
) -> VariantComparison:
    """Median-peak deltas of each variant against the baseline.

    A key (observable, level, quantity) is compared only when each side holds
    exactly one (distribution source, law) group, so sources are never pooled;
    other keys are listed in ``unmatched`` with the reason.

    Raises:
        TypeError: wrong argument types.
        ValueError: ``variants`` empty.
    """
    if not isinstance(baseline, LevelSummaryTable):
        raise TypeError("baseline must be a LevelSummaryTable")
    if not variants:
        raise ValueError("compare needs at least one of variants")
    base = _by_key(baseline)
    deltas: list[VariantDelta] = []
    unmatched: list[tuple[str, str, str, str, str]] = []
    for name in sorted(variants):
        other = _by_key(variants[name])
        for key in sorted(
            set(base) | set(other), key=lambda k: (k[0], _LEVEL_ORDER[k[1]], k[2])
        ):
            left, right = base.get(key, []), other.get(key, [])
            reason = _unmatched_reason(left, right)
            if reason:
                unmatched.append((name, *key, reason))
                continue
            deltas.append(_delta(name, key, left[0], right[0]))
    return VariantComparison(tuple(deltas), tuple(unmatched))


def _unmatched_reason(left: list[GroupStat], right: list[GroupStat]) -> str:
    if not left or not right:
        return "missing in baseline" if not left else "missing in variant"
    if len(left) > 1 or len(right) > 1:
        return "multiple distribution sources; not pooled"
    return ""


def _delta(
    variant: str, key: tuple[str, str, str], left: GroupStat, right: GroupStat
) -> VariantDelta:
    ratio = None if left.median_peak == 0 else right.median_peak / left.median_peak
    return VariantDelta(
        variant=variant,
        observable=key[0],
        level=key[1],
        quantity=key[2],
        units=left.units,
        baseline_source=left.distribution_source,
        variant_source=right.distribution_source,
        baseline_law=left.law,
        variant_law=right.law,
        baseline_peak=left.median_peak,
        variant_peak=right.median_peak,
        delta=right.median_peak - left.median_peak,
        ratio=ratio,
    )


def _num(value: float | None) -> float | None:
    return None if value is None else round_sig(value)


def _csv_header() -> list[str]:
    head = [
        "study_id",
        "observable",
        "level",
        "distribution_source",
        "law",
        "window_start[s]",
        "window_end[s]",
    ]
    for name, units, impulse_units in _QUANTITY_UNITS:
        head += [
            f"peak_{name}[{units}]",
            f"t_peak_{name}[s]",
            f"impulse_{name}[{impulse_units}]",
        ]
    return [*head, "peak_compression[BW]", "context_ratios[1]"]


def _csv_row(row: LevelRow) -> list[Any]:
    stats = {s.name: s for s in row.stats}
    out: list[Any] = [
        row.study_id,
        row.observable,
        row.level,
        row.distribution_source,
        row.law,
        _num(row.window_s[0]),
        _num(row.window_s[1]),
    ]
    for name, _, _ in _QUANTITY_UNITS:
        stat = stats.get(name)
        out += (
            ["", "", ""]
            if stat is None
            else [
                _num(stat.peak),
                _num(stat.t_peak_s),
                _num(stat.impulse),
            ]
        )
    out.append("" if row.peak_compression_bw is None else _num(row.peak_compression_bw))
    out.append(";".join(f"{c.get('band_id')}={c.get('ratio')}" for c in row.context))
    return out


def write_csv(table: LevelSummaryTable, path: str | Path) -> Path:
    """Write the row table as CSV; every numeric header carries ``[units]``.

    The bytes depend only on ``table`` (LF newlines, fixed column order).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_csv_header())
        for row in table.rows:
            writer.writerow(_csv_row(row))
    return out


def _row_dict(row: LevelRow) -> dict[str, Any]:
    return {
        "study_id": row.study_id,
        "observable": row.observable,
        "level": row.level,
        "distribution_source": row.distribution_source,
        "law": row.law,
        "window_s": [_num(row.window_s[0]), _num(row.window_s[1])],
        "peak_compression_bw": _num(row.peak_compression_bw),
        "stats": [
            {
                "quantity": s.name,
                "units": s.units,
                "peak": _num(s.peak),
                "t_peak_s": _num(s.t_peak_s),
                "impulse": _num(s.impulse),
            }
            for s in row.stats
        ],
        "context": list(row.context),
    }


def _group_dict(group: GroupStat) -> dict[str, Any]:
    return {
        "observable": group.observable,
        "level": group.level,
        "quantity": group.quantity,
        "units": group.units,
        "distribution_source": group.distribution_source,
        "law": group.law,
        "n_runs": group.n_runs,
        "min_peak": _num(group.min_peak),
        "median_peak": _num(group.median_peak),
        "max_peak": _num(group.max_peak),
        "mean_impulse": _num(group.mean_impulse),
    }


def write_json(table: LevelSummaryTable, path: str | Path) -> Path:
    """Write the table as canonical JSON (sorted keys, 10 significant figures)."""
    payload = {
        "schema_version": "spine-study-aggregate/1",
        "scientific_status": table.scientific_status,
        "n_included": table.n_included,
        "n_excluded": table.n_excluded,
        "excluded": [
            {"study_id": e.study_id, "status": e.status, "reasons": list(e.reasons)}
            for e in table.excluded
        ],
        "rows": [_row_dict(r) for r in table.rows],
        "groups": [_group_dict(g) for g in table.groups],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    return out
