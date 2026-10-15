"""Distribution sensitivity within one model (SPN-14a, #12769).

Per-level angles and loads depend on the assumed coupling (distribution) law.
This module quantifies that dependence before any cross-model comparison:
the same golf input runs through ONE segmental model under several shipped
``coupling_data`` laws, and only ``coupling_law_id`` changes between the runs.

For each law ``k`` an SPN-13a study (:func:`~.study.run_spine_study`) is run
with ``coupling_law_id=k`` and ``restrict_law_to_model=True`` (a law that
covers more levels than the model is restricted to the model's mapped range
by :func:`~src.shared.python.spine.coupling.restrict_to_levels`; one that
covers fewer fails and is excluded).  The receipts must agree on every input
hash (motion, golf model, segmental model, external loads, configuration) and
on every request field except ``study_id`` and ``coupling_law_id``: geometry,
stiffness, muscles and motion are held fixed by construction and verified
afterwards (:class:`SensitivityIntegrityError` otherwise).

Spread of a per-level quantity ``q`` over the set ``K`` of qualified laws:

    spread(q)          = max_k p_k(q) - min_k p_k(q)
    relative_spread(q) = spread(q) / max_k |p_k(q)|      (0 when all p_k = 0)

with ``p_k`` the signed peak (value of largest magnitude) over the record.
Kinematic quantities are the per-level ISB ZXY angles (rad) read back from the
hashed ``.mot`` artifact; they also report the largest pairwise RMS
difference of the angle time series.  Load quantities are the SPN-13c median
peaks per observable (``generalized_effort``, ``joint_reaction``), level and
quantity, compared through :func:`~.study_aggregation.compare` so laws are
never pooled.  A quantity is labelled ``robust`` when
``relative_spread <= robust_threshold`` and ``sensitive`` otherwise; the label
is descriptive.

Fail-closed: only receipts with status ``pass`` enter a spread; ``unqualified``
and ``failed`` runs are listed in ``excluded`` with their reasons and are
never tuned into passing.  ``scientific_status`` is always ``unqualified`` and
``is_acceptance`` always false: a spread is never acceptance evidence.

CLI: ``python3 -m src.engines.physics_engines.opensim.python.spine.\
distribution_sensitivity --request base.json [--laws ID ...] [--out DIR]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.spine.read_kinematics import (
    read_segmental_kinematics,
)
from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyRequest,
    run_spine_study,
)
from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    LevelSummaryTable,
    VariantDelta,
    aggregate_receipts,
    compare,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    FileRecord,
    SpineStudyReceipt,
    load_receipt,
    round_sig,
)
from src.shared.python.contracts import ensure
from src.shared.python.pose_estimation.model_files import sha256_of
from src.shared.python.spine.coupling_data import (
    available_laws,
    coupling_law,
    load_coupling_laws,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import SegmentalKinematicsSeries

__all__ = [
    "DEFAULT_ROBUST_THRESHOLD",
    "KINEMATICS",
    "LOADS",
    "LawRun",
    "QuantitySpread",
    "SensitivityIntegrityError",
    "SensitivityReport",
    "build_sensitivity_report",
    "classify_spread",
    "kinematic_spreads",
    "load_spreads",
    "main",
    "run_distribution_sensitivity",
    "write_report_csv",
    "write_report_json",
]

KINEMATICS = "kinematics"
LOADS = "loads"
#: Declared reporting threshold of ``relative_spread`` (descriptive only).
DEFAULT_ROBUST_THRESHOLD = 0.1
SCHEMA_VERSION = "spine-distribution-sensitivity/1"
_ISB_ANGLE = "isb_angle"
_AXES = ("flexion_extension", "lateral_bending", "axial_rotation")
_MOT_ROLE = "coordinates_mot"
#: Request fields allowed to differ between the runs of one sensitivity study.
_VARYING = ("study_id", "coupling_law_id")


class SensitivityIntegrityError(ValueError):
    """The runs do not hold everything but the law fixed, or an artifact fails."""


@dataclass(frozen=True)
class LawRun:
    """One law's run with its provenance and receipt status.

    ``law`` is the receipt's ``<law_id>@<version>`` (a restriction shows in
    the version); ``citation`` and ``verified_from_primary`` come from
    ``src/config/spine_coupling_laws.json``.  ``drive_*_rad`` is the geodesic
    residual of the distribution (``None`` when the drive did not run).
    """

    law_id: str
    law: str
    citation: str
    verified_from_primary: bool
    distribution_source: str
    status: str
    status_reasons: tuple[str, ...]
    drive_max_rad: float | None
    drive_rms_rad: float | None
    receipt_name: str

    @property
    def included(self) -> bool:
        """Only ``pass`` runs enter a spread."""
        return self.status == "pass"


@dataclass(frozen=True)
class QuantitySpread:
    """Across-law spread of one per-level quantity (see the module docstring)."""

    kind: str
    observable: str
    level: str
    quantity: str
    units: str
    values: tuple[tuple[str, float], ...]
    spread: float
    relative_spread: float
    classification: str
    max_pairwise_rms: float | None


@dataclass(frozen=True)
class SensitivityReport:
    """Per-law provenance, exclusions, held-fixed inputs and spreads."""

    study_id: str
    model_id: str
    robust_threshold: float
    laws: tuple[LawRun, ...]
    held_fixed: tuple[FileRecord, ...]
    spreads: tuple[QuantitySpread, ...]
    spread_note: str
    scientific_status: str = "unqualified"
    is_acceptance: bool = False

    @property
    def excluded(self) -> tuple[LawRun, ...]:
        """Runs kept out of every spread (status other than ``pass``)."""
        return tuple(law for law in self.laws if not law.included)

    @property
    def included_law_ids(self) -> tuple[str, ...]:
        """Ids of the laws that enter the spreads, sorted."""
        return tuple(law.law_id for law in self.laws if law.included)


# --------------------------------------------------------------- pure helpers


def _check_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("robust_threshold must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"robust_threshold must be finite and > 0, got {value!r}")
    return float(value)


def classify_spread(
    values: Mapping[str, float], robust_threshold: float
) -> tuple[float, float, str]:
    """``(spread, relative_spread, label)`` of per-law values of one quantity.

    Preconditions: ``values`` is a ``Mapping`` (``TypeError``) of at least two
    laws with finite values; ``robust_threshold`` finite and > 0
    (``ValueError``).

    Postconditions: ``spread >= 0``; ``relative_spread`` is ``spread`` over the
    largest magnitude (0 when every value is 0); ``label`` is ``robust`` iff
    ``relative_spread <= robust_threshold``, otherwise ``sensitive``.
    """
    if not isinstance(values, Mapping):
        raise TypeError("values must be a Mapping of law id to value")
    if len(values) < 2:
        raise ValueError("a spread needs at least two laws")
    threshold = _check_threshold(robust_threshold)
    data = np.array([float(v) for v in values.values()])
    if not np.all(np.isfinite(data)):
        raise ValueError("spread values must be finite")
    spread = float(np.max(data) - np.min(data))
    scale = float(np.max(np.abs(data)))
    relative = 0.0 if scale == 0.0 else spread / scale
    label = "robust" if relative <= threshold else "sensitive"
    ensure(spread >= 0.0, "spread must be non-negative", spread)
    return spread, relative, label


def _spread(
    head: tuple[str, str, str, str, str],
    values: Mapping[str, float],
    threshold: float,
    rms: float | None = None,
) -> QuantitySpread:
    spread, relative, label = classify_spread(values, threshold)
    return QuantitySpread(
        kind=head[0],
        observable=head[1],
        level=head[2],
        quantity=head[3],
        units=head[4],
        values=tuple(sorted((k, float(v)) for k, v in values.items())),
        spread=spread,
        relative_spread=relative,
        classification=label,
        max_pairwise_rms=rms,
    )


def _level_ordinal(name: str) -> int:
    return SpinalLevel[name].ordinal


def _deltas_by_key(
    deltas: Sequence[VariantDelta],
) -> dict[tuple[str, str, str], list[VariantDelta]]:
    keyed: dict[tuple[str, str, str], list[VariantDelta]] = {}
    for delta in deltas:
        key = (delta.observable, delta.level, delta.quantity)
        keyed.setdefault(key, []).append(delta)
    return keyed


def load_spreads(
    tables: Mapping[str, LevelSummaryTable], robust_threshold: float
) -> tuple[QuantitySpread, ...]:
    """Per (observable, level, quantity) spread of the median peaks across laws.

    ``tables`` maps a law id to that law's aggregate (one law per table).  The
    first id in sorted order is the :func:`compare` baseline; a key enters
    only when every law holds exactly one group for it (never pooled).

    Raises:
        TypeError: a value is not a :class:`LevelSummaryTable`.
        ValueError: fewer than two laws, or a bad threshold.

    Postcondition: ordered by observable, caudal-to-cranial level, quantity.
    """
    if len(tables) < 2:
        raise ValueError("load spreads need at least two laws")
    for name, table in tables.items():
        if not isinstance(table, LevelSummaryTable):
            raise TypeError(f"tables[{name!r}] must be a LevelSummaryTable")
    threshold = _check_threshold(robust_threshold)
    names = sorted(tables)
    baseline = names[0]
    comparison = compare(tables[baseline], {n: tables[n] for n in names[1:]})
    out = []
    for key, deltas in _deltas_by_key(comparison.deltas).items():
        if len(deltas) != len(names) - 1:
            continue
        values = {baseline: deltas[0].baseline_peak}
        values.update({d.variant: d.variant_peak for d in deltas})
        head = (LOADS, key[0], key[1], key[2], deltas[0].units)
        out.append(_spread(head, values, threshold))
    out.sort(key=lambda s: (s.observable, _level_ordinal(s.level), s.quantity))
    return tuple(out)


def _signed_peak(series: np.ndarray) -> float:
    return float(series[int(np.argmax(np.abs(series)))])


def _max_pairwise_rms(traces: Mapping[str, np.ndarray]) -> float:
    names = sorted(traces)
    worst = 0.0
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            diff = traces[left] - traces[right]
            worst = max(worst, float(np.sqrt(np.mean(diff**2))))
    return worst


def kinematic_spreads(
    levels: Sequence[SpinalLevel],
    angles_by_law: Mapping[str, np.ndarray],
    robust_threshold: float,
) -> tuple[QuantitySpread, ...]:
    """Per (level, ISB axis) spread of the signed peak angle across laws.

    ``angles_by_law`` maps a law id to an ``(n, len(levels), 3)`` array of ISB
    ZXY angles [rad] on one shared time grid.

    Raises:
        ValueError: fewer than two laws, a shape mismatch or a bad threshold.

    Postcondition: ``len(levels) * 3`` spreads, caudal-to-cranial, axes in
    (flexion_extension, lateral_bending, axial_rotation) order, each with the
    largest pairwise RMS difference of the time series.
    """
    if len(angles_by_law) < 2:
        raise ValueError("kinematic spreads need at least two laws")
    threshold = _check_threshold(robust_threshold)
    arrays = {k: np.asarray(v, dtype=float) for k, v in angles_by_law.items()}
    shape = next(iter(arrays.values())).shape
    for name, arr in arrays.items():
        if arr.shape != shape or arr.ndim != 3 or arr.shape[1:] != (len(levels), 3):
            raise ValueError(
                f"angles of {name!r} have shape {arr.shape}; every law needs "
                f"(n, {len(levels)}, 3) on the same grid"
            )
    out = []
    for j, level in enumerate(levels):
        for axis, quantity in enumerate(_AXES):
            traces = {k: arr[:, j, axis] for k, arr in arrays.items()}
            peaks = {k: _signed_peak(t) for k, t in traces.items()}
            head = (KINEMATICS, _ISB_ANGLE, level.name, quantity, "rad")
            out.append(_spread(head, peaks, threshold, _max_pairwise_rms(traces)))
    return tuple(out)


# ------------------------------------------------------------ receipt reading


def _law_run(law_id: str, receipt: SpineStudyReceipt, name: str) -> LawRun:
    entry = load_coupling_laws()[law_id]
    residual = receipt.residual
    return LawRun(
        law_id=law_id,
        law=receipt.law,
        citation=coupling_law(law_id).citation,
        verified_from_primary=bool(entry["verified_from_primary"]),
        distribution_source=receipt.distribution_source,
        status=receipt.status,
        status_reasons=tuple(receipt.status_reasons),
        drive_max_rad=residual.get("drive_max_rad"),
        drive_rms_rad=residual.get("drive_rms_rad"),
        receipt_name=name,
    )


def _check_held_fixed(receipts: Mapping[str, SpineStudyReceipt]) -> None:
    """Every input hash and request field except the law must agree."""
    items = sorted(receipts.items())
    first_id, first = items[0]
    fixed = {k: v for k, v in first.request.items() if k not in _VARYING}
    for law_id, receipt in items:
        if receipt.request.get("coupling_law_id") != law_id:
            raise SensitivityIntegrityError(
                f"receipt {receipt.study_id!r} ran law "
                f"{receipt.request.get('coupling_law_id')!r}, not {law_id!r}"
            )
        if receipt.inputs != first.inputs:
            raise SensitivityIntegrityError(
                f"inputs of {law_id!r} differ from {first_id!r}: only the "
                "distribution law may vary"
            )
        others = {k: v for k, v in receipt.request.items() if k not in _VARYING}
        if others != fixed:
            raise SensitivityIntegrityError(
                f"request of {law_id!r} differs from {first_id!r} beyond the law"
            )


def _artifact(receipt: SpineStudyReceipt, folder: Path, role: str) -> Path:
    found = [a for a in receipt.artifacts if a.role == role]
    if len(found) != 1:
        raise SensitivityIntegrityError(
            f"{receipt.study_id}: expected one {role!r} artifact, found {len(found)}"
        )
    path = folder / found[0].name
    if not path.is_file() or sha256_of(path) != found[0].sha256:
        raise SensitivityIntegrityError(
            f"{receipt.study_id}: {found[0].name!r} is missing or its sha256 "
            "does not match the receipt"
        )
    return path


def _angles(series: SegmentalKinematicsSeries) -> np.ndarray:
    """``(n, n_levels, 3)`` ISB angles [rad] of a kinematics series."""
    frames = []
    for pose in series.poses:
        frames.append([item.isb_angles() for item in pose.transforms])
    return np.asarray(frames, dtype=float)


def _kinematics(
    receipts: Mapping[str, SpineStudyReceipt],
    folders: Mapping[str, Path],
    model_path: Path,
    model_id: str,
) -> tuple[tuple[SpinalLevel, ...], dict[str, np.ndarray]]:
    levels: tuple[SpinalLevel, ...] = ()
    times: np.ndarray | None = None
    out = {}
    for law_id, receipt in sorted(receipts.items()):
        mot = _artifact(receipt, folders[law_id], _MOT_ROLE)
        series = read_segmental_kinematics(
            mot,
            model_path,
            model_id,
            distribution_source=receipt.distribution_source,
        )
        levels = series.poses[0].levels
        if times is not None and not np.array_equal(times, series.time_s):
            raise SensitivityIntegrityError(f"{law_id!r} uses another time grid")
        times = series.time_s
        out[law_id] = _angles(series)
    return levels, out


def _check_receipt_paths(
    receipt_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not isinstance(receipt_paths, Mapping) or not receipt_paths:
        raise ValueError("receipt_paths must map at least one law id to a receipt")
    known = load_coupling_laws()
    for law_id in receipt_paths:
        if law_id not in known:
            raise ValueError(f"unknown coupling law {law_id!r}")
    return {k: Path(v) for k, v in receipt_paths.items()}


def build_sensitivity_report(
    study_id: str,
    receipt_paths: Mapping[str, str | Path],
    model_path: str | Path,
    model_id: str,
    robust_threshold: float = DEFAULT_ROBUST_THRESHOLD,
) -> SensitivityReport:
    """Build the report from per-law SPN-13a receipts (artifacts beside them).

    Args:
        study_id: label of the sensitivity study.
        receipt_paths: law id -> receipt JSON of that law's run.
        model_path / model_id: the segmental model the runs used (needed to
            read the ``.mot`` back into per-level angles; never recorded).
        robust_threshold: declared ``relative_spread`` reporting threshold.

    Raises:
        ValueError: empty or unknown law ids, a bad threshold.
        SensitivityIntegrityError: inputs or request fields differ beyond the
            law, a receipt ran another law, or an artifact hash mismatches.

    Postconditions: deterministic for the same receipts and artifacts;
    ``scientific_status`` is ``unqualified`` and ``is_acceptance`` false; only
    ``pass`` runs enter ``spreads``; with fewer than two of them ``spreads``
    is empty and ``spread_note`` says why.
    """
    threshold = _check_threshold(robust_threshold)
    paths = _check_receipt_paths(receipt_paths)
    receipts = {k: load_receipt(p) for k, p in paths.items()}
    _check_held_fixed(receipts)
    runs = tuple(_law_run(k, receipts[k], paths[k].name) for k in sorted(receipts))
    passed = {r.law_id: receipts[r.law_id] for r in runs if r.included}
    spreads: tuple[QuantitySpread, ...] = ()
    note = ""
    if len(passed) < 2:
        note = f"{len(passed)} qualified law(s); a spread needs at least two"
    else:
        folders = {k: paths[k].parent for k in passed}
        levels, angles = _kinematics(passed, folders, Path(model_path), model_id)
        tables = {k: aggregate_receipts([paths[k]]) for k in passed}
        spreads = kinematic_spreads(levels, angles, threshold) + load_spreads(
            tables, threshold
        )
    report = SensitivityReport(
        study_id=study_id,
        model_id=model_id,
        robust_threshold=threshold,
        laws=runs,
        held_fixed=tuple(receipts[runs[0].law_id].inputs),
        spreads=spreads,
        spread_note=note,
    )
    ensure(
        all(set(dict(s.values)) == set(passed) for s in spreads),
        "every spread covers exactly the qualified laws",
    )
    return report


# -------------------------------------------------------------------- running


def _check_law_ids(law_ids: Sequence[str] | None) -> tuple[str, ...]:
    chosen = available_laws() if law_ids is None else tuple(law_ids)
    if any(not isinstance(k, str) for k in chosen):
        raise TypeError("law_ids must be strings")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate law ids in {chosen}")
    if len(chosen) < 2:
        raise ValueError("a sensitivity study needs at least two laws")
    shipped = set(available_laws())
    pending = set(available_laws(include_pending=True)) - shipped
    for law_id in chosen:
        if law_id in pending:
            raise ValueError(f"law {law_id!r} is pending and has no shipped numbers")
        if law_id not in shipped:
            raise ValueError(f"unknown coupling law {law_id!r}")
    return tuple(sorted(chosen))


def run_distribution_sensitivity(
    base: SpineStudyRequest,
    law_ids: Sequence[str] | None = None,
    *,
    robust_threshold: float = DEFAULT_ROBUST_THRESHOLD,
    code_version: str | None = None,
    env: Mapping[str, str] | None = None,
) -> SensitivityReport:
    """Run ``base`` once per law, varying only ``coupling_law_id``.

    Each run is ``base`` with ``study_id=<base>__<law>``,
    ``coupling_law_id=<law>`` and ``restrict_law_to_model=True``; receipts and
    artifacts go to ``base.output_dir``.  ``law_ids`` defaults to every
    shipped (non-pending) law.

    Preconditions: ``base`` is a :class:`SpineStudyRequest` (``TypeError``);
    at least two distinct shipped law ids; ``base.model_path`` is given
    (``ValueError``) so the per-level angles can be read back.

    Returns:
        The :class:`SensitivityReport` (see :func:`build_sensitivity_report`).
    """
    if not isinstance(base, SpineStudyRequest):
        raise TypeError(f"base must be a SpineStudyRequest, got {type(base).__name__}")
    chosen = _check_law_ids(law_ids)
    if base.model_path is None:
        raise ValueError("base.model_path is required to read per-level angles")
    receipts = {}
    for law_id in chosen:
        request = replace(
            base,
            study_id=f"{base.study_id}__{law_id}",
            coupling_law_id=law_id,
            restrict_law_to_model=True,
        )
        run_spine_study(request, code_version=code_version, env=env)
        receipts[law_id] = base.output_dir / f"{request.study_id}_receipt.json"
    return build_sensitivity_report(
        base.study_id, receipts, base.model_path, base.model_id, robust_threshold
    )


# -------------------------------------------------------------------- writing


def _num(value: float | None) -> float | None:
    return None if value is None else round_sig(value)


def _law_dict(law: LawRun) -> dict[str, Any]:
    return {
        "law_id": law.law_id,
        "law": law.law,
        "citation": law.citation,
        "verified_from_primary": law.verified_from_primary,
        "distribution_source": law.distribution_source,
        "status": law.status,
        "status_reasons": list(law.status_reasons),
        "included": law.included,
        "drive_max_rad": _num(law.drive_max_rad),
        "drive_rms_rad": _num(law.drive_rms_rad),
        "receipt": law.receipt_name,
    }


def _spread_dict(item: QuantitySpread) -> dict[str, Any]:
    return {
        "kind": item.kind,
        "observable": item.observable,
        "level": item.level,
        "quantity": item.quantity,
        "units": item.units,
        "values": {k: _num(v) for k, v in item.values},
        "spread": _num(item.spread),
        "relative_spread": _num(item.relative_spread),
        "classification": item.classification,
        "max_pairwise_rms": _num(item.max_pairwise_rms),
    }


def write_report_json(report: SensitivityReport, path: str | Path) -> Path:
    """Write the report as canonical JSON (sorted keys, 10 significant figures).

    The bytes depend only on ``report``; no absolute path is written.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": report.study_id,
        "model_id": report.model_id,
        "scientific_status": report.scientific_status,
        "is_acceptance": report.is_acceptance,
        "robust_threshold": report.robust_threshold,
        "laws": [_law_dict(law) for law in report.laws],
        "excluded": [law.law_id for law in report.excluded],
        "held_fixed": [r.as_dict() for r in report.held_fixed],
        "spread_note": report.spread_note,
        "spreads": [_spread_dict(s) for s in report.spreads],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


def write_report_csv(report: SensitivityReport, path: str | Path) -> Path:
    """Write one row per spread; value columns are ``value[<law_id>]``."""
    laws = report.included_law_ids
    header = ["kind", "observable", "level", "quantity", "units"]
    header += [f"value[{law}]" for law in laws]
    header += ["spread", "relative_spread", "max_pairwise_rms", "classification"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for item in report.spreads:
            values = dict(item.values)
            rms = item.max_pairwise_rms
            writer.writerow(
                [item.kind, item.observable, item.level, item.quantity, item.units]
                + [_num(values[law]) for law in laws]
                + [
                    _num(item.spread),
                    _num(item.relative_spread),
                    "" if rms is None else _num(rms),
                    item.classification,
                ]
            )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run a base request under each law; write JSON and CSV reports.

    Exit status 0 when at least two laws qualified, 1 otherwise.
    """
    parser = argparse.ArgumentParser(prog="spine.distribution_sensitivity")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--laws", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_ROBUST_THRESHOLD)
    args = parser.parse_args(argv)
    raw = json.loads(args.request.read_text(encoding="utf-8"))
    if args.out is not None:
        raw = {**raw, "output_dir": str(args.out.resolve())}
    base = SpineStudyRequest.from_dict(raw, args.request.resolve().parent)
    report = run_distribution_sensitivity(
        base, args.laws, robust_threshold=args.threshold
    )
    stem = base.output_dir / f"{base.study_id}_sensitivity"
    write_report_csv(report, stem.with_suffix(".csv"))
    written = write_report_json(report, stem.with_suffix(".json"))
    sys.stdout.write(written.read_text(encoding="utf-8"))
    return 0 if len(report.included_law_ids) >= 2 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
