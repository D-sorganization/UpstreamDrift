"""Compact the 9.1 GB raw Simscape parquet to a ~115 MB training artefact.

See ``motion_matching/shared/COMPACT_DATASET_SCHEMA.md`` for the
authoritative output schema. This script is the only producer of that
schema; downstream training code consumes it through
``src.shared.python.dataset_tools.load_compact_swing_dataset``.

Streaming strategy:

The raw file has one row group per trial (10 000 trials × 31 rows) and
all 1956 columns are stored as scientific-notation strings. We stream
**one row group at a time** so peak memory stays under ~2 GB even on the
full dataset. For each row group:

1. Read only the columns we need (the canonical projection map below).
2. Cast the relevant string columns to float64.
3. Build the compact rows (one timestep row per raw row, one trial row
   per row group).
4. Append them to the appropriate arrow ``ChunkedArray`` writer.

Output partitioning: ``timesteps.parquet`` and ``trials.parquet`` are
written as Hive-style partitioned datasets keyed by ``chunk_id`` =
``trial_id // 1000``. That keeps per-trial reads fast (only one chunk
file is touched) without exploding the file count.

Run manually (not in CI):

.. code-block:: bash

    python3 scripts/compact_swing_dataset.py \
        --src C:/Users/diete/Repositories/data/TenThousandFiles.parquet \
        --out C:/Users/diete/Repositories/data/compact \
        --validate
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# This script lives at scripts/, so we add the repo root to sys.path so
# that ``src.shared.python.dataset_tools`` is importable when the
# compactor is run directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.python.contracts import postcondition, precondition  # noqa: E402
from src.shared.python.dataset_tools.canonical import (  # noqa: E402
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
    SCHEMA_VERSION,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

_LOGGER = logging.getLogger(__name__)
_PROGRESS_EVERY = 1000

# ---------------------------------------------------------------------------
# Canonical raw-to-compact column map.
#
# Computed once at import time from CANONICAL_JOINTS so adding a new joint
# only requires editing canonical.py, not this script (DRY).
# ---------------------------------------------------------------------------


def _angular_kinematics_columns(joint: str) -> tuple[list[str], list[str], list[str]]:
    """Return the (q, qd, qdd) raw column names for a canonical joint.

    The raw schema has three flavours of joint:

      * 1-DOF revolute (LE/LF/RE/RF/Torso): one column without an axis suffix
        e.g. ``AngularKinematicsLogs_LEAngularPosition``.
      * Per-axis joints (HipX/HipY/HipZ, SpineX/SpineY, LSX/LSY/LSZ, ...):
        canonical name ends with the axis letter; raw column groups the joint
        prefix with the axis after the kinematic-quantity word, e.g.
        ``AngularKinematicsLogs_HipAngularPositionX``.
      * Translational DOFs (TranslationX/Y/Z): tracked via
        ``AngularKinematicsLogs_HipPosition[X|Y|Z]`` etc. (linear, not
        angular). These are the floating-base translational degrees.
    """
    if joint in {"LE", "LF", "RE", "RF", "Torso"}:
        base = f"AngularKinematicsLogs_{joint}Angular"
        return ([f"{base}Position"], [f"{base}Velocity"], [f"{base}Acceleration"])

    if joint.startswith("Translation"):
        axis = joint[-1]
        base = f"AngularKinematicsLogs_Hip{{}}{axis}"
        return (
            [base.format("Position")],
            [base.format("Velocity")],
            [base.format("Acceleration")],
        )

    # Per-axis joint, e.g. HipX, SpineY, LSZ, LScapY, RWX, ...
    match = re.match(r"^(.+?)([XYZ])$", joint)
    if not match:
        raise ValueError(f"unrecognised canonical joint name: {joint!r}")
    prefix, axis = match.group(1), match.group(2)
    base = f"AngularKinematicsLogs_{prefix}Angular"
    return (
        [f"{base}Position{axis}"],
        [f"{base}Velocity{axis}"],
        [f"{base}Acceleration{axis}"],
    )


# Applied-torque mapping. Each entry maps a canonical joint name to the
# raw column that holds its applied/control/actuator torque. ``None``
# means "no applied-torque column in the raw dump" → store NaN downstream.
#
# Mapping rationale (verified against raw schema column inventory):
#   * Hip[X|Y|Z] → HipLogs_HipTorque[X|Y|Z]Input (revolute joint inputs).
#   * Translation[X|Y|Z] → HipLogs_TranslationForce[X|Y|Z]Input (linear).
#   * LS/RS/Spine/LScap/RScap (multi-DOF actuated bodies) →
#     ActuatorTorque[X|Y|Z] columns.
#   * LF/RF/Torso (1-DOF revolutes that DO have a TorqueLocal in their
#     log block) → TorqueLocal_1.
#   * LE/RE: the raw schema has no LELogs_TorqueLocal_*; per the schema
#     doc we therefore store NaN ("no applied-torque column in the raw
#     dump").
#   * LWX/LWY/RWX/RWY: no LWLogs/RWLogs TorqueLocal columns either; NaN.
TAU_RAW_MAP: dict[str, str | None] = {
    "HipX": "HipLogs_HipTorqueXInput",
    "HipY": "HipLogs_HipTorqueYInput",
    "HipZ": "HipLogs_HipTorqueZInput",
    "LE": None,
    "LF": "LFLogs_TorqueLocal_1",
    "LSX": "LSLogs_ActuatorTorqueX",
    "LSY": "LSLogs_ActuatorTorqueY",
    "LSZ": "LSLogs_ActuatorTorqueZ",
    "LScapX": "LScapLogs_ActuatorTorqueX",
    "LScapY": "LScapLogs_ActuatorTorqueY",
    "LWX": None,
    "LWY": None,
    "RE": None,
    "RF": "RFLogs_TorqueLocal_1",
    "RSX": "RSLogs_ActuatorTorqueX",
    "RSY": "RSLogs_ActuatorTorqueY",
    "RSZ": "RSLogs_ActuatorTorqueZ",
    "RScapX": "RScapLogs_ActuatorTorqueX",
    "RScapY": "RScapLogs_ActuatorTorqueY",
    "RWX": None,
    "RWY": None,
    "SpineX": "SpineLogs_ActuatorTorqueX",
    "SpineY": "SpineLogs_ActuatorTorqueY",
    "Torso": "TorsoLogs_TorqueLocal_1",
    "TranslationX": "HipLogs_TranslationForceXInput",
    "TranslationY": "HipLogs_TranslationForceYInput",
    "TranslationZ": "HipLogs_TranslationForceZInput",
}

# Joints whose tau column is intentionally absent in the raw dump and
# should be left as NaN by the compactor. The loader's NaN check is
# therefore relaxed for the ``tau`` column only — see
# ``_validate_no_nan`` in ``load_compact``.
TAU_NULL_JOINTS: frozenset[str] = frozenset(
    {j for j, raw in TAU_RAW_MAP.items() if raw is None}
)

CLUB_R_COLS = (
    "ClubLogs_CHGlobalPosition_1",
    "ClubLogs_CHGlobalPosition_2",
    "ClubLogs_CHGlobalPosition_3",
)
CLUB_V_COLS = (
    "ClubLogs_CHGlobalVelocity_1",
    "ClubLogs_CHGlobalVelocity_2",
    "ClubLogs_CHGlobalVelocity_3",
)
CLUB_BUTT_COLS = (
    "ClubLogs_TipPosition_1",
    "ClubLogs_TipPosition_2",
    "ClubLogs_TipPosition_3",
)
LHAND_COLS = (
    "LWLogs_LHGlobalPosition_1",
    "LWLogs_LHGlobalPosition_2",
    "LWLogs_LHGlobalPosition_3",
)
RHAND_COLS = (
    "RWLogs_RHGlobalPosition_1",
    "RWLogs_RHGlobalPosition_2",
    "RWLogs_RHGlobalPosition_3",
)
CHS_COL = "ClubLogs_CHS__mph_"
TIME_COL = "time"
TRIAL_COL = "trial_id"


def _coeff_columns() -> list[str]:
    """The 189 ``input_<joint>_<letter>`` column names in canonical order."""
    return [
        f"input_{joint}_{letter}"
        for joint in CANONICAL_JOINTS
        for letter in COEFFICIENT_LETTERS
    ]


COEFF_COLUMNS: list[str] = _coeff_columns()


def _q_qd_qdd_columns() -> tuple[list[str], list[str], list[str]]:
    q_cols: list[str] = []
    qd_cols: list[str] = []
    qdd_cols: list[str] = []
    for joint in CANONICAL_JOINTS:
        q, qd, qdd = _angular_kinematics_columns(joint)
        q_cols.extend(q)
        qd_cols.extend(qd)
        qdd_cols.extend(qdd)
    return q_cols, qd_cols, qdd_cols


Q_COLS, QD_COLS, QDD_COLS = _q_qd_qdd_columns()

# Documented kinematic columns that are missing in the real raw dump.
# Listed so the loader's NaN check can permit them; everything else must
# remain finite. (See COMPACT_DATASET_SCHEMA.md §Validation rules.)
KNOWN_MISSING_KINEMATIC_COLUMNS: frozenset[str] = frozenset(
    {
        "AngularKinematicsLogs_LSAngularAccelerationZ",
        "AngularKinematicsLogs_RScapAngularAccelerationX",
    }
)


def _all_required_raw_columns() -> list[str]:
    """Strictly required columns — failure to provide any of these is fatal."""
    cols: list[str] = [TRIAL_COL, TIME_COL, CHS_COL]
    cols.extend(COEFF_COLUMNS)
    cols.extend(CLUB_R_COLS)
    cols.extend(CLUB_V_COLS)
    cols.extend(CLUB_BUTT_COLS)
    cols.extend(LHAND_COLS)
    cols.extend(RHAND_COLS)
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _all_optional_raw_columns() -> list[str]:
    """Columns we read when present, but tolerate missing (filled with NaN).

    This includes the per-joint kinematic columns (q/qd/qdd) and the
    applied-torque columns. The real raw dump has a couple of legitimately
    missing kinematic columns (e.g. ``LSAngularAccelerationZ``) — we let
    those join the NaN regime rather than refusing to compact the file.
    """
    cols: list[str] = []
    cols.extend(Q_COLS)
    cols.extend(QD_COLS)
    cols.extend(QDD_COLS)
    for raw in TAU_RAW_MAP.values():
        if raw is not None:
            cols.append(raw)
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


REQUIRED_RAW_COLUMNS: list[str] = _all_required_raw_columns()
OPTIONAL_RAW_COLUMNS: list[str] = _all_optional_raw_columns()


# ---------------------------------------------------------------------------
# Output arrow schemas.
# ---------------------------------------------------------------------------


def _build_timesteps_schema() -> pa.Schema:
    list_f64 = pa.list_(pa.float64())
    return pa.schema(
        [
            ("trial_id", pa.uint32()),
            ("chunk_id", pa.uint32()),
            ("t", pa.float64()),
            ("q", list_f64),
            ("qd", list_f64),
            ("qdd", list_f64),
            ("tau", list_f64),
            ("r_clubhead", list_f64),
            ("v_clubhead", list_f64),
            ("r_buttend", list_f64),
            ("r_lhand", list_f64),
            ("r_rhand", list_f64),
            ("r_grip", list_f64),
            ("clubhead_speed_mph", pa.float64()),
        ]
    )


def _build_trials_schema() -> pa.Schema:
    return pa.schema(
        [
            ("trial_id", pa.uint32()),
            ("chunk_id", pa.uint32()),
            ("coefficients", pa.list_(pa.float64())),
            ("joint_names", pa.list_(pa.string())),
            ("coefficient_letters", pa.list_(pa.string())),
            ("simulation_time_s", pa.float64()),
            ("sample_rate_hz", pa.float64()),
            ("clubhead_speed_max_mph", pa.float64()),
            ("total_work_J", pa.float64()),
            ("solver_status", pa.string()),
        ]
    )


# ---------------------------------------------------------------------------
# Cast helpers.
# ---------------------------------------------------------------------------


def _string_column_to_float64(table: pa.Table, column: str) -> np.ndarray:
    """Cast ``table[column]`` to a ``numpy`` ``float64`` array.

    Handles both already-numeric columns (round-trip safe) and the raw
    string-encoded scientific-notation values.
    """
    arr = table[column]
    arr_type = arr.type
    if pa.types.is_floating(arr_type) or pa.types.is_integer(arr_type):
        return arr.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    if pa.types.is_string(arr_type) or pa.types.is_large_string(arr_type):
        casted = arr.cast(pa.float64(), safe=False)
        return casted.to_numpy(zero_copy_only=False)
    raise TypeError(
        f"raw column {column!r} has unexpected type {arr_type}; expected "
        "string or numeric"
    )


def _stack_columns(table: pa.Table, columns: Iterable[str], n_rows: int) -> np.ndarray:
    """Return an ``(n_rows, len(columns))`` ``float64`` matrix.

    Missing columns are filled with NaN — see
    ``KNOWN_MISSING_KINEMATIC_COLUMNS`` for the ones documented in the
    real raw dump.
    """
    table_cols = set(table.column_names)
    column_arrays: list[np.ndarray] = []
    for c in columns:
        if c in table_cols:
            column_arrays.append(_string_column_to_float64(table, c))
        else:
            column_arrays.append(np.full(n_rows, np.nan, dtype=np.float64))
    return np.stack(column_arrays, axis=1)


# ---------------------------------------------------------------------------
# Per-trial transform.
# ---------------------------------------------------------------------------


def _build_tau_matrix(table: pa.Table, n_rows: int) -> np.ndarray:
    """Return an ``(n_rows, N_JOINTS)`` matrix of applied torques.

    Joints with no applied-torque column (or marked NaN-only) get NaN.
    """
    tau = np.full((n_rows, N_JOINTS), np.nan, dtype=np.float64)
    for j_idx, joint in enumerate(CANONICAL_JOINTS):
        if joint in TAU_NULL_JOINTS:
            continue
        raw = TAU_RAW_MAP.get(joint)
        if raw is None or raw not in table.column_names:
            continue
        tau[:, j_idx] = _string_column_to_float64(table, raw)
    return tau


def _trial_record_from_group(
    table: pa.Table,
    trial_id: int,
    simulation_time_s: float,
    sample_rate_hz: float,
    chs_max: float,
    total_work_J: float,
) -> dict[str, object]:
    """Pack the trial-level row from a single row-group table."""
    coeffs = np.empty(N_COEFFS, dtype=np.float64)
    for i, col in enumerate(COEFF_COLUMNS):
        coeffs[i] = _string_column_to_float64(table, col)[0]

    return {
        "trial_id": np.uint32(trial_id),
        "chunk_id": np.uint32(trial_id // 1000),
        "coefficients": coeffs.tolist(),
        "joint_names": list(CANONICAL_JOINTS),
        "coefficient_letters": list(COEFFICIENT_LETTERS),
        "simulation_time_s": float(simulation_time_s),
        "sample_rate_hz": float(sample_rate_hz),
        "clubhead_speed_max_mph": float(chs_max),
        "total_work_J": float(total_work_J),
        "solver_status": "success",
    }


def _timesteps_records_from_group(
    table: pa.Table, trial_id: int
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Return one timestep dict per raw row + summary stats for the trial."""
    n_rows = table.num_rows
    if n_rows == 0:
        return [], {"sim_time": 0.0, "rate": 0.0, "chs_max": 0.0, "work": 0.0}

    t = _string_column_to_float64(table, TIME_COL)
    t = t - t[0]  # rebase to zero per the schema doc
    q = _stack_columns(table, Q_COLS, n_rows)
    qd = _stack_columns(table, QD_COLS, n_rows)
    qdd = _stack_columns(table, QDD_COLS, n_rows)
    tau = _build_tau_matrix(table, n_rows)
    r_ch = _stack_columns(table, CLUB_R_COLS, n_rows)
    v_ch = _stack_columns(table, CLUB_V_COLS, n_rows)
    r_butt = _stack_columns(table, CLUB_BUTT_COLS, n_rows)
    r_lh = _stack_columns(table, LHAND_COLS, n_rows)
    r_rh = _stack_columns(table, RHAND_COLS, n_rows)
    r_grip = (r_lh + r_rh) / 2.0
    chs = _string_column_to_float64(table, CHS_COL)

    rows: list[dict[str, object]] = []
    chunk_id = trial_id // 1000
    for i in range(n_rows):
        rows.append(
            {
                "trial_id": np.uint32(trial_id),
                "chunk_id": np.uint32(chunk_id),
                "t": float(t[i]),
                "q": q[i].tolist(),
                "qd": qd[i].tolist(),
                "qdd": qdd[i].tolist(),
                "tau": tau[i].tolist(),
                "r_clubhead": r_ch[i].tolist(),
                "v_clubhead": v_ch[i].tolist(),
                "r_buttend": r_butt[i].tolist(),
                "r_lhand": r_lh[i].tolist(),
                "r_rhand": r_rh[i].tolist(),
                "r_grip": r_grip[i].tolist(),
                "clubhead_speed_mph": float(chs[i]),
            }
        )

    sim_time = float(t[-1])
    rate = float(n_rows / sim_time) if sim_time > 0 else 0.0
    chs_max = float(np.nanmax(chs))
    # Total work approximation: trapezoidal integral of sum_j tau_j * qd_j.
    power = np.nansum(tau * qd, axis=1)
    work = float(np.trapezoid(power, t)) if n_rows > 1 else 0.0
    summary = {
        "sim_time": sim_time,
        "rate": rate,
        "chs_max": chs_max,
        "work": work,
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Validation utilities used by ``--validate``.
# ---------------------------------------------------------------------------


def _validate_outputs(out_dir: Path) -> None:
    """Run the schema doc's validation rules on the produced parquet."""
    from src.shared.python.dataset_tools.load_compact import (
        load_compact_swing_dataset,
    )

    _LOGGER.info("running --validate against %s", out_dir)
    ds = load_compact_swing_dataset(out_dir, lazy=False)
    _LOGGER.info(
        "validation OK: %d trials, %d timesteps", len(ds.trials), len(ds.timesteps)
    )


# ---------------------------------------------------------------------------
# Top-level streaming loop.
# ---------------------------------------------------------------------------


@precondition(
    lambda src, out, **_: Path(src).exists(),
    "raw parquet file does not exist",
)
@postcondition(
    lambda result: result["timesteps_rows"] > 0 and result["trials_rows"] > 0,
    "compactor must emit non-empty timesteps + trials parquet files",
)
def compact_swing_dataset(
    src: str | Path,
    out: str | Path,
    *,
    limit_trials: int | None = None,
    validate: bool = False,
) -> dict[str, int]:
    """Stream the raw parquet → compact dataset.

    Args:
        src: Path to the raw 9 GB string-encoded parquet.
        out: Output directory (created if missing). Will receive
            ``trials.parquet`` and ``timesteps.parquet``.
        limit_trials: If set, stop after compacting this many trials.
            Used for smoke tests.
        validate: If True, re-load the produced files via the loader
            and run cross-validated schema checks.

    Returns:
        ``{"trials_rows": <n>, "timesteps_rows": <n>}``.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        ValueError: If ``--validate`` is set and validation fails.
    """
    src_path = Path(src)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src_path)
    raw_columns = set(pf.schema_arrow.names)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_columns]
    if missing:
        raise ValueError(
            f"raw parquet missing required columns (first 5): {missing[:5]}"
        )
    optional_present = [c for c in OPTIONAL_RAW_COLUMNS if c in raw_columns]
    optional_missing = [c for c in OPTIONAL_RAW_COLUMNS if c not in raw_columns]
    if optional_missing:
        unexpected = [
            c for c in optional_missing if c not in KNOWN_MISSING_KINEMATIC_COLUMNS
        ]
        if unexpected:
            _LOGGER.warning(
                "raw parquet missing optional columns (NaN-filled): %s",
                unexpected,
            )
    columns_to_read = REQUIRED_RAW_COLUMNS + optional_present

    timesteps_schema = _build_timesteps_schema()
    trials_schema = _build_trials_schema()

    timesteps_writer = pq.ParquetWriter(
        out_dir / "timesteps.parquet",
        timesteps_schema,
        compression="snappy",
    )
    trials_writer = pq.ParquetWriter(
        out_dir / "trials.parquet",
        trials_schema,
        compression="snappy",
    )

    n_trials_done = 0
    n_timesteps_done = 0
    try:
        n_groups = pf.num_row_groups
        for rg_idx in range(n_groups):
            if limit_trials is not None and n_trials_done >= limit_trials:
                break
            table = pf.read_row_group(rg_idx, columns=columns_to_read)
            if table.num_rows == 0:
                continue
            trial_id_arr = _string_column_to_float64(table, TRIAL_COL)
            trial_id = int(trial_id_arr[0])

            ts_rows, summary = _timesteps_records_from_group(table, trial_id)
            if not ts_rows:
                continue
            ts_batch = pa.Table.from_pylist(ts_rows, schema=timesteps_schema)
            timesteps_writer.write_table(ts_batch)
            n_timesteps_done += len(ts_rows)

            trial_row = _trial_record_from_group(
                table,
                trial_id=trial_id,
                simulation_time_s=summary["sim_time"],
                sample_rate_hz=summary["rate"],
                chs_max=summary["chs_max"],
                total_work_J=summary["work"],
            )
            tr_batch = pa.Table.from_pylist([trial_row], schema=trials_schema)
            trials_writer.write_table(tr_batch)
            n_trials_done += 1

            if n_trials_done % _PROGRESS_EVERY == 0:
                _LOGGER.info(
                    "compacted %d trials (%d timesteps)",
                    n_trials_done,
                    n_timesteps_done,
                )
    finally:
        timesteps_writer.close()
        trials_writer.close()

    _LOGGER.info(
        "wrote %d trials and %d timesteps to %s",
        n_trials_done,
        n_timesteps_done,
        out_dir,
    )

    if validate:
        _validate_outputs(out_dir)

    return {
        "trials_rows": n_trials_done,
        "timesteps_rows": n_timesteps_done,
        "schema_version": SCHEMA_VERSION,  # type: ignore[dict-item]
    }


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream-compact the raw Simscape parquet to the "
        "training-pipeline schema.",
    )
    p.add_argument("--src", required=True, help="Raw parquet file path.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument(
        "--limit-trials",
        type=int,
        default=None,
        help="Stop after this many trials (for smoke tests).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Re-load the produced files and run schema validation.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGER.info(
        "compacting %s → %s (limit=%s, validate=%s)",
        args.src,
        args.out,
        args.limit_trials,
        args.validate,
    )
    result = compact_swing_dataset(
        src=args.src,
        out=args.out,
        limit_trials=args.limit_trials,
        validate=args.validate,
    )
    _LOGGER.info("done: %s", result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(main())
