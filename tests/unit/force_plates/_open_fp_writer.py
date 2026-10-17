"""UpstreamDrift-owned generator of synthetic swing files in the open-fp layout.

Written from the documented file format only.  Every number is hand-chosen: six
samples, the left load dropping below the 20 N CoP threshold at the end, so the
net CoP switches from a two-foot to a one-foot value.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

SWING_ID = "20260101-120000-123"
THRESHOLD_N = 20.0
NAN2 = [np.nan, np.nan]
T_S = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.025])
FZ_L = np.array([400.0, 300.0, 100.0, 50.0, 10.0, 5.0])
FZ_R = np.array([400.0, 500.0, 600.0, 700.0, 790.0, 795.0])
COP_L_MM = np.array(
    [[-185.0, 20.0], [-180.0, 0.0], [-170.0, 5.0], [-160.0, 8.0], NAN2, NAN2]
)
COP_R_MM = np.array(
    [
        [185.0, -20.0],
        [190.0, 10.0],
        [195.0, 12.5],
        [200.0, 30.0],
        [201.0, 31.0],
        [202.0, 32.0],
    ]
)
EVENTS = {
    "motion_start": 0.0,
    "max_trail": 0.005,
    "transition": 0.010,
    "max_lead": None,
    "finish": 0.025,
    "impact": 0.015,
    "trigger": None,
}
CSV_HEADER = (
    "t_s,lead_pct,fz_L_n,fz_R_n,cop_L_x_mm,cop_L_y_mm,cop_R_x_mm,cop_R_y_mm,"
    "cop_C_x_mm,cop_C_y_mm,L1_n,L2_n,L3_n,L4_n,R1_n,R2_n,R3_n,R4_n"
)


def combined_cop_mm(
    fz_l: np.ndarray, fz_r: np.ndarray, cop_l: np.ndarray, cop_r: np.ndarray
) -> np.ndarray:
    """Load-weighted CoP of the plates at/above the threshold, NaN if neither."""
    out = np.full((len(fz_l), 2), np.nan)
    for i in range(len(fz_l)):
        parts = [
            (w, c)
            for w, c in ((fz_l[i], cop_l[i]), (fz_r[i], cop_r[i]))
            if w >= THRESHOLD_N
        ]
        if parts:
            weights = np.array([w for w, _ in parts])
            out[i] = weights @ np.array([c for _, c in parts]) / weights.sum()
    return out


def lead_pct(fz_l: np.ndarray, fz_r: np.ndarray, golfer: str) -> np.ndarray:
    """Lead share of the total load in percent, NaN below the threshold."""
    total = fz_l + fz_r
    lead = fz_l if golfer == "RH" else fz_r
    return np.where(total < THRESHOLD_N, np.nan, 100.0 * lead / total)


def corner_forces(fz_l: np.ndarray, fz_r: np.ndarray) -> np.ndarray:
    """Eight corner channels, each plate split evenly over its four corners."""
    return np.column_stack([fz_l / 4.0] * 4 + [fz_r / 4.0] * 4)


def default_arrays(golfer: str = "RH") -> dict[str, np.ndarray]:
    """The documented NPZ arrays for the default swing."""
    return {
        "t": T_S.copy(),
        "forces": corner_forces(FZ_L, FZ_R),
        "fz_L": FZ_L.copy(),
        "fz_R": FZ_R.copy(),
        "cop_L": COP_L_MM.copy(),
        "cop_R": COP_R_MM.copy(),
        "cop_C": combined_cop_mm(FZ_L, FZ_R, COP_L_MM, COP_R_MM),
        "lead_pct": lead_pct(FZ_L, FZ_R, golfer),
    }


def default_meta(golfer: str = "RH", swing_id: str = SWING_ID) -> dict[str, Any]:
    """The documented JSON object for the default swing."""
    return {
        "id": swing_id,
        "created": "2026-01-01T12:00:00",
        "golfer": golfer,
        "bw_n": 800.0,
        "t0_host": 1234.5,
        "events": dict(EVENTS),
        "metrics": {"anything": 1},
        "impact_source": "manual",
        "path": [[0.0, 0.0]],
    }


def write_swing(
    directory: Path,
    *,
    golfer: str = "RH",
    swing_id: str = SWING_ID,
    meta: dict[str, Any] | None = None,
    arrays: dict[str, Any] | None = None,
    raw_json: str | None = None,
) -> Path:
    """Write ``<id>.json`` and ``<id>.npz``; return the JSON path.

    ``meta`` / ``arrays`` replace the defaults wholesale when given;
    ``raw_json`` bypasses ``json.dumps`` to plant malformed text.
    """
    body = default_meta(golfer, swing_id) if meta is None else meta
    data = default_arrays(golfer) if arrays is None else arrays
    json_path = directory / f"{swing_id}.json"
    text = json.dumps(body, allow_nan=False) if raw_json is None else raw_json
    json_path.write_text(text, encoding="utf-8")
    np.savez(directory / f"{swing_id}.npz", **data)
    return json_path


def write_csv(
    directory: Path, *, header: str = CSV_HEADER, name: str = "swing.csv"
) -> Path:
    """Write the CSV export of the default arrays (4 decimals for t, else 2)."""
    arrays = default_arrays("RH")
    cols = [arrays["t"], arrays["lead_pct"], arrays["fz_L"], arrays["fz_R"]]
    for key in ("cop_L", "cop_R", "cop_C"):
        cols += [arrays[key][:, 0], arrays[key][:, 1]]
    cols += list(arrays["forces"].T)
    path = directory / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(header + "\n")
        writer = csv.writer(handle)
        for i in range(len(T_S)):
            row = []
            for j, col in enumerate(cols):
                decimals = 4 if j == 0 else 2
                row.append("" if np.isnan(col[i]) else f"{col[i]:.{decimals}f}")
            writer.writerow(row)
    return path


def npy_bytes(array: np.ndarray) -> bytes:
    """The ``.npy`` serialisation of ``array``."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def lying_npy_bytes(shape: tuple[int, ...]) -> bytes:
    """A ``.npy`` whose header claims ``shape`` but that holds no data."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": "<f8", "fortran_order": False, "shape": shape}
    )
    return buffer.getvalue()


def write_npz_members(
    path: Path,
    members: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
    flag_bits: int = 0,
) -> None:
    """Write a zip of raw ``members`` (``<key>.npy`` entries) at ``path``."""
    with zipfile.ZipFile(path, "w", compression) as archive:
        for key, payload in members.items():
            info = zipfile.ZipInfo(f"{key}.npy")
            info.compress_type = compression
            info.flag_bits |= flag_bits
            archive.writestr(info, payload)


def default_members() -> dict[str, bytes]:
    """The default arrays as raw ``.npy`` members."""
    return {key: npy_bytes(value) for key, value in default_arrays().items()}
