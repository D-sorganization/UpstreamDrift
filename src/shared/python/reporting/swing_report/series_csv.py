"""Series CSV files of a swing report package (#12327, epic #11419).

One file per advertised series: ``series/<quantity>__<engine>[__<channel>].csv``.
The provenance header reuses ``data_io.provenance.add_provenance_header``;
header lines start with ``#``. Missing samples are *empty cells*, never zero.
Withheld quantities hold no series (I5), so they can never reach a CSV.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
import re

from src.shared.python.data_io.provenance import (
    ProvenanceInfo,
    add_provenance_header,
)

from .contract import Quantity, Series, SwingReport

logger = logging.getLogger(__name__)

SERIES_DIR = "series"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_NOT_RECORDED = "not recorded"


class SeriesCsvError(ValueError):
    """Raised when a series CSV is malformed or disagrees with its report."""


def _safe(text: str) -> str:
    return _UNSAFE.sub("_", text).strip("._") or "x"


def plan_series_files(report: SwingReport) -> list[tuple[str, Quantity, Series]]:
    """Return ``(relative_path, quantity, series)`` for every series, in order.

    Postcondition: paths are unique, ASCII-safe and deterministic.
    """
    planned: list[tuple[str, Quantity, Series]] = []
    used: set[str] = set()
    for quantity in report.quantities:
        for series in quantity.series:
            parts = [quantity.quantity_id, series.engine]
            if series.channel:
                parts.append(series.channel)
            stem = "__".join(_safe(p) for p in parts)
            name, n = stem, 1
            while name.lower() in used:
                n += 1
                name = f"{stem}_{n}"
            used.add(name.lower())
            planned.append((f"{SERIES_DIR}/{name}.csv", quantity, series))
    return planned


def _header_info(report: SwingReport, series: Series) -> ProvenanceInfo:
    prov = report.provenance
    return ProvenanceInfo(
        timestamp_utc=prov.timestamp_utc,
        timestamp_local=prov.timestamp_utc,
        software_name=prov.software_name,
        software_version=prov.software_version,
        git_commit_sha=prov.git_commit_sha,
        git_is_dirty=bool(prov.git_is_dirty),
        engine_name=series.engine,
        run_id=series.run_id,
        python_version=_NOT_RECORDED,
        numpy_version=_NOT_RECORDED,
    )


def _cell(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _all_missing(quantity: Quantity, index: int) -> bool:
    return all(s.values[index] is None for s in quantity.series)


def series_csv_text(report: SwingReport, quantity: Quantity, series: Series) -> str:
    """Render one series (plus its spread bounds, when present) as CSV text.

    Spread bounds are left empty where *no* engine has a value at that sample,
    because the report stores a placeholder there that is not data.
    """
    if len(series.time_s) != len(series.values):
        raise ValueError("series time_s and values must have equal length")
    buffer = io.StringIO()
    add_provenance_header(buffer, _header_info(report, series))
    buffer.write(f"# Quantity: {quantity.quantity_id}\n")
    buffer.write(f"# Units: {series.units}\n# Frame: {series.frame}\n")
    unc = quantity.uncertainty
    header = ["time_s", "value"] + (["lower", "upper"] if unc else [])
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for i, (t, v) in enumerate(zip(series.time_s, series.values, strict=True)):
        row = [_cell(t), _cell(v)]
        if unc:
            gap = _all_missing(quantity, i)
            row += ["", ""] if gap else [_cell(unc.lower[i]), _cell(unc.upper[i])]
        writer.writerow(row)
    return buffer.getvalue()


def write_series_csv(
    path: Path, report: SwingReport, quantity: Quantity, series: Series
) -> bytes:
    """Write one series CSV to ``path`` and return its bytes."""
    data = series_csv_text(report, quantity, series).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _parse_cell(cell: str) -> float | None:
    return None if cell == "" else float(cell)


def parse_series_csv(
    data: bytes,
) -> tuple[
    list[float | None], list[float | None], list[float | None], list[float | None]
]:
    """Parse CSV bytes into ``(time_s, value, lower, upper)`` columns.

    ``lower``/``upper`` are empty lists when the file has no bounds columns.
    """
    lines = [ln for ln in data.decode("utf-8").splitlines() if not ln.startswith("#")]
    rows = list(csv.reader(lines))
    if not rows or rows[0][:2] != ["time_s", "value"]:
        raise SeriesCsvError("missing 'time_s,value' header row")
    has_bounds = rows[0][2:] == ["lower", "upper"]
    if not has_bounds and len(rows[0]) != 2:
        raise SeriesCsvError(f"unexpected columns {rows[0]}")
    cols: list[list[float | None]] = [[], [], [], []]
    for row in rows[1:]:
        if len(row) != len(rows[0]):
            raise SeriesCsvError(f"row has {len(row)} cells, expected {len(rows[0])}")
        for idx, cell in enumerate(row):
            cols[idx].append(_parse_cell(cell))
    return cols[0], cols[1], cols[2], cols[3]


def _bounds_match(
    csv_col: list[float | None], expected: tuple[float, ...], gaps: list[bool]
) -> bool:
    if len(csv_col) != len(expected):
        return False
    return all(
        (got is None and gap) or got == want
        for got, want, gap in zip(csv_col, expected, gaps, strict=True)
    )


def check_series_csv(data: bytes, quantity: Quantity, series: Series) -> None:
    """Raise ``SeriesCsvError`` unless the CSV equals ``series`` in the report."""
    time_s, values, lower, upper = parse_series_csv(data)
    if tuple(time_s) != series.time_s or tuple(values) != series.values:
        raise SeriesCsvError("series csv differs from report (time_s/value)")
    unc = quantity.uncertainty
    if unc is None:
        if lower or upper:
            raise SeriesCsvError("series csv differs from report (unexpected bounds)")
        return
    gaps = [_all_missing(quantity, i) for i in range(len(series.values))]
    if not (
        _bounds_match(lower, unc.lower, gaps) and _bounds_match(upper, unc.upper, gaps)
    ):
        raise SeriesCsvError("series csv differs from report (lower/upper)")
