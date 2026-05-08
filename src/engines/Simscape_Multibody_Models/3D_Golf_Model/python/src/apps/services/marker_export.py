"""Selective marker export to CSV / JSON / NPZ.

Produces long-format CSV (one row per ``(frame, marker)``), a JSON object
with a ``metadata`` block, or an NPZ archive (one array per marker plus a
``_meta`` JSON-encoded string).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.models import C3DDataModel

_VALID_FORMATS = ("csv", "json", "npz")
_VALID_COMPONENTS = ("x", "y", "z")


def _sanitize_csv_cell(value: Any) -> Any:
    """Defang Excel-style formula injection in CSV cells."""
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def _file_sha256(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_components(components: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(components, str):
        if components.lower() == "all":
            return _VALID_COMPONENTS
        components = (components,)
    out: list[str] = []
    for c in components:
        c_low = str(c).lower()
        if c_low not in _VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {_VALID_COMPONENTS} or 'all', got {c!r}"
            )
        if c_low not in out:
            out.append(c_low)
    if not out:
        raise ValueError("at least one component must be selected")
    return tuple(out)


def _resolve_frame_range(
    frame_range: tuple[int, int] | None, n_frames: int
) -> tuple[int, int]:
    if frame_range is None:
        return (0, max(0, n_frames - 1))
    if not isinstance(frame_range, tuple) or len(frame_range) != 2:
        raise ValueError("frame_range must be a (start, end) tuple")
    start, end = int(frame_range[0]), int(frame_range[1])
    if start < 0 or end < start or end >= n_frames:
        raise ValueError(
            f"frame_range {frame_range!r} out of bounds [0, {n_frames - 1}]"
        )
    return (start, end)


def _build_metadata_block(
    model: C3DDataModel,
    marker_names: Sequence[str],
    components: Sequence[str],
    frame_range: tuple[int, int],
    include_time: bool,
    include_residual: bool,
) -> dict[str, Any]:
    return {
        "file": os.path.basename(model.filepath) if model.filepath else None,
        "sha256": _file_sha256(model.filepath),
        "frame_range": list(frame_range),
        "n_frames": frame_range[1] - frame_range[0] + 1,
        "point_rate_hz": float(model.point_rate),
        "markers": list(marker_names),
        "components": list(components),
        "include_time": include_time,
        "include_residual": include_residual,
        "units": model.metadata.get("Units (POINT)", ""),
    }


def _column_index_for(component: str) -> int:
    return _VALID_COMPONENTS.index(component)


def export_markers(
    model: C3DDataModel,
    marker_names: Sequence[str],
    components: Sequence[str] | str,
    frame_range: tuple[int, int] | None,
    fmt: str,
    path: Path | str,
    *,
    include_time: bool = True,
    include_residual: bool = False,
) -> Path:
    """Export the named markers to ``path`` in the requested format.

    Args:
        model: Loaded C3D model.
        marker_names: Markers to export. Must all exist in ``model.markers``.
        components: Subset of ``("x", "y", "z")`` or the literal string ``"all"``.
        frame_range: ``(start, end)`` inclusive frame indices, or ``None`` for full.
        fmt: One of ``"csv"``, ``"json"``, ``"npz"``.
        path: Destination file.
        include_time: Include a ``time_s`` column (CSV/JSON) — NPZ stores time
            separately under ``_time``.
        include_residual: Include the per-frame residual where available.
    """
    if model is None:
        raise ValueError("model must be provided")
    if not marker_names:
        raise ValueError("at least one marker must be selected")
    fmt_low = str(fmt).lower()
    if fmt_low not in _VALID_FORMATS:
        raise ValueError(f"fmt must be one of {_VALID_FORMATS}, got {fmt!r}")
    missing = [n for n in marker_names if n not in model.markers]
    if missing:
        raise ValueError(f"unknown markers: {missing!r}")

    comps = _resolve_components(components)
    n_frames = (
        len(model.point_time)
        if model.point_time is not None
        else max((m.position.shape[0] for m in model.markers.values()), default=0)
    )
    if n_frames <= 0:
        raise ValueError("model has no frames to export")
    fr_range = _resolve_frame_range(frame_range, n_frames)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _build_metadata_block(
        model, marker_names, comps, fr_range, include_time, include_residual
    )

    if fmt_low == "csv":
        return _write_csv(
            model,
            marker_names,
            comps,
            fr_range,
            out_path,
            include_time,
            include_residual,
        )
    if fmt_low == "json":
        return _write_json(
            model,
            marker_names,
            comps,
            fr_range,
            out_path,
            metadata,
            include_time,
            include_residual,
        )
    return _write_npz(
        model,
        marker_names,
        comps,
        fr_range,
        out_path,
        metadata,
    )


def _iter_rows(
    model: C3DDataModel,
    marker_names: Sequence[str],
    components: Sequence[str],
    frame_range: tuple[int, int],
    include_time: bool,
    include_residual: bool,
):
    start, end = frame_range
    time_arr = model.point_time
    for fr in range(start, end + 1):
        t = float(time_arr[fr]) if (time_arr is not None and fr < len(time_arr)) else ""
        for name in marker_names:
            m = model.markers[name]
            if m.position.size == 0 or fr >= m.position.shape[0]:
                vals: list[Any] = ["" for _ in components]
                residual: Any = ""
            else:
                vals = [float(m.position[fr, _column_index_for(c)]) for c in components]
                if (
                    include_residual
                    and m.residuals is not None
                    and fr < len(m.residuals)
                ):
                    residual = float(m.residuals[fr])
                else:
                    residual = ""
            row: list[Any] = [fr]
            if include_time:
                row.append(t)
            row.append(name)
            row.extend(vals)
            if include_residual:
                row.append(residual)
            yield row


def _write_csv(
    model: C3DDataModel,
    marker_names: Sequence[str],
    components: Sequence[str],
    frame_range: tuple[int, int],
    path: Path,
    include_time: bool,
    include_residual: bool,
) -> Path:
    header: list[str] = ["frame"]
    if include_time:
        header.append("time_s")
    header.append("marker")
    header.extend(components)
    if include_residual:
        header.append("residual")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in _iter_rows(
            model,
            marker_names,
            components,
            frame_range,
            include_time,
            include_residual,
        ):
            writer.writerow([_sanitize_csv_cell(v) for v in row])
    return path


def _write_json(
    model: C3DDataModel,
    marker_names: Sequence[str],
    components: Sequence[str],
    frame_range: tuple[int, int],
    path: Path,
    metadata: dict[str, Any],
    include_time: bool,
    include_residual: bool,
) -> Path:
    rows: list[dict[str, Any]] = []
    start, end = frame_range
    time_arr = model.point_time
    for fr in range(start, end + 1):
        t = (
            float(time_arr[fr])
            if (time_arr is not None and fr < len(time_arr))
            else None
        )
        for name in marker_names:
            m = model.markers[name]
            entry: dict[str, Any] = {"frame": fr, "marker": name}
            if include_time:
                entry["time_s"] = t
            if m.position.size == 0 or fr >= m.position.shape[0]:
                for c in components:
                    entry[c] = None
            else:
                for c in components:
                    entry[c] = float(m.position[fr, _column_index_for(c)])
            if include_residual:
                if m.residuals is not None and fr < len(m.residuals):
                    entry["residual"] = float(m.residuals[fr])
                else:
                    entry["residual"] = None
            rows.append(entry)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "data": rows}, f, indent=2)
    return path


def _write_npz(
    model: C3DDataModel,
    marker_names: Sequence[str],
    components: Sequence[str],
    frame_range: tuple[int, int],
    path: Path,
    metadata: dict[str, Any],
) -> Path:
    start, end = frame_range
    n = end - start + 1
    arrays: dict[str, np.ndarray] = {}
    cols = [_column_index_for(c) for c in components]
    for name in marker_names:
        m = model.markers[name]
        out = np.full((n, len(cols)), np.nan, dtype=float)
        if m.position.size > 0:
            up_to = min(end + 1, m.position.shape[0])
            slice_n = max(0, up_to - start)
            if slice_n > 0:
                out[:slice_n, :] = m.position[start:up_to, cols]
        arrays[name] = out
    if model.point_time is not None:
        time_arr = np.asarray(model.point_time, dtype=float)
        end_t = min(end + 1, len(time_arr))
        slice_n = max(0, end_t - start)
        time_out = np.full(n, np.nan, dtype=float)
        if slice_n > 0:
            time_out[:slice_n] = time_arr[start:end_t]
        arrays["_time"] = time_out
    arrays["_meta"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)
    if path.suffix.lower() != ".npz":
        return path.with_suffix(path.suffix + ".npz")
    return path


__all__ = ["export_markers"]
