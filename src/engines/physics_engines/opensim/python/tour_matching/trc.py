"""TRC marker files for OpenSim tools (InverseKinematicsTool, MocoTrack).

The writer emits the OpenSim TRC layout: a three-line header, the label row,
the X/Y/Z row, a blank line, then one tab-separated row per frame. Invalid
markers are written as explicit NaN cells, which OpenSim treats as missing
(empty cells are trimmed at row ends and rejected). The
reader restores the same :class:`TourCapture` (NaN and ``valid=False`` for
blanks) so exported files can be checked by roundtrip rather than trust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.tour_capture_contract import TourCapture

_ALLOWED_UNITS = ("m", "mm")


def write_trc(
    capture: TourCapture, path: Path, *, rate_hz: float, units: str = "m"
) -> Path:
    """Write ``capture`` as a TRC file; returns the path.

    ``rate_hz`` must be positive and consistent with the capture clock to
    within one sample; units must be metres (the capture unit) or millimetres.
    """
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("TRC data rate must be positive")
    if units not in _ALLOWED_UNITS:
        raise ValueError("TRC units must be 'm' or 'mm'")
    if capture.frames > 1:
        dt = float(np.mean(np.diff(capture.time_s)))
        if abs(dt * rate_hz - 1.0) > 1e-6:
            raise ValueError("TRC data rate disagrees with the capture clock")
    scale = 1000.0 if units == "mm" else 1.0
    path = Path(path)
    n = len(capture.labels)
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
        "OrigDataStartFrame\tOrigNumFrames",
        f"{rate_hz:.2f}\t{rate_hz:.2f}\t{capture.frames}\t{n}\t{units}\t"
        f"{rate_hz:.2f}\t1\t{capture.frames}",
        "Frame#\tTime\t" + "\t".join(f"{label}\t\t" for label in capture.labels),
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, n + 1)),
        "",
    ]
    rows = []
    for frame in range(capture.frames):
        cells = [str(frame + 1), f"{capture.time_s[frame]:.9f}"]
        for marker in range(n):
            if capture.valid[frame, marker]:
                cells.extend(
                    f"{value * scale:.6f}" for value in capture.points_m[frame, marker]
                )
            else:
                # Explicit NaN cells: OpenSim's TRCFileAdapter trims trailing
                # empty cells and then rejects the short row.
                cells.extend(("NaN", "NaN", "NaN"))
        rows.append("\t".join(cells))
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    return path


def read_trc(path: Path) -> TourCapture:
    """Parse a TRC file written by :func:`write_trc` (or OpenSim) into a capture."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 6 or not lines[0].startswith("PathFileType"):
        raise ValueError("Not a TRC file")
    meta = lines[2].split("\t")
    try:
        frames, n, units = int(meta[2]), int(meta[3]), meta[4]
    except (IndexError, ValueError) as error:
        raise ValueError("Malformed TRC metadata row") from error
    if units not in _ALLOWED_UNITS:
        raise ValueError("Unsupported TRC units")
    labels = tuple(cell for cell in lines[3].split("\t")[2:] if cell)
    if len(labels) != n:
        raise ValueError("TRC label count disagrees with metadata")
    scale = 0.001 if units == "mm" else 1.0
    data = [row.split("\t") for row in lines[5:] if row.strip()]
    if len(data) != frames:
        raise ValueError("TRC frame count disagrees with metadata")
    time = np.empty(frames)
    points = np.full((frames, n, 3), np.nan)
    for f, cells in enumerate(data):
        if len(cells) != 2 + 3 * n:
            raise ValueError("TRC data row has the wrong number of cells")
        time[f] = float(cells[1])
        for m in range(n):
            triple = cells[2 + 3 * m : 5 + 3 * m]
            if all(triple) and not any(v.lower() == "nan" for v in triple):
                points[f, m] = [float(value) * scale for value in triple]
    valid = np.isfinite(points).all(axis=2)
    return TourCapture(time, labels, points, valid)
