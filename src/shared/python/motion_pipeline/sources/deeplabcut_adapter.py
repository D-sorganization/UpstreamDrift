"""DeepLabCut output adapter (custom-keypoint 2D pose estimates).

DeepLabCut (DLC) writes per-video pose estimates as a pandas DataFrame
with a three-level column MultiIndex ``(scorer, bodyparts, coords)``
where ``coords`` is ``x`` / ``y`` / ``likelihood``. The DataFrame is
stored either:

- as HDF5 via ``DataFrame.to_hdf`` (key usually ``df_with_missing``),
  in either the pandas "fixed" or "table" on-disk layout, or
- as CSV with three header rows (``scorer`` / ``bodyparts`` / ``coords``).

Bodyparts are arbitrary user-defined names (e.g. ``clubhead``, ``hosel``,
``ball``), so the loaded :class:`KeypointSequence` uses
``schema_name="custom"`` with keypoint names preserved verbatim and the
DLC ``likelihood`` mapped to keypoint confidence.

Frames are row-indexed; DLC files carry no timestamps, so timestamps are
synthesized from the ``fps`` adapter option (default 30.0). When the row
index is a strictly increasing integer frame counter, those frame
numbers are honoured (``timestamp = frame_number / fps``); otherwise
row position is used.

The HDF5 reader is implemented directly on ``h5py`` so it does not
require the optional PyTables dependency; only the pandas layouts that
DeepLabCut itself produces are recognised. Multi-animal DLC files (an
extra ``individuals`` column level) are detected and rejected with a
descriptive error rather than silently misparsed. The ``deeplabcut``
package itself is never imported.
"""

from __future__ import annotations

import io
import math
import pickle
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pandas as pd

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.motion_pipeline.sources.base import (
    MocapSourceAdapter,
    SourceMetadata,
)
from src.shared.python.motion_pipeline.sources.registry import register_adapter

#: Coordinate level values DLC writes for every bodypart, in storage order.
_COORDS = ("x", "y", "likelihood")


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Unpickler that refuses to resolve any global.

    The pandas "table" layout stores column labels as a pickled payload
    of plain lists / tuples / strings, which needs no global lookup.
    Refusing ``find_class`` outright makes loading attacker-controlled
    files no more dangerous than reading their strings.
    """

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"Refusing to unpickle global {module}.{name} from HDF5 attribute"
        )


def _safe_unpickle(payload: bytes) -> object:
    """Unpickle *payload* while refusing any global object references."""
    return _NoGlobalsUnpickler(io.BytesIO(payload)).load()


def _decode(value: object) -> str:
    """Decode HDF5 byte strings to ``str`` (pass-through otherwise)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_h5_multi_index(group: h5py.Group, key: str) -> list[tuple[str, ...]]:
    """Reconstruct a pandas MultiIndex stored in fixed layout under *key*.

    Pandas stores each level's unique values as ``{key}_level{i}`` and the
    per-entry integer codes as ``{key}_label{i}``.
    """
    nlevels = int(group.attrs[f"{key}_nlevels"])
    levels: list[list[str]] = []
    codes: list[np.ndarray] = []
    for i in range(nlevels):
        levels.append([_decode(v) for v in group[f"{key}_level{i}"][()]])
        codes.append(np.asarray(group[f"{key}_label{i}"][()], dtype=int))
    length = len(codes[0])
    return [
        tuple(levels[lvl][codes[lvl][j]] for lvl in range(nlevels))
        for j in range(length)
    ]


def _read_h5_fixed(
    group: h5py.Group,
) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
    """Read a pandas fixed-layout frame: (columns, values, row_index)."""
    columns = _read_h5_multi_index(group, "axis0")
    nblocks = int(group.attrs.get("nblocks", 1))
    col_data: dict[tuple[str, ...], np.ndarray] = {}
    for b in range(nblocks):
        items = _read_h5_multi_index(group, f"block{b}_items")
        values = np.asarray(group[f"block{b}_values"][()], dtype=float)
        if values.ndim != 2 or values.shape[1] != len(items):
            raise ValueError(
                f"HDF5 block{b}_values shape {values.shape} does not match "
                f"{len(items)} block item columns"
            )
        for j, col in enumerate(items):
            col_data[col] = values[:, j]
    try:
        stacked = np.column_stack([col_data[c] for c in columns])
    except KeyError as e:
        raise ValueError(f"HDF5 frame is missing data for column {e.args[0]!r}") from e
    row_index = np.asarray(group["axis1"][()])
    return columns, stacked, row_index


def _read_h5_table(
    group: h5py.Group,
) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
    """Read a pandas table-layout frame: (columns, values, row_index)."""
    table = group["table"][()]
    names = table.dtype.names or ()
    if "index" not in names or "values_block_0" not in names:
        raise ValueError(
            "HDF5 table layout is missing 'index'/'values_block_0' fields; "
            f"found fields {list(names)}"
        )
    raw_axes = _safe_unpickle(bytes(group.attrs["non_index_axes"]))
    columns: list[tuple[str, ...]] | None = None
    if isinstance(raw_axes, list):
        for entry in raw_axes:
            axis_no, labels = entry
            if axis_no == 1:
                columns = [tuple(_decode(part) for part in col) for col in labels]
                break
    if columns is None:
        raise ValueError("HDF5 table layout carries no column axis metadata")
    values = np.asarray(table["values_block_0"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(
            f"HDF5 table values shape {values.shape} does not match "
            f"{len(columns)} columns"
        )
    return columns, values, np.asarray(table["index"])


@register_adapter
class DeepLabCutAdapter(MocapSourceAdapter):
    """DeepLabCut ``.h5`` / ``.csv`` pose-estimate adapter.

    Emits a 2D :class:`KeypointSequence` with ``schema_name="custom"``:
    keypoint names are the DLC bodyparts verbatim, and DLC ``likelihood``
    becomes keypoint confidence (clamped to ``[0, 1]``).

    Postconditions (verified via :meth:`load_checked`): at least one
    frame, monotonically increasing synthesized timestamps, and all
    finite coordinate values (keypoints with NaN coordinates are
    dropped from their frame; fully-NaN frames are skipped).
    """

    format_name = "deeplabcut"
    file_extensions = (".h5", ".hdf5", ".csv")

    DEFAULT_FPS: ClassVar[float] = 30.0

    def __init__(self, fps: float = DEFAULT_FPS) -> None:
        """Create an adapter synthesizing timestamps at *fps* Hz.

        Precondition: ``fps`` must be a finite positive number.
        """
        if not isinstance(fps, int | float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be a finite positive number, got {fps!r}")
        self.fps = float(fps)

    # ------------------------------------------------------------------
    # Sniffing

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Return True iff *path* looks like DeepLabCut output.

        Conservative by design: CSV files must show the DLC header rows
        (``scorer`` / ``bodyparts`` / ``coords`` first cells) and HDF5
        files must contain a pandas-layout frame whose column MultiIndex
        carries DLC coordinate labels. Plain trajectory CSVs are left to
        the generic CSV adapter.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix not in cls.file_extensions or not p.is_file():
            return False
        if suffix == ".csv":
            return cls._sniff_csv(p)
        return cls._sniff_hdf5(p)

    @staticmethod
    def _sniff_csv(path: Path) -> bool:
        """Check for the DLC 3-row (or multi-animal 4-row) CSV header."""
        try:
            with open(path, encoding="utf-8", newline="") as f:
                first_cells = [
                    f.readline().split(",")[0].strip().lower() for _ in range(4)
                ]
        except (OSError, UnicodeDecodeError):
            return False
        return (
            first_cells[0] == "scorer"
            and "bodyparts" in first_cells[1:3]
            and "coords" in first_cells[1:4]
        )

    @classmethod
    def _sniff_hdf5(cls, path: Path) -> bool:
        """Check for a pandas-layout DLC frame inside the HDF5 file."""
        try:
            with h5py.File(path, "r") as f:
                return cls._find_dlc_group(f) is not None
        except OSError:
            return False

    @staticmethod
    def _find_dlc_group(f: h5py.File) -> tuple[str, str] | None:
        """Return ``(group_name, layout)`` of the first DLC frame, if any.

        ``layout`` is ``"fixed"`` or ``"table"``.
        """
        for name, node in f.items():
            if not isinstance(node, h5py.Group):
                continue
            pandas_type = _decode(node.attrs.get("pandas_type", b""))
            if pandas_type == "frame":
                nlevels = int(node.attrs.get("axis0_nlevels", 0))
                if nlevels != 3:
                    continue
                coords_ds = node.get("axis0_level2")
                if coords_ds is None:
                    continue
                coords = {_decode(v).lower() for v in coords_ds[()]}
                if {"x", "y"} <= coords and "likelihood" in coords:
                    return name, "fixed"
            elif pandas_type == "frame_table":
                raw = bytes(node.attrs.get("non_index_axes", b""))
                if b"likelihood" in raw:
                    return name, "table"
        return None

    # ------------------------------------------------------------------
    # Parsing

    def _parse(self, path: Path) -> tuple[str, list[str], np.ndarray, np.ndarray]:
        """Parse *path* into ``(scorer, bodyparts, data, frame_numbers)``.

        ``data`` has shape ``(n_frames, n_bodyparts, 3)`` in x/y/likelihood
        order; ``frame_numbers`` is a strictly increasing integer array.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"DeepLabCut file not found: {p}")
        if p.suffix.lower() == ".csv":
            columns, values, row_index = self._parse_csv(p)
        else:
            columns, values, row_index = self._parse_hdf5(p)
        scorer, bodyparts, data = self._extract(columns, values, p)
        return scorer, bodyparts, data, self._frame_numbers(row_index)

    @staticmethod
    def _parse_csv(
        path: Path,
    ) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
        with open(path, encoding="utf-8", newline="") as f:
            second_cell = [f.readline().split(",")[0].strip().lower() for _ in range(2)]
        if second_cell[1] == "individuals":
            raise ValueError(
                f"DeepLabCut CSV {path} is a multi-animal file (an "
                "'individuals' header row); multi-animal DLC output is not "
                "supported by this adapter"
            )
        try:
            df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        except (OSError, ValueError) as e:
            raise ValueError(f"Malformed DeepLabCut CSV {path}: {e}") from e
        if df.empty:
            raise ValueError(f"DeepLabCut CSV {path} has no data rows")
        if df.columns.nlevels != 3:
            raise ValueError(
                f"DeepLabCut CSV {path} must have a 3-level "
                f"(scorer, bodyparts, coords) header; got {df.columns.nlevels} levels"
            )
        columns = [tuple(str(part) for part in col) for col in df.columns]
        try:
            values = df.to_numpy(dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"DeepLabCut CSV {path} contains non-numeric data cells: {e}"
            ) from e
        return columns, values, df.index.to_numpy()

    def _parse_hdf5(
        self, path: Path
    ) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
        try:
            with h5py.File(path, "r") as f:
                found = self._find_dlc_group(f)
                if found is None:
                    raise ValueError(
                        f"HDF5 file {path} contains no DeepLabCut-style pandas "
                        "frame (3-level scorer/bodyparts/coords columns)"
                    )
                name, layout = found
                group = f[name]
                if layout == "fixed":
                    return _read_h5_fixed(group)
                return _read_h5_table(group)
        except (OSError, KeyError, pickle.UnpicklingError) as e:
            raise ValueError(f"Malformed DeepLabCut HDF5 file {path}: {e}") from e

    @staticmethod
    def _extract(
        columns: list[tuple[str, ...]],
        values: np.ndarray,
        path: Path,
    ) -> tuple[str, list[str], np.ndarray]:
        """Regroup flat columns into ``(scorer, bodyparts, (n, nbp, 3))``."""
        bad = [c for c in columns if len(c) != 3]
        if bad:
            raise ValueError(
                f"DeepLabCut file {path} has non 3-tuple column labels "
                f"(e.g. {bad[0]!r}); expected (scorer, bodypart, coord)"
            )
        scorers = list(dict.fromkeys(c[0] for c in columns))
        if len(scorers) != 1:
            raise ValueError(
                f"DeepLabCut file {path} mixes multiple scorers {scorers}; "
                "expected exactly one"
            )
        bodyparts = list(dict.fromkeys(c[1] for c in columns))
        lookup = {(c[1], c[2].lower()): j for j, c in enumerate(columns)}
        n_frames = values.shape[0]
        data = np.empty((n_frames, len(bodyparts), 3), dtype=float)
        for b, bodypart in enumerate(bodyparts):
            for k, coord in enumerate(_COORDS):
                j = lookup.get((bodypart, coord))
                if j is None:
                    raise ValueError(
                        f"DeepLabCut file {path} is missing the {coord!r} column "
                        f"for bodypart {bodypart!r}"
                    )
                data[:, b, k] = values[:, j]
        return scorers[0], bodyparts, data

    @staticmethod
    def _frame_numbers(row_index: np.ndarray) -> np.ndarray:
        """Coerce the row index to strictly increasing integer frame numbers.

        Falls back to positional numbering when the index is non-numeric
        (e.g. image-path indices) or not strictly increasing.
        """
        try:
            numbers = np.asarray(row_index, dtype=float)
        except (TypeError, ValueError):
            return np.arange(len(row_index))
        as_int = numbers.astype(int)
        if (
            np.all(np.isfinite(numbers))
            and np.array_equal(as_int, numbers)
            and np.all(np.diff(as_int) > 0)
            and (len(as_int) == 0 or as_int[0] >= 0)
        ):
            return as_int
        return np.arange(len(row_index))

    # ------------------------------------------------------------------
    # Adapter API

    def metadata(self, path: Path) -> SourceMetadata:
        """Return metadata for *path* (fps is the adapter option)."""
        scorer, bodyparts, data, _ = self._parse(Path(path))
        return SourceMetadata(
            format_name=self.format_name,
            fps=self.fps,
            frame_count=int(data.shape[0]),
            unit_system="pixels",
            keypoint_schema="custom",
            notes=f"scorer={scorer}; bodyparts={len(bodyparts)}",
        )

    def load(
        self,
        path: Path,
        calibration: Calibration | None = None,
    ) -> KeypointSequence:
        """Load *path* into a 2D custom-schema :class:`KeypointSequence`.

        Keypoints with non-finite coordinates are dropped; frames with
        no finite keypoints are skipped. Raises :class:`ValueError` with
        file context for malformed inputs.
        """
        p = Path(path)
        scorer, bodyparts, data, frame_numbers = self._parse(p)
        frames: list[KeypointFrame] = []
        for i in range(data.shape[0]):
            keypoints: list[Keypoint] = []
            for b, bodypart in enumerate(bodyparts):
                x, y, likelihood = data[i, b]
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                confidence = (
                    max(0.0, min(1.0, float(likelihood)))
                    if math.isfinite(likelihood)
                    else 0.0
                )
                keypoints.append(
                    Keypoint(
                        x=float(x), y=float(y), confidence=confidence, name=bodypart
                    )
                )
            if not keypoints:
                continue
            frame_no = int(frame_numbers[i])
            frames.append(
                KeypointFrame(
                    timestamp=frame_no / self.fps,
                    keypoints=keypoints,
                    schema_name="custom",
                    frame_index=frame_no,
                )
            )
        if not frames:
            raise ValueError(
                f"DeepLabCut file {p} produced no usable frames "
                "(all keypoints missing or non-finite)"
            )
        return KeypointSequence(
            id=f"deeplabcut-{p.stem}",
            frames=frames,
            calibration=calibration,
            metadata={
                "source_file": str(p),
                "scorer": scorer,
                "bodyparts": bodyparts,
                "fps": self.fps,
            },
        )
