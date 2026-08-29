"""Export functionality for plots and figures.

Provides utilities to save matplotlib figures and plot data to multiple
formats (PNG, PDF, SVG, CSV, JSON) with consistent naming and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.plotting.identity import PlotIdentity

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Matplotlib backends validate metadata keys per format and warn (but do not
# raise) on unrecognized ones. Map our generic timestamp/identity fields onto
# each format's accepted vocabulary so exports stay warning-free.
# PNG (via Pillow) accepts arbitrary keys, so identity fields are embedded
# directly there for straightforward readback (e.g. via PIL.Image.open().info).
_SOFTWARE_NAME = "UpstreamDrift"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExportConfig:
    """Settings for figure / data export.

    Attributes:
        output_dir: Root directory for exported files.
        image_format: Default raster format (``"png"``, ``"jpg"``).
        vector_format: Default vector format (``"pdf"``, ``"svg"``).
        dpi: Resolution for raster exports.
        transparent: Use transparent background.
        bbox_inches: Matplotlib bounding-box mode.
        include_metadata: Embed timestamp / source info in exports.
    """

    output_dir: str | Path = "exports"
    image_format: str = "png"
    vector_format: str = "pdf"
    dpi: int = 300
    transparent: bool = False
    bbox_inches: str = "tight"
    include_metadata: bool = True


# ---------------------------------------------------------------------------
# Figure export
# ---------------------------------------------------------------------------


def _build_savefig_metadata(
    fmt: str, identity: PlotIdentity | None, timestamp: datetime
) -> dict[str, Any]:
    """Build a ``fig.savefig(metadata=...)`` dict appropriate for *fmt*.

    Each matplotlib backend recognizes a different metadata key vocabulary
    (see the matplotlib docs for ``print_png``/``print_pdf``/``print_svg``).
    Standard keys (timestamp, software) are always populated; identity
    fields (engine/model/run) are included only when genuinely known.
    """
    fmt = fmt.lower()
    identity = identity or PlotIdentity()
    title = identity.label()

    if fmt == "pdf":
        # The PDF backend requires an actual datetime.datetime for
        # CreationDate, not a string (unlike PNG/SVG).
        meta: dict[str, Any] = {"Creator": _SOFTWARE_NAME, "CreationDate": timestamp}
        if title:
            meta["Title"] = title
            meta["Subject"] = title
        return meta

    if fmt == "svg":
        meta = {"Creator": _SOFTWARE_NAME, "Date": timestamp.isoformat()}
        if title:
            meta["Title"] = title
        return meta

    # PNG (and anything else routed through Pillow) accepts arbitrary text
    # chunks, so identity fields can be embedded directly for readback via
    # PIL.Image.open(path).info.
    meta = {"Software": _SOFTWARE_NAME, "Creation Time": timestamp.isoformat()}
    if title:
        meta["Title"] = title
    meta.update(identity.as_metadata_dict())
    return meta


def export_figure(
    fig: Figure,
    name: str,
    config: ExportConfig | None = None,
    formats: list[str] | None = None,
    identity: PlotIdentity | None = None,
) -> list[Path]:
    """Save a matplotlib ``Figure`` to one or more formats.

    Args:
        fig: The figure to export.
        name: Base filename (without extension).
        config: Export configuration (uses defaults if ``None``).
        formats: List of formats to export.  Defaults to the image and
            vector formats specified in *config*.
        identity: Optional engine/model/run identity. When
            ``config.include_metadata`` is True, this (plus a UTC
            timestamp and the UpstreamDrift software name) is embedded in
            each saved file's format-appropriate metadata.

    Returns:
        List of paths to the saved files.
    """
    if fig is None:
        raise ValueError("fig must be provided")
    config = config or ExportConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = [config.image_format, config.vector_format]

    timestamp = datetime.now(tz=timezone.utc)

    saved: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        savefig_kwargs: dict[str, Any] = {
            "format": fmt,
            "dpi": config.dpi,
            "transparent": config.transparent,
            "bbox_inches": config.bbox_inches,
        }
        if config.include_metadata:
            savefig_kwargs["metadata"] = _build_savefig_metadata(
                fmt, identity, timestamp
            )
        fig.savefig(str(path), **savefig_kwargs)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------


def export_plot_data(  # noqa: C901
    data: dict[str, Any],
    name: str,
    config: ExportConfig | None = None,
    fmt: str = "json",
    identity: PlotIdentity | None = None,
) -> Path:
    """Export the raw data behind a plot to CSV or JSON.

    Args:
        data: Mapping of series names to numpy arrays or lists.
        name: Base filename (without extension).
        config: Export configuration.
        fmt: ``"json"`` or ``"csv"``.
        identity: Optional engine/model/run identity. When
            ``config.include_metadata`` is True and ``fmt == "json"``, its
            populated fields are merged into the ``_meta`` block alongside
            the export timestamp and source name.

    Returns:
        Path to the exported file.
    """
    if data is None:
        raise ValueError("data must be provided")
    config = config or ExportConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.{fmt}"

    if fmt == "json":
        payload: dict[str, Any] = {}
        if config.include_metadata:
            meta: dict[str, str] = {
                "exported_at": datetime.now(tz=timezone.utc).isoformat(),
                "source": _SOFTWARE_NAME,
            }
            if identity is not None:
                meta.update(identity.as_metadata_dict())
            payload["_meta"] = meta
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                payload[key] = val.tolist()
            else:
                payload[key] = val
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    elif fmt == "csv":
        import csv

        # Flatten dict to columns
        columns: dict[str, list] = {}
        for key, val in data.items():
            arr = np.asarray(val)
            if arr.ndim == 1:
                columns[key] = arr.tolist()
            elif arr.ndim == 2:
                for col in range(arr.shape[1]):
                    columns[f"{key}_{col}"] = arr[:, col].tolist()

        max_rows = max(len(v) for v in columns.values()) if columns else 0
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(columns.keys()))
            for row in range(max_rows):
                writer.writerow(
                    [columns[k][row] if row < len(columns[k]) else "" for k in columns]
                )
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}")

    return path


# ---------------------------------------------------------------------------
# Batch export helper
# ---------------------------------------------------------------------------


def export_all_figures(
    figures: dict[str, Figure],
    config: ExportConfig | None = None,
    formats: list[str] | None = None,
    identity: PlotIdentity | None = None,
) -> dict[str, list[Path]]:
    """Export multiple named figures at once.

    Args:
        figures: ``{name: Figure}`` mapping.
        config: Shared export configuration.
        formats: Formats for each figure.
        identity: Optional engine/model/run identity applied to every
            figure's export metadata (see ``export_figure``).

    Returns:
        ``{name: [paths]}`` mapping.
    """
    if figures is None:
        raise ValueError("figures must be provided")
    results: dict[str, list[Path]] = {}
    for name, fig in figures.items():
        results[name] = export_figure(
            fig, name, config=config, formats=formats, identity=identity
        )
    return results
