"""Export functionality for plots and figures.

Provides utilities to save matplotlib figures and plot data to multiple
formats (PNG, PDF, SVG, CSV, JSON) with consistent naming and metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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


def export_figure(
    fig: Figure,
    name: str,
    config: ExportConfig | None = None,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save a matplotlib ``Figure`` to one or more formats.

    Args:
        fig: The figure to export.
        name: Base filename (without extension).
        config: Export configuration (uses defaults if ``None``).
        formats: List of formats to export.  Defaults to the image and
            vector formats specified in *config*.

    Returns:
        List of paths to the saved files.
    """
    config = config or ExportConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = [config.image_format, config.vector_format]

    saved: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(
            str(path),
            format=fmt,
            dpi=config.dpi,
            transparent=config.transparent,
            bbox_inches=config.bbox_inches,
        )
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------


def export_plot_data(
    data: dict[str, Any],
    name: str,
    config: ExportConfig | None = None,
    fmt: str = "json",
) -> Path:
    """Export the raw data behind a plot to CSV or JSON.

    Args:
        data: Mapping of series names to numpy arrays or lists.
        name: Base filename (without extension).
        config: Export configuration.
        fmt: ``"json"`` or ``"csv"``.

    Returns:
        Path to the exported file.
    """
    config = config or ExportConfig()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{name}.{fmt}"

    if fmt == "json":
        payload: dict[str, Any] = {}
        if config.include_metadata:
            payload["_meta"] = {
                "exported_at": datetime.now(tz=UTC).isoformat(),
                "source": "UpstreamDrift",
            }
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
) -> dict[str, list[Path]]:
    """Export multiple named figures at once.

    Args:
        figures: ``{name: Figure}`` mapping.
        config: Shared export configuration.
        formats: Formats for each figure.

    Returns:
        ``{name: [paths]}`` mapping.
    """
    results: dict[str, list[Path]] = {}
    for name, fig in figures.items():
        results[name] = export_figure(fig, name, config=config, formats=formats)
    return results
