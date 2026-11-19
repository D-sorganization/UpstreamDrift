"""Plot the three-row swing wrist and forearm read-out from a motion bundle (#15275)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.shared.python.anatomical_angles.swing_wrist_readout import (  # noqa: E402
    DEFAULT_LEAD_SIDE,
    SwingWristReadout,
    SwingWristReadoutSettings,
    swing_wrist_readout,
)

_TITLE = "Swing Wrist and Forearm Read-Out (Model Joint Angles)"
_ROW_SPECS = (
    ("Wrist Flexion and Extension", "Flexion (+) / Extension (-) (deg)", "flexion_deg"),
    ("Wrist Radial and Ulnar Deviation", "Ulnar (+) / Radial (-) (deg)", "ulnar_deg"),
    (
        "Forearm Pronation and Supination",
        "Pronation (+) / Supination (-) (deg)",
        "pronation_deg",
    ),
)


@dataclass(frozen=True)
class PlotSettings:
    """Visual and side-mapping settings for :func:`plot_swing_wrist_readout`."""

    lead_side: str = DEFAULT_LEAD_SIDE
    figsize: tuple[float, float] = (11.0, 8.5)
    dpi: int = 150
    line_width: float = 2.0
    zero_line_width: float = 0.9
    grid_alpha: float = 0.3
    y_margin: float = 0.18
    legend_cols: int = 2
    legend_alpha: float = 0.92
    title_size: int = 13
    row_title_size: int = 11
    lead_color: str = "#1f77b4"
    trail_color: str = "#d62728"
    zero_color: str = "#666666"
    mid_divisor: int = 2


def load_bundle_readout(
    bundle_path: Path | str, *, lead_side: str = DEFAULT_LEAD_SIDE
) -> SwingWristReadout:
    """Load a ``.npz`` motion bundle and compute :class:`SwingWristReadout`."""
    path = Path(bundle_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bundle file not found: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        spec = json.loads(bytes(bundle["spec"]).decode("utf-8"))
        q_traj = np.asarray(bundle["reference_q"], dtype=float)
        manifest = json.loads(str(bundle["manifest"])) if "manifest" in bundle else {}
        raw = manifest.get("dt_s")
        dt_s = float(raw) if isinstance(raw, (int, float)) and raw > 0 else None
    cfg = SwingWristReadoutSettings(lead_side=lead_side, dt_s=dt_s)
    return swing_wrist_readout(spec, q_traj, settings=cfg)


def plot_swing_wrist_readout(
    readout: SwingWristReadout,
    output_path: Path | str,
    *,
    plot_settings: PlotSettings | None = None,
) -> Path:
    """Render the three-row wrist and forearm read-out figure to ``output_path``."""
    cfg = PlotSettings() if plot_settings is None else plot_settings
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x_vals = readout.time_s if readout.time_s is not None else readout.frame_index
    fig, axes = plt.subplots(
        len(_ROW_SPECS), 1, figsize=cfg.figsize, sharex=True, constrained_layout=True
    )
    fig.suptitle(_TITLE, fontsize=cfg.title_size, fontweight="bold")
    lead_lbl = f"Lead Wrist ({readout.lead_side})"
    trail_lbl = f"Trail Wrist ({readout.trail_side})"
    for ax, (title, ylabel, attr) in zip(axes, _ROW_SPECS, strict=True):
        ax.axhline(0.0, color=cfg.zero_color, ls="--", lw=cfg.zero_line_width)
        lead_s, trail_s = getattr(readout.lead, attr), getattr(readout.trail, attr)
        ax.plot(x_vals, lead_s, color=cfg.lead_color, lw=cfg.line_width, label=lead_lbl)
        ax.plot(
            x_vals, trail_s, color=cfg.trail_color, lw=cfg.line_width, label=trail_lbl
        )
        ax.set_title(title, fontsize=cfg.row_title_size, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.margins(y=cfg.y_margin)
        ax.grid(True, alpha=cfg.grid_alpha)
        ax.legend(loc="upper left", ncol=cfg.legend_cols, framealpha=cfg.legend_alpha)
    axes[-1].set_xlabel("Time (s)" if readout.time_s is not None else "Frame Index")
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return out


def summarize_readout(
    readout: SwingWristReadout,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return address, mid-frame, minimum and maximum degrees per side and series."""
    mid = int(len(readout.frame_index) // PlotSettings.mid_divisor)
    addr = readout.address_frame
    out: dict[str, dict[str, dict[str, float]]] = {}
    for role, arm in (("lead", readout.lead), ("trail", readout.trail)):
        side_map: dict[str, dict[str, float]] = {}
        for _, _, attr in _ROW_SPECS:
            a = getattr(arm, attr)
            side_map[attr] = {
                "address_deg": float(a[addr]),
                "mid_deg": float(a[mid]),
                "min_deg": float(np.nanmin(a)),
                "max_deg": float(np.nanmax(a)),
            }
        out[f"{role}_{arm.side}"] = side_map
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entry point: compute readout, write PNG, and return summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Path to the input .npz motion bundle")
    parser.add_argument("output", help="Path to the output .png plot")
    parser.add_argument("--lead-side", default=DEFAULT_LEAD_SIDE, choices=("L", "R"))
    args = parser.parse_args(argv)
    readout = load_bundle_readout(args.bundle, lead_side=args.lead_side)
    cfg = PlotSettings(lead_side=args.lead_side)
    plot_swing_wrist_readout(readout, args.output, plot_settings=cfg)
    return summarize_readout(readout)


if __name__ == "__main__":
    main()
