"""Golden-image snapshot tests for the plot_style matplotlib renderer (#4814).

For each ``(shape, color-scale)`` combination — five built-in shapes
(``sphere``, ``cube``, ``cross``, ``star``, ``diamond``) crossed with the
three ``ColorScale`` variants (``StaticColor``, ``PaletteColor``,
``DataDrivenColor``) — a 5-marker scene is rendered to a small PNG at
fixed DPI on the ``Agg`` backend and compared against a committed
fixture under ``tests/snapshots/plot_style/``.

Comparison metric: per-pixel RMS in 8-bit space, normalised by 255.
Tolerance: 0.5 % (i.e. ``rms / 255 < 0.005``).

On the very first run (or with ``PLOT_STYLE_REGEN_SNAPSHOTS=1``) the
baseline PNG is written and the test ``skip``s with a "regenerated"
message so a follow-up run validates the freshly committed baseline.

The CUSTOM_MESH variant is intentionally skipped — its rendered output
has too many free parameters (vertex / face geometry) to be a useful
snapshot at this layer.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.shared.python.plot_style import (  # noqa: E402
    ColormapId,
    DataChannel,
    DataDrivenColor,
    MarkerShape,
    MarkerStyle,
    MatplotlibMarkerRenderer,
    PaletteColor,
    StaticColor,
)

pytestmark = pytest.mark.integration

# Tolerance: 0.5 % of full 8-bit dynamic range.
RMS_TOLERANCE_FRAC = 0.005
SNAPSHOT_DPI = 100
SNAPSHOT_FIGSIZE = (3.0, 3.0)
N_MARKERS = 5

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots" / "plot_style"

REGEN = os.environ.get("PLOT_STYLE_REGEN_SNAPSHOTS", "").strip() not in (
    "",
    "0",
    "false",
    "False",
)

# Five evenly spaced 2-D positions in [0, 1].
_FIXED_POSITIONS = np.stack(
    [
        np.linspace(0.1, 0.9, N_MARKERS),
        np.full(N_MARKERS, 0.5),
    ],
    axis=1,
)


SHAPE_CASES: tuple[tuple[str, MarkerShape], ...] = (
    ("sphere", MarkerShape.SPHERE),
    ("cube", MarkerShape.CUBE),
    ("cross", MarkerShape.CROSS),
    ("star", MarkerShape.STAR),
    ("diamond", MarkerShape.DIAMOND),
)


def _static_fill() -> StaticColor:
    return StaticColor("#1f77b4")


def _palette_fill() -> PaletteColor:
    # Palette index 2 -> "tab10"[2] = green-ish.
    return PaletteColor(palette_name="tab10", palette_index=2)


def _data_driven_fill() -> DataDrivenColor:
    """A linear ramp 0 -> 1 across N_MARKERS via a 2-D channel."""
    values = np.linspace(0.0, 1.0, N_MARKERS, dtype=np.float64)
    # Shape (1, N_MARKERS) so the channel is per-(frame, marker).
    channel = DataChannel(name="ramp", values=values.reshape(1, N_MARKERS))
    return DataDrivenColor(
        channel=channel,
        colormap=ColormapId.VIRIDIS,
        vmin=0.0,
        vmax=1.0,
    )


COLOR_CASES: tuple[tuple[str, object], ...] = (
    ("static", _static_fill),
    ("palette", _palette_fill),
    ("data_driven", _data_driven_fill),
)


# ---------------------------------------------------------------------------
# Helpers (mirror the body_part_viz snapshot suite — see #4814 in CLAUDE.md).
# ---------------------------------------------------------------------------


def _render_to_array(fig: Figure) -> np.ndarray:
    """Render figure to an ``(H, W, 4)`` RGBA uint8 array."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(
        fig.canvas.buffer_rgba(),  # type: ignore[attr-defined]
        dtype=np.uint8,
    )
    return buf.reshape(int(height), int(width), 4).copy()


def _save_png(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SNAPSHOT_DPI)


def _load_png(path: Path) -> np.ndarray:
    img = plt.imread(str(path))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
    if img.shape[-1] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def _rms_normalised(a: np.ndarray, b: np.ndarray) -> float:
    """Issue-spec metric: ``sqrt(mean((a - b)**2)) / 255``."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)) / 255.0)


def _setup_axes() -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=SNAPSHOT_FIGSIZE, dpi=SNAPSHOT_DPI)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])  # type: ignore[arg-type,operator]
    ax.set_yticks([])  # type: ignore[arg-type,operator]
    ax.set_aspect("equal")
    return fig, ax


def _compare_or_regen(fig: Figure, golden_path: Path) -> None:
    if REGEN or not golden_path.is_file():
        _save_png(fig, golden_path)
        if not REGEN:
            pytest.skip(
                f"Generated missing golden {golden_path.name} on first run; "
                "rerun the test to compare."
            )
        return
    actual = _render_to_array(fig)
    expected = _load_png(golden_path)
    rms = _rms_normalised(actual, expected)
    assert rms < RMS_TOLERANCE_FRAC, (
        f"snapshot {golden_path.name} normalised RMS {rms:.4f} "
        f"exceeds tolerance {RMS_TOLERANCE_FRAC:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fig_ax():
    fig, ax = _setup_axes()
    yield fig, ax
    plt.close(fig)


@pytest.mark.parametrize(
    ("shape_key", "shape"),
    SHAPE_CASES,
    ids=[name for name, _ in SHAPE_CASES],
)
@pytest.mark.parametrize(
    ("color_key", "color_factory"),
    COLOR_CASES,
    ids=[name for name, _ in COLOR_CASES],
)
def test_shape_color_snapshot(
    fig_ax,
    shape_key: str,
    shape: MarkerShape,
    color_key: str,
    color_factory,
) -> None:
    """5 markers x (shape x color-scale) -> committed PNG.

    Cross-product yields 5 x 3 = 15 baseline snapshots.
    """
    fig, ax = fig_ax
    fill = color_factory()
    style = MarkerStyle(shape=shape, size_px=18.0, fill_color=fill)

    renderer = MatplotlibMarkerRenderer()
    if isinstance(fill, DataDrivenColor):
        # Resolve the data-driven RGBA explicitly so the snapshot is
        # deterministic regardless of the renderer's internal default.
        rgba = np.zeros((N_MARKERS, 4), dtype=np.float64)
        for i in range(N_MARKERS):
            rgba[i] = np.asarray(fill.resolve(0, i), dtype=np.float64)
    else:
        single = np.asarray(fill.resolve(0, 0), dtype=np.float64)
        rgba = np.broadcast_to(single, (N_MARKERS, 4)).copy()

    renderer.draw(ax, _FIXED_POSITIONS, style, rgba)

    fname = f"{shape_key}__{color_key}.png"
    _compare_or_regen(fig, SNAPSHOT_DIR / fname)
