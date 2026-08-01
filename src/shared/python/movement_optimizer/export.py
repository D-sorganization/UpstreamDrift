# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Export helpers for animation GIFs, plot images, and JSON result data.

Design Principles:
    DBC  -- preconditions checked at function entry.
    DRY  -- common save logic factored into each function.
    LoD  -- callers pass only figure + path, no internal state.

Security:
    Export paths are validated to prevent path traversal attacks. When a
    ``base_dir`` is supplied, the resolved output path must reside inside
    that directory; absolute paths and ``..`` components that escape the
    base are rejected. When no ``base_dir`` is supplied (e.g. the GUI
    passes a path picked by the user via the OS file dialog), the path is
    still normalized and checked for embedded null bytes.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Version tag embedded in every JSON result file so that importers can
# detect format changes.  Bump when the on-disk layout changes incompatibly.
EXPORT_FORMAT_VERSION = "1.0"


def _validate_export_path(path: str, base_dir: str | os.PathLike[str] | None) -> Path:
    """Normalize ``path`` and validate it for safe export.

    When ``base_dir`` is provided, the resolved path must be located inside
    ``base_dir``. Otherwise only basic hygiene checks are applied (the path
    must be a non-empty string with no embedded null bytes).

    Args:
        path: Caller-supplied output path.
        base_dir: Optional containment directory. If given, ``path`` is
            resolved relative to it and rejected if it escapes the base.

    Returns:
        The resolved absolute :class:`pathlib.Path` to write to.

    Raises:
        ValueError: If ``path`` is empty, contains a null byte, or escapes
            ``base_dir`` when one is supplied.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")

    candidate = Path(path)

    if base_dir is None:
        # GUI flow: user already picked the location via the OS file dialog.
        # Normalize and return without containment enforcement.
        return candidate.resolve(strict=False)

    base = Path(base_dir).resolve(strict=False)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (base / candidate).resolve(strict=False)

    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path {path!r} escapes base directory {str(base)!r}") from exc
    return resolved


def export_animation_gif(
    fig: Figure,
    draw_frame_fn: Callable[[int], None],
    n_frames: int,
    path: str,
    fps: int = 15,
    base_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Export a matplotlib animation as a GIF file.

    Args:
        fig: Matplotlib Figure to animate.
        draw_frame_fn: Callable that redraws the figure for a given frame index.
        n_frames: Total number of animation frames (must be > 0).
        path: Output file path (should end in ``.gif``).
        fps: Frames per second for playback (default 15).
        base_dir: Optional directory the resolved ``path`` must be inside.

    Raises:
        ValueError: If ``n_frames <= 0``, ``fps <= 0``, or ``path`` fails
            validation (empty, null byte, or escapes ``base_dir``).
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    safe_path = _validate_export_path(path, base_dir)
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    # matplotlib stubs type FuncAnimation callback as (int, ...) -> Iterable
    # but Callable[[int], None] is compatible at runtime.
    anim = FuncAnimation(fig, cast(Any, draw_frame_fn), frames=n_frames, blit=False)
    writer = PillowWriter(fps=fps)
    # matplotlib stubs type AbstractMovieWriter narrowly; PillowWriter is
    # compatible at runtime.
    anim.save(str(safe_path), writer=cast(Any, writer))
    logger.info("Exported GIF animation to %s (%d frames, %d fps)", safe_path, n_frames, fps)


def export_plots_png(
    fig: Figure,
    path: str,
    base_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Export a matplotlib figure as a PNG image.

    Args:
        fig: Matplotlib Figure to save.
        path: Output file path.
        base_dir: Optional directory the resolved ``path`` must be inside.

    Raises:
        ValueError: If ``path`` fails validation.
    """
    safe_path = _validate_export_path(path, base_dir)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(safe_path), format="png", dpi=150, bbox_inches="tight")
    logger.info("Exported PNG to %s", safe_path)


def export_plots_pdf(
    fig: Figure,
    path: str,
    base_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Export a matplotlib figure as a PDF file.

    Args:
        fig: Matplotlib Figure to save.
        path: Output file path.
        base_dir: Optional directory the resolved ``path`` must be inside.

    Raises:
        ValueError: If ``path`` fails validation.
    """
    safe_path = _validate_export_path(path, base_dir)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(safe_path), format="pdf", bbox_inches="tight")
    logger.info("Exported PDF to %s", safe_path)


def export_result_json(
    data: dict[str, Any],
    path: str,
    base_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Export an optimization result dict to a JSON file.

    A ``"format_version"`` key is automatically injected so that
    :func:`movement_optimizer.import_results.import_result_from_json` can
    verify compatibility when the file is reloaded.

    Args:
        data: Arbitrary JSON-serializable dict containing result data.
        path: Output file path (should end in ``.json``).
        base_dir: Optional directory the resolved ``path`` must be inside.

    Raises:
        ValueError: If ``data`` is not a dict or ``path`` fails validation.
    """
    if not isinstance(data, dict):
        raise ValueError(f"data must be a dict, got {type(data).__name__}")
    safe_path = _validate_export_path(path, base_dir)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": EXPORT_FORMAT_VERSION, **data}
    safe_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Exported result JSON to %s", safe_path)
