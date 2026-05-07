"""High-level :func:`visualize_fit` entry point per issue #4133.

Bundles the three canonical figures plus the Meshcat overlay URL into a
single dict so callers (e.g. ``fit_swing_pinocchio``) do not need to know
about the per-view modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .._types import ClubTargetLike, FitResult
from .error_timecourse import plot_error_timecourse
from .fit_quality_card import plot_fit_quality_card
from .meshcat_overlay import meshcat_overlay
from .trajectory_overlay import plot_trajectory_overlay

logger = logging.getLogger(__name__)


def visualize_fit(
    target: ClubTargetLike,
    result: FitResult,
    *,
    out_dir: Path | None = None,
    interactive: bool = False,
) -> dict[str, Path | str]:
    """Emit the three canonical figures + (optional) Meshcat overlay.

    Args:
        target: Measured trajectory.
        result: Pinocchio fit result.
        out_dir: Output directory for PNG figures. If ``None``, no files
            are written and the returned dict is empty.
        interactive: If ``True``, also publish a Meshcat overlay and add
            its URL under the key ``"meshcat_url"``. Skipped silently if
            meshcat cannot be imported.

    Returns:
        A dict mapping view name → output ``Path`` for figures, plus
        ``"meshcat_url"`` → ``str`` when ``interactive=True``.
    """
    artefacts: dict[str, Path | str] = {}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib  # noqa: F401  (presence check)
        except ImportError as exc:  # pragma: no cover - defensive
            logger.warning("matplotlib unavailable, skipping 2D figures: %s", exc)
        else:
            traj_path = out_dir / "trajectory_overlay.png"
            err_path = out_dir / "error_timecourse.png"
            card_path = out_dir / "fit_quality_card.png"

            import matplotlib.pyplot as plt

            fig1 = plot_trajectory_overlay(target, result, out_path=traj_path)
            plt.close(fig1)
            fig2 = plot_error_timecourse(target, result, out_path=err_path)
            plt.close(fig2)
            fig3 = plot_fit_quality_card(target, result, out_path=card_path)
            plt.close(fig3)

            artefacts["trajectory_overlay"] = traj_path
            artefacts["error_timecourse"] = err_path
            artefacts["fit_quality_card"] = card_path

    if interactive:
        try:
            artefacts["meshcat_url"] = meshcat_overlay(target, result)
        except ImportError as exc:
            logger.info("Meshcat unavailable, skipping 3D overlay: %s", exc)

    return artefacts
