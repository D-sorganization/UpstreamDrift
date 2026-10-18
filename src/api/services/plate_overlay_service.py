"""Measured plate overlay for ``/analysis/ground-reaction`` (OF-07, #13087).

A request names a swing ``<id>.json`` *relative to* the server's plate-trial
root, configured with the ``UPSTREAMDRIFT_PLATE_TRIAL_ROOT`` environment
variable.  No root, no overlay: an arbitrary server path is never read.  A
path that is absolute, climbs out of the root, or passes through a symlink is
refused.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from src.api.models.plate_overlay import PlateOverlayRequest
from src.api.services.ground_reaction_service import compute_ground_reaction_plot
from src.shared.python.biomechanics.ground_reaction_plot_model import (
    series_from_payload,
)
from src.shared.python.force_plates import PlateTimeAlignment
from src.shared.python.force_plates.open_fp_format.declared_import import (
    DeclaredPlate,
    DeclaredReading,
    DeclaredStance,
)
from src.shared.python.force_plates.open_fp_format.overlay_file import (
    overlay_swing_file,
)

__all__ = [
    "PLATE_TRIAL_ROOT_ENV",
    "PlateRootNotConfiguredError",
    "compute_ground_reaction_overlay",
    "resolve_trial_path",
]

PLATE_TRIAL_ROOT_ENV = "UPSTREAMDRIFT_PLATE_TRIAL_ROOT"


class PlateRootNotConfiguredError(RuntimeError):
    """The server has no plate-trial root, so plate overlays are disabled."""


def _configured_root(root: str | Path | None) -> Path:
    value = root if root is not None else os.environ.get(PLATE_TRIAL_ROOT_ENV)
    if not value:
        raise PlateRootNotConfiguredError(
            f"plate overlays are disabled: set {PLATE_TRIAL_ROOT_ENV}"
        )
    resolved = Path(value).resolve()
    if not resolved.is_dir():
        raise PlateRootNotConfiguredError("plate-trial root is not a directory")
    return resolved


def _relative_parts(trial: str) -> tuple[str, ...]:
    if (
        not isinstance(trial, str)
        or not trial.strip()
        or "\\" in trial
        or "\0" in trial
    ):
        raise ValueError("trial must be a relative POSIX path")
    path = PurePosixPath(trial)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("trial must stay inside the plate-trial root")
    return path.parts


def resolve_trial_path(trial: str, root: str | Path | None = None) -> Path:
    """The file for ``trial`` under the allow-listed root.

    Raises ``PlateRootNotConfiguredError`` without a root and ``ValueError`` for
    an absolute path, a ``..`` or ``.`` component, a symlink anywhere on the
    path, a path that resolves outside the root, or a missing regular file.
    """
    base = _configured_root(root)
    current = base
    for part in _relative_parts(trial):
        current = current / part
        if current.is_symlink():
            raise ValueError("trial must not pass through a symbolic link")
    if not current.is_file() or not current.resolve().is_relative_to(base):
        raise ValueError("trial is not a file inside the plate-trial root")
    return current


def _stance(request: PlateOverlayRequest) -> DeclaredStance:
    return DeclaredStance(
        handedness=request.handedness,
        origin_m=tuple(request.origin_m),  # type: ignore[arg-type]
        target_dir=tuple(request.target_dir),  # type: ignore[arg-type]
    )


def compute_ground_reaction_overlay(
    run: Any, request: PlateOverlayRequest, *, root: str | Path | None = None
) -> dict[str, Any]:
    """Ground-reaction payload of ``run`` with the measured trial overlaid.

    A run with no ground reaction stays unavailable (nothing to overlay on).

    Raises ``PlateRootNotConfiguredError`` without a configured root and
    ``ValueError`` / ``TypeError`` for refused paths or invalid declarations.
    """
    path = resolve_trial_path(request.trial, root)
    payload = compute_ground_reaction_plot(run, impact_time_s=request.impact_time_s)
    if not payload["available"]:
        return payload
    alignment = PlateTimeAlignment(**request.alignment.model_dump())
    overlaid = overlay_swing_file(
        series_from_payload(payload),
        path,
        declared=DeclaredPlate(**request.declared.model_dump()).to_declarations(),
        placement=_stance(request).to_placement(frame_label="request"),
        alignment=alignment,
        options=DeclaredReading(**request.reading.model_dump()).to_options(
            request.coverage
        ),
    )
    return overlaid.to_dict()
