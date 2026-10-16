"""Read an OpenSim motion into per-level segmental kinematics (SPN-9d, #12759).

Inverse of :func:`~.drive.export_mot`.  A ``.mot``/``.sto`` coordinate table
(from SPN-9b driving or a marker-based IK run) is turned into a
:class:`~src.shared.python.spine.segmental.SegmentalKinematicsSeries`:

    theta_i = s_i * q_i   (s_i the level-map axis sign, +/-1; q_i the coordinate)
    T_i     = T_i0 @ T(R_ZXY(theta_i), 0)

with ``T_i0`` the neutral offset the inventory reads natively from the model
and angles in radians (``inDegrees=yes`` tables are converted).  Reading never
re-distributes anything: the result is exactly what the table states.

Provenance.  ``distribution_source`` is required and non-empty.  IK results
must say so (for example ``"ik:<file>"``) and are measured/IK kinematics, never
coupling outputs.  When the table header carries a ``distribution_source`` key
(written by :func:`~.drive.export_mot`) it must equal the argument.

Native verification (``verify_native=True``) realises each frame in the model
and requires the terminal-in-pelvis rotation to equal ``B @ compose @ E`` (the
chain identity of :mod:`.drive`) to ``1e-9``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    body_transform_in,
    set_coordinate_values,
)
from src.engines.physics_engines.opensim.python.spine.drive import (
    DISTRIBUTION_SOURCE_KEY,
    chain_bodies,
    _resolve_level_map,
    chain_offsets_from_model,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import _read_table
from src.engines.physics_engines.opensim.python.spine.inventory import (
    ModelLevelMap,
    SpineInventory,
    inventory_spine,
)
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.segmental import (
    LevelTransform,
    SegmentalKinematicsSeries,
    SegmentalPose,
)

__all__ = ["read_segmental_kinematics"]

_NATIVE_TOL = 1e-9


def _header_source(path: Path) -> str | None:
    table = import_opensim().TimeSeriesTable(str(path))
    if not table.hasTableMetaDataKey(DISTRIBUTION_SOURCE_KEY):
        return None
    return str(table.getTableMetaDataAsString(DISTRIBUTION_SOURCE_KEY))


def _check_inputs(motion: Path, model: Path, source: object) -> str:
    if not isinstance(source, str):
        raise TypeError("distribution_source must be a str")
    if not source.strip():
        raise ValueError(
            "distribution_source must be non-empty (e.g. 'ik:<file>' or "
            "'coupling:<law>@<version>')"
        )
    if not motion.is_file():
        raise ValueError(f"motion file not found: {motion.name}")
    if not model.is_file():
        raise ValueError(f"model file not found: {model.name}")
    header = _header_source(motion)
    if header is not None and header != source:
        raise ValueError(
            f"distribution_source {source!r} disagrees with the motion header "
            f"value {header!r}"
        )
    return source


def _level_angles(
    inventory: SpineInventory,
    entry: ModelLevelMap,
    values: dict[str, np.ndarray],
    scale: float,
) -> dict:
    """Per-level (n, 3) ISB angles [rad]; raises naming any missing column."""
    out = {}
    for level in inventory.levels:
        names = inventory.coordinate_names[level]
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(
                f"motion lacks coordinate column(s) {missing} for level {level.name}"
            )
        signs = np.array(entry.levels[level].axis_sign, dtype=float)
        out[level] = np.stack([values[name] for name in names], axis=1) * scale * signs
    return out


def _build_poses(inventory: SpineInventory, angles: dict, n: int) -> tuple:
    poses = []
    for frame in range(n):
        poses.append(
            SegmentalPose(
                tuple(
                    LevelTransform.from_isb_angles(
                        level, angles[level][frame], inventory.neutral[level]
                    )
                    for level in inventory.levels
                )
            )
        )
    return tuple(poses)


def _verify_native(
    model_path: Path,
    model_id: str,
    entry: ModelLevelMap,
    inventory: SpineInventory,
    values: dict[str, np.ndarray],
    poses: tuple,
) -> None:
    model = import_opensim().Model(str(model_path))
    state = model.initSystem()
    fixed = chain_offsets_from_model(model_path, model_id, level_map=entry)
    names = [n for lv in inventory.levels for n in inventory.coordinate_names[lv]]
    worst = 0.0
    for frame, pose in enumerate(poses):
        set_coordinate_values(model, state, {n: values[n][frame] for n in names})
        pelvis_body, terminal_body = chain_bodies(entry)
        native = body_transform_in(model, state, terminal_body, pelvis_body)
        expect = fixed.base_rotation @ pose.compose()[:3, :3] @ fixed.terminal_rotation
        worst = max(worst, float(np.max(np.abs(native[:3, :3] - expect))))
    if worst > _NATIVE_TOL:
        raise ValueError(
            f"native verification failed: composed pose differs from the model "
            f"by {worst:.3e} (tolerance {_NATIVE_TOL:.0e})"
        )


def read_segmental_kinematics(
    motion_path: str | Path,
    model_path: str | Path,
    model_id: str,
    *,
    distribution_source: str,
    level_map: ModelLevelMap | None = None,
    verify_native: bool = False,
) -> SegmentalKinematicsSeries:
    """Build per-level segmental kinematics from an OpenSim motion file.

    Preconditions: ``distribution_source`` is a non-empty ``str`` (``TypeError``
    / ``ValueError``); both files exist; ``model_id`` has a verified level map
    or ``level_map`` is given; the motion has a column for every mapped level
    coordinate (the error names the missing ones); a header
    ``distribution_source`` equals the argument (``ValueError``).

    Postconditions: ``time_s`` is the table time; level angles are radians with
    the level-map axis signs applied; every frame is marked valid (a reader
    cannot judge feasibility); the series carries ``distribution_source``.
    With ``verify_native`` the realised model rotation must equal the composed
    pose to ``1e-9`` or ``ValueError`` is raised.
    """
    motion, model = Path(motion_path), Path(model_path)
    source = _check_inputs(motion, model, distribution_source)
    entry = _resolve_level_map(model_id, level_map)
    inventory = inventory_spine(model, model_id, level_map=entry)
    wanted = {n for lv in inventory.levels for n in inventory.coordinate_names[lv]}
    time_s, values, in_degrees, _events = _read_table(motion, wanted)
    scale = float(np.pi / 180.0) if in_degrees else 1.0
    angles = _level_angles(inventory, entry, values, scale)
    poses = _build_poses(inventory, angles, time_s.shape[0])
    if verify_native:
        scaled = {k: v * scale for k, v in values.items()}
        _verify_native(model, model_id, entry, inventory, scaled, poses)
    series = SegmentalKinematicsSeries(
        time_s, poses, source, np.ones(time_s.shape[0], dtype=bool)
    )
    ensure(series.distribution_source == source, "distribution_source not carried")
    return series
