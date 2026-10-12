"""Public-explorer export of the 3-D putt simulation (issue #12319).

The public explorer's Putting tab consumes the putt simulation through the
*mirror pattern* (``D-sorganization/public-web-management``): the canonical
artifact is produced here, in the private monorepo, and copied byte-for-byte
into the public mirror's vendored directory.  No new public package is
created; this module only writes and reads that one artifact.

The artifact is the ``ud.putt_scene/1`` record: a plain JSON document holding
the terrain grid, the simulated trajectory, the aim / target lines and the
summary metrics of a :class:`PuttScene`.  It is serialised with the repo's
RFC 8785 canonical JSON writer, so the same scene always yields the same
bytes (the mirror diff is then a real content diff).

Design by Contract:
    * :func:`export_putt_scene` validates the scene (finite numbers,
      consistent array shapes, named roll model) before touching the disk and
      raises :class:`PuttExportError` -- nothing is written on failure.
    * :func:`read_putt_export` fails closed on an unknown schema, a missing
      field, malformed JSON or a missing file.
    * Postcondition: ``read_putt_export(export_putt_scene(scene, p)) == scene``
      field for field.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bunkershot3d.provenance.hashing import canonical_json
from src.tools.putting_green_gui._scene_builder import PuttScene

logger = logging.getLogger(__name__)

SCHEMA_ID = "ud.putt_scene/1"

_UNITS = {"length": "m", "time": "s", "speed": "m/s"}
_FRAME = {
    "x": "down the nominal ball-to-cup line",
    "y": "lateral",
    "z": "elevation",
}

_ARRAY_FIELDS = (
    "grid_x",
    "grid_y",
    "grid_z",
    "trajectory_xyz",
    "start_xyz",
    "cup_xyz",
    "aim_line_xyz",
    "target_line_xyz",
)
_SCALAR_FIELDS = (
    "hole_radius_m",
    "final_distance_to_cup_m",
    "total_roll_m",
    "duration_s",
    "peak_break_m",
    "launch_speed_ms",
)
_REQUIRED_KEYS = (
    "schema",
    "units",
    "frame",
    "roll_model",
    "terrain",
    "trajectory",
    "lines",
    "green_size_m",
    "metrics",
)


class PuttExportError(ValueError):
    """Raised when a putt scene or export artifact violates its contract."""


def _validate_scene(scene: PuttScene) -> None:
    if not isinstance(scene, PuttScene):
        raise PuttExportError(f"scene must be a PuttScene; got {type(scene).__name__}")
    if not isinstance(scene.roll_model, str) or not scene.roll_model.strip():
        raise PuttExportError("roll_model must be a non-empty string")
    for name in _ARRAY_FIELDS:
        arr = np.asarray(getattr(scene, name), dtype=float)
        if not np.all(np.isfinite(arr)):
            raise PuttExportError(f"{name} must be finite (no NaN/inf)")
    for name in (*_SCALAR_FIELDS, "green_size"):
        values = np.atleast_1d(np.asarray(getattr(scene, name), dtype=float))
        if not np.all(np.isfinite(values)):
            raise PuttExportError(f"{name} must be finite (no NaN/inf)")
    ny, nx = np.asarray(scene.grid_y).size, np.asarray(scene.grid_x).size
    if np.asarray(scene.grid_z).shape != (ny, nx):
        raise PuttExportError(
            f"grid_z shape {np.asarray(scene.grid_z).shape} != (ny, nx) = ({ny}, {nx})"
        )
    traj = np.asarray(scene.trajectory_xyz)
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 1:
        raise PuttExportError(
            f"trajectory_xyz must be a non-empty (N, 3) array; got {traj.shape}"
        )
    if len(scene.roll_modes) != traj.shape[0]:
        raise PuttExportError(
            f"roll_modes length {len(scene.roll_modes)} != trajectory length "
            f"{traj.shape[0]}"
        )


def scene_to_record(scene: PuttScene) -> dict[str, Any]:
    """Convert a scene to the ``ud.putt_scene/1`` record (pure data).

    Args:
        scene: A scene from :func:`build_putt_scene`.

    Returns:
        A JSON-compatible mapping.

    Raises:
        PuttExportError: If the scene violates the export contract.
    """
    _validate_scene(scene)
    return {
        "schema": SCHEMA_ID,
        "units": dict(_UNITS),
        "frame": dict(_FRAME),
        "roll_model": scene.roll_model,
        "terrain": {
            "x": np.asarray(scene.grid_x, dtype=float).tolist(),
            "y": np.asarray(scene.grid_y, dtype=float).tolist(),
            "z": np.asarray(scene.grid_z, dtype=float).tolist(),
        },
        "trajectory": {
            "xyz": np.asarray(scene.trajectory_xyz, dtype=float).tolist(),
            "roll_modes": list(scene.roll_modes),
        },
        "lines": {
            "start": np.asarray(scene.start_xyz, dtype=float).tolist(),
            "cup": np.asarray(scene.cup_xyz, dtype=float).tolist(),
            "aim": np.asarray(scene.aim_line_xyz, dtype=float).tolist(),
            "target": np.asarray(scene.target_line_xyz, dtype=float).tolist(),
        },
        "green_size_m": [float(scene.green_size[0]), float(scene.green_size[1])],
        "metrics": {
            "holed": bool(scene.holed),
            **{name: float(getattr(scene, name)) for name in _SCALAR_FIELDS},
        },
    }


def export_putt_scene(scene: PuttScene, path: Path) -> Path:
    """Write the scene as a byte-deterministic mirror artifact.

    Args:
        scene: A scene from :func:`build_putt_scene`.
        path: Destination file; parent directories are created.

    Returns:
        ``path``.

    Raises:
        PuttExportError: If ``path`` is not a :class:`~pathlib.Path` or the
            scene violates the export contract (nothing is written).
    """
    if not isinstance(path, Path):
        raise PuttExportError(f"path must be a pathlib.Path; got {type(path).__name__}")
    text = canonical_json(scene_to_record(scene)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    logger.info("Wrote %s putt export to %s", SCHEMA_ID, path)
    return path


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise PuttExportError(f"export is missing required field {where}{key!r}")
    return mapping[key]


def read_putt_export(path: Path) -> PuttScene:
    """Read a ``ud.putt_scene/1`` artifact back into a :class:`PuttScene`.

    Raises:
        PuttExportError: On a missing file, malformed JSON, an unknown schema
            or a missing field.
    """
    if not isinstance(path, Path):
        raise PuttExportError(f"path must be a pathlib.Path; got {type(path).__name__}")
    if not path.is_file():
        raise PuttExportError(f"export file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PuttExportError(f"export is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PuttExportError("export root must be a JSON object")
    schema = data.get("schema")
    if schema != SCHEMA_ID:
        raise PuttExportError(f"unsupported schema {schema!r}; expected {SCHEMA_ID!r}")
    for key in _REQUIRED_KEYS:
        _require(data, key, "")
    terrain, traj = data["terrain"], data["trajectory"]
    lines, metrics = data["lines"], data["metrics"]
    size = data["green_size_m"]
    if not isinstance(size, list) or len(size) != 2:
        raise PuttExportError("green_size_m must be a two-element list")

    def arr(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=float)

    scalars = {
        name: float(_require(metrics, name, "metrics.")) for name in _SCALAR_FIELDS
    }
    for name, value in scalars.items():
        if not math.isfinite(value):
            raise PuttExportError(f"metrics.{name} must be finite")
    scene = PuttScene(
        grid_x=arr(_require(terrain, "x", "terrain.")),
        grid_y=arr(_require(terrain, "y", "terrain.")),
        grid_z=arr(_require(terrain, "z", "terrain.")),
        trajectory_xyz=arr(_require(traj, "xyz", "trajectory.")),
        roll_modes=[str(m) for m in _require(traj, "roll_modes", "trajectory.")],
        start_xyz=arr(_require(lines, "start", "lines.")),
        cup_xyz=arr(_require(lines, "cup", "lines.")),
        aim_line_xyz=arr(_require(lines, "aim", "lines.")),
        target_line_xyz=arr(_require(lines, "target", "lines.")),
        green_size=(float(size[0]), float(size[1])),
        holed=bool(_require(metrics, "holed", "metrics.")),
        roll_model=str(data["roll_model"]),
        **scalars,
    )
    _validate_scene(scene)
    return scene
