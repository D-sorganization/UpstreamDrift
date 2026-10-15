"""Golf humanoid motion reader for the segmental spine pipeline (SPN-9a).

The matched swings drive ``golf_humanoid.osim`` whose single ``back`` joint
(coordinates ``lumbar_extension``, ``lumbar_bending``, ``lumbar_rotation``;
frames ``pelvis_offset`` -> ``torso_offset``; child body ``torso``) yields the
global pelvis->thorax rotation that later stages distribute over the levels.

Convention (ADR-0055 item 4): the ``back`` joint carries TransformAxes
``(0 0 1)``, ``(1 0 0)``, ``(0 1 0)`` on the three lumbar coordinates, so the
joint rotation is the intrinsic ZXY product (see :func:`back_joint_rotation`)
and, with the offset-frame rotations ``R_po`` and ``R_to``::

    R_pelvis->torso = R_po @ R_ZXY(ext, bend, rot) @ R_to^T

``pelvis_to_thorax`` is evaluated natively (state realisation), not from this
formula; the formula documents and cross-checks the mapping. Frames: X
anterior, Y cranial, Z right; radians, metres, seconds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim

__all__ = [
    "GolfTrunkInput",
    "back_joint_rotation",
    "read_golf_humanoid_motion",
    "resample",
]

_LUMBAR = ("lumbar_extension", "lumbar_bending", "lumbar_rotation")
_PELVIS = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
)
_EVENT_PREFIX = "event_"
_ROT_TOL = 1e-9


def _frozen(value: object) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    out.setflags(write=False)
    return out


def _frozen_map(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return {str(k): _frozen(v) for k, v in values.items()}


def back_joint_rotation(
    lumbar_extension: float, lumbar_bending: float, lumbar_rotation: float
) -> np.ndarray:
    """Return ``Rz(ext) @ Rx(bend) @ Ry(rot)`` (radians) as a (3, 3) rotation.

    Pure (no OpenSim). This is the ``back`` joint rotation of ``golf_humanoid``
    for TransformAxes ``(0 0 1)``, ``(1 0 0)``, ``(0 1 0)`` (ADR-0055 item 4).

    Raises:
        TypeError: an angle is not a real number.
        ValueError: an angle is not finite.
    """
    angles = []
    for name, value in zip(
        _LUMBAR, (lumbar_extension, lumbar_bending, lumbar_rotation), strict=True
    ):
        try:
            angle = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number: {exc}") from exc
        if not np.isfinite(angle):
            raise ValueError(f"{name} must be finite")
        angles.append(angle)
    return Rotation.from_euler("ZXY", angles).as_matrix()


@dataclass(frozen=True)
class GolfTrunkInput:
    """Global trunk target and passthrough coordinates of one golf swing.

    Attributes:
        time_s: (n,) strictly increasing sample times [s].
        pelvis_to_thorax: (n, 3, 3) rotation of the torso frame expressed in the
            pelvis frame (native, including the joint offset frames).
        pelvis_pose_ground: (n, 4, 4) pelvis body pose in ground.
        other_coordinates: remaining model coordinates (arms, legs, ...) by name,
            each (n,), radians or metres, unchanged.
        events: event name -> time [s]; empty when the file carries none.
        source: ``{"file": basename, "sha256": hex}`` only (no absolute paths).

    Preconditions (``TypeError``/``ValueError``): consistent leading length,
    finite arrays, strictly increasing time, proper rotations. Postconditions:
    all arrays are copies and read-only.
    """

    time_s: np.ndarray
    pelvis_to_thorax: np.ndarray
    pelvis_pose_ground: np.ndarray
    other_coordinates: Mapping[str, np.ndarray]
    events: Mapping[str, float]
    source: Mapping[str, str]

    def __post_init__(self) -> None:
        time_s = _frozen(self.time_s)
        rots = _frozen(self.pelvis_to_thorax)
        poses = _frozen(self.pelvis_pose_ground)
        n = time_s.shape[0] if time_s.ndim == 1 else -1
        if n < 1:
            raise ValueError("time_s must be a non-empty 1-D array")
        if rots.shape != (n, 3, 3):
            raise ValueError(f"pelvis_to_thorax must have shape ({n}, 3, 3)")
        if poses.shape != (n, 4, 4):
            raise ValueError(f"pelvis_pose_ground must have shape ({n}, 4, 4)")
        if not (np.all(np.isfinite(time_s)) and np.all(np.isfinite(rots))):
            raise ValueError("time_s and pelvis_to_thorax must be finite")
        if not np.all(np.isfinite(poses)):
            raise ValueError("pelvis_pose_ground must be finite")
        if n > 1 and not np.all(np.diff(time_s) > 0.0):
            raise ValueError("time_s must be strictly increasing")
        gram = np.einsum("nij,nkj->nik", rots, rots)
        if not np.allclose(gram, np.eye(3), atol=_ROT_TOL, rtol=0):
            raise ValueError("pelvis_to_thorax must contain proper rotations")
        others = _frozen_map(self.other_coordinates)
        for name, arr in others.items():
            if arr.shape != (n,):
                raise ValueError(f"other_coordinates[{name!r}] must have shape ({n},)")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "pelvis_to_thorax", rots)
        object.__setattr__(self, "pelvis_pose_ground", poses)
        object.__setattr__(self, "other_coordinates", others)
        object.__setattr__(
            self, "events", {str(k): float(v) for k, v in self.events.items()}
        )
        object.__setattr__(
            self, "source", {str(k): str(v) for k, v in self.source.items()}
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate_name(label: str, names: set[str]) -> str | None:
    """Map a table column label to a model coordinate name (value columns only)."""
    if label in names:
        return label
    parts = label.strip("/").split("/")
    if len(parts) >= 2 and parts[-1] == "value" and parts[-2] in names:
        return parts[-2]
    return None


def _column(table: Any, index: int, n_rows: int) -> np.ndarray:
    vec = table.getDependentColumnAtIndex(index)
    return np.array([vec[r] for r in range(n_rows)], dtype=float)


def _read_table(
    path: Path, names: set[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], bool, dict[str, float]]:
    osim = import_opensim()
    table = osim.TimeSeriesTable(str(path))
    n_rows = int(table.getNumRows())
    in_degrees = (
        table.hasTableMetaDataKey("inDegrees")
        and table.getTableMetaDataAsString("inDegrees") == "yes"
    )
    time_s = np.array(
        [table.getIndependentColumn()[r] for r in range(n_rows)], dtype=float
    )
    values: dict[str, np.ndarray] = {}
    for idx, label in enumerate(table.getColumnLabels()):
        name = _coordinate_name(str(label), names)
        if name is not None:
            values[name] = _column(table, idx, n_rows)
    events: dict[str, float] = {}
    for key in table.getTableMetaDataKeys():
        if str(key).startswith(_EVENT_PREFIX):
            events[str(key)[len(_EVENT_PREFIX) :]] = float(
                table.getTableMetaDataAsString(str(key))
            )
    return time_s, values, in_degrees, events


def _is_rotational(coord: Any, osim: Any) -> bool:
    return bool(coord.getMotionType() == osim.Coordinate.Rotational)


def _mat33(rotation: Any) -> np.ndarray:
    return np.array([[rotation.get(i, j) for j in range(3)] for i in range(3)])


def _body_transform(body: Any, state: Any) -> np.ndarray:
    """Body pose in ground as a (4, 4) matrix."""
    transform = body.getTransformInGround(state)
    out = np.eye(4)
    out[:3, :3] = _mat33(transform.R())
    position = transform.p()
    out[:3, 3] = [position.get(i) for i in range(3)]
    return out


def _evaluate_frames(
    model: Any,
    values: Mapping[str, np.ndarray],
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = model.initSystem()
    coords = model.updCoordinateSet()
    pelvis = model.getBodySet().get("pelvis")
    torso = model.getBodySet().get("torso")
    rots = np.empty((n_rows, 3, 3))
    poses = np.empty((n_rows, 4, 4))
    for row in range(n_rows):
        for name, series in values.items():
            coords.get(name).setValue(state, float(series[row]), False)
        model.realizePosition(state)
        pelvis_pose = _body_transform(pelvis, state)
        torso_pose = _body_transform(torso, state)
        poses[row] = pelvis_pose
        rots[row] = pelvis_pose[:3, :3].T @ torso_pose[:3, :3]
    return rots, poses


def read_golf_humanoid_motion(motion_path: Path, model_path: Path) -> GolfTrunkInput:
    """Read a ``.sto``/``.mot`` golf swing and evaluate the trunk natively.

    Coordinates are matched to the model by name or by ``/.../<name>/value``
    state paths; ``inDegrees=yes`` rotational columns are converted to radians.
    Coordinates absent from the file keep their model defaults. Events are read
    from ``event_<name>`` table metadata when present (else empty).

    Preconditions: both files exist with suffix ``.sto``/``.mot``/``.osim``;
    the file holds the three lumbar coordinates and a strictly increasing time
    column (``FileNotFoundError``/``ValueError``).

    Postconditions: ``pelvis_to_thorax`` is the torso frame in the pelvis frame
    including offset frames; ``source`` has the basename and SHA-256 only.
    """
    motion, model_file = Path(motion_path), Path(model_path)
    for path in (motion, model_file):
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path.name}")
    if motion.suffix.lower() not in (".sto", ".mot"):
        raise ValueError(f"motion file must be .sto or .mot, got {motion.suffix!r}")
    osim = import_opensim()
    model = osim.Model(str(model_file))
    coord_set = model.getCoordinateSet()
    names = {coord_set.get(i).getName() for i in range(coord_set.getSize())}
    time_s, values, in_degrees, events = _read_table(motion, names)
    missing = [n for n in _LUMBAR if n not in values]
    if missing:
        raise ValueError(f"motion file lacks lumbar coordinates: {missing}")
    if in_degrees:
        for name, series in values.items():
            if _is_rotational(coord_set.get(name), osim):
                values[name] = np.radians(series)
    rots, poses = _evaluate_frames(model, values, time_s.shape[0])
    skip = set(_LUMBAR) | set(_PELVIS)
    others = {k: v for k, v in values.items() if k not in skip}
    result = GolfTrunkInput(
        time_s=time_s,
        pelvis_to_thorax=rots,
        pelvis_pose_ground=poses,
        other_coordinates=others,
        events=events,
        source={"file": motion.name, "sha256": _sha256(motion)},
    )
    ensure(
        result.pelvis_to_thorax.shape[0] == time_s.shape[0],
        "one rotation per sample",
        result.pelvis_to_thorax.shape,
    )
    return result


def resample(input: GolfTrunkInput, rate_hz: float) -> GolfTrunkInput:  # noqa: A002
    """Resample to a uniform rate: SLERP for rotations, linear for the rest.

    The pelvis pose rotation block is SLERPed and its translation interpolated
    linearly. Endpoints are preserved (the grid is ``t0 + k/rate`` with the
    final time appended if it is not on the grid).

    Raises:
        TypeError: ``input`` is not a :class:`GolfTrunkInput`.
        ValueError: ``rate_hz`` is not a positive finite number or the input has
            fewer than two samples.
    """
    if not isinstance(input, GolfTrunkInput):
        raise TypeError(f"input must be a GolfTrunkInput, got {type(input).__name__}")
    rate = float(rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"rate_hz must be positive and finite, got {rate_hz!r}")
    t = input.time_s
    if t.shape[0] < 2:
        raise ValueError("resample needs at least two samples")
    count = int(np.floor((t[-1] - t[0]) * rate + 1e-9)) + 1
    grid = t[0] + np.arange(count) / rate
    if t[-1] - grid[-1] > 1e-9:
        grid = np.append(grid, t[-1])
    grid = np.clip(grid, t[0], t[-1])
    grid[-1] = min(grid[-1], t[-1]) if grid[-1] - t[-1] < 1e-9 else grid[-1]
    rots = Slerp(t, Rotation.from_matrix(input.pelvis_to_thorax))(grid).as_matrix()
    pose_rot = Slerp(t, Rotation.from_matrix(input.pelvis_pose_ground[:, :3, :3]))(
        grid
    ).as_matrix()
    poses = np.tile(np.eye(4), (grid.shape[0], 1, 1))
    poses[:, :3, :3] = pose_rot
    for axis in range(3):
        poses[:, axis, 3] = np.interp(grid, t, input.pelvis_pose_ground[:, axis, 3])
    others = {k: np.interp(grid, t, v) for k, v in input.other_coordinates.items()}
    return GolfTrunkInput(
        time_s=grid,
        pelvis_to_thorax=rots,
        pelvis_pose_ground=poses,
        other_coordinates=others,
        events=input.events,
        source=input.source,
    )
