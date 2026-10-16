"""Drive a coupled segmental spine model from golf trunk input (SPN-9b, #12757).

Pipeline: :class:`GolfTrunkInput` (SPN-9a) -> per-frame distribution of the
global pelvis->thorax rotation over the spinal levels
(:func:`~src.shared.python.spine.distribution.distribute_series_detailed`,
SPN-2b) -> level coordinate table + pelvis free-joint coordinates -> OpenSim
``.mot`` (radians, ``inDegrees=no``).

Frames and the chain identity (ADR-0055).  For a model with levels
``1..n`` (caudal first) the terminal body pose in the pelvis frame is

    R_total(phi) = B @ R_1 @ ... @ R_n @ E,

with ``R_i = T_i0 @ R(theta_i)`` (``T_i0`` the inventory neutral offset),
``B`` the rotation of the first inferior body in the pelvis body frame and
``E`` the rotation of the terminal body in the last superior vertebra frame.
``B`` and ``E`` are read natively from the model at zero coordinates by
:func:`chain_offsets_from_model` and enter the solve through
:class:`~src.shared.python.spine.distribution.ChainOffsets`.

Frame-alignment assumption (stated, not hidden).  The golf target is the
``golf_humanoid`` torso frame in the pelvis frame.  The segmental terminal
body frame (e.g. ``thorax_proxy``) is assumed to coincide with that torso
frame: ``R_target = R_pelvis->torso @ A`` with ``A = I`` by default.  If the
terminal frame is rotated by a known fixed amount ``A`` (the terminal frame
expressed in the golf torso frame) pass it as ``DriveOptions.torso_alignment``.
The pelvis body frames are assumed to coincide the same way.  The
alignment is a modelling choice and is not validated here: a wrong ``A`` shows
up only as a biased distribution, not as a residual.  ADR-0055 item 2 forbids
assuming ``A = I`` for a reported result, so the study layer
(:mod:`spine.study`) requires a calibrated ``A`` with a provenance string and
records an unsupplied one as ``assumed_identity``, never ``pass``.

Pelvis coordinates.  The pelvis pose in ground is encoded as intrinsic Euler
angles of the model's ground-pelvis joint (``PelvisCoordinates.sequence``,
unwrapped over time) plus the translation of the pelvis origin; the joint's
parent/child offset frames are assumed to be identity.

The coupling law is never altered: an unreachable target gives a positive
geodesic residual that is reported per frame, frames beyond ``tolerance_rad``
are flagged invalid, and nothing is rescaled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    body_transform_in,
    set_coordinate_values,
)
from src.engines.physics_engines.opensim.python.spine.coupler_extraction import (
    extract_coupling_law,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import GolfTrunkInput
from src.engines.physics_engines.opensim.python.spine.inventory import (
    ModelLevelMap,
    SpineInventory,
    inventory_spine,
    load_level_maps,
)
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.coupling import CouplingLaw
from src.shared.python.spine.distribution import (
    ChainOffsets,
    distribute_series_detailed,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import SegmentalKinematicsSeries

__all__ = [
    "DISTRIBUTION_SOURCE_KEY",
    "FIXTURE_PELVIS",
    "GOLF_PELVIS",
    "DriveOptions",
    "DriveResult",
    "PelvisCoordinates",
    "chain_offsets_from_model",
    "drive_segmental_model",
    "export_mot",
]

_ROT_TOL = 1e-9

#: ``.mot`` header key carrying the per-level ``distribution_source`` label.
DISTRIBUTION_SOURCE_KEY = "distribution_source"


@dataclass(frozen=True)
class PelvisCoordinates:
    """Names and Euler sequence of a model's ground-pelvis joint coordinates.

    ``rotation`` are listed in the order of ``sequence`` (intrinsic, upper
    case as in :class:`scipy.spatial.transform.Rotation`), ``translation`` is
    the x, y, z position of the pelvis origin in ground.
    """

    rotation: tuple[str, str, str]
    translation: tuple[str, str, str]
    sequence: str

    def __post_init__(self) -> None:
        if len(self.rotation) != 3 or len(self.translation) != 3:
            raise ValueError("rotation and translation need three names each")
        if len(set(self.rotation + self.translation)) != 6:
            raise ValueError("pelvis coordinate names must be distinct")
        if sorted(self.sequence) != ["X", "Y", "Z"]:
            raise ValueError(
                f"sequence must be a permutation of XYZ, got {self.sequence!r}"
            )


#: ``golf_humanoid`` ground-pelvis CustomJoint: tilt Z, list X, rotation Y.
GOLF_PELVIS = PelvisCoordinates(
    ("pelvis_tilt", "pelvis_list", "pelvis_rotation"),
    ("pelvis_tx", "pelvis_ty", "pelvis_tz"),
    "ZXY",
)
#: Synthetic fixture ground-pelvis FreeJoint: body-fixed X, Y, Z.
FIXTURE_PELVIS = PelvisCoordinates(
    ("pelvis_rx", "pelvis_ry", "pelvis_rz"),
    ("pelvis_tx", "pelvis_ty", "pelvis_tz"),
    "XYZ",
)


def _proper_rotation(value: object, name: str) -> np.ndarray:
    rot = np.array(value, dtype=float, copy=True)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
        raise ValueError(f"{name} must be a finite (3, 3) proper rotation")
    ortho = np.allclose(rot.T @ rot, np.eye(3), atol=_ROT_TOL, rtol=0)
    if not ortho or abs(float(np.linalg.det(rot)) - 1.0) > _ROT_TOL:
        raise ValueError(f"{name} must be a proper rotation (orthonormal, det +1)")
    rot.setflags(write=False)
    return rot


@dataclass(frozen=True)
class DriveOptions:
    """Optional settings of :func:`drive_segmental_model`.

    Attributes:
        tolerance_rad: frames with a larger geodesic residual are flagged
            invalid (never repaired); ``None`` flags only unconverged solves.
        pelvis: pelvis coordinate encoding of the target model.
        torso_alignment: ``A`` in ``R_target = R_pelvis->torso @ A`` (terminal
            frame expressed in the golf torso frame); identity when ``None``.
        bounds: forwarded to the distribution (independent-name -> (lo, hi)).
        level_map: level map of the model; defaults to the registry entry of
            ``model_id`` (``src/config/spine_model_level_maps.json``).

    Preconditions (``ValueError``/``TypeError``): ``tolerance_rad`` finite and
    positive; ``torso_alignment`` a proper rotation; ``pelvis`` a
    :class:`PelvisCoordinates`.
    """

    tolerance_rad: float | None = None
    pelvis: PelvisCoordinates = GOLF_PELVIS
    torso_alignment: np.ndarray | None = field(default=None, compare=False)
    bounds: Mapping[str, tuple[float, float]] | None = None
    level_map: ModelLevelMap | None = None

    def __post_init__(self) -> None:
        tol = self.tolerance_rad
        if tol is not None and not (np.isfinite(tol) and tol > 0.0):
            raise ValueError(f"tolerance_rad must be finite and > 0, got {tol}")
        if not isinstance(self.pelvis, PelvisCoordinates):
            raise TypeError("pelvis must be a PelvisCoordinates")
        if self.torso_alignment is not None:
            align = _proper_rotation(self.torso_alignment, "torso_alignment")
            object.__setattr__(self, "torso_alignment", align)


def _frozen(value: object) -> np.ndarray:
    out = np.array(value, dtype=float, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class DriveResult:
    """Outcome of driving a segmental model; arrays are read-only.

    Attributes:
        kinematics: per-level poses with ``distribution_source`` and ``valid``.
        residual_rad: (n,) geodesic distance target -> realised rotation.
        phi: (n, n_phi) independent coordinates of the law per frame.
        independent_names: names of the ``phi`` columns.
        coordinates: model coordinate name -> (n,) value (radians, metres):
            the level coordinates (inventory names, axis-sign applied) and the
            six pelvis coordinates.
        model_id / model_sha256 / law_id / law_version / input_source:
            provenance (basenames and hashes only).
    """

    kinematics: SegmentalKinematicsSeries
    residual_rad: np.ndarray
    phi: np.ndarray
    independent_names: tuple[str, ...]
    coordinates: Mapping[str, np.ndarray]
    model_id: str
    model_sha256: str
    law_id: str
    law_version: str
    input_source: Mapping[str, str]

    def __post_init__(self) -> None:
        time_s = self.kinematics.time_s
        n = time_s.shape[0]
        resid = _frozen(self.residual_rad)
        if resid.shape != (n,):
            raise ValueError(f"residual_rad must have shape ({n},)")
        phi = _frozen(self.phi)
        if phi.shape != (n, len(self.independent_names)):
            raise ValueError("phi must have shape (n, len(independent_names))")
        coords = {str(k): _frozen(v) for k, v in self.coordinates.items()}
        if any(v.shape != (n,) for v in coords.values()):
            raise ValueError(f"every coordinate series must have shape ({n},)")
        object.__setattr__(self, "residual_rad", resid)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "input_source", dict(self.input_source))

    @property
    def time_s(self) -> np.ndarray:
        """Sample times [s]."""
        return self.kinematics.time_s

    @property
    def valid(self) -> np.ndarray:
        """Per-frame validity mask (converged and within tolerance)."""
        return self.kinematics.valid

    @property
    def max_residual_rad(self) -> float:
        """Largest per-frame geodesic residual [rad]."""
        return float(np.max(self.residual_rad))


# --------------------------------------------------------------- native I/O


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chain_bodies(entry: ModelLevelMap) -> tuple[str, str]:
    """``(pelvis_body, terminal_body)`` of a level map.

    Raises:
        ValueError: either body name is missing.
    """
    if not entry.pelvis_body or not entry.terminal_body:
        raise ValueError("level map lacks pelvis/terminal bodies")
    return entry.pelvis_body, entry.terminal_body


def _resolve_level_map(model_id: str, level_map: ModelLevelMap | None):
    if level_map is not None:
        entry = level_map
    else:
        maps = load_level_maps()
        if model_id not in maps:
            raise ValueError(f"unknown model_id {model_id!r}; known {sorted(maps)}")
        entry = maps[model_id]
    if not entry.verified or not entry.levels:
        raise ValueError(f"level map for {model_id!r} is unverified")
    if not entry.pelvis_body or not entry.terminal_body:
        raise ValueError(f"level map for {model_id!r} lacks pelvis/terminal bodies")
    return entry


def _joint_base_bodies(model: object, joint_name: str) -> tuple[str, str]:
    """Names of the (parent, child) bodies a joint connects."""
    joints = model.getJointSet()  # type: ignore[attr-defined]
    joint = joints.get(joint_name)
    parent = joint.getParentFrame().findBaseFrame()
    child = joint.getChildFrame().findBaseFrame()
    return str(parent.getName()), str(child.getName())


def _chain_offsets(model: object, entry: ModelLevelMap) -> ChainOffsets:
    ordered = sorted(entry.levels, key=lambda lv: lv.ordinal)
    first_inferior, _ = _joint_base_bodies(model, entry.levels[ordered[0]].joint)
    _, last_superior = _joint_base_bodies(model, entry.levels[ordered[-1]].joint)
    state = model.initSystem()  # type: ignore[attr-defined]
    set_coordinate_values(model, state, {})
    pelvis_body, terminal_body = chain_bodies(entry)
    base = body_transform_in(model, state, first_inferior, pelvis_body)
    term = body_transform_in(model, state, terminal_body, last_superior)
    return ChainOffsets(base[:3, :3], term[:3, :3])


def chain_offsets_from_model(
    model_path: str | Path, model_id: str, *, level_map: ModelLevelMap | None = None
) -> ChainOffsets:
    """Read the fixed base/terminal rotations ``B`` and ``E`` of a model.

    ``B`` is the first inferior body in the pelvis body frame and ``E`` the
    terminal body in the last superior vertebra frame, both evaluated at zero
    coordinates (the connections are assumed rigid).

    Raises:
        ValueError: missing file, unknown or unverified ``model_id``.
    """
    path = Path(model_path)
    if not path.is_file():
        raise ValueError(f"model file not found: {path.name}")
    entry = _resolve_level_map(model_id, level_map)
    return _chain_offsets(import_opensim().Model(str(path)), entry)


# --------------------------------------------------------------- the drive


def _law_levels(law: CouplingLaw) -> set[SpinalLevel]:
    return set(law.expand(np.zeros(len(law.independent_names()))))


def _pelvis_columns(poses: np.ndarray, enc: PelvisCoordinates) -> dict[str, np.ndarray]:
    # Copy: SciPy < 1.15 ``Rotation`` rejects the read-only pose arrays.
    rotation = Rotation.from_matrix(np.array(poses[:, :3, :3]))
    euler = rotation.as_euler(cast(Any, enc.sequence))
    angles: np.ndarray = np.unwrap(euler, axis=0)
    out = {name: angles[:, k] for k, name in enumerate(enc.rotation)}
    out.update({name: poses[:, k, 3] for k, name in enumerate(enc.translation)})
    return out


def _level_columns(
    results: tuple, inventory: SpineInventory, entry: ModelLevelMap
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for level in inventory.levels:
        angles = np.stack([r.level_angles[level] for r in results])
        signs = entry.levels[level].axis_sign
        for k, name in enumerate(inventory.coordinate_names[level]):
            out[name] = signs[k] * angles[:, k]
    return out


def _check_drive_inputs(golf_input: object, model_path: Path) -> None:
    if not isinstance(golf_input, GolfTrunkInput):
        raise TypeError(
            f"golf_input must be a GolfTrunkInput, got {type(golf_input).__name__}"
        )
    if not model_path.is_file():
        raise ValueError(f"model file not found: {model_path.name}")


def _prepare_law(
    law: CouplingLaw | None,
    model: object,
    inventory: SpineInventory,
    path: Path,
    sha: str,
) -> CouplingLaw:
    if law is None:
        law = extract_coupling_law(
            model,
            inventory,
            law_id=f"{inventory.model_id}:model_couplers",
            version=f"sha256:{sha[:12]}",
            citation=f"couplers of {path.name}",
        )
    if not isinstance(law, CouplingLaw):
        raise TypeError(f"law must be a CouplingLaw, got {type(law).__name__}")
    if _law_levels(law) != set(inventory.levels):
        raise ValueError(
            "law levels "
            f"{sorted(lv.name for lv in _law_levels(law))} differ from model "
            f"levels {sorted(lv.name for lv in inventory.levels)}"
        )
    return law


def drive_segmental_model(
    golf_input: GolfTrunkInput,
    *,
    model_path: str | Path,
    model_id: str,
    law: CouplingLaw | None = None,
    options: DriveOptions | None = None,
) -> DriveResult:
    """Distribute the golf trunk rotation over a model's spinal levels.

    Each frame solves ``phi* = argmin || log(R_target^T R(phi)) ||`` with
    ``R_target = R_pelvis->torso @ A`` (see module docstring), the inventory
    neutral offsets and the model's native chain offsets ``B``, ``E``.
    ``law`` defaults to the law extracted from the model's own couplers.

    Preconditions: ``golf_input`` is a :class:`GolfTrunkInput` (``TypeError``);
    ``model_path`` exists, ``model_id`` has a verified level map (or
    ``options.level_map`` is given), and the law covers exactly the model's
    levels (``ValueError``).

    Postconditions: the law is untouched; ``residual_rad`` is the geodesic
    distance per frame and ``max_residual_rad`` its maximum; frames over
    ``options.tolerance_rad`` are flagged invalid, never repaired; the
    kinematics carry a non-empty ``distribution_source``.
    """
    opts = DriveOptions() if options is None else options
    path = Path(model_path)
    _check_drive_inputs(golf_input, path)
    entry = _resolve_level_map(model_id, opts.level_map)
    model = import_opensim().Model(str(path))
    inventory = inventory_spine(model, model_id, level_map=entry)
    sha = _sha256(path)
    chosen = _prepare_law(law, model, inventory, path, sha)
    align = np.eye(3) if opts.torso_alignment is None else opts.torso_alignment
    targets = golf_input.pelvis_to_thorax @ align
    detailed = distribute_series_detailed(
        targets,
        golf_input.time_s,
        chosen,
        tolerance_rad=opts.tolerance_rad,
        neutral=inventory.neutral,
        fixed=_chain_offsets(model, entry),
        bounds=opts.bounds,
    )
    coords = _level_columns(detailed.results, inventory, entry)
    coords.update(_pelvis_columns(golf_input.pelvis_pose_ground, opts.pelvis))
    result = DriveResult(
        kinematics=detailed.series,
        residual_rad=detailed.residual_rad,
        phi=np.stack([r.phi for r in detailed.results]),
        independent_names=chosen.independent_names(),
        coordinates=coords,
        model_id=model_id,
        model_sha256=sha,
        law_id=detailed.results[0].law_id,
        law_version=detailed.results[0].law_version,
        input_source=golf_input.source,
    )
    source = result.kinematics.distribution_source
    ensure(
        source.strip() != "",
        "distribution_source must be non-empty",
    )
    return result


def export_mot(
    result: DriveResult, path: str | Path, *, include_independent: bool = False
) -> Path:
    """Write the coordinate table as an OpenSim ``.mot`` (radians, metres).

    Columns are the level coordinates and the six pelvis coordinates (plus the
    law's independent coordinates when ``include_independent``); the table is
    written natively by ``STOFileAdapter`` with ``inDegrees=no`` and the
    kinematics' ``distribution_source`` in the header (read back by SPN-9d).

    Raises:
        TypeError: ``result`` is not a :class:`DriveResult`.
        ValueError: ``path`` does not end in ``.mot``.
    """
    if not isinstance(result, DriveResult):
        raise TypeError(f"result must be a DriveResult, got {type(result).__name__}")
    out = Path(path)
    if out.suffix != ".mot":
        raise ValueError(f"path must end in .mot, got {out.name!r}")
    columns = dict(result.coordinates)
    if include_independent:
        for k, name in enumerate(result.independent_names):
            columns[name] = result.phi[:, k]
    osim = import_opensim()
    labels = osim.StdVectorString()
    for name in columns:
        labels.append(name)
    table = osim.TimeSeriesTable()
    table.setColumnLabels(labels)
    matrix = np.column_stack(list(columns.values()))
    for time, row in zip(result.time_s, matrix, strict=True):
        table.appendRow(float(time), osim.RowVector([float(v) for v in row]))
    table.addTableMetaDataString("inDegrees", "no")
    table.addTableMetaDataString(
        DISTRIBUTION_SOURCE_KEY, result.kinematics.distribution_source
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    osim.STOFileAdapter.write(table, str(out))
    return out
