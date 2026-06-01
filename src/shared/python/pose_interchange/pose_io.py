"""Save / load engine-native initial-state files for the canonical pose.

This module is the Subtask 6 deliverable of EPIC #4895 (issue #4900).
It builds on the canonical pose dataclass (Subtask 1, #4896) and the
per-engine adapter registry (Subtask 2, #4897) to produce engine-native
artifacts that downstream tools (Pose Studio, Starting-Pose Matcher,
:mod:`motion_matching.load_body_target`) consume directly.

Public API
----------

* :func:`save_initial_state` / :func:`load_initial_state` - per-engine
  starting-state I/O. ``engine`` selects one of
  ``{"drake", "mujoco", "pinocchio", "opensim", "simscape"}`` and each
  engine has its own native file shape:

  ===========  ====================================================
  engine       file format
  ===========  ====================================================
  drake        pickle of ``{q, v, model_metadata}``
  mujoco       JSON ``{qpos, qvel}``
  pinocchio    ``np.savez`` ``{q, v}``
  opensim      ``.sto`` row format (``initial_state``)
  simscape     JSON ``{Tx, Ty, Tz, Rx, Ry, Rz, Scale, jointAngles}``
  ===========  ====================================================

  Round-trip parity with :class:`~pose_interchange.canonical.CanonicalPose`
  is held to 1e-9 across every engine; see
  ``tests/unit/pose_interchange/pose_io/test_starting_state_roundtrip.py``.

* :func:`save_motion_match_target` writes a single-frame ``BodyTarget``
  JSON that loads via
  :func:`motion_matching.load_body_target.load_body_target` (the JSON
  loader is registered in this module, see
  :mod:`motion_matching.loaders.body_json`).

* :func:`save_reference_pose` / :func:`list_saved_reference_poses` write
  and enumerate canonical poses in the on-disk reference-pose library.
  Reload uses :meth:`CanonicalPose.from_path`.

Velocities are not part of the canonical pose contract - the matcher
always sets ``v = 0`` when materialising an initial state. Each engine
therefore emits a zero-velocity vector sized to match its ``q`` /
``qpos`` slot count.
"""

from __future__ import annotations

import json
import pickle  # noqa: S403 - trusted pickle for engine-native MCAP fixtures
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    forward_kinematics,
)
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.adapters import ADAPTER_REGISTRY
from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
)

SUPPORTED_ENGINES: Final[frozenset[str]] = frozenset(
    {"drake", "mujoco", "pinocchio", "opensim", "simscape"}
)

# ``BodyTarget`` validation requires at least 2 frames; we synthesise a
# second frame at ``_MOTION_MATCH_DT_S`` when emitting a single canonical
# pose. The dt is small enough that it will not perturb downstream cost
# terms but large enough that ``time`` is strictly increasing.
_MOTION_MATCH_DT_S: Final[float] = 1.0e-3

_REFERENCE_POSE_LIBRARY: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "motion_matching"
    / "diagnostics"
    / "reference_pose_library"
)


# ----------------------------------------------------------------------------
# Engine validation
# ----------------------------------------------------------------------------


def _require_supported_engine(engine: str) -> None:
    """Validate *engine* against :data:`SUPPORTED_ENGINES`.

    Raises ``ValueError`` with a descriptive message listing the
    supported engines if *engine* is not one of them. Any other type
    raises ``TypeError``.
    """
    if not isinstance(engine, str):
        raise TypeError(
            f"engine must be a string, got {type(engine).__name__}: {engine!r}"
        )
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"engine={engine!r} is not supported; "
            f"expected one of {sorted(SUPPORTED_ENGINES)}"
        )


def _require_canonical_pose(pose: object) -> None:
    if not isinstance(pose, CanonicalPose):
        raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
    if pose.convention_tag != CONVENTION_TAG:
        raise ValueError(
            f"pose.convention_tag must be {CONVENTION_TAG!r}, "
            f"got {pose.convention_tag!r}"
        )


def _resolve_path(output_path: Path | str) -> Path:
    p = Path(output_path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# Drake: pickle of {q, v, model_metadata}
# ----------------------------------------------------------------------------


def _save_drake(pose: CanonicalPose, path: Path) -> None:
    adapter_cls = ADAPTER_REGISTRY["drake"]
    adapter = adapter_cls()
    q = adapter.from_canonical(pose)
    v = np.zeros_like(q)
    payload: dict[str, Any] = {
        "q": q,
        "v": v,
        "model_metadata": {
            "engine": "drake",
            "convention_tag": pose.convention_tag,
            "pelvis_prefix": 6,
            "pelvis_layout": "xyz_rpy_rad",
            "joint_names": tuple(REFERENCE_GOLFER_FIELDS),
        },
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def _load_drake(path: Path) -> CanonicalPose:
    """Load a Drake initial-state ``.drake`` file.

    Trust boundary (issue #6929)
    ----------------------------
    The ``.drake`` interchange format is a Python ``pickle`` (see
    :func:`_save_drake`). ``pickle.load`` executes arbitrary code embedded in
    the stream, so a malicious ``.drake`` file is an arbitrary-code-execution
    vector. This is **deliberately accepted** because the format is a
    *local-only, desktop/CLI* artifact: it is **never** loaded from
    network/API input (the FastAPI surface uses the JSON/``.npz`` engine
    formats), and the file is produced by :func:`_save_drake` on the same
    machine. Callers must therefore treat ``.drake`` files exactly like any
    other executable they run: only load files you produced or fully trust.
    Do **not** expose this loader on an untrusted ingest path; if that ever
    becomes necessary, migrate ``.drake`` to ``np.savez``/JSON like the
    MuJoCo and Pinocchio adapters, which avoid pickle entirely.
    """
    with path.open("rb") as fh:
        # noqa: S301 — trusted, locally-produced file only; see trust-boundary
        # note above. Not reachable from any API/network input.
        payload = pickle.load(fh)  # noqa: S301
    if not isinstance(payload, dict) or "q" not in payload:
        raise ValueError(f"{path}: drake pickle missing required 'q' field")
    adapter = ADAPTER_REGISTRY["drake"]()
    return adapter.to_canonical(np.asarray(payload["q"], dtype=float))


# ----------------------------------------------------------------------------
# MuJoCo: JSON {qpos, qvel}
# ----------------------------------------------------------------------------


def _save_mujoco(pose: CanonicalPose, path: Path) -> None:
    adapter = ADAPTER_REGISTRY["mujoco"]()
    qpos = adapter.from_canonical(pose)
    qvel = np.zeros(qpos.shape[0] - 1, dtype=float)  # MuJoCo v has one less DOF
    payload = {
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "convention_tag": pose.convention_tag,
        "engine": "mujoco",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_mujoco(path: Path) -> CanonicalPose:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "qpos" not in payload:
        raise ValueError(f"{path}: mujoco JSON missing required 'qpos' field")
    adapter = ADAPTER_REGISTRY["mujoco"]()
    return adapter.to_canonical(np.asarray(payload["qpos"], dtype=float))


# ----------------------------------------------------------------------------
# Pinocchio: np.savez {q, v}
# ----------------------------------------------------------------------------


def _save_pinocchio(pose: CanonicalPose, path: Path) -> None:
    adapter = ADAPTER_REGISTRY["pinocchio"]()
    q = adapter.from_canonical(pose)
    v = np.zeros(q.shape[0] - 1, dtype=float)  # Pinocchio v has one less DOF
    np.savez(path, q=q, v=v)


def _load_pinocchio(path: Path) -> CanonicalPose:
    # numpy adds .npz suffix if missing on save; on load, accept either form.
    # np.savez appends ".npz" to the full filename (e.g., "state.pin" -> "state.pin.npz"),
    # so we must also check path with suffix appended, not just replaced.
    candidates = [
        path,
        path.with_suffix(".npz"),  # replaces existing suffix
        Path(str(path) + ".npz"),  # appends .npz (what np.savez does)
    ]
    actual: Path | None = next((c for c in candidates if c.exists()), None)
    if actual is None:
        raise FileNotFoundError(f"pinocchio archive not found: {path}")
    with np.load(actual, allow_pickle=False) as bundle:
        if "q" not in bundle.files:
            raise ValueError(f"{actual}: pinocchio archive missing required 'q' array")
        q = np.asarray(bundle["q"], dtype=float)
    adapter = ADAPTER_REGISTRY["pinocchio"]()
    return adapter.to_canonical(q)


# ----------------------------------------------------------------------------
# OpenSim: .sto row format
# ----------------------------------------------------------------------------


def _opensim_column_names() -> list[str]:
    """Return ``[pelvis_tx, pelvis_ty, pelvis_tz, pelvis_rx, pelvis_ry,
    pelvis_rz, *REFERENCE_GOLFER_FIELDS]``."""
    return [
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "pelvis_rx",
        "pelvis_ry",
        "pelvis_rz",
        *REFERENCE_GOLFER_FIELDS,
    ]


def _save_opensim(pose: CanonicalPose, path: Path) -> None:
    adapter = ADAPTER_REGISTRY["opensim"]()
    q = adapter.from_canonical(pose)
    columns = _opensim_column_names()
    if q.shape[0] != len(columns):
        raise ValueError(
            "OpenSim adapter produced unexpected q size: "
            f"got {q.shape[0]}, expected {len(columns)}"
        )
    n_data = len(columns)
    header_lines = [
        "name=initial_state",
        "version=1",
        f"datacolumns={n_data + 1}",
        "endheader",
    ]
    header = "\n".join(header_lines) + "\n"
    body_header = "time\t" + "\t".join(columns) + "\n"
    # Use repr-style float formatting to preserve precision for the
    # round-trip; ``%.17g`` is enough to recover an IEEE-754 double.
    cells = "\t".join(f"{v:.17g}" for v in q)
    body = f"0.0\t{cells}\n"
    path.write_text(header + body_header + body, encoding="utf-8")


def _load_opensim(path: Path) -> CanonicalPose:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if "endheader" not in lines:
        raise ValueError(f"{path}: opensim .sto missing 'endheader' marker")
    end_idx = lines.index("endheader")
    body_lines = lines[end_idx + 1 :]
    if len(body_lines) < 2:
        raise ValueError(f"{path}: opensim .sto missing column header or data row")
    columns = body_lines[0].split("\t")
    if not columns or columns[0].strip().lower() != "time":
        raise ValueError(
            f"{path}: opensim .sto first column must be 'time', got {columns[0]!r}"
        )
    data_cells = body_lines[1].split("\t")
    if len(data_cells) != len(columns):
        raise ValueError(
            f"{path}: opensim .sto row width {len(data_cells)} does not "
            f"match column header width {len(columns)}"
        )
    # Drop the leading 'time' cell - everything after is q.
    q = np.array([float(c) for c in data_cells[1:]], dtype=float)
    adapter = ADAPTER_REGISTRY["opensim"]()
    return adapter.to_canonical(q)


# ----------------------------------------------------------------------------
# Simscape: JSON {Tx, Ty, Tz, Rx, Ry, Rz, Scale, jointAngles: {...}}
# ----------------------------------------------------------------------------


_SIMSCAPE_DEFAULT_SCALE: Final[float] = 1.0


def _save_simscape(pose: CanonicalPose, path: Path) -> None:
    payload: dict[str, Any] = {
        "Tx": float(pose.pelvis_translation_m[0]),
        "Ty": float(pose.pelvis_translation_m[1]),
        "Tz": float(pose.pelvis_translation_m[2]),
        "Rx": float(pose.pelvis_rotation_xyz_deg[0]),
        "Ry": float(pose.pelvis_rotation_xyz_deg[1]),
        "Rz": float(pose.pelvis_rotation_xyz_deg[2]),
        "Scale": _SIMSCAPE_DEFAULT_SCALE,
        "jointAngles": {
            name: float(pose.angle_deg(name)) for name in REFERENCE_GOLFER_FIELDS
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_simscape(path: Path) -> CanonicalPose:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Scale", "jointAngles"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(
            f"{path}: simscape JSON missing required keys: {sorted(missing)}"
        )
    angles_raw = payload["jointAngles"]
    if not isinstance(angles_raw, dict):
        raise ValueError(
            f"{path}: simscape JSON 'jointAngles' must be a dict, got "
            f"{type(angles_raw).__name__}"
        )
    angles = {
        name: float(angles_raw[name])
        for name in REFERENCE_GOLFER_FIELDS
        if name in angles_raw
    }
    return CanonicalPose(
        pelvis_translation_m=np.array(
            [float(payload["Tx"]), float(payload["Ty"]), float(payload["Tz"])],
            dtype=float,
        ),
        pelvis_rotation_xyz_deg=np.array(
            [float(payload["Rx"]), float(payload["Ry"]), float(payload["Rz"])],
            dtype=float,
        ),
        joint_angles_deg=angles,
    )


# ----------------------------------------------------------------------------
# Public dispatch
# ----------------------------------------------------------------------------


def save_initial_state(
    pose: CanonicalPose, engine: str, output_path: Path | str
) -> None:
    """Write *pose* as an engine-native initial-state file.

    Parameters
    ----------
    pose
        Canonical pose to materialise. Must carry ``convention_tag ==
        canonical-v1``.
    engine
        Target engine identifier; one of
        :data:`SUPPORTED_ENGINES`.
    output_path
        Destination path. Parent directories are created if absent.
        For ``engine == "pinocchio"`` ``numpy`` may append the ``.npz``
        suffix.

    Raises
    ------
    ValueError
        If ``engine`` is not supported or ``pose`` carries an unknown
        convention tag.
    TypeError
        If ``pose`` is not a :class:`CanonicalPose` or ``engine`` is not
        a string.
    """
    _require_canonical_pose(pose)
    _require_supported_engine(engine)
    path = _resolve_path(output_path)

    dispatch = {
        "drake": _save_drake,
        "mujoco": _save_mujoco,
        "pinocchio": _save_pinocchio,
        "opensim": _save_opensim,
        "simscape": _save_simscape,
    }
    dispatch[engine](pose, path)


def load_initial_state(engine: str, input_path: Path | str) -> CanonicalPose:
    """Read an engine-native initial-state file back as a canonical pose.

    Inverse of :func:`save_initial_state`. Round-trip parity is held to
    float-equality: ``load_initial_state(engine,
    save_initial_state(pose, engine, p)) == pose`` to ``1e-9``.

    Raises
    ------
    ValueError
        If ``engine`` is not supported or the file is malformed.
    """
    _require_supported_engine(engine)
    path = Path(input_path)

    dispatch = {
        "drake": _load_drake,
        "mujoco": _load_mujoco,
        "pinocchio": _load_pinocchio,
        "opensim": _load_opensim,
        "simscape": _load_simscape,
    }
    return dispatch[engine](path)


# ----------------------------------------------------------------------------
# Motion-match target
# ----------------------------------------------------------------------------

# Marker name set emitted by ``save_motion_match_target`` - matches the
# 13 named landmarks returned by ``forward_kinematics`` in
# :mod:`motion_matching.diagnostics.forward_kinematics`. Order is fixed so
# downstream BodyTarget consumers see deterministic columns.
MOTION_MATCH_LANDMARKS: Final[tuple[str, ...]] = (
    "pelvis",
    "spine_top",
    "torso_top",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
    "butt",
    "clubhead",
)


def save_motion_match_target(pose: CanonicalPose, output_path: Path | str) -> None:
    """Write a single-frame :class:`BodyTarget`-compatible JSON.

    The output emits two identical frames (``BodyTarget`` requires
    ``N >= 2``) at ``t = 0`` and ``t = 1 ms`` carrying the Cartesian
    positions of the canonical pose's landmarks. ``impact_idx`` is set
    to ``0`` so the loader treats frame 0 as impact-aligned.

    The file loads via
    :func:`motion_matching.load_body_target.load_body_target` (JSON
    dispatch is registered by importing this module - see
    :mod:`motion_matching.loaders.body_json`).

    Parameters
    ----------
    pose
        Canonical pose to evaluate via :func:`forward_kinematics`.
    output_path
        Destination JSON path (``.json`` extension recommended).
    """
    _require_canonical_pose(pose)
    path = _resolve_path(output_path)

    fk = forward_kinematics(pose.angles_full_dict_deg())
    frame: list[list[float]] = []
    for name in MOTION_MATCH_LANDMARKS:
        xyz = fk.points[name]
        frame.append(
            [
                float(xyz[0]) + float(pose.pelvis_translation_m[0]),
                float(xyz[1]) + float(pose.pelvis_translation_m[1]),
                float(xyz[2]) + float(pose.pelvis_translation_m[2]),
            ]
        )
    payload: dict[str, Any] = {
        "schema": "body_target_json_v1",
        "convention_tag": pose.convention_tag,
        "time_s": [0.0, _MOTION_MATCH_DT_S],
        "marker_names": list(MOTION_MATCH_LANDMARKS),
        "marker_xyz": [frame, [list(row) for row in frame]],
        "impact_idx": 0,
        "events": [{"label": "address", "frame": 0, "time_s": 0.0}],
        "source": {
            "filename": path.name,
            "format": "synthetic",
            "subject_id": "canonical_pose",
            "trial_id": "single_frame",
            "sha256": "0" * 64,
        },
        "coordinate_frame": "z_up_right_handed",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Reference-pose library
# ----------------------------------------------------------------------------


def _reference_pose_library_dir() -> Path:
    return _REFERENCE_POSE_LIBRARY


def save_reference_pose(pose: CanonicalPose, name: str) -> Path:
    """Persist *pose* into the on-disk reference-pose library.

    The library lives next to the FK / reference-pose helpers in
    :mod:`motion_matching.diagnostics`. The directory is created on
    first use.

    Parameters
    ----------
    pose
        Canonical pose to store.
    name
        Library entry name (no extension). Must be a non-empty string
        free of path separators.

    Returns
    -------
    Path
        Absolute path to the written ``.json`` file. Reload via
        :meth:`CanonicalPose.from_path`.

    Raises
    ------
    ValueError
        If ``name`` is empty or contains a path separator.
    """
    _require_canonical_pose(pose)
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(
            f"name must not contain path separators or be '.'/'..': {name!r}"
        )

    library = _reference_pose_library_dir()
    library.mkdir(parents=True, exist_ok=True)
    out = library / f"{name}.json"
    pose.to_path(out)
    return out


def list_saved_reference_poses() -> list[str]:
    """Return the sorted list of entry names in the reference-pose library.

    Returns ``[]`` if the library directory does not yet exist. Names are
    returned without the ``.json`` extension.
    """
    library = _reference_pose_library_dir()
    if not library.exists():
        return []
    return sorted(p.stem for p in library.glob("*.json") if p.is_file())


__all__ = [
    "MOTION_MATCH_LANDMARKS",
    "SUPPORTED_ENGINES",
    "list_saved_reference_poses",
    "load_initial_state",
    "save_initial_state",
    "save_motion_match_target",
    "save_reference_pose",
]
