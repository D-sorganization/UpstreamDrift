"""Musculoskeletal golf-swing model: kinematics mapping and model assembly.

Part of the OpenSim musculoskeletal swing work (issue #11617, epic #11605).

The golf humanoid used by the tour-matching pipeline is the Rajagopal-Lai-Uhlrich
full-body skeleton with the muscles stripped and a welded club added: the 39
coordinate names are identical.  This module therefore builds the musculoskeletal
swing model by *re-using the original muscle-bearing model* and fitting it to the
golfer:

1. per-body scale factors are derived from the golf humanoid joint-frame offsets,
2. the golf body masses and the welded club are copied across,
3. the wrist and subtalar coordinates are unlocked (the swing drives them),
4. the generic upper-body torque actuators are replaced by named stand-ins and
   reserve actuators are added to every coordinate no muscle can actuate.

The base model has **no arm, shoulder or trunk muscles** (80 Millard lower-limb
muscles only).  Upper-body joint demand is carried by ``upper_*`` torque actuators
that stand in for the absent muscles; this is reported, never hidden.

Pure-numpy helpers (state-table IO, coordinate mapping, smoothing, muscle
grouping) do not import OpenSim so they can be tested in any environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

BASE_MODEL_ENV = "UPSTREAMDRIFT_MSK_BASE_MODEL"
BASE_MODEL_RELATIVE = (
    "shared/models/opensim/opensim-models/Models/Rajagopal/RajagopalLaiUhlrich2023.osim"
)
REPO_ROOT = Path(__file__).resolve().parents[5]

# Coordinates whose torque the model's muscles cannot supply (no muscles
# cross them in the base model).  They get ``upper_*`` stand-in actuators.
UPPER_BODY_COORDINATES: tuple[str, ...] = (
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
    "arm_flex_r",
    "arm_add_r",
    "arm_rot_r",
    "elbow_flex_r",
    "pro_sup_r",
    "wrist_flex_r",
    "wrist_dev_r",
    "arm_flex_l",
    "arm_add_l",
    "arm_rot_l",
    "elbow_flex_l",
    "pro_sup_l",
    "wrist_flex_l",
    "wrist_dev_l",
)
ROOT_COORDINATES: tuple[str, ...] = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
)
# Coordinates unlocked in the base model so the matched kinematics can drive them.
UNLOCKED_COORDINATES: tuple[str, ...] = (
    "subtalar_angle_r",
    "subtalar_angle_l",
    "wrist_flex_r",
    "wrist_dev_r",
    "wrist_flex_l",
    "wrist_dev_l",
)
# Dependent coordinates solved by constraint; they are never reserve-actuated.
DEPENDENT_COORDINATES: tuple[str, ...] = ("knee_angle_r_beta", "knee_angle_l_beta")

# Optimal force of the non-muscle actuators.  Muscle-free joints (upper body, pelvis
# root residuals) get a physical scale so their controls are O(1) and the
# optimiser is well conditioned; leg reserves keep a small value so they are
# expensive and only used when the muscles cannot supply the moment.
UPPER_OPTIMAL_FORCE = 100.0
ROOT_OPTIMAL_FORCE = 100.0
RESERVE_OPTIMAL_FORCE = 1.0

# Welded club copied from the golf humanoid (hand_r_to_club joint, Club body).
CLUB_MASS_KG = 0.32
CLUB_COM_M = (0.0, -0.786, 0.0)
CLUB_INERTIA = (0.1158, 0.0001, 0.1158)
CLUB_GRIP_OFFSET_IN_HAND_M = (0.0, -0.06, 0.0)
CLUB_HEAD_IN_CLUB_M = (0.0, -1.042, 0.0)

# Muscle-name prefix -> functional group (Rajagopal naming).
_MUSCLE_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("glmax", "hip extensors"),
    ("glmed", "hip abductors"),
    ("glmin", "hip abductors"),
    ("tfl", "hip abductors"),
    ("piri", "hip rotators"),
    ("iliacus", "hip flexors"),
    ("psoas", "hip flexors"),
    ("sart", "hip flexors"),
    ("addbrev", "hip adductors"),
    ("addlong", "hip adductors"),
    ("addmag", "hip adductors"),
    ("grac", "hip adductors"),
    ("pect", "hip adductors"),
    ("bflh", "hamstrings"),
    ("bfsh", "hamstrings"),
    ("semimem", "hamstrings"),
    ("semiten", "hamstrings"),
    ("recfem", "quadriceps"),
    ("vas", "quadriceps"),
    ("gas", "plantarflexors"),
    ("soleus", "plantarflexors"),
    ("tibpost", "plantarflexors"),
    ("fdl", "plantarflexors"),
    ("fhl", "plantarflexors"),
    ("perlong", "evertors"),
    ("perbrev", "evertors"),
    ("tibant", "dorsiflexors"),
    ("edl", "dorsiflexors"),
    ("ehl", "dorsiflexors"),
)


class MappingError(ValueError):
    """Raised when kinematics cannot be mapped onto the model coordinates."""


@dataclass(frozen=True)
class CoordinateMapping:
    """Result of mapping source coordinate columns onto model coordinates."""

    mapped: dict[str, str]
    unmapped_model: tuple[str, ...]
    unmapped_source: tuple[str, ...]

    def coverage(self, model_names: Sequence[str]) -> float:
        """Fraction of ``model_names`` that have a source column."""
        require(len(model_names) > 0, "model_names must be non-empty")
        return float(len(self.mapped)) / float(len(model_names))


def muscle_group(name: str) -> str:
    """Return the functional group of a Rajagopal muscle name.

    The trailing side suffix (``_r``/``_l``) is ignored for the group but
    reported by :func:`muscle_side`.

    Raises:
        ValueError: if ``name`` is empty.
    """
    require(bool(name), "muscle name must be non-empty")
    base = name.rsplit("_", 1)[0] if name.endswith(("_r", "_l")) else name
    for prefix, group in _MUSCLE_GROUP_PREFIXES:
        if base.startswith(prefix):
            return group
    return "other"


def muscle_side(name: str) -> str:
    """Return ``"r"``, ``"l"`` or ``"?"`` from the muscle name suffix."""
    require(bool(name), "muscle name must be non-empty")
    return name[-1] if name.endswith(("_r", "_l")) else "?"


def read_states_table(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read an OpenSim ``.sto``/``.mot`` file into ``(times, {label: column})``.

    Only the plain tab-separated text format is supported.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the header or data block is malformed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"states file not found: {p}")
    lines = p.read_text().splitlines()
    try:
        header_end = next(
            i for i, ln in enumerate(lines) if ln.strip().lower() == "endheader"
        )
    except StopIteration as exc:
        raise ValueError(f"{p}: missing 'endheader'") from exc
    labels = lines[header_end + 1].split("\t")
    rows = [ln.split("\t") for ln in lines[header_end + 2 :] if ln.strip()]
    if not rows or any(len(r) != len(labels) for r in rows):
        raise ValueError(f"{p}: ragged or empty data block")
    data = np.array(rows, dtype=float)
    ensure(data.shape[1] == len(labels), "column count mismatch", data.shape)
    return data[:, 0], {lab: data[:, i] for i, lab in enumerate(labels) if i > 0}


def coordinate_name(label: str) -> str | None:
    """Return the coordinate name of a ``.../<coord>/value`` label, else ``None``."""
    parts = label.split("/")
    if label.endswith("/value") and len(parts) >= 2:
        return parts[-2]
    if "/" not in label and label:
        return label
    return None


def map_coordinates(
    source_labels: Sequence[str], model_coordinates: Sequence[str]
) -> CoordinateMapping:
    """Map source column labels to model coordinate names (by coordinate name).

    The source may be an OpenSim states table (``/jointset/<j>/<c>/value``) or a
    plain motion file (``<c>``).  Speed columns are ignored.

    Raises:
        MappingError: if two source columns resolve to the same coordinate.
    """
    mapped: dict[str, str] = {}
    unmapped_source: list[str] = []
    model_set = set(model_coordinates)
    for label in source_labels:
        name = coordinate_name(label)
        if name is None:
            continue
        if name not in model_set:
            unmapped_source.append(label)
            continue
        if name in mapped:
            raise MappingError(f"duplicate source column for coordinate {name!r}")
        mapped[name] = label
    unmapped_model = tuple(c for c in model_coordinates if c not in mapped)
    return CoordinateMapping(mapped, unmapped_model, tuple(unmapped_source))


def smooth_kinematics(
    times: np.ndarray, columns: Mapping[str, np.ndarray], cutoff_hz: float
) -> dict[str, np.ndarray]:
    """Zero-phase 4th-order (2nd-order filtfilt) Butterworth low-pass of each column.

    Raises:
        ValueError: if sampling is non-uniform or the cutoff is not below Nyquist.
    """
    t = np.asarray(times, dtype=float)
    require(t.ndim == 1 and t.size > 12, "need at least 13 samples to filter")
    dt = np.diff(t)
    require(bool(np.all(dt > 0)), "times must be strictly increasing")
    require(
        float(dt.max() - dt.min()) < 1e-3 * float(dt.mean()),
        "times must be uniformly sampled",
    )
    fs = 1.0 / float(dt.mean())
    require(0.0 < cutoff_hz < fs / 2.0, "cutoff must be in (0, Nyquist)")
    b, a = butter(2, cutoff_hz / (fs / 2.0))
    return {k: filtfilt(b, a, np.asarray(v, dtype=float)) for k, v in columns.items()}


def write_sto(
    path: str | Path,
    times: np.ndarray,
    columns: Mapping[str, np.ndarray],
    *,
    in_degrees: bool = False,
) -> Path:
    """Write ``columns`` to an OpenSim version-3 ``.sto`` file."""
    require(len(columns) > 0, "columns must be non-empty")
    n = len(times)
    for k, v in columns.items():
        require(len(v) == n, f"column {k!r} length {len(v)} != {n}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    labels = list(columns)
    data = np.column_stack([times] + [np.asarray(columns[k]) for k in labels])
    with p.open("w") as fh:
        fh.write(
            f"{p.stem}\nversion=1\nnRows={n}\nnColumns={len(labels) + 1}\n"
            f"inDegrees={'yes' if in_degrees else 'no'}\nendheader\n"
        )
        fh.write("time\t" + "\t".join(labels) + "\n")
        for row in data:
            fh.write("\t".join(f"{x:.9g}" for x in row) + "\n")
    return p


def resolve_base_model(explicit: str | Path | None = None) -> Path:
    """Locate the muscle-bearing base model (RajagopalLaiUhlrich2023.osim).

    Search order: ``explicit``, ``$UPSTREAMDRIFT_MSK_BASE_MODEL``, the repo
    submodule path, then ``src/``-relative submodule layout.

    Raises:
        FileNotFoundError: listing every probed path.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(BASE_MODEL_ENV):
        candidates.append(Path(os.environ[BASE_MODEL_ENV]))
    candidates.append(REPO_ROOT / BASE_MODEL_RELATIVE)
    candidates.append(REPO_ROOT / "src" / BASE_MODEL_RELATIVE)
    for cand in candidates:
        if cand.is_file():
            return cand
    probed = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "RajagopalLaiUhlrich2023.osim not found; run "
        "`git submodule update --init shared/models/opensim/opensim-models` or set "
        f"${BASE_MODEL_ENV}. Probed: {probed}"
    )


def _require_opensim() -> Any:
    try:
        import opensim
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("OpenSim python bindings are required") from exc
    return opensim


def derive_body_scale_factors(reference: Any, target: Any) -> dict[str, float]:
    """Per-body uniform scale of ``reference`` that reproduces ``target``.

    For every body the ratio of the summed joint-frame offset norms (target over
    reference) is taken over frames that exist in both models.  Bodies with no
    measurable offsets get ``1.0``.

    Raises:
        ValueError: if either model has no joints.
    """
    osim = _require_opensim()

    def offsets(model: Any) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for joint in model.getJointSet():
            for frame in (joint.getParentFrame(), joint.getChildFrame()):
                pof = osim.PhysicalOffsetFrame.safeDownCast(frame)
                if pof is None:
                    continue
                body = frame.findBaseFrame().getName()
                norm = float(np.linalg.norm(pof.get_translation().to_numpy()))
                out.setdefault(body, {})[f"{joint.getName()}/{frame.getName()}"] = norm
        return out

    ref, tgt = offsets(reference), offsets(target)
    require(bool(ref) and bool(tgt), "models must contain joints")
    scales: dict[str, float] = {}
    for body, frames in ref.items():
        common = [k for k in frames if k in tgt.get(body, {})]
        r = sum(frames[k] for k in common)
        t = sum(tgt[body][k] for k in common)
        scales[body] = float(t / r) if r > 1e-4 else 1.0
    return scales


def _scale_set(scales: Mapping[str, float]) -> Any:
    osim = _require_opensim()
    scale_set = osim.ScaleSet()
    for body, factor in scales.items():
        sc = osim.Scale()
        sc.setSegmentName(body)
        sc.setScaleFactors(osim.Vec3(float(factor)))
        sc.setApply(True)
        scale_set.cloneAndAppend(sc)
    return scale_set


def _copy_masses(model: Any, golf: Any) -> None:
    """Scale each body's mass and inertia to the golf humanoid's body mass."""
    osim = _require_opensim()
    for body in golf.getBodySet():
        name = body.getName()
        if name == "Club" or not model.getBodySet().hasComponent(name):
            continue
        mine = model.updBodySet().get(name)
        if mine.getMass() <= 0.0:
            continue
        ratio = body.getMass() / mine.getMass()
        inertia = [mine.get_inertia().get(i) for i in range(6)]
        mine.setMass(float(body.getMass()))
        mine.set_inertia(osim.Vec6(*[float(x * ratio) for x in inertia]))


def _add_club(model: Any) -> None:
    osim = _require_opensim()
    club = osim.Body(
        "Club",
        CLUB_MASS_KG,
        osim.Vec3(*CLUB_COM_M),
        osim.Inertia(*CLUB_INERTIA),
    )
    model.addBody(club)
    weld = osim.WeldJoint(
        "hand_r_to_club",
        model.getBodySet().get("hand_r"),
        osim.Vec3(*CLUB_GRIP_OFFSET_IN_HAND_M),
        osim.Vec3(0.0),
        club,
        osim.Vec3(0.0),
        osim.Vec3(0.0),
    )
    model.addJoint(weld)


def _replace_actuators(
    model: Any, reserve_optimal_force: float
) -> tuple[dict[str, str], dict[str, float]]:
    """Replace torque actuators; return ``(kinds, optimal_forces)`` by name.

    ``upper_<coord>`` actuators stand in for absent upper-body muscles and
    ``reserve_<coord>`` actuators cover every other non-dependent coordinate.
    """
    osim = _require_opensim()
    fset = model.updForceSet()
    for i in reversed(range(fset.getSize())):
        if fset.get(i).getConcreteClassName() == "CoordinateActuator":
            fset.remove(i)
    kinds: dict[str, str] = {}
    optimal: dict[str, float] = {}
    upper = set(UPPER_BODY_COORDINATES)
    root = set(ROOT_COORDINATES)
    for coord in model.getCoordinateSet():
        name = coord.getName()
        if name in DEPENDENT_COORDINATES:
            continue
        if coord.get_locked():
            continue
        kind = "upper" if name in upper else "reserve"
        act = osim.CoordinateActuator(name)
        act.setName(f"{kind}_{name}")
        scale = (
            UPPER_OPTIMAL_FORCE
            if kind == "upper"
            else ROOT_OPTIMAL_FORCE
            if name in root
            else float(reserve_optimal_force)
        )
        act.setOptimalForce(scale)
        optimal[act.getName()] = scale
        act.setMinControl(-np.inf)
        act.setMaxControl(np.inf)
        model.addForce(act)
        kinds[act.getName()] = kind
    return kinds, optimal


def build_musculoskeletal_model(
    golf_model_path: str | Path,
    base_model_path: str | Path | None = None,
    *,
    reserve_optimal_force: float = RESERVE_OPTIMAL_FORCE,
) -> tuple[Any, dict[str, Any]]:
    """Fit the Rajagopal-Lai-Uhlrich muscle model to the golf humanoid.

    Returns ``(model, info)`` where ``info`` holds the body scale factors, the
    actuator kinds and the model counts.  The model is *not* yet initialised.

    Raises:
        FileNotFoundError: if the golf or base model is missing.
        ValueError: if ``reserve_optimal_force`` is not positive.
    """
    osim = _require_opensim()
    require(reserve_optimal_force > 0, "reserve_optimal_force must be positive")
    golf_path = Path(golf_model_path)
    if not golf_path.is_file():
        raise FileNotFoundError(f"golf humanoid model not found: {golf_path}")
    base_path = resolve_base_model(base_model_path)
    golf = osim.Model(str(golf_path))
    golf.initSystem()
    model = osim.Model(str(base_path))
    state = model.initSystem()
    scales = derive_body_scale_factors(model, golf)
    model.scale(state, _scale_set(scales), True)
    _copy_masses(model, golf)
    for coord_name in UNLOCKED_COORDINATES:
        model.updCoordinateSet().get(coord_name).set_locked(False)
    _add_club(model)
    kinds, optimal = _replace_actuators(model, reserve_optimal_force)
    model.setName("golf_musculoskeletal")
    model.finalizeConnections()
    model.initSystem()
    info = {
        "base_model": str(base_path),
        "golf_model": str(golf_path),
        "body_scale_factors": scales,
        "actuator_kinds": kinds,
        "actuator_optimal_force": optimal,
        "n_muscles": int(model.getMuscles().getSize()),
        "n_coordinates": int(model.getCoordinateSet().getSize()),
        "total_mass_kg": float(sum(b.getMass() for b in model.getBodySet())),
    }
    ensure(info["n_muscles"] > 0, "musculoskeletal model has no muscles")
    return model, info


def state_labels_for(model: Any, coordinates: Sequence[str]) -> dict[str, str]:
    """Return ``{coordinate: '/jointset/.../value'}`` state paths for ``coordinates``."""
    paths: dict[str, str] = {}
    for coord in model.getCoordinateSet():
        if coord.getName() in coordinates:
            paths[coord.getName()] = coord.getAbsolutePathString() + "/value"
    return paths
