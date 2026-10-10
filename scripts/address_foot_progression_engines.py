"""Address toe-out per engine from each engine's own forward kinematics (OSV-6).

Issue #11737 asks for both feet within 2 degrees of the capture's toe-out at
address in every engine. The ground-support pipeline fits the address with a
MuJoCo (or Drake) plant and records the full fitted coordinate vector in
``address_report.json`` (``address_coordinates_deg`` and
``address_translations_m``). This script evaluates that single vector with the
FK of the other engines, so each engine's foot yaw is measured natively
instead of assumed:

* ``mujoco`` and ``pinocchio``: ``get_plant(engine, scaled_spec)`` and the plant
  body poses at the recorded coordinates.
* ``myosuite``: the bundled spec-to-MyoSuite retarget map
  (``coordinate_map_anthro.json``) drives the named joints of the MyoSuite
  ``myolegs`` MJCF. The map carries no pelvis orientation, so the pelvis
  rotation (``HipInputX/Y/Z``) is applied to the model's pelvis body here.
  Run it with the MyoSuite virtual environment.

An engine that cannot evaluate the coordinates is reported as unavailable with
the exact reason, never as 0. Two subcommands::

    python3 -m scripts.address_foot_progression_engines evaluate \
        --run-dir RUN --label driver --engines mujoco pinocchio --out OUT.json
    python3 -m scripts.address_foot_progression_engines render \
        --json OUT.json --out-dir DIR --engines pinocchio myosuite

Stills from ``render`` are FK diagrams (calcn and toes origins from the
engine's FK), not engine renders.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
SCALED_SPEC = "full_body_spec_hipcal_scaled.json"
REPORT = "address_report.json"
BODIES = ("femur", "tibia", "calcn", "toes")
SIDES = (("left", "l"), ("right", "r"))
COLORS = {"left": "#1f77b4", "right": "#d62728"}
#: OSV-6 acceptance: each foot within this many degrees of the capture.
TOLERANCE_DEG = 2.0
#: Pelvis orientation coordinates of the Simscape spec (Rx, Ry, Rz chain).
PELVIS_ROTATION = ("HipInputX", "HipInputY", "HipInputZ")


def score_engine(
    points: Mapping[str, np.ndarray],
    targets_deg: Mapping[str, float],
    handedness: str = "right",
) -> dict[str, Any]:
    """Model toe-out and error per foot from the engine's body points.

    Pure: ``points`` holds world positions (Z up) of ``calcn_{r,l}`` and
    ``toes_{r,l}``. Error is model minus target; ``within_tolerance`` uses
    ``TOLERANCE_DEG``. Raises ``ValueError`` for a missing body or target.
    """
    from src.shared.python.motion_matching.pipeline.address_feet import (
        feet_deg_from_positions,
    )

    for side, _ in SIDES:
        if side not in targets_deg:
            raise ValueError(f"targets_deg needs a '{side}' entry")
    model = feet_deg_from_positions(points, handedness)
    error = {s: float(model[s] - targets_deg[s]) for s, _ in SIDES}
    return {
        "available": True,
        "reason": None,
        "model_deg": {s: float(model[s]) for s, _ in SIDES},
        "error_deg": error,
        "within_tolerance": all(abs(e) <= TOLERANCE_DEG for e in error.values()),
    }


def unavailable(reason: str) -> dict[str, Any]:
    """An engine entry that could not be evaluated; never a zero angle."""
    if not reason:
        raise ValueError("an unavailable engine needs a reason")
    return {
        "available": False,
        "reason": reason,
        "model_deg": None,
        "error_deg": None,
        "within_tolerance": None,
    }


def coordinate_vector_rad(
    order: Sequence[str],
    angles_deg: Mapping[str, float],
    translations_m: Mapping[str, float],
) -> np.ndarray:
    """Coordinate vector in ``order`` (radians, metres) from the report blocks.

    Raises ``ValueError`` naming the coordinates the report does not carry.
    """
    missing = [n for n in order if n not in angles_deg and n not in translations_m]
    if missing:
        raise ValueError(f"report lacks coordinates {missing}")
    return np.array(
        [
            translations_m[n] if n in translations_m else np.radians(angles_deg[n])
            for n in order
        ],
        dtype=float,
    )


def pelvis_rotation(angles_deg: Mapping[str, float]) -> np.ndarray:
    """Pelvis rotation matrix ``Rx(a) Ry(b) Rz(c)`` of the spec's hip chain."""
    a, b, c = (np.radians(angles_deg[n]) for n in PELVIS_ROTATION)
    ca, sa, cb, sb, cc, sc = (
        np.cos(a),
        np.sin(a),
        np.cos(b),
        np.sin(b),
        np.cos(c),
        np.sin(c),
    )
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx @ ry @ rz


def plant_points(
    engine: str,
    spec: Mapping[str, Any],
    angles_deg: Mapping[str, float],
    translations_m: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Leg body origins from ``get_plant(engine, spec)`` at the recorded pose.

    Returns ``(points, notes)``. Raises ``ValueError`` (with the reason) when
    the plant lacks a recorded coordinate.
    """
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    plant = get_plant(engine, spec)
    order = list(plant.coordinate_order)
    q = coordinate_vector_rad(order, angles_deg, translations_m)
    reported = set(angles_deg) | set(translations_m)
    extra = sorted(reported - set(order))
    notes = [f"plant has {len(order)} coordinates, report has {len(reported)}"]
    if extra:
        nonzero = [n for n in extra if abs(angles_deg.get(n, 0.0)) > 1e-9]
        notes.append(f"report coordinates the plant lacks: {extra}")
        if nonzero:
            raise ValueError(
                f"{engine} plant has no coordinate for non-zero values {nonzero}"
            )
    names = [f"{b}_{s}" for b in BODIES for _, s in SIDES]
    poses = plant.frame_poses({n: (n, (0.0, 0.0, 0.0)) for n in names}, q)
    return {n: np.asarray(poses[n][1], dtype=float) for n in names}, notes


def solve_hip_rotation(
    foot_deg: Callable[[Mapping[str, float]], Mapping[str, float]],
    angles_deg: Mapping[str, float],
    targets_deg: Mapping[str, float],
    *,
    passes: int = 6,
    tolerance_deg: float = 0.05,
) -> dict[str, float]:
    """``hip_rotation_{l,r}`` (deg) that put each foot on its target in one engine.

    ``foot_deg(angles)`` is the engine's own FK toe-out. The gain per foot is a
    central difference of that FK, so the solve is sign-safe. Raises
    ``ValueError`` when a foot barely responds or the solve does not converge.
    """
    current = dict(angles_deg)
    names = {"left": "hip_rotation_l", "right": "hip_rotation_r"}
    for _ in range(passes):
        now = foot_deg(current)
        errors = {s: targets_deg[s] - now[s] for s in names}
        if max(abs(e) for e in errors.values()) <= tolerance_deg:
            break
        for side, name in names.items():
            hi = foot_deg({**current, name: current[name] + 1.0})[side]
            lo = foot_deg({**current, name: current[name] - 1.0})[side]
            gain = (hi - lo) / 2.0
            if abs(gain) < 0.1:
                raise ValueError(f"{name} barely turns the {side} foot")
            current[name] += errors[side] / gain
    final = foot_deg(current)
    if any(abs(targets_deg[s] - final[s]) > 10 * tolerance_deg for s in names):
        raise ValueError(f"hip_rotation solve did not converge: {dict(final)}")
    return {name: float(current[name]) for name in names.values()}


def pinocchio_points(
    spec: Mapping[str, Any],
    angles_deg: Mapping[str, float],
    translations_m: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Leg body origins from the native Pinocchio model's own FK.

    ``get_plant("pinocchio", ...).frame_poses`` goes through
    ``PinocchioFullBodyIK``, which hard-codes 41 coordinates while the anthro
    document has 44, so it cannot be used. The native model
    (``FullBodyPinocchioModel``, all 44 primitive joints) is evaluated instead,
    with an identity-placed frame added on each leg body so the public
    ``frame_poses`` returns its origin.
    """
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    order = list(spec["coordinate_order"])
    q = coordinate_vector_rad(order, angles_deg, translations_m)
    names = [f"{b}_{s}" for b in BODIES for _, s in SIDES]
    identity = np.eye(4).tolist()
    probed = dict(spec)
    probed["frames"] = list(spec["frames"]) + [
        {"name": f"probe_{n}", "body": n, "placement": identity} for n in names
    ]
    model = FullBodyPinocchioModel(probed)
    poses = model.frame_poses(dict(zip(order, q, strict=True)))
    notes = [
        f"native FullBodyPinocchioModel, {len(order)} primitive coordinates "
        "(PinocchioFullBodyIK needs 41 and is not used)"
    ]
    return {n: np.asarray(poses[f"probe_{n}"])[:3, 3] for n in names}, notes


def _myolegs_xml() -> Path:
    import myosuite

    root = Path(myosuite.__file__).resolve().parent
    xml = root / "simhive" / "myo_sim" / "leg" / "myolegs.xml"
    if not xml.is_file():
        raise FileNotFoundError(f"MyoSuite myolegs.xml not found at {xml}")
    return xml


def myosuite_points(
    spec: Mapping[str, Any],
    angles_deg: Mapping[str, float],
    translations_m: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Leg body origins from the MyoSuite ``myolegs`` MJCF via the retarget map.

    Leg joints come from ``retarget_frame`` (the map's ``hip_rotation_*``,
    ``hip_adduction_*``, ``hip_flexion_*``, knee, ankle, subtalar and mtp
    entries). The pelvis body is rotated by the spec's pelvis chain because
    the map has no pelvis orientation entry.
    """
    import mujoco

    from src.engines.physics_engines.myosuite.python.retarget import (
        default_retarget_map,
        retarget_frame,
        source_coordinate_index,
    )

    order = list(spec["coordinate_order"])
    q_source = coordinate_vector_rad(order, angles_deg, translations_m)
    rmap = default_retarget_map()
    leg_prefixes = {"hip", "knee", "ankle", "subtalar", "mtp"}
    leg_targets = [t for t in rmap.target_names if t.split("_")[0] in leg_prefixes]
    mapped = {rmap.target_names[i] for i, _ in rmap.source_to_target.values()}
    unmapped = [t for t in leg_targets if t not in mapped]
    if unmapped:
        raise ValueError(f"retarget map has no source for leg joints {unmapped}")
    gather = source_coordinate_index(order, rmap)
    q_target = retarget_frame(q_source[gather], rmap)

    model = mujoco.MjModel.from_xml_path(str(_myolegs_xml()))
    data = mujoco.MjData(model)
    notes = [
        f"model {_myolegs_xml().name} from the installed myosuite package",
        "knee translation/rotation helper joints left at 0 (equality-coupled in sim)",
    ]
    for name, value in zip(rmap.target_names, q_target, strict=True):
        if name not in leg_targets:
            continue
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"MyoSuite model has no joint {name!r}")
        data.qpos[model.jnt_qposadr[joint]] = float(value)
    # The MJCF root is a free joint whose default pose is yawed (-1.57 rad) and
    # lifted 1 m; put it at the origin so world axes match the spec world.
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root < 0:
        raise ValueError("MyoSuite model has no 'root' free joint")
    adr = model.jnt_qposadr[root]
    data.qpos[adr : adr + 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    base = np.zeros(9)
    mujoco.mju_quat2Mat(base, model.body_quat[pelvis])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(
        quat, (pelvis_rotation(angles_deg) @ base.reshape(3, 3)).ravel()
    )
    model.body_quat[pelvis] = quat
    mujoco.mj_forward(model, data)
    points = {}
    for body in (f"{b}_{s}" for b in BODIES for _, s in SIDES):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise ValueError(f"MyoSuite model has no body {body!r}")
        points[body] = np.array(data.xpos[bid], dtype=float)
    return points, notes


def _myosuite_hip_solve(
    spec: Mapping[str, Any],
    angles_deg: Mapping[str, float],
    translations_m: Mapping[str, float],
    targets_deg: Mapping[str, float],
) -> dict[str, Any]:
    """Hip rotations that put the MyoSuite feet on target, against the map's own.

    Diagnostic for the retarget map: the identity leg mapping hands the spec's
    hip rotation to a model with different hip and foot geometry, so the map
    alone need not reproduce the capture toe-out.
    """
    from src.shared.python.motion_matching.pipeline.address_feet import (
        feet_deg_from_positions,
    )

    def feet(angles: Mapping[str, float]) -> Mapping[str, float]:
        points, _ = myosuite_points(spec, angles, translations_m)
        return feet_deg_from_positions(points, "right")

    solved = solve_hip_rotation(feet, angles_deg, targets_deg)
    achieved = feet({**angles_deg, **solved})
    return {
        "map_hip_rotation_deg": {
            k: float(angles_deg[k]) for k in ("hip_rotation_l", "hip_rotation_r")
        },
        "solved_hip_rotation_deg": solved,
        "solved_error_deg": {s: float(achieved[s] - targets_deg[s]) for s in achieved},
    }


def evaluate(
    run_dir: Path, label: str, engines: Sequence[str], out: Path
) -> dict[str, Any]:
    """Evaluate ``engines`` on a run directory and merge them into ``out``."""
    report = json.loads((run_dir / REPORT).read_text(encoding="utf-8"))
    spec = json.loads((run_dir / SCALED_SPEC).read_text(encoding="utf-8"))
    feet = report["foot_progression"]["feet"]
    targets = {s: float(feet[s]["target_deg"]) for s, _ in SIDES}
    doc: dict[str, Any] = {}
    if out.is_file():
        doc = json.loads(out.read_text(encoding="utf-8"))
    doc.update(
        {
            "label": label,
            "run_dir": str(run_dir),
            "tolerance_deg": TOLERANCE_DEG,
            "targets_deg": targets,
            "target_is_default": {s: feet[s]["target_is_default"] for s, _ in SIDES},
            "pipeline_model_deg": {s: feet[s]["model_deg"] for s, _ in SIDES},
        }
    )
    engines_doc = doc.setdefault("engines", {})
    angles = report.get("address_coordinates_deg")
    translations = report.get("address_translations_m")
    for engine in engines:
        if angles is None or translations is None:
            engines_doc[engine] = unavailable(
                f"{REPORT} has no address_coordinates_deg (re-run --address-only)"
            )
            continue
        try:
            if engine == "myosuite":
                points, notes = myosuite_points(spec, angles, translations)
            elif engine == "pinocchio":
                points, notes = pinocchio_points(spec, angles, translations)
            else:
                points, notes = plant_points(engine, spec, angles, translations)
            entry = score_engine(points, targets)
            entry["notes"] = notes
            if engine == "myosuite":
                entry["engine_hip_rotation_solve"] = _myosuite_hip_solve(
                    spec, angles, translations, targets
                )
            entry["leg_points_m"] = {
                k: [float(x) for x in v] for k, v in points.items()
            }
        except Exception as exc:  # noqa: BLE001 - an engine failure is reported, not hidden
            LOGGER.exception("%s unavailable", engine)
            entry = unavailable(f"{type(exc).__name__}: {exc}")
        engines_doc[engine] = entry
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def draw(doc: Mapping[str, Any], engines: Sequence[str], out_dir: Path) -> list[Path]:
    """Overhead and face-on FK diagrams, one PNG per available engine."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.shared.python.motion_matching.foot_progression import (
        model_long_axis,
    )
    from src.shared.python.motion_matching.pipeline.address_feet import (
        NATIVE_UP_AXIS,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for engine in engines:
        entry = doc["engines"].get(engine)
        if not entry or not entry["available"]:
            LOGGER.warning("skipping %s: %s", engine, entry and entry["reason"])
            continue
        pts = {k: np.asarray(v) for k, v in entry["leg_points_m"].items()}
        fig, (top, front) = plt.subplots(1, 2, figsize=(11, 5.4))
        for side, sfx in SIDES:
            chain = np.array([pts[f"{b}_{sfx}"] for b in BODIES])
            top.plot(chain[:, 0], chain[:, 1], "-o", color=COLORS[side], ms=3)
            front.plot(chain[:, 1], chain[:, 2], "-o", color=COLORS[side], ms=3)
            heel, toe = pts[f"calcn_{sfx}"], pts[f"toes_{sfx}"]
            axis = model_long_axis(heel, toe, NATIVE_UP_AXIS)
            top.plot(
                [heel[0], heel[0] + 0.35 * axis[0]],
                [heel[1], heel[1] + 0.35 * axis[1]],
                color="k",
                lw=2,
            )
            top.annotate(
                f"{side}: model {entry['model_deg'][side]:+.1f} deg\n"
                f"capture {doc['targets_deg'][side]:+.1f} deg\n"
                f"error {entry['error_deg'][side]:+.1f} deg",
                (heel[0], heel[1]),
                xytext=(6, 10 if side == "left" else -34),
                textcoords="offset points",
                color=COLORS[side],
                fontsize=9,
            )
        top.set_title(f"{engine}: overhead (world axes, Z up)")
        top.set_aspect("equal")
        top.set_xlabel("world x (m)")
        top.set_ylabel("world y (m)")
        front.set_title(f"{engine}: side view of the legs (y vs z)")
        front.set_aspect("equal")
        front.set_xlabel("world y (m)")
        front.set_ylabel("up (m)")
        fig.suptitle(
            f"FK diagram: {engine} address foot progression, {doc['label']} "
            "(legs from the engine's own FK at the fitted address)"
        )
        path = out_dir / f"{doc['label']}_{engine}_fk_diagram.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--run-dir", type=Path, required=True)
    ev.add_argument("--label", required=True)
    ev.add_argument("--engines", nargs="+", required=True)
    ev.add_argument("--out", type=Path, required=True)
    rd = sub.add_parser("render")
    rd.add_argument("--json", type=Path, required=True)
    rd.add_argument("--out-dir", type=Path, required=True)
    rd.add_argument("--engines", nargs="+", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "evaluate":
        doc = evaluate(args.run_dir, args.label, args.engines, args.out)
        for name, entry in doc["engines"].items():
            LOGGER.info("%s: %s", name, entry["model_deg"] or entry["reason"])
    else:
        doc = json.loads(args.json.read_text(encoding="utf-8"))
        for path in draw(doc, args.engines, args.out_dir):
            LOGGER.info("%s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
