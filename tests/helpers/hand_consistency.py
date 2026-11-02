"""Per-engine hand mass and length readers for HRG-HX7 (#13764).

Each reader builds the separated-hand flagship document with that engine's
own exporter and reads the hand quantities back from the engine (or, for
OpenSim, from the exported XML and the grip-hand meshes it writes). An engine
that is not installed skips with a reason. A quantity an export cannot carry
is an :class:`Unrepresentable` with a reason, never a passing value.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.shared.python.model_appearance.hand_spec import resolved_hand_spec
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.full_body_v2 import (
    LEFT_HAND_BODY,
    derive_full_body_v2_spec,
    segment_boundary_locations,
)

ROOT = Path(__file__).resolve().parents[2]
FLAGSHIP_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
HANDS = ("L", "R")
GATING_ENGINES = ("mujoco", "drake", "pinocchio", "opensim")
REPORTED_ENGINES = ("simscape", "myosuite")
MEASURED = "measured"


@dataclass(frozen=True)
class Unrepresentable:
    """A quantity an engine's flagship export cannot carry, with the reason."""

    engine: str
    quantity: str
    reason: str


@dataclass(frozen=True)
class HandReading:
    """Hand mass (kg) and length (m) per hand as one engine exports them."""

    mass_kg: dict[str, float]
    length_m: dict[str, float] | Unrepresentable


_NO_HAND_GEOMETRY = "flagship export draws no grip hands; hand length is visual only"
_CAPABILITIES: dict[str, dict[str, Any]] = {
    "mujoco": {"gating": True, "mass_kg": MEASURED, "length_m": _NO_HAND_GEOMETRY},
    "drake": {"gating": True, "mass_kg": MEASURED, "length_m": _NO_HAND_GEOMETRY},
    "pinocchio": {"gating": True, "mass_kg": MEASURED, "length_m": _NO_HAND_GEOMETRY},
    "opensim": {"gating": True, "mass_kg": MEASURED, "length_m": MEASURED},
    "simscape": {
        "gating": False,
        "mass_kg": "parity is non-gating; needs MATLAB R2025b (#13764)",
        "length_m": "parity is non-gating; needs MATLAB R2025b (#13764)",
    },
    "myosuite": {
        "gating": False,
        "mass_kg": "parity is non-gating; MyoFullBody hands are not spec bodies",
        "length_m": "parity is non-gating; MyoFullBody hands are not spec bodies",
    },
}


def capability_report() -> dict[str, dict[str, Any]]:
    """Every engine's hand capabilities: ``MEASURED`` or an :class:`Unrepresentable`."""
    return {
        engine: {
            "gating": caps["gating"],
            **{
                q: caps[q]
                if caps[q] == MEASURED
                else Unrepresentable(engine, q, caps[q])
                for q in ("mass_kg", "length_m")
            },
        }
        for engine, caps in _CAPABILITIES.items()
    }


def separated_flagship_doc() -> dict[str, Any]:
    """The flagship driver spec as ``full-body-v2`` with separated hand bodies."""
    v1 = json.loads(FLAGSHIP_SPEC.read_text(encoding="utf-8"))
    return derive_full_body_v2_spec(
        v1,
        segment_boundary_locations(v1),
        provenance="HRG-HX7 cross-engine hand consistency",
        separate_hands=True,
    )


def hand_bodies(doc: dict[str, Any]) -> dict[str, str]:
    """Spec body name of each hand: the left-wrist child and the right closure body."""
    return {"L": LEFT_HAND_BODY, "R": doc["closure"]["body_a"]}


def _body(doc: dict[str, Any], name: str) -> dict[str, Any]:
    return next(b for b in doc["bodies"] if b["name"] == name)


def spec_hand_masses(doc: dict[str, Any]) -> dict[str, float]:
    """Summed solid mass of each hand body in the specification."""
    return {
        hand: sum(float(s["mass_kg"]) for s in _body(doc, name)["solids"])
        for hand, name in hand_bodies(doc).items()
    }


def spec_hand_length_m(doc: dict[str, Any]) -> float:
    """The one resolved hand length every emitter must draw (#13995)."""
    return resolved_hand_spec(doc).hand_length_m


def _doc_bytes(doc: dict[str, Any]) -> bytes:
    return json.dumps(doc, sort_keys=True).encode("utf-8")


def _mujoco_masses(doc: dict[str, Any]) -> dict[str, float]:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    model = mujoco.MjModel.from_xml_string(export_full_body_mjcf(_doc_bytes(doc))[0])
    return {
        hand: float(model.body(name).mass[0]) for hand, name in hand_bodies(doc).items()
    }


def _drake_masses(doc: dict[str, Any]) -> dict[str, float]:
    pytest.importorskip("pydrake.all")
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    model = FullBodyDrakeModel(doc)
    links = model.metadata["solid_links"]
    plant = model.plant
    return {
        hand: sum(
            plant.GetBodyByName(links[s["name"]]).default_mass()
            for s in _body(doc, name)["solids"]
        )
        for hand, name in hand_bodies(doc).items()
    }


def _pinocchio_masses(doc: dict[str, Any]) -> dict[str, float]:
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    model = FullBodyPinocchioModel(doc).model
    wrist = {j["child"]: j["primitives"][-1]["coordinate"] for j in doc["joints"]}
    return {
        hand: float(model.inertias[model.getJointId(wrist[name])].mass)
        for hand, name in hand_bodies(doc).items()
    }


def _opensim_masses(doc: dict[str, Any]) -> dict[str, float]:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        clean_osim_body_name,
        export_full_body_osim,
    )

    xml, _ = export_full_body_osim(doc, club_geometry_ref=None)
    # OpenSim tags such as ``HuntCrossleyForce::ContactParametersSet`` are not XML names.
    root = ET.fromstring(xml.replace("::", "__"))
    bodies = {b.get("name"): b for b in root.iter("Body")}
    return {
        hand: float(bodies[clean_osim_body_name(name)].findtext("mass"))
        for hand, name in hand_bodies(doc).items()
    }


def _opensim_lengths(doc: dict[str, Any]) -> dict[str, float]:
    """Hand length the OpenSim export passes to its grip-hand meshes."""
    from src.engines.physics_engines.opensim.python import club_visuals, full_body_osim

    seen: list[float] = []
    original = club_visuals.grip_hand_meshes

    def spy(club: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["hand_length_m"])
        return original(club, **kwargs)

    with pytest.MonkeyPatch.context() as patch, TemporaryDirectory() as out:
        patch.setattr(club_visuals, "grip_hand_meshes", spy)
        full_body_osim._write_club_assets(doc, Path(out))
    assert len(seen) == 1, "OpenSim export must draw the grip hands exactly once"
    return dict.fromkeys(HANDS, float(seen[0]))


_MASS_READERS: dict[str, Callable[[dict[str, Any]], dict[str, float]]] = {
    "mujoco": _mujoco_masses,
    "drake": _drake_masses,
    "pinocchio": _pinocchio_masses,
    "opensim": _opensim_masses,
}


def read_hand(engine: str, doc: dict[str, Any]) -> HandReading:
    """Hand mass and length of ``engine``'s flagship export (gating engines only)."""
    if engine not in _MASS_READERS:
        raise ValueError(f"{engine!r} is not a gating engine {GATING_ENGINES}")
    length = capability_report()[engine]["length_m"]
    return HandReading(
        mass_kg=_MASS_READERS[engine](doc),
        length_m=_opensim_lengths(doc) if length == MEASURED else length,
    )


_FRAME_SEED = 1376402
# Import order matters: pydrake before pinocchio, as in the cross-engine FK suite.
_ENGINE_MODULES = ("mujoco", "pydrake.all", "pinocchio", "opensim")


def preload_engines() -> None:
    """Import every installed engine once, before any test runs.

    ``tests/conftest.py::_protect_engine_modules`` drops engine modules a test
    imported; a second test re-importing pydrake or pinocchio then fails inside
    their C extensions. Loading them first puts them in that fixture's snapshot.
    Missing engines are left for the per-test ``importorskip`` to report.
    """
    for name in _ENGINE_MODULES:
        try:
            importlib.import_module(name)
        except (ImportError, OSError):
            continue


def _frame_names(hand: str) -> tuple[str, str]:
    return f"HX7_{hand}_forearm", f"HX7_{hand}_hand"


def with_hand_frames(doc: dict[str, Any]) -> dict[str, Any]:
    """``doc`` plus identity frames on each hand body and its parent (forearm) body.

    Every exporter already emits spec frames, so this reads M through each
    engine's own forward kinematics without touching engine internals.
    """
    out = json.loads(json.dumps(doc))
    parents = {j["child"]: j["parent"] for j in out["joints"]}
    for hand, body in hand_bodies(out).items():
        forearm, hand_frame = _frame_names(hand)
        for name, on in ((forearm, parents[body]), (hand_frame, body)):
            out["frames"].append(
                {"name": name, "body": on, "placement": np.eye(4).tolist()}
            )
    return out


def wrist_configurations(doc: dict[str, Any], count: int) -> list[dict[str, float]]:
    """Seeded configurations: translations in +-0.5 m, rotations in +-0.6 rad."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(_FRAME_SEED)
    return [
        {
            name: float(
                rng.uniform(-0.5, 0.5)
                if name.endswith(("Px", "Py", "Pz")) or name.startswith("Translation")
                else rng.uniform(-0.6, 0.6)
            )
            for name in doc["coordinate_order"]
        }
        for _ in range(count)
    ]


def _mujoco_frames(doc: dict[str, Any]) -> Callable[[dict[str, float]], dict]:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(_doc_bytes(doc))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    adr = {n: model.joint(n).qposadr[0] for n in doc["coordinate_order"]}
    sites = {
        f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
        for f, s in meta["frame_sites"].items()
    }

    def poses(q: dict[str, float]) -> dict[str, np.ndarray]:
        for name, value in q.items():
            data.qpos[adr[name]] = value
        mujoco.mj_kinematics(model, data)
        out = {}
        for frame, sid in sites.items():
            pose = np.eye(4)
            pose[:3, :3] = data.site_xmat[sid].reshape(3, 3)
            pose[:3, 3] = data.site_xpos[sid]
            out[frame] = pose
        return out

    return poses


def _drake_frames(doc: dict[str, Any]) -> Callable[[dict[str, float]], dict]:
    pytest.importorskip("pydrake.all")
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    return FullBodyDrakeModel(doc).frame_poses


def _pinocchio_frames(doc: dict[str, Any]) -> Callable[[dict[str, float]], dict]:
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    return FullBodyPinocchioModel(doc).frame_poses


def _opensim_frames(doc: dict[str, Any]) -> Callable[[dict[str, float]], dict]:
    from tests.helpers.opensim_support import require_opensim

    require_opensim()
    from src.engines.physics_engines.opensim.python.full_body_parity import (
        OpenSimFullBodyParityAdapter,
    )

    return OpenSimFullBodyParityAdapter(_doc_bytes(doc)).frame_poses


_FRAME_READERS: dict[str, Callable[[dict[str, Any]], Callable]] = {
    "mujoco": _mujoco_frames,
    "drake": _drake_frames,
    "pinocchio": _pinocchio_frames,
    "opensim": _opensim_frames,
}


def _spec_frames(doc: dict[str, Any], q: dict[str, float]) -> dict[str, np.ndarray]:
    poses = spec_body_poses(doc, q)
    return {
        f["name"]: poses[f["body"]] @ np.asarray(f["placement"], dtype=float)
        for f in doc["frames"]
    }


def _hand_on_forearm(frames: dict[str, Any], hand: str) -> np.ndarray:
    forearm, hand_frame = (
        np.asarray(frames[n], dtype=float) for n in _frame_names(hand)
    )
    return forearm[:3, :3].T @ hand_frame[:3, :3]


def rotation_error_deg(expected: np.ndarray, got: np.ndarray) -> float:
    """Geodesic angle between two rotations, precise near zero (chord form)."""
    chord = float(np.linalg.norm(expected.T @ got - np.eye(3)))
    return float(np.degrees(2.0 * np.arcsin(min(1.0, chord / (2.0 * np.sqrt(2.0))))))


def worst_hand_frame_error_deg(
    engine: str, doc: dict[str, Any], configs: list[dict[str, float]]
) -> dict[str, float]:
    """Worst M error (deg) per hand of ``engine`` against the spec over ``configs``."""
    if engine not in _FRAME_READERS:
        raise ValueError(f"{engine!r} is not a gating engine {GATING_ENGINES}")
    framed = with_hand_frames(doc)
    poses_of = _FRAME_READERS[engine](framed)
    worst = dict.fromkeys(HANDS, 0.0)
    for q in configs:
        expected, got = _spec_frames(framed, q), poses_of(q)
        for hand in HANDS:
            error = rotation_error_deg(
                _hand_on_forearm(expected, hand), _hand_on_forearm(got, hand)
            )
            worst[hand] = max(worst[hand], error)
    return worst
