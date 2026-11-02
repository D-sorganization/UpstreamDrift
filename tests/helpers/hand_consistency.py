"""Per-engine hand mass and length readers for HRG-HX7 (#13764).

Each reader builds the separated-hand flagship document with that engine's
own exporter and reads the hand quantities back from the engine (or, for
OpenSim, from the exported XML and the grip-hand meshes it writes). An engine
that is not installed skips with a reason. A quantity an export cannot carry
is an :class:`Unrepresentable` with a reason, never a passing value.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from defusedxml import ElementTree as ET

from src.shared.python.model_appearance.hand_spec import resolved_hand_spec
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
