"""Build the golf-humanoid `.osim` from the Rajagopal2015 OpenSense base.

This script is the deterministic generator for
``src/engines/physics_engines/opensim/models/golf_humanoid.osim`` (issue #4110,
OpenSim Parity Spec §3.2).

Source base
-----------
``shared/models/opensim/opensim-models/Models/Rajagopal_OpenSense/``
``Rajagopal2015_opensense.osim``

The OpenSense variant is derived from Rajagopal et al. (2016) and is **already
muscle-stripped** — its ``ForceSet`` is empty. Using the OpenSense variant lets
us skip the muscle-removal pass entirely, which keeps the build deterministic
without depending on the OpenSim Python bindings (which are not pip-installable
on every platform — see PR body for details).

Modifications applied
---------------------
1. Rename ``Model name="OpenSense_Subject"`` → ``Model name="golf_humanoid"``.
2. Attach the improved club held in both hands (OSV-9, #11756) through
   ``msk_club.attach_club``: the shared head, shaft and grip meshes with the
   address-square face roll, the shared club mass and inertia, the lead
   ``WeldJoint hand_l_to_club`` plus the trail ``WeldConstraint
   hand_r_to_club`` (or, with ``--grip-model bushing``, a free club and one
   ``BushingForce`` per hand). Hand-side grip frames and the default (address)
   pose come from the committed ``msk_club_grip_calibration.json``.
3. The same club replaces the one in ``golf_humanoid_scaled.osim``.
4. Add a ``CoordinateActuator`` to the ``ForceSet`` for every skeleton
   ``Coordinate`` (the free club coordinates of the bushing grip stay
   passive). Naming convention: ``tau_<coord_name>``. ``optimal_force=1``,
   ``min_control=-Inf``, ``max_control=+Inf`` (we control torque directly in
   N·m via the polynomial controller — see OPENSIM_PARITY_SPEC §3.2 step 4).

The script emits **byte-identical** output across runs (no timestamps, no
non-deterministic iteration order) so the committed artifact is reproducible.

Usage
-----
::

    python3 scripts/build_humanoid_osim.py

Writes ``golf_humanoid.osim`` and rewrites the club of
``golf_humanoid_scaled.osim`` in ``src/engines/physics_engines/opensim/models``.
To recalibrate the grip after a skeleton change::

    python3 scripts/build_humanoid_osim.py --skeleton-only /tmp/golf_humanoid.osim
    python3 -m src.engines.physics_engines.opensim.python.msk_club_calibration \\
        /tmp/golf_humanoid.osim src/engines/physics_engines/opensim/models/golf_humanoid_scaled.osim

Notes
-----
- We deliberately **do not** import ``opensim`` here. The Rajagopal OpenSense
  XML is already a valid ``OpenSimDocument Version="40000"`` document; pure-XML
  manipulation keeps the builder runnable in any environment that has Python
  3.10+. The committed model is then validated by
  ``tests/test_opensim_model_loads.py`` whenever the OpenSim Python bindings
  are available.
- The Simscape body-chain coordinate naming alignment (cross-engine spec §2.6)
  is documented in ``models/README.md`` — the OpenSim coordinate names used
  here are the canonical Rajagopal names that the Simscape→OpenSim
  ``coordinate_map`` (issue ``OPENSIM-COORD-MAP``) translates.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import defusedxml.ElementTree as DefusedET

# Repository root, derived from this script's location.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.physics_engines.opensim.python import msk_club  # noqa: E402

_SUBMODULE_BASE_OSIM = (
    REPO_ROOT
    / "shared"
    / "models"
    / "opensim"
    / "opensim-models"
    / "Models"
    / "Rajagopal_OpenSense"
    / "Rajagopal2015_opensense.osim"
)

#: ``UPSTREAMDRIFT_RAJAGOPAL_OPENSENSE`` points at a local copy of the base when
#: the opensim-models submodule is not checked out.
BASE_OSIM = Path(
    os.environ.get("UPSTREAMDRIFT_RAJAGOPAL_OPENSENSE", str(_SUBMODULE_BASE_OSIM))
)

OUTPUT_OSIM = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid.osim"
)
SCALED_OSIM = OUTPUT_OSIM.with_name("golf_humanoid_scaled.osim")

GOLF_UNLOCKED_COORDINATES: tuple[str, ...] = (
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
    "subtalar_angle_r",
    "subtalar_angle_l",
    "mtp_angle_r",
    "mtp_angle_l",
)

GOLF_COORDINATE_RANGES: dict[str, tuple[float, float]] = {
    "arm_flex_r": (math.radians(-120.0), math.radians(180.0)),
    "arm_flex_l": (math.radians(-120.0), math.radians(180.0)),
    "lumbar_rotation": (math.radians(-120.0), math.radians(120.0)),
    "wrist_dev_r": (math.radians(-45.0), math.radians(45.0)),
    "wrist_dev_l": (math.radians(-45.0), math.radians(45.0)),
}


def _parse(path: Path) -> ET.ElementTree:
    """Parse an XML file preserving the original declaration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Base OSIM not found: {path}. "
            "Run `git submodule update --init shared/models/opensim/opensim-models`."
        )
    return DefusedET.parse(path)


def _find_one(parent: ET.Element, tag: str) -> ET.Element:
    """Find a single child by tag; raise if missing or duplicated."""
    matches = parent.findall(tag)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one <{tag}> under <{parent.tag}>, found {len(matches)}."
        )
    return matches[0]


def _make_coordinate_actuator(coord_name: str) -> ET.Element:
    """Build a <CoordinateActuator> for a single coordinate."""
    act = ET.Element("CoordinateActuator", attrib={"name": f"tau_{coord_name}"})
    ET.SubElement(act, "appliesForce").text = "true"
    ET.SubElement(act, "min_control").text = "-Inf"
    ET.SubElement(act, "max_control").text = "Inf"
    ET.SubElement(act, "coordinate").text = coord_name
    ET.SubElement(act, "optimal_force").text = "1"
    return act


def _collect_coordinate_names(model: ET.Element) -> list[str]:
    """Return coordinate names in document order (deterministic)."""
    jointset = _find_one(model, "JointSet")
    objects = _find_one(jointset, "objects")
    names: list[str] = []
    for joint in objects:
        coords = joint.find("coordinates")
        if coords is None:
            continue
        for coord in coords.findall("Coordinate"):
            cname = coord.get("name")
            if cname is None:
                continue
            names.append(cname)
    return names


def _apply_golf_coordinate_adjustments(model: ET.Element) -> None:
    """Unlock coordinates required for golf swing and widen excursions."""
    unlocked_set = set(GOLF_UNLOCKED_COORDINATES)
    jointset = _find_one(model, "JointSet")
    objects = _find_one(jointset, "objects")
    for joint in objects:
        coords = joint.find("coordinates")
        if coords is None:
            continue
        for coord in coords.findall("Coordinate"):
            cname = coord.get("name")
            if not cname:
                continue
            if cname in unlocked_set:
                locked = coord.find("locked")
                if locked is not None:
                    locked.text = "false"
            if cname in GOLF_COORDINATE_RANGES:
                lo, hi = GOLF_COORDINATE_RANGES[cname]
                range_elem = coord.find("range")
                if range_elem is not None:
                    range_elem.text = f"{lo:.16g} {hi:.16g}"


def _indent(elem: ET.Element, level: int = 0, *, tab: str = "\t") -> None:
    """Pretty-print indenter that matches the Rajagopal source style (tabs)."""
    pad = "\n" + tab * level
    child_pad = pad + tab
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_pad
        for i, child in enumerate(elem):
            _indent(child, level + 1, tab=tab)
            if i < len(elem) - 1:
                if not child.tail or not child.tail.strip():
                    child.tail = child_pad
            else:
                if not child.tail or not child.tail.strip():
                    child.tail = pad
    else:
        if level and (elem.tail is None or not elem.tail.strip()):
            elem.tail = pad


def _root(tree: ET.ElementTree) -> ET.Element:
    """Root element of ``tree``; raises ``ValueError`` for an empty tree."""
    root = tree.getroot()
    if root is None:
        raise ValueError("XML tree has no root element")
    return root


def build_skeleton(base_osim: Path | None = None) -> ET.ElementTree:
    """The golf skeleton: renamed, golf coordinates unlocked, one actuator each."""
    tree = _parse(base_osim if base_osim is not None else BASE_OSIM)
    root = _root(tree)
    if root.tag != "OpenSimDocument":
        raise ValueError(f"Unexpected root element: {root.tag}")
    model = _find_one(root, "Model")
    model.set("name", "golf_humanoid")
    _apply_golf_coordinate_adjustments(model)
    coord_names = _collect_coordinate_names(model)
    if not coord_names:
        raise ValueError("No coordinates found in the base model — aborting.")
    force_objects = _find_one(_find_one(model, "ForceSet"), "objects")
    for cname in coord_names:
        force_objects.append(_make_coordinate_actuator(cname))
    return tree


def attach_improved_club(
    tree: ET.ElementTree,
    model_name: str,
    *,
    club: str = "driver",
    grip_model: str = msk_club.DEFAULT_GRIP_MODEL,
    calibration_path: Path = msk_club.CALIBRATION_PATH,
) -> ET.ElementTree:
    """Put the shared club in both hands of ``tree`` (replacing any club).

    Raises ``KeyError`` when ``model_name`` has no committed grip calibration
    for ``club`` (run ``msk_club_calibration`` on the model first).
    """
    model = _find_one(_root(tree), "Model")
    calibration = msk_club.load_calibration(model_name, club, calibration_path)
    msk_club.attach_club(
        model, msk_club.load_msk_club(club), calibration, grip_model=grip_model
    )
    return tree


def write_model(tree: ET.ElementTree, output_path: Path) -> Path:
    """Write ``tree`` deterministically (tab indentation, fixed declaration)."""
    root = _root(tree)
    for elem in root.iter():  # re-indent from scratch so rebuilds are byte-stable
        if len(elem):
            elem.text = None
        elem.tail = None
    _indent(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    declaration = b'<?xml version="1.0" encoding="UTF-8" ?>\n'
    output_path.write_bytes(declaration + xml_bytes + b"\n")
    return output_path


def build(
    *,
    output_path: Path = OUTPUT_OSIM,
    club: str = "driver",
    grip_model: str = msk_club.DEFAULT_GRIP_MODEL,
    base_osim: Path | None = None,
    calibration_path: Path = msk_club.CALIBRATION_PATH,
) -> Path:
    """Build golf_humanoid.osim with the improved two-hand club; returns the path."""
    tree = build_skeleton(base_osim)
    attach_improved_club(
        tree,
        output_path.stem,
        club=club,
        grip_model=grip_model,
        calibration_path=calibration_path,
    )
    return write_model(tree, output_path)


def rebuild_club(
    model_path: Path,
    *,
    club: str = "driver",
    grip_model: str = msk_club.DEFAULT_GRIP_MODEL,
    output_path: Path | None = None,
) -> Path:
    """Replace the club of an existing Rajagopal model (e.g. the scaled one)."""
    tree = _parse(model_path)
    attach_improved_club(tree, model_path.stem, club=club, grip_model=grip_model)
    return write_model(tree, output_path or model_path)


def _summary(output: Path) -> str:
    size_kb = output.stat().st_size / 1024.0
    return f"Wrote {output.relative_to(REPO_ROOT)} ({size_kb:.1f} KiB)."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the golf humanoid models.")
    parser.add_argument("--base", type=Path, default=None, help="Rajagopal base")
    parser.add_argument("--club", default="driver", choices=("driver", "iron7"))
    parser.add_argument(
        "--grip-model",
        default=msk_club.DEFAULT_GRIP_MODEL,
        choices=msk_club.GRIP_MODELS,
    )
    parser.add_argument(
        "--skeleton-only",
        type=Path,
        default=None,
        help="Write the club-less skeleton here (input to the calibration)",
    )
    args = parser.parse_args(argv)
    if args.skeleton_only is not None:
        write_model(build_skeleton(args.base), args.skeleton_only)
        return 0
    outputs = [
        build(club=args.club, grip_model=args.grip_model, base_osim=args.base),
        rebuild_club(SCALED_OSIM, club=args.club, grip_model=args.grip_model),
    ]
    for output in outputs:
        sys.stdout.write(_summary(output) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
