"""Export a finished ground-support run as a package for MuJoCo MJX (MM-7b, #10109, #10520).

Writes mjx_package.xml, mjx_package.npz, mjx_package.json to the run directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DET
import numpy as np

from src.engines.physics_engines.mujoco.python.full_body_ik import (
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ROOT_COORDINATES,
)
from src.shared.python.motion_matching.pipeline import (
    BALANCE,
    CAPTURES,
    OMEGA_RAD_S,
    RATE_HZ,
    TRACKING_CUTOFF_HZ,
    Lane,
    smooth_reference,
)

log = logging.getLogger("mjx_export")


def stiffen_weld(xml: str) -> str:
    """Return MJCF with exported site weld ('native_grip') made stiff for MJX."""
    root = DET.fromstring(xml)
    equality = root.find("equality")
    if equality is None:
        raise ValueError("Exported MJCF carries no equality block")
    welds = [w for w in equality if w.tag == "weld"]
    if len(welds) != 1:
        raise ValueError("Expected exactly one closure weld in the MJCF")
    welds[0].set("solref", "0.002 1")
    welds[0].set("solimp", "0.99 0.999 0.001")
    return ET.tostring(root, encoding="unicode")


def _load_reference_trajectory(
    run: Path, receipt: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand_file = run / "candidate.npz"
    if cand_file.is_file():
        from src.shared.python.motion_matching.candidate_io import load_candidate

        cand = load_candidate(cand_file)
        times, q_ref = cand.time_s, cand.q
        ik = (
            np.load(run / "ik_trajectory.npz")
            if (run / "ik_trajectory.npz").is_file()
            else None
        )
        valid = (
            cand.marker_validity
            if cand.marker_validity is not None
            else (
                ik["valid"]
                if ik is not None
                else np.ones((len(times), len(receipt["labels"])), dtype=bool)
            )
        )
        return times, q_ref, valid

    ik = np.load(run / "ik_trajectory.npz")
    return ik["time_s"], ik["q_ref"], ik["valid"]


def _write_mjx_outputs(
    run: Path,
    xml: str,
    meta: dict[str, Any],
    package_arrays: dict[str, Any],
) -> None:
    np.savez(run / "mjx_package.npz", **package_arrays)
    (run / "mjx_package.xml").write_text(xml, encoding="utf-8")
    (run / "mjx_package.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def export_mjx_package(run: Path | str, timestep: float = 5e-4) -> dict[str, Any]:
    run_path = Path(run)
    receipt = json.loads((run_path / "receipt.json").read_text(encoding="utf-8"))
    spec_bytes = (run_path / "full_body_spec_hipcal_scaled.json").read_bytes()
    times, q_ref, valid = _load_reference_trajectory(run_path, receipt)
    labels = tuple(receipt["labels"])
    lane = Lane(labels, CAPTURES[receipt["capture"]])
    adapter = NativeMujocoFullBodyModel(spec_bytes)
    attachments = {
        label: (a["body"], a["offset_m"])
        for label, a in receipt["ik"]["attachments_m"].items()
    }
    kin = FullBodyMarkerKinematics(adapter, {k: attachments[k] for k in labels})
    model = adapter.model
    q_track = smooth_reference(q_ref, RATE_HZ, TRACKING_CUTOFF_HZ)

    xml = stiffen_weld(adapter.xml)
    root = DET.fromstring(xml)
    option = root.find("option")
    if option is None:
        raise ValueError("Exported MJCF carries no option tag")
    option.set("timestep", f"{timestep:g}")
    option.set("integrator", "Euler")
    xml = ET.tostring(root, encoding="unicode")

    names = list(adapter.coordinate_order)
    qpos_adr = np.array([model.joint(n).qposadr[0] for n in names])
    dof_adr = np.array([model.joint(n).dofadr[0] for n in names])
    root_mask = np.array([n in ROOT_COORDINATES for n in names])
    spheres = adapter._spheres
    sphere_names = list(spheres)
    contact = adapter.contact_parameters
    ground = adapter.ground_plane
    package_arrays = {
        "time_s": times,
        "q_track": q_track,
        "q_ref": q_ref,
        "targets_m": lane.points,
        "valid": valid,
        "marker_body_ids": np.array(kin._body_ids),
        "marker_local_m": np.array(kin._local),
        "qpos_adr": qpos_adr,
        "dof_adr": dof_adr,
        "root_mask": root_mask,
        "sphere_site_ids": np.array([spheres[n]["site_id"] for n in sphere_names]),
        "sphere_body_ids": np.array([spheres[n]["body_id"] for n in sphere_names]),
        "sphere_radii_m": np.array([spheres[n]["radius"] for n in sphere_names]),
        "ground_normal": np.asarray(ground.normal, dtype=float),
        "ground_height_m": np.array(ground.height_m),
    }
    meta = {
        "run": run_path.name,
        "capture": receipt["capture"],
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "coordinate_order": names,
        "labels": list(labels),
        "sphere_names": sphere_names,
        "contact": contact.as_document(),
        "controller": {
            "omega_rad_s": OMEGA_RAD_S,
            "zeta": 1.0,
            "balance": list(BALANCE),
            "note": "computed torque on the actuated coordinates with the root free; the MJX port omits the centre-of-mass balance term",
        },
        "timestep_s": timestep,
        "rate_hz": RATE_HZ,
        "mass_kg": float(np.sum(model.body_mass)),
        "gravity_m_s2": list(map(float, model.opt.gravity)),
        "closure": {
            "site_a": int(adapter._closure[0]),
            "site_b": int(adapter._closure[1]),
        },
        "baseline": {
            "replay_marker_rms_m": receipt["dynamics"]["marker_rms_m"],
            "reference_marker_rms_m": receipt["ik"]["reference"]["marker_rms_m"],
        },
    }
    _write_mjx_outputs(run_path, xml, meta, package_arrays)
    log.info(
        "exported %s: %d frames, %d markers, %d spheres, %d qpos",
        run_path.name,
        len(times),
        len(labels),
        len(sphere_names),
        model.nq,
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--timestep", type=float, default=5e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    export_mjx_package(args.run, args.timestep)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
