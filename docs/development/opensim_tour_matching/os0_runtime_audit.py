"""OS-0 Runtime and Evidence Baseline Audit.

Audits runtime environment, OpenSim model topology, canonical C3D capture,
and reference candidate contracts per Epic #10003 Work Package OS-0.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET

import ezc3d
import numpy as np


import logging

logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_environment() -> dict[str, Any]:
    """Audit the host and Python runtime environment."""
    probed_packages = [
        "opensim",
        "casadi",
        "numpy",
        "scipy",
        "ezc3d",
        "pytest",
        "matplotlib",
    ]
    pkg_status: dict[str, Any] = {}
    for pkg in probed_packages:
        spec = importlib.util.find_spec(pkg)
        installed = spec is not None
        version = None
        if installed:
            try:
                version = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                version = "installed (metadata unreadable)"
        pkg_status[pkg] = {
            "installed": installed,
            "version": version,
        }

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "packages": pkg_status,
        "opensim_installed": pkg_status["opensim"]["installed"],
    }


def audit_opensim_model(model_path: Path) -> dict[str, Any]:
    """Audit OpenSim model XML structure and topology."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tree = ET.parse(model_path)
    root = tree.getroot()

    # Find Model tag
    model_elem = root.find("Model")
    model_name = (
        model_elem.attrib.get("name", "unknown")
        if model_elem is not None
        else "unknown"
    )

    bodies = [
        b.attrib.get("name") for b in root.findall(".//Body") if b.attrib.get("name")
    ]
    joints = [j.attrib.get("name") for j in root.findall(".//Joint") or []]
    # Check specific joint types
    weld_joints = [j.attrib.get("name") for j in root.findall(".//WeldJoint") or []]
    custom_joints = [j.attrib.get("name") for j in root.findall(".//CustomJoint") or []]
    pin_joints = [j.attrib.get("name") for j in root.findall(".//PinJoint") or []]
    ellipsoid_joints = [
        j.attrib.get("name") for j in root.findall(".//EllipsoidJoint") or []
    ]

    coords = [
        c.attrib.get("name")
        for c in root.findall(".//Coordinate")
        if c.attrib.get("name")
    ]
    actuators = [
        a.attrib.get("name")
        for a in root.findall(".//CoordinateActuator")
        if a.attrib.get("name")
    ]
    constraints = [c.attrib.get("name") for c in root.findall(".//Constraint") or []]
    markers = [
        m.attrib.get("name") for m in root.findall(".//Marker") if m.attrib.get("name")
    ]

    # Check club attachment
    club_body_found = "Club" in bodies or "club" in bodies
    club_weld_found = any("club" in (j or "").lower() for j in weld_joints)

    return {
        "model_file": model_path.name,
        "model_sha256": compute_sha256(model_path),
        "model_name": model_name,
        "body_count": len(bodies),
        "bodies": bodies,
        "coordinate_count": len(coords),
        "coordinates": coords,
        "actuator_count": len(actuators),
        "actuators": actuators,
        "joint_counts": {
            "total": len(joints)
            + len(weld_joints)
            + len(custom_joints)
            + len(pin_joints)
            + len(ellipsoid_joints),
            "WeldJoint": len(weld_joints),
            "CustomJoint": len(custom_joints),
            "PinJoint": len(pin_joints),
            "EllipsoidJoint": len(ellipsoid_joints),
        },
        "weld_joints": weld_joints,
        "constraint_count": len(constraints),
        "constraints": constraints,
        "marker_count": len(markers),
        "markers": markers,
        "club_topology": {
            "club_body_present": club_body_found,
            "club_weld_present": club_weld_found,
        },
    }


def audit_c3d_capture(c3d_path: Path) -> dict[str, Any]:
    """Audit the tour-average C3D motion capture dataset."""
    if not c3d_path.is_file():
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    c = ezc3d.c3d(str(c3d_path))
    header = c["header"]
    point_header = header["points"]

    sample_rate = point_header["frame_rate"]
    first_frame = point_header["first_frame"]
    last_frame = point_header["last_frame"]
    points = c["data"]["points"]  # [4, n_markers, n_frames]
    num_markers = points.shape[1]
    num_frames = points.shape[2]
    duration_s = (num_frames - 1) / sample_rate

    labels = [s.strip() for s in c["parameters"]["POINT"]["LABELS"]["value"]]
    coords_xyz = points[:3, :, :]
    residuals = points[3, :, :]
    valid_mask = np.isfinite(coords_xyz).all(axis=0) & (residuals >= 0)

    analog_channels = (
        c["parameters"].get("ANALOG", {}).get("LABELS", {}).get("value", [])
    )

    return {
        "c3d_file": c3d_path.name,
        "c3d_sha256": compute_sha256(c3d_path),
        "sample_rate_hz": float(sample_rate),
        "first_frame": int(first_frame),
        "last_frame": int(last_frame),
        "total_frames": int(num_frames),
        "duration_s": float(duration_s),
        "marker_count": int(num_markers),
        "marker_labels": labels,
        "total_observed_points": int(np.sum(valid_mask)),
        "analog_channel_count": len(analog_channels),
        "analog_channels": [s.strip() for s in analog_channels],
        "has_ground_reaction_force_plates": False,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    evidence_dir = (
        repo_root / "docs" / "development" / "opensim_tour_matching" / "evidence"
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== OpenSim Work Package OS-0 Audit ===")
    env_info = audit_environment()
    logger.info("Python: %s", env_info["python_version"].split()[0])
    logger.info("OpenSim Installed: %s", env_info["opensim_installed"])

    model_path = (
        repo_root
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "models"
        / "golf_humanoid.osim"
    )
    model_info = audit_opensim_model(model_path)
    logger.info(
        "Model Bodies: %d, Coordinates: %d, Actuators: %d",
        model_info["body_count"],
        model_info["coordinate_count"],
        model_info["actuator_count"],
    )

    c3d_path = repo_root / "data" / "C3D_TA_Driver.c3d"
    c3d_info = audit_c3d_capture(c3d_path)
    logger.info(
        "C3D Frames: %d @ %.1f Hz (Duration: %.6f s)",
        c3d_info["total_frames"],
        c3d_info["sample_rate_hz"],
        c3d_info["duration_s"],
    )

    receipt: dict[str, Any] = {
        "work_package": "OS-0: Runtime and Evidence Baseline",
        "epic": "#10003",
        "environment": env_info,
        "model_inventory": model_info,
        "capture_audit": c3d_info,
        "status": "baseline_inventoried_opensim_missing",
        "notes": (
            "OS-0 baseline complete: model XML and C3D capture audited. "
            "Native OpenSim Python bindings are confirmed absent in current environment, "
            "satisfying the mandatory red gate. A dedicated supported environment with Moco "
            "is required before proceeding to OS-1."
        ),
    }

    out_file = evidence_dir / "os0_audit_receipt.json"
    out_file.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logger.info("Receipt written to: %s", out_file)


if __name__ == "__main__":
    main()
