"""Portable research extraction snapshots bound to exact trajectory bytes.

The sidecar leaves the six-field flight wire and aero provenance untouched. It
restores retained state without SDKs, replay, impact or flight recomputation.
Recorded parent hashes are provenance claims, not fresh Library authentication;
loading never promotes source-clock, contact or physical qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.motion_matching.ledger import find_repo_root
from src.shared.python.physics import SwingState
from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID
from src.shared.python.shadow_tracker import check_sha256

from .necromatcher_impact import BASIS_PROBE_LENGTH_M
from .necromatcher_impact_contracts import (
    ReplayImpactGeometry,
    ReplayImpactSelection,
    finite_array,
    nonempty,
)

__all__ = ["export_replay_impact_receipt", "load_replay_impact_receipt"]

RECEIPT_SCHEMA = "necromatcher/replay-impact-receipt/1"
_STATE_FIELDS = frozenset(
    {
        "clubhead_velocity",
        "clubhead_angular_velocity",
        "clubhead_orientation",
        "clubhead_mass",
        "clubhead_loft_deg",
        "clubhead_moi",
        "impact_offset",
        "engine_name",
        "metadata",
    }
)


def _json_copy(value: Any) -> Any:
    """Detach JSON data, refusing NaN or silent tuple/non-string-key conversion."""
    clone = json.loads(json.dumps(value, allow_nan=False))
    if clone != value:
        raise ValueError("Extraction metadata must already use JSON-native values")
    return clone


def _metadata(value: object) -> dict[str, Any]:
    """Validate supported retained extraction declarations without re-admission."""
    if not isinstance(value, dict):
        raise ValueError("Extraction metadata must be an object")
    data: dict[str, Any] = _json_copy(value)
    literals = {
        "frame_id": FLIGHT_FRAME_ID,
        "impact_extraction_schema": "necromatcher/replay-impact/1",
        "replay_clock_policy": "authored_simulation_seconds",
    }
    for name, expected in literals.items():
        if data.get(name) != expected:
            raise ValueError(f"Unsupported extraction {name}")
    for name in (
        "scientific_qualified",
        "physical_source_time_qualified",
        "capture_pts_used_for_velocity",
    ):
        if data.get(name) is not False:
            raise ValueError(f"Extraction {name} must be explicitly false")
    for name in ("recorded_sample_index", "capture_initial_frame_index"):
        if type(data.get(name)) is not int or data[name] < 0:
            raise ValueError(f"Extraction {name} requires a nonnegative integer")
    time = finite_array(
        data.get("recorded_time_s"), (), "Recorded authored time"
    ).item()
    if time < 0:
        raise ValueError("Recorded authored time must be nonnegative")
    for name in ("replay", "profile", "fit", "model", "capture"):
        nonempty(data.get(name + "_id"), name + " identity")
        digest = data.get(name + "_hash")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("Parent hashes require the canonical sha256 prefix")
        check_sha256(digest[7:], name + " hash")
    geometry_record = data.get("geometry")
    selection_record = data.get("selection")
    if not isinstance(geometry_record, dict) or not isinstance(selection_record, dict):
        raise ValueError("Extraction geometry and selection require typed records")
    ReplayImpactGeometry.from_record(geometry_record)
    selection = ReplayImpactSelection.from_record(selection_record)
    if selection.recorded_sample_index != data["recorded_sample_index"]:
        raise ValueError("Selection and retained sample index differ")
    finite_array(data.get("head_point_flight_m"), (3,), "Flight head point")
    probe = finite_array(
        data.get("basis_probe_length_m"), (), "Basis probe length"
    ).item()
    if probe != BASIS_PROBE_LENGTH_M:
        raise ValueError("Unsupported derivative basis probe length")
    if (
        not isinstance(data.get("capture_initial_frame"), dict)
        or not data["capture_initial_frame"]
    ):
        raise ValueError("Retained capture frame identity must be a nonempty object")
    return data


def _restore_state(record: object) -> SwingState:
    """Return detached SI arrays and metadata, refusing malformed state fields."""
    if not isinstance(record, dict) or set(record) != _STATE_FIELDS:
        raise ValueError("Receipt state requires exactly the supported state fields")
    data = dict(record)
    for name in (
        "clubhead_velocity",
        "clubhead_angular_velocity",
        "clubhead_orientation",
    ):
        data[name] = finite_array(data[name], (3,), name)
    if not np.isclose(
        np.linalg.norm(data["clubhead_orientation"]), 1, rtol=0, atol=1e-12
    ):
        raise ValueError("Club face normal must be unit length")
    for name in ("clubhead_mass", "clubhead_moi", "clubhead_loft_deg"):
        data[name] = float(finite_array(data[name], (), name).item())
    if data["clubhead_mass"] <= 0 or data["clubhead_moi"] <= 0:
        raise ValueError("Effective impact mass and MOI must be positive")
    if data["impact_offset"] is not None:
        data["impact_offset"] = finite_array(
            data["impact_offset"], (2,), "Impact offset"
        )
    nonempty(data["engine_name"], "Engine name")
    data["metadata"] = _metadata(data["metadata"])
    geometry = ReplayImpactGeometry.from_record(data["metadata"]["geometry"])
    if (
        data["clubhead_mass"] != geometry.mass_kg
        or data["clubhead_moi"] != geometry.moi_kg_m2
    ):
        raise ValueError("Retained mass/MOI differs from authored geometry assumptions")
    return SwingState(**data)


def _state_record(state: SwingState) -> dict[str, Any]:
    if not isinstance(state, SwingState):
        raise ValueError("Receipt export requires an extracted SwingState")
    copied = _restore_state(
        {field.name: getattr(state, field.name) for field in fields(SwingState)}
    )
    record = {field.name: getattr(copied, field.name) for field in fields(SwingState)}
    for name in (
        "clubhead_velocity",
        "clubhead_angular_velocity",
        "clubhead_orientation",
        "impact_offset",
    ):
        if record[name] is not None:
            record[name] = record[name].tolist()
    return record


def _trajectory_pin(raw: bytes) -> dict[str, Any]:
    """Admit the same buffer through the canonical public viewer wire reader."""
    from src.launchers.tools_repo_path import ensure_tools_importable

    ensure_tools_importable(
        find_repo_root(Path(__file__)), os.environ.get("TOOLS_REPO_PATH")
    )
    from shared.python.swing_sim.flight_interchange import (
        ball_flight_trajectory_from_json,
    )

    trajectory = ball_flight_trajectory_from_json(raw.decode("utf-8"))
    if trajectory.frame_id != FLIGHT_FRAME_ID:
        raise ValueError(
            "Trajectory and extraction must use the canonical flight frame"
        )
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _publish_exclusive(destination: Path, raw: bytes) -> Path:
    """Flush an owned stage and atomically link without replacing any destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False
        ) as stream:
            staged = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, destination)
    except BaseException as primary:
        if staged is not None:
            try:
                staged.unlink(missing_ok=True)
            except OSError as cleanup:
                primary.add_note(f"Owned receipt stage cleanup failed: {cleanup}")
        raise
    staged.unlink()
    return destination


def export_replay_impact_receipt(
    state: SwingState, trajectory: Path, destination: Path
) -> Path:
    """Exclusively publish a versioned retained-state sidecar for exact wire bytes.

    Postconditions: trajectory bytes are unchanged; only the new sidecar is
    published. Existing destinations fail closed. Hashes bind serialized bytes,
    not a newly verified replay/contact event or calibrated historical clock.
    """
    record = {
        "schema": RECEIPT_SCHEMA,
        "state": _state_record(state),
        "trajectory": _trajectory_pin(Path(trajectory).read_bytes()),
    }
    raw = (json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return _publish_exclusive(Path(destination), raw)


def load_replay_impact_receipt(receipt: Path, trajectory: Path) -> SwingState:
    """Verify a portable receipt against exact trajectory bytes and restore state.

    No SDK, Library, simulation or viewer launch occurs. Returned arrays and
    nested metadata are detached. Retained research flags must remain false;
    the receipt provides serialized provenance, not fresh parent authentication.
    """
    record = json.loads(Path(receipt).read_bytes())
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "state", "trajectory"}
        or record["schema"] != RECEIPT_SCHEMA
    ):
        raise ValueError("Unsupported replay impact receipt envelope")
    pin = record["trajectory"]
    if not isinstance(pin, dict) or set(pin) != {"sha256", "size_bytes"}:
        raise ValueError("Receipt trajectory pin requires exact hash and size")
    check_sha256(pin["sha256"], "Trajectory hash")
    if type(pin["size_bytes"]) is not int or pin["size_bytes"] <= 0:
        raise ValueError("Trajectory byte size must be a positive integer")
    raw = Path(trajectory).read_bytes()
    if (
        len(raw) != pin["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != pin["sha256"]
    ):
        raise ValueError("Trajectory bytes differ from receipt hash or size")
    if _trajectory_pin(raw) != pin:
        raise ValueError("Trajectory pin differs")
    return _restore_state(record["state"])
