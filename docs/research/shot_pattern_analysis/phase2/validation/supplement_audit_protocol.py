"""Independent strict inventory checks; no generic small-matrix exemption."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOLS = {
    "pw-primary-face-bounds/1": "pw_within_lean_face",
    "pw-control-face-bounds/1": "pw_fixed_target_face",
    "driver-control-speed-bounds/1": "driver_fixed_target_speed",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_strict(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def _load_review_inputs(evidence: Path) -> tuple[dict, dict]:
    reviews = evidence / "review_followups"
    inventory = load_json_strict(reviews / "SUPPLEMENTAL_EXECUTION_INVENTORY.json")
    for name, digest in inventory["protocol_plan_sha256"].items():
        if _sha(reviews / name) != digest:
            raise ValueError("predeclared protocol hash mismatch")
    original_path = evidence / "final_v1/matrix_plan.json"
    face = load_json_strict(reviews / "SUPPLEMENTAL_BOUNDS_PLAN.json")
    if _sha(original_path) != face["original_primary_plan_sha256"]:
        raise ValueError("original comparison plan hash mismatch")
    return inventory, load_json_strict(original_path)


def validate_supplement_plan(plan: dict, expected_stage: str, evidence: Path) -> dict:
    """Compare every config field with reviewed inputs plus exact overrides."""
    protocol = plan.get("supplement_protocol")
    if protocol not in PROTOCOLS or expected_stage not in ("pilot", "final"):
        raise ValueError("explicit reviewed supplement protocol and stage required")
    if plan.get("stage") != expected_stage:
        raise ValueError("supplement stage mismatch")
    inventory, original = _load_review_inputs(evidence)
    entries = [
        row for row in inventory["entries"] if row["protocol"] == PROTOCOLS[protocol]
    ]
    names = [row["cell"] for row in entries]
    if len(names) != (18 if protocol == "pw-primary-face-bounds/1" else 4) or len(
        set(names)
    ) != len(names):
        raise ValueError("invalid reviewed inventory")
    selected = plan.get("selected_cell_names", [])
    if (
        len(selected) != len(names)
        or set(selected) != set(names)
        or set(plan["cells"]) != set(names)
    ):
        raise ValueError("missing, extra, or duplicate supplement cells")
    for entry in entries:
        expected = deepcopy(original["cells"][entry["cell"]])
        speed = expected["club"]["club_speed_mps"]
        expected.update(
            n_shots=32 if expected_stage == "pilot" else 10000,
            seed=entry["seed"],
            target_x_m=entry["target_x_m"],
            match_tolerance_m=entry["match_tolerance_m"],
            match_face_bounds_deg=entry["face_bounds_deg"],
            match_speed_factors=[v / speed for v in entry["speed_bounds_mps"]],
        )
        if plan["cells"][entry["cell"]] != expected:
            raise ValueError(
                f"supplement configuration differs from predeclaration: {entry['cell']}"
            )
    return {
        "protocol": protocol,
        "stage": expected_stage,
        "unique_cells": len(names),
        "scope": "Per-protocol unique executions; control aliases require separate verification",
    }


def validate_control_aliases(path: Path, expected_stage: str, evidence: Path) -> dict:
    """Verify four PW0 references without treating them as fresh simulations."""
    import math

    envelope = load_json_strict(path)
    inventory, _ = _load_review_inputs(evidence)
    mapping = inventory["control_aliases_to_primary"]
    refs = envelope.get("references", [])
    if envelope.get("schema") != "matched-study-control-aliases/1" or len(refs) != 4:
        raise ValueError("exact four-reference alias envelope required")
    names = [ref.get("alias_cell") for ref in refs]
    if len(set(names)) != 4 or set(names) != set(mapping):
        raise ValueError("missing, extra, or duplicate control aliases")
    allowed = {
        "protocol_id",
        "alias_cell",
        "alias_of_primary",
        "execution_kind",
        "source_root",
        "source_plan_sha256",
        "source_cell_manifest_sha256",
        "source_csv_sha256",
        "requested_config",
    }
    for ref in refs:
        if (
            set(ref) != allowed
            or ref["execution_kind"] != "provenance_alias"
            or ref["protocol_id"] != "supplement_control_alias"
        ):
            raise ValueError(
                "alias must not claim independent execution or extra results"
            )
        name = ref["alias_of_primary"]
        if name != mapping[ref["alias_cell"]]:
            raise ValueError("alias source identity mismatch")
        root = Path(ref["source_root"])
        plan = load_json_strict(root / "matrix_plan.json")
        validate_supplement_plan(plan, expected_stage, evidence)
        if (
            plan["supplement_protocol"] != "pw-primary-face-bounds/1"
            or _sha(root / "matrix_plan.json") != ref["source_plan_sha256"]
        ):
            raise ValueError("alias source plan mismatch")
        folder = root / name
        if _sha(folder / "manifest.json") != ref["source_cell_manifest_sha256"]:
            raise ValueError("alias source manifest mismatch")
        manifest = load_json_strict(folder / "manifest.json")
        for filename, digest in manifest["files_sha256"].items():
            if _sha(folder / filename) != digest:
                raise ValueError("alias source bytes changed")
        if _sha(folder / "shots.csv.gz") != ref["source_csv_sha256"]:
            raise ValueError("alias sample hash mismatch")
        summary = load_json_strict(folder / "summary.json")
        config = summary["config"]
        if config != plan["cells"][name] or config["target_x_m"] is not None:
            raise ValueError("alias source config mismatch")
        target = summary["target_x_m"]
        if not math.isfinite(target) or abs(target - 77.40247724866299) > 1e-9:
            raise ValueError("alias physical target mismatch")
        requested = deepcopy(config)
        requested["target_x_m"] = target
        if ref["requested_config"] != requested:
            raise ValueError("alias requested physical/noise/scoring config mismatch")
        for pattern in ("Straight", "Draw", "Fade"):
            shot = summary["nominals"][pattern]["outcome"]
            if (
                math.hypot(shot["carry_x_m"] - target, shot["carry_y_m"])
                > config["match_tolerance_m"]
            ):
                raise ValueError("alias nominal residual fails")
    return {
        "verified_aliases": 4,
        "new_executions": 0,
        "reference_file_sha256": _sha(path),
    }
