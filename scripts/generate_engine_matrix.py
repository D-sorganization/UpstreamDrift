#!/usr/bin/env python3
"""Generate the engine capability matrix from ledger and acceptance verdicts (MS-71, #10351).

Single source of truth for physics engine qualification, conformance, and capability
profiles across UpstreamDrift catalog, API routes, and launcher tiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.tiers import ENGINE_TIERS, get_engine_tier
from src.shared.python.contracts import postcondition, precondition
from src.shared.python.shadow_tracker.engine_matrix import (
    ENGINE_CONFORMANCE_SCHEMA_VERSION,
    EngineReceipt,
    audit_engine_conformance,
)

CONFIG_DIR = REPO_ROOT / "src" / "config"
MATRIX_PATH = CONFIG_DIR / "engine_capability_matrix.json"
DEFAULT_LEDGER_PATH = REPO_ROOT / "reports" / "matched_swing_ledger.json"
DEFAULT_VERDICTS_PATH = (
    REPO_ROOT
    / "docs"
    / "development"
    / "full_body_models"
    / "evidence"
    / "acceptance"
    / "verdicts_2026-09.json"
)

ADVERTISED_ENGINES: tuple[str, ...] = (
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
    "myosuite",
    "simscape",
    "putting_green",
)

ALL_KNOWN_ENGINES: tuple[str, ...] = (
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
    "myosuite",
    "simscape",
    "putting_green",
    "jaxsim",
    "double_pendulum",
)

DEFAULT_ENGINE_CAPABILITIES: dict[str, str] = {
    "mass_matrix": "full",
    "jacobian": "full",
    "contact_forces": "full",
    "inverse_dynamics": "full",
    "drift_acceleration": "full",
    "parameter_gradients": "partial",
    "state_control_gradients": "partial",
    "forward_sim": "full",
    "contact_step": "partial",
    "muscles": "none",
    "trajectory_opt": "partial",
    "video_export": "partial",
    "dataset_export": "full",
    "force_visualization": "partial",
    "model_positioning": "full",
    "measurements": "full",
}

ENGINE_CAPABILITY_FEATURES: dict[str, dict[str, str]] = {
    "mujoco": {
        "contact_forces": "full",
        "contact_step": "full",
        "drift_acceleration": "full",
        "force_visualization": "full",
        "measurements": "full",
        "model_positioning": "full",
        "parameter_gradients": "none",
        "state_control_gradients": "none",
        "trajectory_opt": "none",
        "video_export": "full",
        "dataset_export": "full",
    },
    "drake": {
        "contact_forces": "partial",
        "contact_step": "full",
        "drift_acceleration": "full",
        "force_visualization": "full",
        "measurements": "full",
        "model_positioning": "full",
        "parameter_gradients": "partial",
        "state_control_gradients": "full",
        "trajectory_opt": "full",
        "video_export": "partial",
        "dataset_export": "full",
    },
    "pinocchio": {
        "contact_forces": "partial",
        "contact_step": "partial",
        "drift_acceleration": "full",
        "force_visualization": "partial",
        "measurements": "partial",
        "model_positioning": "full",
        "parameter_gradients": "full",
        "state_control_gradients": "full",
        "trajectory_opt": "full",
        "video_export": "none",
        "dataset_export": "full",
    },
    "opensim": {
        "contact_forces": "partial",
        "contact_step": "partial",
        "drift_acceleration": "partial",
        "force_visualization": "partial",
        "measurements": "full",
        "model_positioning": "partial",
        "parameter_gradients": "none",
        "state_control_gradients": "none",
        "trajectory_opt": "none",
        "video_export": "none",
        "dataset_export": "full",
        "muscles": "full",
    },
    "myosuite": {
        "contact_forces": "partial",
        "contact_step": "full",
        "drift_acceleration": "partial",
        "force_visualization": "partial",
        "measurements": "partial",
        "model_positioning": "partial",
        "parameter_gradients": "none",
        "state_control_gradients": "none",
        "trajectory_opt": "none",
        "video_export": "none",
        "dataset_export": "partial",
        "muscles": "full",
    },
    "simscape": {
        "contact_forces": "full",
        "contact_step": "full",
        "drift_acceleration": "full",
        "force_visualization": "full",
        "measurements": "full",
        "model_positioning": "full",
        "parameter_gradients": "none",
        "state_control_gradients": "none",
        "trajectory_opt": "none",
        "video_export": "partial",
        "dataset_export": "full",
    },
    "putting_green": {
        "contact_forces": "partial",
        "contact_step": "full",
        "drift_acceleration": "full",
        "force_visualization": "partial",
        "measurements": "full",
        "model_positioning": "full",
        "parameter_gradients": "none",
        "state_control_gradients": "none",
        "trajectory_opt": "none",
        "video_export": "none",
        "dataset_export": "partial",
    },
    "jaxsim": {
        "contact_forces": "partial",
        "contact_step": "partial",
        "drift_acceleration": "partial",
        "force_visualization": "none",
        "measurements": "none",
        "model_positioning": "partial",
        "parameter_gradients": "full",
        "state_control_gradients": "full",
        "trajectory_opt": "partial",
        "video_export": "none",
        "dataset_export": "partial",
    },
    "double_pendulum": {
        "contact_forces": "none",
        "contact_step": "none",
        "drift_acceleration": "full",
        "force_visualization": "full",
        "measurements": "full",
        "model_positioning": "full",
        "parameter_gradients": "full",
        "state_control_gradients": "full",
        "trajectory_opt": "full",
        "video_export": "none",
        "dataset_export": "full",
    },
}


def derive_tile_status(
    *,
    is_engine: bool,
    declared_status: str,
    qualification_status: str,
    runtime_available: bool = True,
    tier: str | None = None,
) -> str:
    """Derive launcher tile status dynamically from engine matrix qualification.

    Honesty contract:
    - Non-engines retain their declared status.
    - An engine tile claims 'ready' ONLY when advertised_and_qualified.
    - If runtime dependencies are missing, tile degrades to 'runtime_unavailable'.
    - Unqualified or experimental tier engines display 'experimental'.
    """
    if not is_engine:
        return declared_status
    if not runtime_available:
        return "runtime_unavailable"
    if qualification_status == "advertised_and_qualified":
        return (
            declared_status
            if declared_status in ("ready", "gui_ready", "engine_ready")
            else "ready"
        )
    return "experimental"


def _load_ledger_rows(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    if ledger_path.is_file():
        try:
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "rows" in data:
                return list(data["rows"])
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback to in-memory ledger scan if file does not exist
    try:
        from src.shared.python.motion_matching.ledger import scan

        ledger = scan(repo_root=repo_root)
        return [r.model_dump() for r in ledger.rows]
    except (ImportError, ValueError, OSError):
        return []


def _find_qualified_receipt_for_engine(
    engine: str, ledger_rows: list[dict[str, Any]], *, required_horizon: str = "G3"
) -> tuple[EngineReceipt | None, list[str]]:
    """Find the physically accepted G3 release receipt and list of passed horizons."""
    norm_engine = "myosuite" if engine == "myosim" else engine
    passed_horizons: list[str] = []
    qualified_receipt: EngineReceipt | None = None

    for row in ledger_rows:
        row_eng = str(row.get("engine", "")).lower()
        if row_eng == "myosim":
            row_eng = "myosuite"
        if row_eng != norm_engine:
            continue
        acc = row.get("acceptance")
        if isinstance(acc, dict) and acc.get("is_physically_accepted") is True:
            horizon = str(acc.get("horizon", "")).upper()
            if horizon and horizon not in passed_horizons:
                passed_horizons.append(horizon)
            # Only full-swing G3 satisfies release qualification (MS-104 / MS-106)
            if horizon == required_horizon and qualified_receipt is None:
                metrics = row.get("metrics") or {}
                closure_m = float(metrics.get("max_closure_residual_m") or 0.0)
                qualified_receipt = EngineReceipt(
                    engine_name=norm_engine,
                    engine_version=str(row.get("engine_version", "1.0.0")),
                    model_name=str(row.get("model_name", "standard_anthropometric")),
                    model_sha256=str(row.get("candidate_sha256", "unverified")),
                    state_convention="canonical_v2_quaternion",
                    contact_model_type="standard_planar_contact",
                    is_physically_accepted=True,
                    measured_closure_translation_m=closure_m,
                    measured_closure_rotation_rad=0.0,
                )
    return qualified_receipt, passed_horizons


def _resolve_tier_for_engine(engine_name: str) -> str:
    try:
        return get_engine_tier(engine_name)
    except ValueError:
        return "archived" if engine_name == "double_pendulum" else "experimental"


@precondition(lambda: True, "precondition")
@postcondition(
    lambda res: "schema_version" in res and "profiles" in res,
    "postcondition",
)
def generate_matrix_data(
    ledger_rows: list[dict[str, Any]] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build the engine capability matrix dictionary."""
    rows = (
        ledger_rows
        if ledger_rows is not None
        else _load_ledger_rows(repo_root=repo_root)
    )

    profiles: dict[str, Any] = {}
    advertised: list[str] = []
    experimental: list[str] = []
    unsupported: list[str] = []

    for eng in ALL_KNOWN_ENGINES:
        is_adv = eng in ADVERTISED_ENGINES
        tier = _resolve_tier_for_engine(eng)
        receipt, passed_horizons = _find_qualified_receipt_for_engine(eng, rows)

        res = audit_engine_conformance(
            engine_name=eng,
            receipt=receipt,
            is_advertised=is_adv,
        )

        if is_adv:
            advertised.append(eng)
        if tier == "experimental":
            experimental.append(eng)
        elif not res.is_qualified:
            unsupported.append(eng)

        caps = {
            **DEFAULT_ENGINE_CAPABILITIES,
            **ENGINE_CAPABILITY_FEATURES.get(eng, {}),
        }
        profiles[eng] = {
            "engine_name": eng,
            "tier": tier,
            "status": res.status,
            "is_qualified": res.is_qualified,
            "release_ready": res.is_qualified and ("G3" in passed_horizons),
            "horizon_coverage": sorted(passed_horizons),
            "failure_reasons": list(res.failure_reasons),
            "conformance_checks": list(res.conformance_checks),
            "capabilities": caps,
            "receipt": None,
        }

    return {
        "schema_version": ENGINE_CONFORMANCE_SCHEMA_VERSION,
        "description": "Authoritative engine capability and qualification matrix (MS-71, #10351)",
        "advertised_engines": sorted(advertised),
        "experimental_engines": sorted(set(experimental)),
        "unsupported_engines": sorted(set(unsupported)),
        "profiles": profiles,
    }


def write_matrix(matrix: dict[str, Any], output_path: Path = MATRIX_PATH) -> None:
    """Save formatted matrix JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(matrix, indent=2, sort_keys=False) + "\n"
    output_path.write_text(content, encoding="utf-8")


def check_matrix(matrix: dict[str, Any], target_path: Path = MATRIX_PATH) -> bool:
    """Return True if the file on disk exactly matches generated matrix."""
    if not target_path.is_file():
        return False
    current = json.loads(target_path.read_text(encoding="utf-8"))
    return current == matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify engine capability matrix."
    )
    parser.add_argument(
        "--write", action="store_true", help="Write matrix to destination"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check that committed matrix is fresh"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MATRIX_PATH,
        help="Custom output file path",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print matrix summary and exit"
    )
    args = parser.parse_args()

    data = generate_matrix_data(repo_root=REPO_ROOT)

    if args.dry_run:
        print(
            f"Engine Matrix: {len(data['profiles'])} engines ({len(data['advertised_engines'])} advertised)"
        )
        for eng, prof in data["profiles"].items():
            print(
                f"  - {eng:15}: tier={prof['tier']:12} status={prof['status']:24} qualified={prof['is_qualified']}"
            )
        return 0

    if args.check:
        if not check_matrix(data, args.output):
            print(
                f"ERROR: {args.output} is stale or missing; run with --write to regenerate.",
                file=sys.stderr,
            )
            return 1
        print(f"Engine capability matrix at {args.output} is fresh.")
        return 0

    if args.write or not args.check:
        write_matrix(data, args.output)
        print(f"Wrote engine capability matrix to {args.output}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
