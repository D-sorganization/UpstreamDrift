"""Compare two clean CRBA requalification replays and record provenance."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np

SCHEMA_VERSION = "proximal-distal-articulated-crba-replay-report/v2"
PRIMARY_ARTIFACTS = (
    "articulated_inertia_cross_engine",
    "articulated_native_constraint_discrepancy",
    "articulated_contact_projection",
    "articulated_drift_contact_attribution",
    "articulated_forward_contact",
    "articulated_distributed_grip_atlas",
)
FIGURE_PATHS = (
    "fig_articulated_inertia_cross_engine.pdf",
    "fig_articulated_inertia_cross_engine.svg",
    "fig_articulated_native_constraint_discrepancy.pdf",
    "fig_articulated_contact_projection.pdf",
    "fig_articulated_contact_projection.svg",
    "fig_articulated_drift_contact_attribution.pdf",
    "fig_articulated_drift_contact_attribution.svg",
    "fig_articulated_forward_contact.pdf",
    "fig_articulated_forward_contact.svg",
    "fig_articulated_distributed_grip_atlas.pdf",
    "fig_articulated_distributed_grip_atlas.svg",
)
_REGISTRATION = "data/articulated_crba_requalification.json"
_CLAIM_REGISTRY = "data/claim_audit_registry.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing replay output: {path.as_posix()}")
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _require_equal_bytes(first: Path, second: Path, label: str) -> None:
    if first.read_bytes() != second.read_bytes():
        raise ValueError(f"{label} replay mismatch")


def _compare_json(first: Path, second: Path, label: str) -> dict[str, object]:
    _require_equal_bytes(first, second, label)
    first_object = json.loads(first.read_text(encoding="utf-8"))
    second_object = json.loads(second.read_text(encoding="utf-8"))
    if first_object != second_object:
        raise ValueError(f"{label} canonical JSON replay mismatch")
    return _file_record(first)


def _validate_claim_registry(path: Path) -> dict[str, object]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    claims = registry.get("claims")
    reviews = registry.get("candidate_reviews")
    if not isinstance(claims, list) or not isinstance(reviews, list):
        raise ValueError("claim registry requires claims and candidate_reviews")
    claims_by_id = {claim.get("claim_id"): claim for claim in claims}
    reviews_by_id = {review.get("candidate_id"): review for review in reviews}
    if len(claims_by_id) != len(claims) or None in claims_by_id:
        raise ValueError("claim registry has duplicate or missing claim IDs")
    if len(reviews_by_id) != len(reviews) or None in reviews_by_id:
        raise ValueError("claim registry has duplicate or missing candidate review IDs")
    for claim_id, claim in claims_by_id.items():
        candidate_ids = claim.get("candidate_ids", [])
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"claim registry duplicates candidates for {claim_id}")
        for candidate_id in candidate_ids:
            review = reviews_by_id.get(candidate_id)
            if review is None or claim_id not in review.get("claim_ids", []):
                raise ValueError(
                    f"claim registry is not reciprocal: {claim_id} -> {candidate_id}"
                )
    for candidate_id, review in reviews_by_id.items():
        claim_ids = review.get("claim_ids", [])
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                f"claim registry duplicates claims for candidate {candidate_id}"
            )
        for claim_id in claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None or candidate_id not in claim.get("candidate_ids", []):
                raise ValueError(
                    f"claim registry is not reciprocal: {candidate_id} -> {claim_id}"
                )
    return {
        "claim_count": len(claims),
        "candidate_review_count": len(reviews),
        "reciprocal_links_valid": True,
    }


def _array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    if first.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _compare_npz(first: Path, second: Path, label: str) -> dict[str, object]:
    with np.load(first) as first_archive, np.load(second) as second_archive:
        if first_archive.files != second_archive.files:
            raise ValueError(f"{label} NPZ replay mismatch: member order")
        members: dict[str, dict[str, object]] = {}
        for name in first_archive.files:
            first_array = np.asarray(first_archive[name])
            second_array = np.asarray(second_archive[name])
            if not _array_equal(first_array, second_array):
                raise ValueError(f"{label} NPZ replay mismatch: {name}")
            members[name] = {
                "dtype": str(first_array.dtype),
                "shape": list(first_array.shape),
                "sha256": hashlib.sha256(first_array.tobytes()).hexdigest(),
            }
    return {**_file_record(first), "member_count": len(members), "members": members}


def _validate_environment(environment: dict[str, object]) -> None:
    python_version = environment.get("python_version")
    if not isinstance(python_version, str) or not python_version.startswith(
        ("3.11.", "3.12.")
    ):
        raise ValueError("replay environment requires Python 3.11 or 3.12")
    if environment.get("platform") != "Linux" or environment.get("machine") != (
        "x86_64"
    ):
        raise ValueError("replay environment requires Linux x86_64")
    if (
        environment.get("mujoco_version") != "3.8.0"
        or environment.get("pinocchio_version") != "3.8.0"
    ):
        raise ValueError("replay environment engine identity drifted")
    limits = environment.get("thread_limits")
    if not isinstance(limits, dict) or set(limits.values()) != {"1"}:
        raise ValueError("replay environment must limit every numerical thread pool")
    records = environment.get("distribution_record_sha256")
    if not isinstance(records, dict) or not {"pin", "mujoco"}.issubset(records):
        raise ValueError("replay environment distribution hashes are incomplete")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in records.values()):
        raise ValueError("replay environment distribution hash is invalid")


def _distribution_record_sha256(name: str) -> str:
    distribution = metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if record is None:
        raise ValueError(f"installed distribution lacks RECORD: {name}")
    return hashlib.sha256(record.encode("utf-8")).hexdigest()


def capture_environment() -> dict[str, object]:
    """Capture and validate the native environment executing the comparison."""
    import mujoco
    import pinocchio as pin

    if not all(hasattr(pin, name) for name in ("Model", "crba", "rnea")):
        raise ValueError("pinocchio import is not the robotics distribution")
    distributions = (
        "pin",
        "mujoco",
        "numpy",
        "cmeel-urdfdom",
        "cmeel-tinyxml2",
    )
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "mujoco_version": mujoco.__version__,
        "pinocchio_version": pin.__version__,
        "distribution_versions": {
            name: metadata.version(name) for name in distributions
        },
        "distribution_record_sha256": {
            name: _distribution_record_sha256(name) for name in distributions
        },
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    _validate_environment(environment)
    return environment


def build_replay_report(
    first_article: Path,
    second_article: Path,
    *,
    source_revision: str,
    environment: dict[str, object],
) -> dict[str, object]:
    """Return a fail-closed comparison report for two independent replays."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be a 40-character lowercase SHA")
    _validate_environment(environment)
    first_article = Path(first_article).resolve()
    second_article = Path(second_article).resolve()
    registration = _compare_json(
        first_article / _REGISTRATION,
        second_article / _REGISTRATION,
        "registration",
    )
    artifacts: dict[str, dict[str, object]] = {}
    npz_member_count = 0
    for name in PRIMARY_ARTIFACTS:
        json_record = _compare_json(
            first_article / "data" / f"{name}.json",
            second_article / "data" / f"{name}.json",
            f"{name} JSON",
        )
        npz_record = _compare_npz(
            first_article / "data" / f"{name}.npz",
            second_article / "data" / f"{name}.npz",
            name,
        )
        npz_member_count += int(npz_record["member_count"])
        artifacts[name] = {"json": json_record, "npz": npz_record}
    figures: dict[str, dict[str, object]] = {}
    for name in FIGURE_PATHS:
        first = first_article / "figures" / name
        second = second_article / "figures" / name
        if first.read_bytes() != second.read_bytes():
            raise ValueError(f"figure replay mismatch: {name}")
        figures[name] = _file_record(first)
    first_registry = first_article / _CLAIM_REGISTRY
    second_registry = second_article / _CLAIM_REGISTRY
    _require_equal_bytes(first_registry, second_registry, "claim registry")
    registry_validation = _validate_claim_registry(first_registry)
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "native_synthetic_replay_qualification",
        "source_revision": source_revision,
        "registration": registration,
        "environment": environment,
        "execution_contract": {
            "clean_replay_count": 2,
            "maximum_workers": 1,
            "outcomes_hidden_until_both_replays_completed": True,
        },
        "artifacts": artifacts,
        "figures": figures,
        "claim_registry": {
            **_file_record(first_registry),
            **registry_validation,
        },
        "summary": {
            "primary_artifact_count": len(artifacts),
            "npz_member_count": npz_member_count,
            "figure_count": len(figures),
            "claim_registry_exact": True,
            "claim_registry_reciprocal": True,
        },
        "all_replay_gates_passed": True,
        "promotion_status": (
            "replay_qualified_pending_claim_pdf_release_and_protected_ci"
        ),
        "inference_boundary": (
            "This report qualifies deterministic execution only for the declared "
            "synthetic articulated models. It is not human, anatomical, equipment, "
            "injury, population, or coaching evidence."
        ),
    }


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--replay-one", type=Path, required=True)
    parser.add_argument("--replay-two", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Write a deterministic report after both clean replays complete."""
    args = _parser().parse_args()
    report = build_replay_report(
        args.replay_one,
        args.replay_two,
        source_revision=args.source_revision,
        environment=capture_environment(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
