"""Fail-closed aggregation of the preregistered #9153 rigid smoke."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from scripts.research.proximal_distal_energy.articulated_forward_attribution_runner import (
    CaseCheckpoint,
    load_registered_checkpoints,
    manifest_sha256,
)
from scripts.research.proximal_distal_energy.articulated_forward_attribution_study import (
    assess_closure_refinement,
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _payload(checkpoint: CaseCheckpoint) -> Mapping[str, Any]:
    try:
        value = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"validated checkpoint became unreadable: {checkpoint.path}"
        ) from exc
    return _mapping(value, name="checkpoint")


def _checkpoint_inventory(
    checkpoints: tuple[CaseCheckpoint, ...],
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.path.name):
        try:
            content = checkpoint.path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"validated checkpoint became unreadable: {checkpoint.path}"
            ) from exc
        files.append(
            {
                "name": checkpoint.path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "count": len(files),
        "checkpoint_set_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _completed_group(
    *,
    checkpoints: list[CaseCheckpoint],
    payloads: list[Mapping[str, Any]],
    tolerances: Mapping[str, Any],
) -> dict[str, object]:
    cases = [item.case for item in checkpoints]
    steps = tuple(case.time_step_s for case in cases)
    momentum: list[float] = []
    work: list[float] = []
    case_failures: list[str] = []
    for payload in payloads:
        outcome = _mapping(payload.get("outcome"), name="outcome")
        result = _mapping(outcome.get("result"), name="outcome.result")
        closure = _mapping(result.get("closure"), name="outcome.result.closure")
        momentum.append(
            _finite_number(
                closure.get("momentum_relative_residual"),
                name="momentum_relative_residual",
            )
        )
        work.append(
            _finite_number(
                closure.get("work_relative_residual"),
                name="work_relative_residual",
            )
        )
        raw_failures = closure.get("failure_codes")
        if not isinstance(raw_failures, list) or any(
            not isinstance(code, str) or not code for code in raw_failures
        ):
            raise ValueError("closure.failure_codes must be a list of strings")
        case_failures.extend(raw_failures)
        expected_pass = not raw_failures
        if closure.get("passes_registered_tolerances") is not expected_pass:
            raise ValueError("closure pass flag disagrees with its failure codes")
    assessment = assess_closure_refinement(
        time_steps_s=steps,
        momentum_relative_residuals=tuple(momentum),
        work_relative_residuals=tuple(work),
        momentum_tolerance=_finite_number(
            tolerances.get("momentum_relative"), name="momentum_relative tolerance"
        ),
        work_tolerance=_finite_number(
            tolerances.get("work_relative"), name="work_relative tolerance"
        ),
        refinement_ratio_limit=_finite_number(
            tolerances.get("refinement_ratio_limit"),
            name="refinement_ratio_limit",
        ),
    )
    failures = list(dict.fromkeys((*case_failures, *assessment.failure_codes)))
    return {
        "status": "completed",
        "time_steps_s": list(steps),
        "momentum_relative_residuals": momentum,
        "work_relative_residuals": work,
        "momentum_refinement_ratios": list(assessment.momentum_refinement_ratios),
        "work_refinement_ratios": list(assessment.work_refinement_ratios),
        "failure_codes": failures,
        "passes": not failures,
    }


def _unavailable_group(payloads: list[Mapping[str, Any]]) -> dict[str, object]:
    details: list[str] = []
    for payload in payloads:
        outcome = _mapping(payload.get("outcome"), name="outcome")
        failure = _mapping(outcome.get("failure"), name="outcome.failure")
        if failure.get("code") != "native_engine_unavailable":
            raise ValueError("unavailable checkpoint has an unregistered failure code")
        detail = failure.get("detail")
        if not isinstance(detail, str) or not detail:
            raise ValueError("native-unavailability detail must be nonempty")
        details.append(detail)
    return {
        "status": "unavailable",
        "failure_codes": ["native_engine_unavailable"],
        "failure_details": list(dict.fromkeys(details)),
        "passes": False,
    }


def aggregate_registered_smoke(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
) -> dict[str, object]:
    """Aggregate a complete identity-valid run without suppressing failures."""

    checkpoints = load_registered_checkpoints(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    payload_by_path = {item.path: _payload(item) for item in checkpoints}
    counts = Counter(item.status for item in checkpoints)
    grouped: dict[tuple[int, int, float, str, str], list[CaseCheckpoint]] = {}
    for checkpoint in checkpoints:
        case = checkpoint.case
        key = (
            case.source_case_index,
            case.source_sample_index,
            case.source_time_s,
            case.engine,
            case.variant,
        )
        grouped.setdefault(key, []).append(checkpoint)
    tolerances = _mapping(manifest.get("tolerances"), name="tolerances")
    groups: list[dict[str, object]] = []
    for key, items in grouped.items():
        statuses = {item.status for item in items}
        payloads = [payload_by_path[item.path] for item in items]
        if statuses == {"completed"}:
            result = _completed_group(
                checkpoints=items,
                payloads=payloads,
                tolerances=tolerances,
            )
        elif statuses == {"unavailable"}:
            result = _unavailable_group(payloads)
        else:
            result = {
                "status": "failed",
                "failure_codes": ["mixed_or_failed_case_status"],
                "passes": False,
            }
        groups.append(
            {
                "source_case_index": key[0],
                "source_sample_index": key[1],
                "source_time_s": key[2],
                "engine": key[3],
                "variant": key[4],
                **result,
            }
        )
    promotion_failures: list[str] = []
    if counts["unavailable"]:
        promotion_failures.extend(
            ("native_engine_unavailable", "cross_engine_parity_unavailable")
        )
    else:
        promotion_failures.append("cross_engine_parity_not_evaluated")
    if counts["failed"]:
        promotion_failures.append("failed_case")
    if any(
        "closure" in code for group in groups for code in group.get("failure_codes", [])
    ):
        promotion_failures.append("case_closure_failure")
    if any(
        "refinement" in code
        for group in groups
        for code in group.get("failure_codes", [])
    ):
        promotion_failures.append("refinement_failure")
    return {
        "schema_version": "articulated-forward-attribution-summary/v1",
        "identity": {
            "plan_sha256": manifest_sha256(manifest),
            "execution_revision": execution_revision,
        },
        "counts": {
            "registered": len(checkpoints),
            "completed": counts["completed"],
            "unavailable": counts["unavailable"],
            "failed": counts["failed"],
        },
        "checkpoint_inventory": _checkpoint_inventory(checkpoints),
        "groups": groups,
        "promotion": {
            "eligible": not promotion_failures,
            "failure_codes": promotion_failures,
        },
        "claim_boundary": {
            "human_or_coaching_inference": False,
            "cross_engine_parity_established": False,
            "screening_population_evaluated": False,
        },
    }


def aggregate_rigid_smoke(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
) -> dict[str, object]:
    """Compatibility name for the original rigid registered smoke."""

    return aggregate_registered_smoke(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )


def publish_registered_smoke_evidence(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Path:
    """Publish the complete validated checkpoint set and governed summary.

    The output is deterministic for byte-identical inputs. Existing unrelated
    checkpoint files fail closed so a publication cannot silently mix runs.
    """

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    checkpoints = load_registered_checkpoints(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    summary = aggregate_registered_smoke(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
    )
    published_checkpoint_dir = output_dir / "checkpoints"
    expected_names = {checkpoint.path.name for checkpoint in checkpoints}
    if published_checkpoint_dir.exists():
        extras = {
            path.name
            for path in published_checkpoint_dir.glob("case-*.json")
            if path.name not in expected_names
        }
        if extras:
            raise ValueError(
                "publication directory contains unregistered checkpoints: "
                + ", ".join(sorted(extras))
            )
    for checkpoint in checkpoints:
        try:
            content = checkpoint.path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"validated checkpoint became unreadable: {checkpoint.path}"
            ) from exc
        _write_atomic(published_checkpoint_dir / checkpoint.path.name, content)
    summary_path = output_dir / "summary.json"
    serialized = (
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    _write_atomic(summary_path, serialized)
    return summary_path


def publish_rigid_smoke_evidence(
    *,
    manifest: Mapping[str, object],
    execution_revision: str,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Path:
    """Compatibility name for publishing the original rigid smoke."""

    return publish_registered_smoke_evidence(
        manifest=manifest,
        execution_revision=execution_revision,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
    )


__all__ = [
    "aggregate_registered_smoke",
    "aggregate_rigid_smoke",
    "publish_registered_smoke_evidence",
    "publish_rigid_smoke_evidence",
]
