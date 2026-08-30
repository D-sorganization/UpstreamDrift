"""Register a fresh, outcome-blind structural-factorial recovery replay."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_factorial_runner import (
    plan_sha256,
)

RECOVERY_SCHEMA = "articulated-structural-factorial-recovery-registration/1.1.0"
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_SUPERSEDED_RESOURCE_CONTRACT = (
    "DeskComputer serial CPU only; no ControlTower, WSL, or VHDX access"
)
_RUNNER_PLACEMENT = {
    "workflow_runner_label": "ubuntu-latest",
    "runner_environment": "github-hosted",
    "runner_os": "Linux",
    "runner_arch": "X64",
    "image_os_prefix": "ubuntu",
}
_REQUIRED_STOP_CONDITIONS = (
    "runtime_replay_contract_drift",
    "workflow_or_artifact_failure",
    "receipt_or_api_identity_failure",
    "corruption_killswitch_failure",
    "checkpoint_status_or_result_mismatch",
    "legacy_array_mismatch",
    "missing_or_corrupt_enriched_sidecar",
    "non_gap_free_cumulative_prefix",
)
_REQUIRED_EVIDENCE = (
    "raw_run_jobs_artifacts_api_responses",
    "runtime_replay_artifact_and_exact_gate",
    "checkpoint_artifact_zip_and_github_digest",
    "artifact_receipt_schema_1_4",
    "source_derived_corruption_audit",
    "gap_free_collection_manifest_schema_1_3",
    "exact_legacy_enrichment_audit",
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_run_id(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hex(value: object, *, name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


@dataclass(frozen=True)
class StructuralFactorialRecoveryRegistration:
    """Immutable operational registration for the attested prefix recovery."""

    qualified_runtime_audit_run_id: int
    qualified_runtime_identity_sha256: str
    triggering_run_id: int
    triggering_dispatch_head: str
    repeat_run_ids: tuple[int, ...]
    attested_repeat_run_ids: tuple[int, ...]
    repeatability_audit_sha256: str
    superseded_registration_sha256: str
    runner_conflict_run_id: int
    runner_conflict_dispatch_head: str
    case_stop_exclusive: int = 100
    slice_width: int = 20

    def __post_init__(self) -> None:
        _positive_run_id(
            self.qualified_runtime_audit_run_id,
            name="qualified_runtime_audit_run_id",
        )
        _positive_run_id(self.triggering_run_id, name="triggering_run_id")
        _hex(
            self.qualified_runtime_identity_sha256,
            name="qualified_runtime_identity_sha256",
            pattern=_HEX_64,
        )
        _hex(
            self.triggering_dispatch_head,
            name="triggering_dispatch_head",
            pattern=_HEX_40,
        )
        _hex(
            self.repeatability_audit_sha256,
            name="repeatability_audit_sha256",
            pattern=_HEX_64,
        )
        _hex(
            self.superseded_registration_sha256,
            name="superseded_registration_sha256",
            pattern=_HEX_64,
        )
        _positive_run_id(self.runner_conflict_run_id, name="runner_conflict_run_id")
        _hex(
            self.runner_conflict_dispatch_head,
            name="runner_conflict_dispatch_head",
            pattern=_HEX_40,
        )
        if not self.repeat_run_ids:
            raise ValueError("repeat_run_ids must not be empty")
        for run_id in self.repeat_run_ids:
            _positive_run_id(run_id, name="repeat_run_ids item")
        if not self.attested_repeat_run_ids:
            raise ValueError("attested_repeat_run_ids must not be empty")
        if not set(self.attested_repeat_run_ids).issubset(self.repeat_run_ids):
            raise ValueError("attested repeat runs must be a subset of repeat runs")
        if self.case_stop_exclusive <= 0 or self.slice_width <= 0:
            raise ValueError("case stop and slice width must be positive")
        if self.case_stop_exclusive % self.slice_width:
            raise ValueError("case stop must be divisible by slice width")

    def to_manifest(
        self,
        *,
        plan: Mapping[str, object],
        launch: Mapping[str, object],
    ) -> dict[str, object]:
        """Return the deterministic recovery registration."""

        expected_plan_sha256 = plan_sha256(plan)
        if launch.get("plan_sha256") != expected_plan_sha256:
            raise ValueError("launch plan identity does not match the canonical plan")
        execution = _mapping(plan.get("execution"), name="plan.execution")
        if execution.get("resource_contract") != _SUPERSEDED_RESOURCE_CONTRACT:
            raise ValueError("plan resource contract is not the registered conflict")
        execution_revision = _hex(
            launch.get("execution_revision"),
            name="launch.execution_revision",
            pattern=_HEX_40,
        )
        slices = []
        for ordinal, case_start in enumerate(
            range(0, self.case_stop_exclusive, self.slice_width), start=1
        ):
            case_stop = case_start + self.slice_width
            slices.append(
                {
                    "ordinal": ordinal,
                    "case_start": case_start,
                    "case_stop": case_stop,
                    "depends_on_case_stop": None if case_start == 0 else case_start,
                    "status": (
                        "authorized_only_after_conflict_run_terminal_and_retained"
                        if case_start == 0
                        else "blocked_pending_prior_cumulative_exact_gate"
                    ),
                }
            )
        return {
            "schema_version": RECOVERY_SCHEMA,
            "classification": "outcome_blind_recovery_preregistration",
            "operational_amendment": {
                "timing": (
                    "after_runner_placement_conflict_before_replacement_execution"
                ),
                "superseded_registration_sha256": (self.superseded_registration_sha256),
                "conflicting_run_id": self.runner_conflict_run_id,
                "conflicting_dispatch_head": self.runner_conflict_dispatch_head,
                "conflicting_plan_resource_contract": (_SUPERSEDED_RESOURCE_CONTRACT),
                "conflicting_run_promotable": False,
                "registered_design_or_numerical_gate_change": False,
                "checkpoint_reuse_permitted": False,
            },
            "identity": {
                "plan_sha256": expected_plan_sha256,
                "execution_revision": execution_revision,
                "qualified_runtime_audit_run_id": self.qualified_runtime_audit_run_id,
                "qualified_runtime_identity_sha256": (
                    self.qualified_runtime_identity_sha256
                ),
            },
            "trigger_evidence": {
                "first_enriched_replay_run_id": self.triggering_run_id,
                "first_enriched_replay_dispatch_head": (self.triggering_dispatch_head),
                "repeat_run_ids": list(self.repeat_run_ids),
                "attested_repeat_run_ids": list(self.attested_repeat_run_ids),
                "repeatability_audit_sha256": self.repeatability_audit_sha256,
                "classification": "first_enriched_replay_anomaly_supported",
            },
            "registered_prefix": {
                "case_start": 0,
                "case_stop": self.case_stop_exclusive,
                "slice_width": self.slice_width,
                "slice_count": len(slices),
            },
            "slices": slices,
            "execution_policy": {
                "maximum_concurrent_structural_runs": 1,
                "dispatch_only_after_prior_slice_is_terminal_and_retained": True,
                "runtime_replay_gate_required_before_every_slice": True,
                "dispatch_head_recorded_before_each_slice": True,
                "automatic_rerun_permitted": False,
                "case_100_to_120_dispatch_permitted": False,
                "runner_placement": dict(_RUNNER_PLACEMENT),
            },
            "evidence_policy": {
                "required_per_slice": list(_REQUIRED_EVIDENCE),
                "reuse_prior_checkpoint_bytes": False,
                "mix_attested_and_unattested_checkpoint_sources": False,
                "exact_equality_only": True,
                "tolerance_substitution_permitted": False,
            },
            "stop_conditions": list(_REQUIRED_STOP_CONDITIONS),
            "claim_boundary": {
                "scientific_outcomes_may_be_inspected": False,
                "effect_direction_may_be_interpreted": False,
                "campaign_promotion_authorized": False,
                "human_or_coaching_inference_authorized": False,
            },
        }


def validate_recovery_registration(
    *,
    manifest: Mapping[str, object],
    plan: Mapping[str, object],
    launch: Mapping[str, object],
) -> None:
    """Fail closed unless a registration is the exact canonical contract."""

    evidence_policy = _mapping(manifest.get("evidence_policy"), name="evidence_policy")
    if evidence_policy.get("reuse_prior_checkpoint_bytes") is not False:
        raise ValueError("reuse prior checkpoint bytes must remain prohibited")
    if (
        evidence_policy.get("mix_attested_and_unattested_checkpoint_sources")
        is not False
    ):
        raise ValueError(
            "mixing attested and unattested checkpoint sources is prohibited"
        )
    if manifest.get("stop_conditions") != list(_REQUIRED_STOP_CONDITIONS):
        raise ValueError("stop conditions do not match the canonical fail-closed set")
    slices = manifest.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("slices must define a gap-free prefix")
    expected_start = 0
    for item in slices:
        record = _mapping(item, name="slice")
        if record.get("case_start") != expected_start:
            raise ValueError("slices must define a gap-free prefix")
        case_stop = record.get("case_stop")
        if not isinstance(case_stop, int) or isinstance(case_stop, bool):
            raise ValueError("slice case stop must be an integer")
        expected_start = case_stop

    identity = _mapping(manifest.get("identity"), name="identity")
    trigger = _mapping(manifest.get("trigger_evidence"), name="trigger_evidence")
    amendment = _mapping(
        manifest.get("operational_amendment"), name="operational_amendment"
    )
    registration = StructuralFactorialRecoveryRegistration(
        qualified_runtime_audit_run_id=_positive_run_id(
            identity.get("qualified_runtime_audit_run_id"),
            name="qualified_runtime_audit_run_id",
        ),
        qualified_runtime_identity_sha256=_hex(
            identity.get("qualified_runtime_identity_sha256"),
            name="qualified_runtime_identity_sha256",
            pattern=_HEX_64,
        ),
        triggering_run_id=_positive_run_id(
            trigger.get("first_enriched_replay_run_id"),
            name="first_enriched_replay_run_id",
        ),
        triggering_dispatch_head=_hex(
            trigger.get("first_enriched_replay_dispatch_head"),
            name="first_enriched_replay_dispatch_head",
            pattern=_HEX_40,
        ),
        repeat_run_ids=tuple(
            _positive_run_id(value, name="repeat_run_ids item")
            for value in _sequence(trigger.get("repeat_run_ids"), name="repeat_run_ids")
        ),
        attested_repeat_run_ids=tuple(
            _positive_run_id(value, name="attested_repeat_run_ids item")
            for value in _sequence(
                trigger.get("attested_repeat_run_ids"),
                name="attested_repeat_run_ids",
            )
        ),
        repeatability_audit_sha256=_hex(
            trigger.get("repeatability_audit_sha256"),
            name="repeatability_audit_sha256",
            pattern=_HEX_64,
        ),
        superseded_registration_sha256=_hex(
            amendment.get("superseded_registration_sha256"),
            name="superseded_registration_sha256",
            pattern=_HEX_64,
        ),
        runner_conflict_run_id=_positive_run_id(
            amendment.get("conflicting_run_id"), name="conflicting_run_id"
        ),
        runner_conflict_dispatch_head=_hex(
            amendment.get("conflicting_dispatch_head"),
            name="conflicting_dispatch_head",
            pattern=_HEX_40,
        ),
    )
    expected = registration.to_manifest(plan=plan, launch=launch)
    if manifest != expected:
        raise ValueError("recovery registration is not the canonical manifest")


def validate_registered_slice(
    *,
    manifest: Mapping[str, object],
    plan: Mapping[str, object],
    launch: Mapping[str, object],
    case_start: int,
    case_stop: int,
    runner_environment: str | None = None,
    runner_os: str | None = None,
    runner_arch: str | None = None,
    runner_image_os: str | None = None,
    runtime_audit_run_id: int | None = None,
) -> Mapping[str, object]:
    """Return the exact preregistered slice record or fail closed."""

    validate_recovery_registration(manifest=manifest, plan=plan, launch=launch)
    placement_values = (
        runner_environment,
        runner_os,
        runner_arch,
        runner_image_os,
    )
    if any(value is not None for value in placement_values):
        if any(value is None for value in placement_values):
            raise ValueError("runner placement validation requires all runtime fields")
        validate_runner_placement(
            manifest=manifest,
            runner_environment=str(runner_environment),
            runner_os=str(runner_os),
            runner_arch=str(runner_arch),
            runner_image_os=str(runner_image_os),
        )
    if runtime_audit_run_id is not None:
        identity = _mapping(manifest.get("identity"), name="identity")
        if runtime_audit_run_id != identity.get("qualified_runtime_audit_run_id"):
            raise ValueError("runtime audit run does not match registration")
    slices = manifest.get("slices")
    if not isinstance(slices, list):  # defended by the full validator
        raise ValueError("slices must define a gap-free prefix")
    matches = [
        _mapping(item, name="slice")
        for item in slices
        if isinstance(item, Mapping)
        and item.get("case_start") == case_start
        and item.get("case_stop") == case_stop
    ]
    if len(matches) != 1:
        raise ValueError("requested range is not an exact registered slice")
    return matches[0]


def validate_runner_placement(
    *,
    manifest: Mapping[str, object],
    runner_environment: str,
    runner_os: str,
    runner_arch: str,
    runner_image_os: str,
) -> None:
    """Fail closed unless the actual hosted runner matches the amendment."""

    execution_policy = _mapping(
        manifest.get("execution_policy"), name="execution_policy"
    )
    placement = _mapping(
        execution_policy.get("runner_placement"), name="runner_placement"
    )
    if runner_environment != placement.get("runner_environment"):
        raise ValueError("runner environment does not match registration")
    if runner_os != placement.get("runner_os"):
        raise ValueError("runner OS does not match registration")
    if runner_arch != placement.get("runner_arch"):
        raise ValueError("runner architecture does not match registration")
    image_prefix = placement.get("image_os_prefix")
    if not isinstance(image_prefix, str) or not runner_image_os.startswith(
        image_prefix
    ):
        raise ValueError("runner image does not match registration")


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _read_mapping(path: Path) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=False, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or validate one recovery registration."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--plan", type=Path, required=True)
    validate_parser.add_argument("--launch", type=Path, required=True)
    validate_parser.add_argument("--registration", type=Path, required=True)
    slice_parser = subparsers.add_parser("validate-slice")
    slice_parser.add_argument("--plan", type=Path, required=True)
    slice_parser.add_argument("--launch", type=Path, required=True)
    slice_parser.add_argument("--registration", type=Path, required=True)
    slice_parser.add_argument("--case-start", type=int, required=True)
    slice_parser.add_argument("--case-stop", type=int, required=True)
    slice_parser.add_argument("--runner-environment", required=True)
    slice_parser.add_argument("--runner-os", required=True)
    slice_parser.add_argument("--runner-arch", required=True)
    slice_parser.add_argument("--runner-image-os", required=True)
    slice_parser.add_argument("--runtime-audit-run-id", type=int, required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--plan", type=Path, required=True)
    generate_parser.add_argument("--launch", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument(
        "--qualified-runtime-audit-run-id", type=int, required=True
    )
    generate_parser.add_argument("--qualified-runtime-identity-sha256", required=True)
    generate_parser.add_argument("--triggering-run-id", type=int, required=True)
    generate_parser.add_argument("--triggering-dispatch-head", required=True)
    generate_parser.add_argument(
        "--repeat-run-id", type=int, action="append", required=True
    )
    generate_parser.add_argument(
        "--attested-repeat-run-id", type=int, action="append", required=True
    )
    generate_parser.add_argument("--repeatability-audit-sha256", required=True)
    generate_parser.add_argument("--superseded-registration-sha256", required=True)
    generate_parser.add_argument("--runner-conflict-run-id", type=int, required=True)
    generate_parser.add_argument("--runner-conflict-dispatch-head", required=True)
    generate_parser.add_argument(
        "--confirm-no-outcome-inspection", action="store_true", required=True
    )
    generate_parser.add_argument(
        "--confirm-no-checkpoint-reuse", action="store_true", required=True
    )
    args = parser.parse_args(argv)
    plan = _read_mapping(args.plan)
    launch = _read_mapping(args.launch)
    if args.command == "validate":
        validate_recovery_registration(
            manifest=_read_mapping(args.registration), plan=plan, launch=launch
        )
        return 0
    if args.command == "validate-slice":
        validate_registered_slice(
            manifest=_read_mapping(args.registration),
            plan=plan,
            launch=launch,
            case_start=args.case_start,
            case_stop=args.case_stop,
            runner_environment=args.runner_environment,
            runner_os=args.runner_os,
            runner_arch=args.runner_arch,
            runner_image_os=args.runner_image_os,
            runtime_audit_run_id=args.runtime_audit_run_id,
        )
        return 0
    registration = StructuralFactorialRecoveryRegistration(
        qualified_runtime_audit_run_id=args.qualified_runtime_audit_run_id,
        qualified_runtime_identity_sha256=args.qualified_runtime_identity_sha256,
        triggering_run_id=args.triggering_run_id,
        triggering_dispatch_head=args.triggering_dispatch_head,
        repeat_run_ids=tuple(args.repeat_run_id),
        attested_repeat_run_ids=tuple(args.attested_repeat_run_id),
        repeatability_audit_sha256=args.repeatability_audit_sha256,
        superseded_registration_sha256=args.superseded_registration_sha256,
        runner_conflict_run_id=args.runner_conflict_run_id,
        runner_conflict_dispatch_head=args.runner_conflict_dispatch_head,
    )
    _write_atomic(args.output, registration.to_manifest(plan=plan, launch=launch))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RECOVERY_SCHEMA",
    "StructuralFactorialRecoveryRegistration",
    "main",
    "validate_recovery_registration",
    "validate_runner_placement",
    "validate_registered_slice",
]
