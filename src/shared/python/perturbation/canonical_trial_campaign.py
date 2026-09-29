"""Bind canonical Tools plan execution directly to a qualified trial bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .canonical_trial_executor import (
    BatchTrialRunner,
    TrialEvidenceCollector,
    TrialRunner,
    VariationSampler,
    execute_batched_variation,
    execute_serial_variation,
)
from .trial_evidence import CanonicalTrialEvidence
from .trial_evidence_bundle import (
    TrialEvidenceBundleSummary,
    _validate_destination,
    write_trial_evidence_bundle,
)


@dataclass(frozen=True)
class CanonicalVariationCampaignResult:
    """In-memory records and their atomically persisted bundle summary."""

    records: tuple[CanonicalTrialEvidence, ...]
    bundle: TrialEvidenceBundleSummary

    def __post_init__(self) -> None:
        if not self.records or any(
            not isinstance(record, CanonicalTrialEvidence) for record in self.records
        ):
            raise TypeError("records must contain CanonicalTrialEvidence")
        if not isinstance(self.bundle, TrialEvidenceBundleSummary):
            raise TypeError("bundle must be TrialEvidenceBundleSummary")
        if len(self.records) != self.bundle.trial_count:
            raise ValueError("bundle trial count must match campaign records")


def execute_serial_variation_campaign(
    *,
    plan: object,
    gateway: VariationSampler,
    runner: TrialRunner,
    collector: TrialEvidenceCollector,
    destination: Path,
) -> CanonicalVariationCampaignResult:
    """Execute every canonical row serially and atomically persist all outcomes."""
    target = _validate_destination(destination)
    records = execute_serial_variation(plan, gateway, runner, collector)
    bundle = write_trial_evidence_bundle(target, records)
    return CanonicalVariationCampaignResult(records, bundle)


def execute_batched_variation_campaign(
    *,
    plan: object,
    gateway: VariationSampler,
    batch_runner: BatchTrialRunner,
    collector: TrialEvidenceCollector,
    destination: Path,
) -> CanonicalVariationCampaignResult:
    """Execute a canonical batch and atomically persist its row-aligned outcomes."""
    target = _validate_destination(destination)
    records = execute_batched_variation(plan, gateway, batch_runner, collector)
    bundle = write_trial_evidence_bundle(target, records)
    return CanonicalVariationCampaignResult(records, bundle)


__all__ = [
    "CanonicalVariationCampaignResult",
    "execute_batched_variation_campaign",
    "execute_serial_variation_campaign",
]
