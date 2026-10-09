"""Common admission through the authoritative Tools replay contract."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle


def validate_native_replay_bundle(
    bundle: ExperimentReplayBundle, contracts: Any
) -> ExperimentReplayBundle:
    """Revalidate integrity and required capabilities using Tools' authority."""
    validated = contracts.load_experiment_replay_bundle(
        contracts.dumps_experiment_replay_bundle(bundle)
    )
    if validated.blocking_capabilities:
        raise ValueError("required native replay capabilities are unavailable")
    return validated


def native_replay_admission_bytes() -> bytes:
    """Bind this executed admission helper into each adapter's provider hash."""
    return Path(__file__).read_bytes()
