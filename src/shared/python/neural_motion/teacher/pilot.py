"""First-model pilot teacher corpus spec (NM-04 #10619)."""

from __future__ import annotations

from pathlib import Path

from src.shared.python.neural_motion.experiment import NESTED_EPISODE_STAGES

from .types import TeacherGenerationSpec

__all__ = ["PILOT_MODEL_ID", "build_pilot_teacher_spec"]

PILOT_MODEL_ID = "driven_double_pendulum"


def build_pilot_teacher_spec(root: Path | str) -> TeacherGenerationSpec:
    """Build the audited first-model pilot corpus layout under ``root``."""
    base = Path(root)
    return TeacherGenerationSpec(
        model_id=PILOT_MODEL_ID,
        campaign_id="nm04.pilot.driven_double_pendulum",
        master_seed=10619,
        nested_stages=NESTED_EPISODE_STAGES,
        max_episodes_per_stage=NESTED_EPISODE_STAGES[0],
        store_root=base / "feasible_episodes",
        rejected_root=base / "rejected_rollouts",
        ledger_path=base / "teacher_ledger.json",
        acquisition_log_path=base / "acquisition.jsonl",
        state_path=base / "generation_state.json",
    )
