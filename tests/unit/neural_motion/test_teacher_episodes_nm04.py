"""NM-04 (#10619): feasible teacher episodes and active-learning candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_JOINTS,
)
from src.shared.python.neural_motion.episodes import (
    EPISODE_STORE_SCHEMA,
    EpisodeRecord,
    EpisodeStore,
    FamilySplitPlan,
    build_family_splits,
)
from src.shared.python.neural_motion.experiment import NESTED_EPISODE_STAGES
from src.shared.python.neural_motion.teacher import (
    TEACHER_CAMPAIGN_SCHEMA,
    AcquisitionMode,
    MockTeacherRolloutBackend,
    TeacherAnchor,
    TeacherGenerationCampaign,
    TeacherGenerationSpec,
    build_pilot_teacher_spec,
    load_generation_state,
    save_generation_state,
    select_acquisition_candidates,
)
from src.shared.python.neural_motion.teacher.generation import (
    RejectedRolloutStore,
    TeacherGenerationState,
)

pytestmark = pytest.mark.unit

_T = 6
_N = N_JOINTS


def _finite(seed: int, shape: tuple[int, ...] = (_T, _N)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float64)


def _anchor(trial_id: str = "TW_wiffle", *, seed_id: str = "seed_a") -> TeacherAnchor:
    return TeacherAnchor(
        anchor_id=seed_id,
        trial_id=trial_id,
        model_id="driven_double_pendulum",
        source="co03_retrieval",
        q0=_finite(1, (_N,))[0],
        coefficients=_finite(2, (189,))[0],
        geometry_stratum="std_driver",
        contact_stratum="no_contact",
        club_stratum="driver",
    )


def _spec(tmp_path: Path, *, max_episodes: int = 12) -> TeacherGenerationSpec:
    return TeacherGenerationSpec(
        model_id="driven_double_pendulum",
        campaign_id="pilot.nm04.test",
        master_seed=42,
        nested_stages=(100, 500, 2000),
        max_episodes_per_stage=max_episodes,
        store_root=tmp_path / "feasible",
        rejected_root=tmp_path / "rejected",
        ledger_path=tmp_path / "teacher_ledger.json",
        acquisition_log_path=tmp_path / "acquisition.jsonl",
        state_path=tmp_path / "state.json",
    )


def _episode(trial_id: str, family_id: str, *, seed: int = 0) -> EpisodeRecord:
    t = np.linspace(0.0, 0.5, _T)
    q = _finite(seed)
    v = _finite(seed + 1)
    u = _finite(seed + 2)
    a = _finite(seed + 3)
    q_next = np.roll(q, -1, axis=0)
    q_next[-1] = q[-1]
    return EpisodeRecord(
        trial_id=trial_id,
        family_id=family_id,
        model_id="driven_double_pendulum",
        control_basis="joint_torque",
        units="SI",
        joint_names=CANONICAL_JOINTS,
        coefficient_letters=COEFFICIENT_LETTERS,
        schema_version=EPISODE_STORE_SCHEMA,
        sample_times_s=t,
        q=q,
        v=v,
        u=u,
        a_native=a,
        q_next=q_next,
        channel_availability={
            "q": "available",
            "v": "available",
            "u": "available",
            "a_native": "available",
            "q_next": "available",
        },
        ancestry=("teacher:nm04",),
        geometry_stratum="std_driver",
        contact_stratum="no_contact",
        club_stratum="driver",
    )


def test_pilot_spec_uses_frozen_nested_stages() -> None:
    spec = build_pilot_teacher_spec(Path("ignored"))
    assert spec.nested_stages == NESTED_EPISODE_STAGES
    assert spec.model_id == "driven_double_pendulum"


def test_duplicate_seed_and_perturbation_skipped(tmp_path: Path) -> None:
    spec = _spec(tmp_path, max_episodes=4)
    backend = MockTeacherRolloutBackend(infeasible_scale=1e9)
    campaign = TeacherGenerationCampaign(spec, backend=backend)
    first = campaign.run_stage(target_count=2, stage_index=0)
    second = campaign.run_stage(target_count=2, stage_index=0)
    assert first.accepted == 2
    assert second.accepted == 2
    store = EpisodeStore(spec.store_root)
    assert len(list(store.iter_episode_ids())) == 4
    # Re-loading state must not re-run already completed attempt keys.
    frozen_keys = set(campaign._known_keys)
    replay = TeacherGenerationCampaign(spec, backend=backend)
    replay._known_keys = set(frozen_keys)
    replay._state = TeacherGenerationState(
        campaign_id=spec.campaign_id,
        completed_attempts=0,
        attempt_keys=tuple(frozen_keys),
        simulation_cost_units=load_generation_state(
            spec.state_path
        ).simulation_cost_units,
    )
    receipt = replay.run_stage(target_count=1, stage_index=0)
    assert receipt.skipped_duplicate >= 1


def test_infeasible_rollouts_accounted_separately(tmp_path: Path) -> None:
    spec = _spec(tmp_path, max_episodes=6)
    backend = MockTeacherRolloutBackend(infeasible_scale=0.85)
    campaign = TeacherGenerationCampaign(spec, backend=backend)
    receipt = campaign.run_stage(target_count=6, stage_index=0)
    assert receipt.rejected >= 1
    assert receipt.accepted >= 1
    assert receipt.accepted == 6
    rejected = RejectedRolloutStore(spec.rejected_root)
    assert rejected.count() >= 1
    feasible = EpisodeStore(spec.store_root)
    assert len(list(feasible.iter_episode_ids())) == receipt.accepted


def test_resume_after_partial_batch(tmp_path: Path) -> None:
    spec = _spec(tmp_path, max_episodes=5)
    backend = MockTeacherRolloutBackend(infeasible_scale=1e9)
    campaign = TeacherGenerationCampaign(spec, backend=backend)
    partial = campaign.run_stage(target_count=3, stage_index=0)
    assert partial.accepted == 3
    state = load_generation_state(spec.state_path)
    assert state is not None
    assert state.completed_attempts == 3
    resumed = TeacherGenerationCampaign(spec, backend=backend)
    finish = resumed.run_stage(target_count=2, stage_index=0)
    assert finish.accepted == 2
    assert len(list(EpisodeStore(spec.store_root).iter_episode_ids())) == 5


def test_teacher_episodes_do_not_inherit_workbook_eval_split(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, max_episodes=3)
    backend = MockTeacherRolloutBackend(infeasible_scale=1e9)
    campaign = TeacherGenerationCampaign(spec, backend=backend)
    campaign.run_stage(target_count=3, stage_index=0)
    store = EpisodeStore(spec.store_root)
    episodes = [store.read_episode(eid, lazy=False) for eid in store.iter_episode_ids()]
    assert episodes
    for ep in episodes:
        assert not any(a.startswith("workbook:") for a in ep.ancestry)
        assert "simulated_body:conditional" in ep.ancestry
    plan = build_family_splits(
        episodes,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        seed=0,
        held_out_strata={"geometry": ("held_out_geom",)},
    )
    assert all(plan.split_of(ep.trial_id) != "real_data_eval" for ep in episodes)


def test_active_acquisition_never_targets_test_labels(tmp_path: Path) -> None:
    episodes = [
        _episode("t_train_a", "fam_train", seed=1),
        _episode("t_train_b", "fam_train", seed=2),
        _episode("t_test", "fam_test", seed=3),
    ]
    plan: FamilySplitPlan = build_family_splits(
        episodes,
        ratios={"train": 0.5, "val": 0.0, "test": 0.5},
        seed=1,
    )
    assert plan.split_of("t_test") == "test"
    pool = select_acquisition_candidates(
        episodes,
        split_plan=plan,
        scores={"t_train_a": 0.9, "t_train_b": 0.2, "t_test": 10.0},
        k=2,
        mode=AcquisitionMode.UNCERTAINTY,
        rng_seed=0,
    )
    assert "t_test" not in pool.selected_trial_ids
    random_pool = select_acquisition_candidates(
        episodes,
        split_plan=plan,
        scores={"t_train_a": 0.1, "t_train_b": 0.1, "t_test": 0.1},
        k=1,
        mode=AcquisitionMode.RANDOM_CONTROL,
        rng_seed=0,
    )
    assert "t_test" not in random_pool.selected_trial_ids


def test_zero_requested_channel_cannot_pass_as_available(tmp_path: Path) -> None:
    spec = _spec(tmp_path, max_episodes=1)
    backend = MockTeacherRolloutBackend(
        infeasible_scale=1e9,
        zero_channel="a_native",
    )
    campaign = TeacherGenerationCampaign(spec, backend=backend)
    with pytest.raises(ValueError, match="a_native|zero|unavailable"):
        campaign.run_stage(target_count=1, stage_index=0)


def test_native_replay_digest_recorded_in_ledger(tmp_path: Path) -> None:
    spec = _spec(tmp_path, max_episodes=2)
    backend = MockTeacherRolloutBackend(infeasible_scale=1e9)
    campaign = TeacherGenerationCampaign(spec, backend=backend)
    campaign.run_stage(target_count=2, stage_index=0)
    ledger = json.loads(spec.ledger_path.read_text(encoding="utf-8"))
    assert ledger["schema"] == TEACHER_CAMPAIGN_SCHEMA
    entries = ledger["entries"]
    assert len(entries) == 2
    for row in entries:
        assert row["teacher_objective"] >= 0.0
        assert row["convergence_iterations"] >= 0
        assert len(row["independent_replay_digest"]) == 64
        assert row["feasible"] is True


def test_generation_state_round_trip(tmp_path: Path) -> None:
    state = TeacherGenerationState(
        campaign_id="c1",
        completed_attempts=3,
        attempt_keys=("42:0:0", "42:1:0"),
        simulation_cost_units=12.5,
    )
    path = tmp_path / "state.json"
    save_generation_state(path, state)
    loaded = load_generation_state(path)
    assert loaded == state
