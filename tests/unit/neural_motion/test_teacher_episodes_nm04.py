"""NM-04 (#10619): feasible teacher episodes and active-learning candidates.

Acceptance (issue copy): native reproducibility, duplicate-seed avoidance,
physical feasibility and replay, teacher leakage prevention, rejection
accounting, active acquisition cannot consume test labels, resume after
partial batch, zero requested channel cannot pass missing-field test.
"""

from __future__ import annotations

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
from src.shared.python.neural_motion.teachers import (
    ACQUISITION_SCHEMA,
    TEACHER_SCHEMA,
    AcquisitionLog,
    AcquisitionStrategy,
    ActiveLearningAcquirer,
    NestedTeacherCorpus,
    PerturbationKind,
    RejectionLedger,
    TeacherEpisodeGenerator,
    TeacherSpec,
)

pytestmark = pytest.mark.unit

_T = 4
_N = N_JOINTS


def _baseline_episode(*, trial_id: str, family_id: str, seed: int) -> EpisodeRecord:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 0.3, _T)
    q = rng.normal(size=(_T, _N))
    v = rng.normal(size=(_T, _N))
    u = rng.normal(size=(_T, _N))
    a = rng.normal(size=(_T, _N))
    q_next = np.roll(q, -1, axis=0)
    q_next[-1] = q[-1]
    return EpisodeRecord(
        trial_id=trial_id,
        family_id=family_id,
        model_id="mock.driven_double_pendulum",
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
        ancestry=("baseline:tour_driver",),
        geometry_stratum="std_driver",
        contact_stratum="no_contact",
        club_stratum="driver",
    )


def _spec(
    *,
    seed: int,
    family_id: str = "fam_a",
    kind: PerturbationKind = PerturbationKind.NEAR_BASELINE,
    channels: tuple[str, ...] = ("q", "v", "u", "a_native"),
) -> TeacherSpec:
    return TeacherSpec(
        seed=seed,
        family_id=family_id,
        model_id="mock.driven_double_pendulum",
        perturbation=kind,
        geometry_stratum="std_driver",
        contact_stratum="no_contact",
        club_stratum="driver",
        duration_s=0.3,
        ancestry=("baseline:tour_driver", f"seed:{seed}"),
        requested_channels=channels,
    )


# ---------------------------------------------------------------------------
# Schema / DbC
# ---------------------------------------------------------------------------


def test_teacher_and_acquisition_schemas_are_versioned() -> None:
    assert TEACHER_SCHEMA == "neural-teacher-episodes/1.0.0"
    assert ACQUISITION_SCHEMA == "neural-acquisition-log/1.0.0"
    assert NESTED_EPISODE_STAGES == (100, 500, 2000)


def test_spec_rejects_nonfinite_duration_and_empty_channels() -> None:
    with pytest.raises(ValueError, match="duration"):
        TeacherSpec(
            seed=0,
            family_id="f",
            model_id="m",
            perturbation=PerturbationKind.NEAR_BASELINE,
            geometry_stratum="g",
            contact_stratum="c",
            club_stratum="cl",
            duration_s=float("nan"),
            ancestry=("a",),
            requested_channels=("q",),
        )
    with pytest.raises(ValueError, match="requested_channels"):
        TeacherSpec(
            seed=0,
            family_id="f",
            model_id="m",
            perturbation=PerturbationKind.NEAR_BASELINE,
            geometry_stratum="g",
            contact_stratum="c",
            club_stratum="cl",
            duration_s=0.1,
            ancestry=("a",),
            requested_channels=(),
        )


def test_zero_requested_channel_cannot_pass_missing_field() -> None:
    """Requested channel absent from rollout must fail closed (no silent zeros)."""
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=1)
    gen = TeacherEpisodeGenerator(baseline=baseline)
    # Request a channel the generator cannot supply for this synthetic path.
    spec = _spec(seed=7, channels=("q", "v", "u", "a_native", "muscle_excitation"))
    outcome = gen.generate(spec)
    assert outcome.feasible is False
    assert outcome.episode is None
    assert "muscle_excitation" in outcome.reason
    assert outcome.rejection_cost > 0.0


# ---------------------------------------------------------------------------
# Reproducibility / duplicate seeds
# ---------------------------------------------------------------------------


def test_native_reproducibility_same_seed_same_content_hash() -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=11)
    gen = TeacherEpisodeGenerator(baseline=baseline)
    a = gen.generate(_spec(seed=42))
    b = gen.generate(_spec(seed=42))
    assert a.feasible and b.feasible
    assert a.episode is not None and b.episode is not None
    assert a.episode.content_sha256 == b.episode.content_sha256
    assert a.replay_digest == b.replay_digest
    assert a.replay_digest is not None


def test_duplicate_seed_avoidance_across_corpus(tmp_path: Path) -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=3)
    corpus = NestedTeacherCorpus(
        store=EpisodeStore(tmp_path / "store"),
        generator=TeacherEpisodeGenerator(baseline=baseline),
        checkpoint_path=tmp_path / "checkpoint.json",
        stage_sizes=(4,),
        compute_cap_cost=100.0,
    )
    first = corpus.run_stage(stage_index=0, seeds=(0, 1, 2, 3))
    assert first.accepted_count == 4
    # Re-submitting the same seeds must not double-write or inflate cost blindly.
    second = corpus.run_stage(stage_index=0, seeds=(0, 1))
    assert second.accepted_count == 0
    assert second.duplicate_seed_count == 2
    assert len(list(corpus.store.iter_episode_ids())) == 4


# ---------------------------------------------------------------------------
# Feasibility / rejection accounting / quarantine
# ---------------------------------------------------------------------------


def test_infeasible_rollout_recorded_separately_with_rejection_cost(
    tmp_path: Path,
) -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=5)

    def _poison(spec: TeacherSpec, _: EpisodeRecord) -> EpisodeRecord:
        # Force nonfinite controls → feasibility fail.
        raise ValueError("forced infeasible dynamics")

    gen = TeacherEpisodeGenerator(baseline=baseline, rollout_fn=_poison)
    corpus = NestedTeacherCorpus(
        store=EpisodeStore(tmp_path / "store"),
        generator=gen,
        checkpoint_path=tmp_path / "checkpoint.json",
        stage_sizes=(3,),
        compute_cap_cost=50.0,
        rejected_store=EpisodeStore(tmp_path / "rejected"),
    )
    result = corpus.run_stage(stage_index=0, seeds=(10, 11, 12))
    assert result.accepted_count == 0
    assert result.rejected_count == 3
    assert result.rejection_ledger.total_rejected_cost > 0.0
    assert len(result.rejection_ledger.entries) == 3
    # Feasible store empty; rejected ledger persisted.
    assert list(corpus.store.iter_episode_ids()) == []
    ledger_path = tmp_path / "rejection_ledger.json"
    result.rejection_ledger.write_json(ledger_path)
    assert ledger_path.is_file()


def test_outlier_quarantined_not_clipped(tmp_path: Path) -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=9)
    gen = TeacherEpisodeGenerator(
        baseline=baseline,
        outlier_torque_norm=1e-6,  # tiny threshold → quarantine normal rollouts
    )
    corpus = NestedTeacherCorpus(
        store=EpisodeStore(tmp_path / "store"),
        generator=gen,
        checkpoint_path=tmp_path / "checkpoint.json",
        stage_sizes=(2,),
        compute_cap_cost=20.0,
        quarantine_store=EpisodeStore(tmp_path / "quarantine"),
    )
    result = corpus.run_stage(stage_index=0, seeds=(1, 2))
    assert result.quarantined_count == 2
    assert result.accepted_count == 0
    assert len(list(corpus.quarantine_store.iter_episode_ids())) == 2


# ---------------------------------------------------------------------------
# Resume / nested stages / ancestry
# ---------------------------------------------------------------------------


def test_resume_after_partial_batch(tmp_path: Path) -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=2)
    store = EpisodeStore(tmp_path / "store")
    ckpt = tmp_path / "checkpoint.json"
    corpus = NestedTeacherCorpus(
        store=store,
        generator=TeacherEpisodeGenerator(baseline=baseline),
        checkpoint_path=ckpt,
        stage_sizes=(5,),
        compute_cap_cost=100.0,
    )
    partial = corpus.run_stage(stage_index=0, seeds=(0, 1, 2), max_new=2)
    assert partial.accepted_count == 2
    assert ckpt.is_file()
    resumed = NestedTeacherCorpus(
        store=store,
        generator=TeacherEpisodeGenerator(baseline=baseline),
        checkpoint_path=ckpt,
        stage_sizes=(5,),
        compute_cap_cost=100.0,
    )
    rest = resumed.run_stage(stage_index=0, seeds=(0, 1, 2, 3, 4))
    # Seeds 0,1 already done; 2 was pending from partial? max_new=2 took 0,1
    # so 2,3,4 remain → 3 new.
    assert rest.accepted_count == 3
    assert len(list(store.iter_episode_ids())) == 5


def test_learning_curve_subsets_nest_and_preserve_ancestry(tmp_path: Path) -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=4)
    corpus = NestedTeacherCorpus(
        store=EpisodeStore(tmp_path / "store"),
        generator=TeacherEpisodeGenerator(baseline=baseline),
        checkpoint_path=tmp_path / "checkpoint.json",
        stage_sizes=(2, 4),
        compute_cap_cost=100.0,
    )
    s0 = corpus.run_stage(stage_index=0, seeds=(0, 1))
    s1 = corpus.run_stage(stage_index=1, seeds=(0, 1, 2, 3))
    assert s0.accepted_count == 2
    # Stage-1 reuses stage-0 seeds (nested) without rewriting; adds 2 new.
    assert s1.accepted_count == 2
    assert s1.nested_reuse_count == 2
    subsets = corpus.learning_curve_subsets()
    assert subsets[0] <= subsets[1]
    assert len(subsets[0]) == 2
    assert len(subsets[1]) == 4
    for eid in subsets[1]:
        ep = corpus.store.read_episode(eid, lazy=False)
        assert "baseline:tour_driver" in ep.ancestry


# ---------------------------------------------------------------------------
# Splits / leakage / active acquisition
# ---------------------------------------------------------------------------


def _build_split_plan(store: EpisodeStore) -> FamilySplitPlan:
    episodes = [store.read_episode(eid, lazy=False) for eid in store.iter_episode_ids()]
    return build_family_splits(
        episodes,
        ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        seed=0,
    )


def test_teacher_leakage_prevention_and_acquisition_spares_test(
    tmp_path: Path,
) -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=8)
    # Distinct families so splits can separate train vs test.
    gen = TeacherEpisodeGenerator(baseline=baseline)
    store = EpisodeStore(tmp_path / "store")
    corpus = NestedTeacherCorpus(
        store=store,
        generator=gen,
        checkpoint_path=tmp_path / "checkpoint.json",
        stage_sizes=(8,),
        compute_cap_cost=100.0,
    )
    # One seed per family so family splits can assign train/test.
    seeds_and_families = [(i, f"fam_{i}") for i in range(8)]
    for seed, family in seeds_and_families:
        outcome = gen.generate(_spec(seed=seed, family_id=family))
        assert outcome.episode is not None
        store.write_episode(outcome.episode)
        corpus.record_seed_done(
            stage_index=0, seed=seed, episode_id=outcome.episode.episode_id
        )

    plan = _build_split_plan(store)
    test_ids = set(plan.splits.get("test", ()))
    assert test_ids, "expected a non-empty test split for leakage check"

    acquirer = ActiveLearningAcquirer(
        store=store,
        split_plan=plan,
        rng_seed=0,
    )
    # Candidate pool includes train + test trial ids; acquisition must ignore test.
    pool = [
        ep.trial_id
        for ep in (store.read_episode(e, lazy=False) for e in store.iter_episode_ids())
    ]
    log = acquirer.select(
        candidate_trial_ids=pool,
        n=3,
        strategies=(
            AcquisitionStrategy.UNCERTAINTY,
            AcquisitionStrategy.COVERAGE,
            AcquisitionStrategy.RANDOM_CONTROL,
        ),
        scores={tid: float(i) for i, tid in enumerate(pool)},
        coverage={tid: float(i % 3) for i, tid in enumerate(pool)},
    )
    assert isinstance(log, AcquisitionLog)
    assert log.schema == ACQUISITION_SCHEMA
    selected = {entry.trial_id for entry in log.selected}
    assert selected.isdisjoint(test_ids)
    assert log.test_labels_consumed == 0
    # Explicit attempt to request a test trial must fail closed.
    with pytest.raises(ValueError, match="test"):
        acquirer.select(
            candidate_trial_ids=sorted(test_ids),
            n=1,
            strategies=(AcquisitionStrategy.RANDOM_CONTROL,),
            scores=dict.fromkeys(test_ids, 1.0),
            coverage=dict.fromkeys(test_ids, 1.0),
        )


def test_random_torque_is_comparison_corpus_not_only_path() -> None:
    baseline = _baseline_episode(trial_id="b0", family_id="fam", seed=6)
    gen = TeacherEpisodeGenerator(baseline=baseline)
    near = gen.generate(_spec(seed=1, kind=PerturbationKind.NEAR_BASELINE))
    strat = gen.generate(_spec(seed=2, kind=PerturbationKind.STRATIFIED))
    sobol = gen.generate(_spec(seed=3, kind=PerturbationKind.LOW_DISCREPANCY))
    rand = gen.generate(_spec(seed=4, kind=PerturbationKind.RANDOM_TORQUE))
    assert all(o.feasible for o in (near, strat, sobol, rand))
    hashes = {
        near.episode.content_sha256,  # type: ignore[union-attr]
        strat.episode.content_sha256,  # type: ignore[union-attr]
        sobol.episode.content_sha256,  # type: ignore[union-attr]
        rand.episode.content_sha256,  # type: ignore[union-attr]
    }
    assert len(hashes) == 4
    assert rand.corpus_role == "comparison"
    assert near.corpus_role == "primary"


def test_rejection_ledger_aggregates_cost() -> None:
    ledger = RejectionLedger.empty()
    ledger = ledger.append(seed=1, reason="nonfinite", cost=1.5)
    ledger = ledger.append(seed=2, reason="replay_mismatch", cost=2.5)
    assert ledger.total_rejected_cost == pytest.approx(4.0)
    assert len(ledger.entries) == 2
