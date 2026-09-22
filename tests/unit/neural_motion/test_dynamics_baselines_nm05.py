"""NM-05 (#10620): classical and small neural dynamics baselines.

Acceptance (issue copy): known 1/2-DOF native labels, trial split leakage
regression, train-only normalization, unavailable torque mask, identity-channel
leakage, deterministic tiny overfit smoke, native long rollout versus one-step
error, optional torch absence.

Synthetic fixtures validate software contracts only — not native training
success or acceleration claims. Episode payloads keep the compact 27-joint
layout; active DOF indices carry the known 1/2-DOF analytic labels.
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
from src.shared.python.neural_motion.baselines import (
    BASELINE_SCHEMA,
    ClassicalMethod,
    DynamicsBaselineTrainer,
    DynamicsTaskKind,
    InverseLabelConditioning,
    PilotCheckpointCard,
    build_trial_matrices,
    evaluate_rollout_vs_onestep,
    identity_channel_leakage_score,
    optional_torch_available,
)
from src.shared.python.neural_motion.episodes import (
    EPISODE_STORE_SCHEMA,
    EpisodeRecord,
    EpisodeStore,
    FamilySplitPlan,
    TrainOnlyNormalizer,
    build_family_splits,
)
from src.shared.python.training.datasets import (
    DatasetRegistry,
    register_dynamics_baseline_corpus,
)
from src.shared.python.training.scheduler import neural_dynamics_baseline_budget

pytestmark = pytest.mark.unit

_T = 16
_N = N_JOINTS
_RATIOS = {"train": 0.5, "val": 0.25, "test": 0.25}


def _pad(active: np.ndarray) -> np.ndarray:
    """Embed low-DOF analytic columns into the compact 27-joint layout."""
    out = np.zeros((_T, _N), dtype=np.float64)
    out[:, : active.shape[1]] = active
    return out


def _pendulum_1dof(*, trial_id: str, family_id: str, seed: int) -> EpisodeRecord:
    """Analytic simple-pendulum labels on DOF 0: a = -sin(q) + u."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 0.75, _T)
    dt = float(t[1] - t[0])
    q0 = rng.uniform(-0.6, 0.6, size=(_T, 1))
    v0 = rng.uniform(-0.4, 0.4, size=(_T, 1))
    u0 = rng.uniform(-0.3, 0.3, size=(_T, 1))
    a0 = -np.sin(q0) + u0
    qn0 = q0 + v0 * dt + 0.5 * a0 * dt * dt
    return EpisodeRecord(
        trial_id=trial_id,
        family_id=family_id,
        model_id="mock.simple_pendulum_1dof",
        control_basis="joint_torque",
        units="SI",
        joint_names=CANONICAL_JOINTS,
        coefficient_letters=COEFFICIENT_LETTERS,
        schema_version=EPISODE_STORE_SCHEMA,
        sample_times_s=t,
        q=_pad(q0),
        v=_pad(v0),
        u=_pad(u0),
        a_native=_pad(a0),
        q_next=_pad(qn0),
        channel_availability={
            "q": "available",
            "v": "available",
            "u": "available",
            "a_native": "available",
            "q_next": "available",
        },
        ancestry=("fixture:analytic_pendulum_1dof",),
        geometry_stratum="unit_length",
        contact_stratum="no_contact",
        club_stratum="none",
    )


def _pendulum_2dof(*, trial_id: str, family_id: str, seed: int) -> EpisodeRecord:
    """Coupled 2-DOF fixture labels on DOFs 0-1 (software plant)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 0.6, _T)
    dt = float(t[1] - t[0])
    q0 = rng.uniform(-0.5, 0.5, size=(_T, 2))
    v0 = rng.uniform(-0.3, 0.3, size=(_T, 2))
    u0 = rng.uniform(-0.25, 0.25, size=(_T, 2))
    a0 = np.column_stack(
        [
            -np.sin(q0[:, 0]) - 0.1 * v0[:, 1] + u0[:, 0],
            -0.5 * np.sin(q0[:, 1]) + u0[:, 1],
        ]
    )
    qn0 = q0 + v0 * dt + 0.5 * a0 * dt * dt
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
        q=_pad(q0),
        v=_pad(v0),
        u=_pad(u0),
        a_native=_pad(a0),
        q_next=_pad(qn0),
        channel_availability={
            "q": "available",
            "v": "available",
            "u": "available",
            "a_native": "available",
            "q_next": "available",
        },
        ancestry=("fixture:analytic_pendulum_2dof",),
        geometry_stratum="unit_length",
        contact_stratum="no_contact",
        club_stratum="none",
    )


def _store_with_splits(
    tmp_path: Path, episodes: list[EpisodeRecord]
) -> tuple[EpisodeStore, FamilySplitPlan]:
    store = EpisodeStore(tmp_path / "corpus")
    for ep in episodes:
        store.write_episode(ep)
    plan = build_family_splits(episodes, ratios=_RATIOS, seed=7)
    return store, plan


def test_known_1dof_and_2dof_labels_classical_beats_or_matches_mlp(
    tmp_path: Path,
) -> None:
    """Analytical / classical methods must be scored honestly vs a tiny MLP."""
    episodes = [
        _pendulum_1dof(trial_id=f"p1_{i}", family_id=f"f1_{i}", seed=10 + i)
        for i in range(8)
    ] + [
        _pendulum_2dof(trial_id=f"p2_{i}", family_id=f"f2_{i}", seed=40 + i)
        for i in range(8)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    trainer = DynamicsBaselineTrainer(
        store=store,
        split=plan,
        task=DynamicsTaskKind.FORWARD_ACCELERATION,
        seeds=(0, 1, 2),
        output_dir=tmp_path / "runs",
        active_dofs=(0, 1),
    )
    card = trainer.run_pilot()
    assert isinstance(card, PilotCheckpointCard)
    assert card.schema == BASELINE_SCHEMA
    assert card.test_untouched is True
    assert ClassicalMethod.ANALYTICAL_PENDULUM.value in card.method_order
    assert ClassicalMethod.RIDGE.value in card.method_order
    assert ClassicalMethod.NEAREST_NEIGHBOR.value in card.method_order
    analytical = card.val_mse_by_method[ClassicalMethod.ANALYTICAL_PENDULUM.value]
    mlp = card.val_mse_by_method.get("small_mlp")
    if mlp is not None:
        assert analytical <= mlp * 1.05 or card.analytical_beats_mlp is True


def test_trial_split_leakage_regression(tmp_path: Path) -> None:
    """Near-duplicate family members must not cross train/val/test."""
    base = _pendulum_1dof(trial_id="leak_a", family_id="fam_leak", seed=1)
    twin = _pendulum_1dof(trial_id="leak_b", family_id="fam_leak", seed=2)
    other = _pendulum_1dof(trial_id="other", family_id="fam_other", seed=3)
    store, plan = _store_with_splits(tmp_path, [base, twin, other])
    mats = build_trial_matrices(
        store,
        plan,
        task=DynamicsTaskKind.FORWARD_ACCELERATION,
        split="train",
        active_dofs=(0,),
    )
    train_trials = set(mats.trial_ids)
    val = build_trial_matrices(
        store,
        plan,
        task=DynamicsTaskKind.FORWARD_ACCELERATION,
        split="val",
        active_dofs=(0,),
    )
    test = build_trial_matrices(
        store,
        plan,
        task=DynamicsTaskKind.FORWARD_ACCELERATION,
        split="test",
        active_dofs=(0,),
    )
    assert train_trials.isdisjoint(val.trial_ids)
    assert train_trials.isdisjoint(test.trial_ids)
    assert plan.split_of("leak_a") == plan.split_of("leak_b")
    assert plan.family_split["fam_leak"] == plan.split_of("leak_a")
    assert len({plan.family_split[f] for f in plan.family_split}) >= 1


def test_train_only_normalization(tmp_path: Path) -> None:
    episodes = [
        _pendulum_1dof(trial_id=f"n{i}", family_id=f"fn{i}", seed=100 + i)
        for i in range(6)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    train_trials = plan.splits["train"]
    trial_to_eid = {t: eid for eid, t in _trial_index(store).items()}
    train_eps = [
        store.read_episode(trial_to_eid[tid], lazy=False) for tid in train_trials
    ]
    normalizer = TrainOnlyNormalizer.fit(
        train_eps, channels=("q", "v", "u", "a_native")
    )
    trainer = DynamicsBaselineTrainer(
        store=store,
        split=plan,
        task=DynamicsTaskKind.INVERSE_CONTROL,
        seeds=(0,),
        output_dir=tmp_path / "norm",
        normalizer=normalizer,
        active_dofs=(0,),
        conditioning=InverseLabelConditioning(
            contact_mode="no_contact",
            actuation_mode="fully_actuated",
            allocation_policy="selection_objective",
        ),
    )
    card = trainer.run_pilot()
    assert card.normalizer_digest == normalizer.stats_digest
    with pytest.raises(ValueError, match="immutable"):
        normalizer.refit(train_eps)


def _trial_index(store: EpisodeStore) -> dict[str, str]:
    """Map episode_id -> trial_id from the store manifest."""
    out: dict[str, str] = {}
    for eid in store.iter_episode_ids():
        ep = store.read_episode(eid, lazy=False)
        out[eid] = ep.trial_id
    return out


def test_unavailable_torque_mask(tmp_path: Path) -> None:
    ep = _pendulum_1dof(trial_id="mask0", family_id="fm0", seed=5)
    ep = EpisodeRecord(
        trial_id=ep.trial_id,
        family_id=ep.family_id,
        model_id=ep.model_id,
        control_basis=ep.control_basis,
        units=ep.units,
        joint_names=ep.joint_names,
        coefficient_letters=ep.coefficient_letters,
        schema_version=ep.schema_version,
        sample_times_s=ep.sample_times_s,
        q=ep.q,
        v=ep.v,
        u=None,
        a_native=ep.a_native,
        q_next=ep.q_next,
        channel_availability={
            "q": "available",
            "v": "available",
            "u": "unavailable",
            "a_native": "available",
            "q_next": "available",
        },
        ancestry=ep.ancestry,
        geometry_stratum=ep.geometry_stratum,
        contact_stratum=ep.contact_stratum,
        club_stratum=ep.club_stratum,
    )
    store = EpisodeStore(tmp_path / "masked")
    store.write_episode(ep)
    plan = build_family_splits(
        [ep], ratios={"train": 1.0, "val": 0.0, "test": 0.0}, seed=0
    )
    with pytest.raises(ValueError, match="unavailable|torque|mask"):
        build_trial_matrices(
            store,
            plan,
            task=DynamicsTaskKind.INVERSE_CONTROL,
            split="train",
            active_dofs=(0,),
            conditioning=InverseLabelConditioning(
                contact_mode="no_contact",
                actuation_mode="fully_actuated",
                allocation_policy="selection_objective",
            ),
        )


def test_identity_channel_leakage_rejected(tmp_path: Path) -> None:
    """Copying q/v into the target must not count as dynamics skill."""
    episodes = [
        _pendulum_1dof(trial_id=f"id{i}", family_id=f"fid{i}", seed=200 + i)
        for i in range(4)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    mats = build_trial_matrices(
        store,
        plan,
        task=DynamicsTaskKind.FORWARD_ACCELERATION,
        split="train",
        active_dofs=(0,),
    )
    leak = identity_channel_leakage_score(
        mats.features, mats.targets, mats.feature_names, mats.target_names
    )
    assert leak["q_in_features_and_scored_as_target"] is False
    assert "q" not in mats.target_names
    assert "v" not in mats.target_names


def test_deterministic_tiny_overfit_smoke(tmp_path: Path) -> None:
    episodes = [
        _pendulum_1dof(trial_id=f"ov{i}", family_id=f"fov{i}", seed=300 + i)
        for i in range(4)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    trainer = DynamicsBaselineTrainer(
        store=store,
        split=plan,
        task=DynamicsTaskKind.FORWARD_NEXT_STATE,
        seeds=(0, 0),
        output_dir=tmp_path / "overfit",
        overfit_smoke=True,
        active_dofs=(0,),
    )
    a = trainer.run_pilot()
    b = trainer.run_pilot()
    assert a.content_digest == b.content_digest
    assert a.best_seed_metrics["train_mse"] < 1.0


def test_rollout_vs_onestep_reported(tmp_path: Path) -> None:
    episodes = [
        _pendulum_1dof(trial_id=f"ro{i}", family_id=f"fro{i}", seed=400 + i)
        for i in range(6)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    mats = build_trial_matrices(
        store,
        plan,
        task=DynamicsTaskKind.FORWARD_NEXT_STATE,
        split="val",
        active_dofs=(0,),
    )
    report = evaluate_rollout_vs_onestep(
        store=store,
        trial_ids=plan.splits["val"],
        predictor="identity_hold",
        horizon=4,
        active_dofs=(0,),
    )
    assert report["one_step_mse"] >= 0.0
    assert report["rollout_mse"] >= report["one_step_mse"] - 1e-9
    assert report["native_qualified"] is False
    assert "software_contract" in report["limitations"]
    assert mats.features.shape[0] > 0


def test_optional_torch_absence_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = [
        _pendulum_1dof(trial_id=f"torch{i}", family_id=f"ft{i}", seed=500 + i)
        for i in range(4)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    monkeypatch.setattr(
        "src.shared.python.neural_motion.baselines.neural.torch_is_available",
        lambda: False,
    )
    assert optional_torch_available() in (True, False)
    trainer = DynamicsBaselineTrainer(
        store=store,
        split=plan,
        task=DynamicsTaskKind.FORWARD_ACCELERATION,
        seeds=(0,),
        output_dir=tmp_path / "no_torch",
        require_mlp=False,
        active_dofs=(0,),
    )
    card = trainer.run_pilot()
    assert "small_mlp" not in card.val_mse_by_method or card.mlp_skipped_reason
    assert ClassicalMethod.RIDGE.value in card.val_mse_by_method


def test_training_scheduler_and_dataset_seams(tmp_path: Path) -> None:
    episodes = [
        _pendulum_1dof(trial_id=f"reg{i}", family_id=f"fr{i}", seed=600 + i)
        for i in range(3)
    ]
    store, _ = _store_with_splits(tmp_path, episodes)
    registry = DatasetRegistry()
    ds = register_dynamics_baseline_corpus(
        registry,
        dataset_id="nm05_pilot",
        name="NM-05 dynamics pilot",
        root=store.root,
    )
    assert ds.format == "hdf5"
    budget = neural_dynamics_baseline_budget()
    assert budget.schema == BASELINE_SCHEMA
    assert budget.seeds == (0, 1, 2)
    assert budget.tasks == (
        DynamicsTaskKind.FORWARD_ACCELERATION,
        DynamicsTaskKind.FORWARD_NEXT_STATE,
        DynamicsTaskKind.INVERSE_CONTROL,
    )


def test_inverse_requires_conditioning(tmp_path: Path) -> None:
    episodes = [
        _pendulum_2dof(trial_id=f"inv{i}", family_id=f"fi{i}", seed=700 + i)
        for i in range(4)
    ]
    store, plan = _store_with_splits(tmp_path, episodes)
    with pytest.raises(ValueError, match="conditioning"):
        build_trial_matrices(
            store,
            plan,
            task=DynamicsTaskKind.INVERSE_CONTROL,
            split="train",
            active_dofs=(0, 1),
            conditioning=None,
        )
