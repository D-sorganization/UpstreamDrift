"""NM-06 (#10621): neural_motion masked trajectory-to-control proposals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.neural_motion.proposals import (
    PROPOSAL_SCHEMA,
    MaskedProposalModel,
    MaskedProposalTrainConfig,
    ProposalConfig,
    ProposalMode,
    ProposalSample,
    load_proposal_checkpoint,
    mean_control_fails_while_modes_succeed,
    refine_proposal_hybrid,
    save_proposal_checkpoint,
    train_masked_proposals,
)
from src.shared.python.neural_motion.tasks import (
    ConditioningSpec,
    MaskedTrajectoryTask,
    dimensions_from_model,
)

pytestmark = pytest.mark.unit

_T = 12
_MODEL = "driven_double_pendulum"


def _conditioning(*, mask: tuple[bool, ...] = (True, True)) -> ConditioningSpec:
    dims = dimensions_from_model(_MODEL)
    return ConditioningSpec(
        geometry_id="geom.fixture.v1",
        q0=tuple(0.0 for _ in range(dims.u_dim)),
        v0=tuple(0.0 for _ in range(dims.u_dim)),
        horizon_s=0.6,
        time_step_s=0.05,
        constraint_profile="model_native.v1",
        contact_profile="no_contact.v1",
        observation_mask=mask,
    )


def _task(*, mask: tuple[bool, ...] = (True, True)) -> MaskedTrajectoryTask:
    dims = dimensions_from_model(_MODEL)
    return MaskedTrajectoryTask(
        task_id=f"mask.{_MODEL}.fixture",
        dimensions=dims,
        conditioning=_conditioning(mask=mask),
        proposal_space="continuous_controls",
    )


def _config(**kwargs: object) -> ProposalConfig:
    task = _task()
    defaults: dict[str, object] = {
        "control_basis": "joint_torque",
        "seq_len": _T,
        "n_proposals": 1,
        "mode": ProposalMode.SELECTION_OBJECTIVE,
        "include_solver_hints": False,
        "seed": 11,
    }
    defaults.update(kwargs)
    return ProposalConfig.from_task(task, **defaults)  # type: ignore[arg-type]


def _trajectory(*, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dims = dimensions_from_model(_MODEL)
    return rng.normal(0.0, 0.1, size=(_T, dims.q_dim)).astype(np.float64)


def _times(*, span: float = 0.6) -> np.ndarray:
    return np.linspace(0.0, span, _T)


def _sample(
    *,
    seed: int = 0,
    mask: tuple[bool, ...] = (True, True),
    span: float = 0.6,
) -> ProposalSample:
    cond = _conditioning(mask=mask)
    if span != 0.6:
        cond = ConditioningSpec(
            geometry_id=cond.geometry_id,
            q0=cond.q0,
            v0=cond.v0,
            horizon_s=span,
            time_step_s=cond.time_step_s,
            constraint_profile=cond.constraint_profile,
            contact_profile=cond.contact_profile,
            observation_mask=mask,
        )
    return ProposalSample(
        trajectory=_trajectory(seed=seed),
        sample_times_s=_times(span=span),
        conditioning=cond,
    )


def test_proposal_config_rejects_hardcoded_189_primary_dim() -> None:
    with pytest.raises(ValueError, match="u_dim"):
        ProposalConfig(
            model_id=_MODEL,
            u_dim=189,
            control_basis="joint_torque",
            seq_len=_T,
            obs_channels=2,
            q_dim=2,
        )


def test_same_target_multiple_controls_mean_fails_modes_succeed() -> None:
    sample = _sample(seed=3)
    mode_a = np.array([0.4, -0.2], dtype=np.float64)
    mode_b = np.array([-0.4, 0.2], dtype=np.float64)
    assert mean_control_fails_while_modes_succeed(
        sample,
        (mode_a, mode_b),
        target_residual_tol=0.05,
    )


def test_masks_and_time_influence_output() -> None:
    pytest.importorskip("torch")
    cfg = _config(n_proposals=2, mode=ProposalMode.MIXTURE_ABLATION, seed=11)
    model = MaskedProposalModel(cfg)
    rich = _sample(mask=(True, True), seed=1)
    club_only = ProposalSample(
        trajectory=rich.trajectory.copy(),
        sample_times_s=rich.sample_times_s.copy(),
        conditioning=_conditioning(mask=(True, False)),
    )
    out_rich = model.propose(rich)
    out_club = model.propose(club_only)
    assert out_rich.controls.shape == (cfg.u_dim,)
    assert not np.allclose(out_rich.controls, out_club.controls)

    stretched = _sample(seed=1, span=1.2)
    out_time = model.propose(stretched)
    assert not np.allclose(out_rich.controls, out_time.controls)


def test_variable_model_dimensions() -> None:
    pytest.importorskip("torch")
    dims = dimensions_from_model(_MODEL)
    assert dims.u_dim == 2
    for n_prop in (1, 2):
        mode = (
            ProposalMode.SELECTION_OBJECTIVE
            if n_prop == 1
            else ProposalMode.MIXTURE_ABLATION
        )
        cfg = _config(n_proposals=n_prop, mode=mode, seed=n_prop)
        model = MaskedProposalModel(cfg)
        out = model.propose(_sample(seed=n_prop))
        assert out.controls.shape == (dims.u_dim,)
        assert out.mode_controls.shape == (n_prop, dims.u_dim)


def test_solver_hints_optional() -> None:
    pytest.importorskip("torch")
    cfg_plain = _config(include_solver_hints=False)
    cfg_hints = _config(include_solver_hints=True, seed=99)
    plain = MaskedProposalModel(cfg_plain).propose(_sample(seed=2))
    hinted = MaskedProposalModel(cfg_hints).propose(_sample(seed=2))
    assert plain.solver_start is None
    assert hinted.solver_start is not None
    assert hinted.solver_start.shape == (cfg_hints.u_dim,)


def test_refine_fail_closed_without_replay() -> None:
    warm = np.array([0.1, -0.05], dtype=np.float64)

    def polish(_target: object, theta: np.ndarray) -> dict[str, object]:
        return {
            "coefficients": theta * 0.5,
            "projection_cost": 0.1,
            "independent_replay": False,
        }

    with pytest.raises(ValueError, match="independent_replay"):
        refine_proposal_hybrid(
            target=_sample(),
            proposal_controls=warm,
            polish_fn=polish,
            expected_control_dim=2,
        )


def test_refine_hybrid_with_replay() -> None:
    warm = np.array([0.1, -0.05], dtype=np.float64)

    def polish(_target: object, theta: np.ndarray) -> dict[str, object]:
        return {
            "coefficients": theta * 0.5,
            "final_rmse_m": 1.0e-3,
            "solver": "fixture_polish",
            "independent_replay": True,
            "projection_cost": 0.42,
        }

    result = refine_proposal_hybrid(
        target=_sample(seed=5),
        proposal_controls=warm,
        polish_fn=polish,
        expected_control_dim=2,
    )
    assert result.independent_replay is True
    assert result.projection_cost == pytest.approx(0.42)
    assert np.allclose(result.controls, warm * 0.5)


def test_checkpoint_mismatch_rejects(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    cfg = _config(seed=3)
    model = MaskedProposalModel(cfg)
    path = tmp_path / "proposal.json"
    save_proposal_checkpoint(path, model)
    with pytest.raises(ValueError, match="incompatible"):
        load_proposal_checkpoint(
            path,
            model_id=_MODEL,
            u_dim=4,
            control_basis="joint_torque",
        )
    loaded, card = load_proposal_checkpoint(
        path,
        model_id=_MODEL,
        u_dim=cfg.u_dim,
        control_basis=cfg.control_basis,
    )
    assert card.schema == PROPOSAL_SCHEMA
    assert loaded.config.model_id == _MODEL


def test_train_losses_use_observation_not_coeff_mse_alone(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    task = _task()
    cfg = _config(
        n_proposals=2,
        mode=ProposalMode.MIXTURE_ABLATION,
        seed=7,
    )
    trajs = [_trajectory(seed=20 + i) for i in range(8)]
    times = [_times() for _ in trajs]
    labels = [
        (
            np.array([0.5, -0.3], dtype=np.float64),
            np.array([-0.5, 0.3], dtype=np.float64),
        )
        for _ in trajs
    ]
    result = train_masked_proposals(
        config=cfg,
        task=task,
        training=MaskedProposalTrainConfig(
            epochs=3,
            batch_size=4,
            lr=1e-2,
            seed=7,
            output_dir=tmp_path / "run",
        ),
        trajectories=trajs,
        sample_times=times,
        teacher_controls=labels,
    )
    assert result.checkpoint_path.is_file()
    assert result.claims_native_success is False
    assert result.used_observation_rollout_loss is True
    assert result.final_observation_loss < result.final_control_loss * 10 + 1.0
    assert "control_mse_alone" not in result.selection_criterion


def test_state_payload_roundtrip() -> None:
    pytest.importorskip("torch")
    cfg = _config(n_proposals=2, mode=ProposalMode.MIXTURE_ABLATION, seed=5)
    model = MaskedProposalModel(cfg)
    sample = _sample(seed=4)
    before = model.propose(sample)
    payload = model.state_payload()
    restored = MaskedProposalModel.from_state_payload(payload)
    after = restored.propose(sample)
    assert np.allclose(before.controls, after.controls)
    assert np.allclose(before.mode_controls, after.mode_controls)
