"""NM-06 (#10621): masked trajectory-to-control proposals with native refinement.

Acceptance (issue copy): same target / multiple controls; mean control can fail
while valid modes succeed; masks and time influence output; variable model
dimensions; coefficient order / time-domain conversion; collapse diagnostics;
native refinement and replay; mismatch rejects checkpoint.

Synthetic fixtures validate software contracts only — not native training
success or acceleration claims. Native/evidence paths must exercise the real
adapter when claimed; these unit tests do not fabricate physics success.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Unit lane has no torch: skip this module cleanly (same pattern as test_hybrid.py).
pytest.importorskip("torch")

from src.shared.python.motion_matching.hybrid import (
    ProposalCheckpointContract,
    assert_proposal_checkpoint_compatible,
    refine_control_proposal,
)
from src.shared.python.motion_matching.inverse.basis_time import (
    coefficients_to_time_domain_torques,
    require_coefficient_letter_order,
)
from src.shared.python.motion_matching.inverse.collapse import (
    CVAE_PLATEAU_EVIDENCE,
    diagnose_mode_collapse,
)
from src.shared.python.motion_matching.inverse.masked_proposal import (
    MaskedObservation,
    MaskedProposalConfig,
    MaskedControlProposal,
    ProposalMode,
    evaluate_control_on_linear_plant,
    mean_control_fails_while_modes_succeed,
)
from src.shared.python.motion_matching.inverse.proposal_training import (
    ProposalTrainingConfig,
    train_masked_control_proposals,
)

pytestmark = pytest.mark.unit

_T = 12
_C_CLUB = 6  # butt(3) + clubhead(3) software-contract layout


def _obs(
    *,
    control_dim: int = 2,
    mask: tuple[bool, ...] | None = None,
    times: np.ndarray | None = None,
    seed: int = 0,
) -> MaskedObservation:
    rng = np.random.default_rng(seed)
    traj = rng.normal(0.0, 0.1, size=(_T, _C_CLUB)).astype(np.float64)
    if mask is None:
        mask = tuple(True for _ in range(_C_CLUB))
    if times is None:
        times = np.linspace(0.0, 0.6, _T)
    return MaskedObservation(
        trajectory=traj,
        observation_mask=mask,
        sample_times_s=times,
        duration_s=float(times[-1] - times[0]),
        q0=tuple(0.0 for _ in range(control_dim)),
        v0=tuple(0.0 for _ in range(control_dim)),
        geometry_id="geom.fixture.v1",
        model_id="driven_double_pendulum",
        control_basis="joint_torque",
        contact_profile="no_contact.v1",
    )


def test_same_target_multiple_controls_mean_fails_modes_succeed() -> None:
    """Ambiguous target: mean of modes fails plant; each feasible mode succeeds."""
    obs = _obs(control_dim=2, seed=3)
    mode_a = np.array([0.4, -0.2], dtype=np.float64)
    mode_b = np.array([-0.4, 0.2], dtype=np.float64)
    assert mean_control_fails_while_modes_succeed(
        obs,
        modes=(mode_a, mode_b),
        target_residual_tol=0.05,
    )


def test_masks_and_time_influence_output() -> None:
    pytest.importorskip("torch")
    cfg = MaskedProposalConfig(
        control_dim=2,
        trajectory_channels=_C_CLUB,
        seq_len=_T,
        n_modes=2,
        hidden=32,
        n_blocks=1,
        seed=11,
    )
    model = MaskedControlProposal(cfg)
    rich = _obs(mask=tuple(True for _ in range(_C_CLUB)), seed=1)
    club_only = MaskedObservation(
        trajectory=rich.trajectory.copy(),
        observation_mask=(True, True, True, False, False, False),
        sample_times_s=rich.sample_times_s.copy(),
        duration_s=rich.duration_s,
        q0=rich.q0,
        v0=rich.v0,
        geometry_id=rich.geometry_id,
        model_id=rich.model_id,
        control_basis=rich.control_basis,
        contact_profile=rich.contact_profile,
    )
    selected_rich = model.propose(rich, mode=ProposalMode.SELECTED)
    selected_club = model.propose(club_only, mode=ProposalMode.SELECTED)
    assert selected_rich.controls.shape == (2,)
    assert not np.allclose(selected_rich.controls, selected_club.controls)

    stretched = MaskedObservation(
        trajectory=rich.trajectory.copy(),
        observation_mask=rich.observation_mask,
        sample_times_s=np.linspace(0.0, 1.2, _T),
        duration_s=1.2,
        q0=rich.q0,
        v0=rich.v0,
        geometry_id=rich.geometry_id,
        model_id=rich.model_id,
        control_basis=rich.control_basis,
        contact_profile=rich.contact_profile,
    )
    selected_time = model.propose(stretched, mode=ProposalMode.SELECTED)
    assert not np.allclose(selected_rich.controls, selected_time.controls)


def test_variable_model_dimensions() -> None:
    pytest.importorskip("torch")
    for control_dim in (1, 2, 4):
        cfg = MaskedProposalConfig(
            control_dim=control_dim,
            trajectory_channels=_C_CLUB,
            seq_len=_T,
            n_modes=2,
            hidden=24,
            n_blocks=1,
            seed=control_dim,
        )
        model = MaskedControlProposal(cfg)
        out = model.propose(_obs(control_dim=control_dim), mode=ProposalMode.SELECTED)
        assert out.controls.shape == (control_dim,)
        mixture = model.propose(
            _obs(control_dim=control_dim), mode=ProposalMode.MIXTURE
        )
        assert mixture.mode_controls.shape == (cfg.n_modes, control_dim)


def test_coefficient_order_and_time_domain_conversion() -> None:
    letters = ("A", "B", "C", "D", "E", "F", "G")
    require_coefficient_letter_order(letters)
    with pytest.raises(ValueError, match="letter"):
        require_coefficient_letter_order(("G", "F", "E", "D", "C", "B", "A"))

    # Constant torque A=1.5, others zero — time-domain must stay 1.5
    coeffs = np.zeros((2, 7), dtype=np.float64)
    coeffs[:, 0] = 1.5
    times = np.array([0.0, 0.25, 0.5], dtype=np.float64)
    torques = coefficients_to_time_domain_torques(coeffs, times, letter_order=letters)
    assert torques.shape == (3, 2)
    assert np.allclose(torques, 1.5)


def test_collapse_diagnostics_and_cvae_plateau_evidence() -> None:
    collapsed = np.ones((8, 3), dtype=np.float64)
    diag = diagnose_mode_collapse(collapsed, min_pairwise_l2=0.05)
    assert diag.is_collapsed is True
    assert diag.effective_mode_count == 1

    diverse = np.eye(3, dtype=np.float64)
    diag_ok = diagnose_mode_collapse(diverse, min_pairwise_l2=0.05)
    assert diag_ok.is_collapsed is False
    assert diag_ok.effective_mode_count >= 2

    # Plateau evidence from the historical cVAE investigation must remain named.
    assert CVAE_PLATEAU_EVIDENCE["val_recon_plateau"] is True
    assert CVAE_PLATEAU_EVIDENCE["mean_prediction_baseline"] is True


def test_native_refinement_and_independent_replay() -> None:
    obs = _obs(control_dim=2, seed=5)
    warm = np.array([0.1, -0.05], dtype=np.float64)

    def polish(_target: object, theta: np.ndarray) -> dict[str, object]:
        return {
            "coefficients": theta * 0.5,
            "final_rmse_m": 1.0e-3,
            "solver": "fixture_polish",
            "independent_replay": True,
            "projection_cost": 0.42,
        }

    result = refine_control_proposal(
        observation=obs,
        proposal_controls=warm,
        polish_fn=polish,
        require_independent_replay=True,
    )
    assert result.independent_replay is True
    assert result.projection_cost == pytest.approx(0.42)
    assert np.allclose(result.controls, warm * 0.5)


def test_mismatch_rejects_checkpoint(tmp_path: Path) -> None:
    contract = ProposalCheckpointContract(
        model_id="driven_double_pendulum",
        control_dim=2,
        control_basis="joint_torque",
        schema_version="neural-masked-proposals/1.0.0",
        weight_digest="abc123",
    )
    bad = ProposalCheckpointContract(
        model_id="driven_double_pendulum",
        control_dim=4,
        control_basis="joint_torque",
        schema_version="neural-masked-proposals/1.0.0",
        weight_digest="abc123",
    )
    with pytest.raises(ValueError, match="incompatible"):
        assert_proposal_checkpoint_compatible(expected=contract, loaded=bad)

    path = tmp_path / "ckpt.json"
    contract.write_json(path)
    loaded = ProposalCheckpointContract.read_json(path)
    assert_proposal_checkpoint_compatible(expected=contract, loaded=loaded)


def test_train_losses_use_observation_after_rollout_not_coeff_mse_alone(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    cfg = MaskedProposalConfig(
        control_dim=2,
        trajectory_channels=_C_CLUB,
        seq_len=_T,
        n_modes=2,
        hidden=32,
        n_blocks=1,
        seed=7,
    )
    train_cfg = ProposalTrainingConfig(
        epochs=3,
        batch_size=4,
        lr=1e-2,
        seed=7,
        output_dir=tmp_path / "run",
        use_observation_rollout_loss=True,
        control_regularization=1e-3,
    )
    samples = [_obs(seed=20 + i) for i in range(8)]
    # Teacher labels: two modes that average to near-zero (ambiguous).
    labels = [(np.array([0.5, -0.3]), np.array([-0.5, 0.3])) for _ in samples]
    result = train_masked_control_proposals(
        config=cfg,
        training=train_cfg,
        observations=samples,
        teacher_mode_controls=labels,
    )
    assert result.checkpoint_path.is_file()
    assert result.used_observation_rollout_loss is True
    assert result.final_observation_loss < result.final_control_mse * 10 + 1.0
    assert "control_mse_alone" not in result.selection_criterion


def test_linear_plant_evaluator_is_deterministic() -> None:
    obs = _obs(control_dim=2, seed=9)
    u = np.array([0.2, -0.1], dtype=np.float64)
    a = evaluate_control_on_linear_plant(obs, u)
    b = evaluate_control_on_linear_plant(obs, u)
    assert np.allclose(a, b)
    assert a.shape[0] == obs.trajectory.shape[0]
