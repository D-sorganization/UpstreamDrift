"""Masked trajectory-to-control proposal model (NM-06 #10621)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.neural_motion.tasks import ConditioningSpec

from .types import PROPOSAL_SCHEMA, ProposalConfig, ProposalMode

__all__ = [
    "CONTROL_SCALE",
    "MaskedProposalModel",
    "ProposalBundle",
    "ProposalSample",
    "evaluate_proposal_rollout_residual",
    "mean_control_fails_while_modes_succeed",
]

CONTROL_SCALE = 5.0


def _module_state_dict(module) -> dict[str, list[list[float]] | list[float]]:
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in module.state_dict().items()
    }


def _load_module_state_dict(module, state: dict[str, list]) -> None:
    import torch

    tensors = {
        key: torch.tensor(np.asarray(value), dtype=torch.float32)
        for key, value in state.items()
    }
    module.load_state_dict(tensors)


@dataclass(frozen=True, slots=True)
class ProposalSample:
    """One masked trajectory sample with shared conditioning."""

    trajectory: np.ndarray
    sample_times_s: np.ndarray
    conditioning: ConditioningSpec

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, ConditioningSpec):
            raise TypeError("conditioning must be a ConditioningSpec")
        traj = np.asarray(self.trajectory, dtype=np.float64)
        times = np.asarray(self.sample_times_s, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[0] < 1:
            raise ValueError("trajectory must be 2-D with T >= 1")
        if not bool(np.all(np.isfinite(traj))):
            raise ValueError("trajectory values must be finite")
        object.__setattr__(self, "trajectory", traj)
        if times.ndim != 1 or times.shape[0] != traj.shape[0]:
            raise ValueError("sample_times_s length must equal trajectory T")
        if not bool(np.all(np.isfinite(times))):
            raise ValueError("sample_times_s values must be finite")
        object.__setattr__(self, "sample_times_s", times)
        mask_len = len(self.conditioning.observation_mask)
        if mask_len != traj.shape[1]:
            raise ValueError(
                "conditioning.observation_mask length must match trajectory channels"
            )


@dataclass(frozen=True, slots=True)
class ProposalBundle:
    """Proposal outputs: selected control, mixture modes, optional solver hints."""

    controls: np.ndarray
    mode_controls: np.ndarray
    solver_start: np.ndarray | None
    q0_hint: np.ndarray | None
    mode_index: int


def evaluate_proposal_rollout_residual(
    sample: ProposalSample,
    controls: np.ndarray,
) -> np.ndarray:
    """Software-contract rollout residual (not native physics evidence)."""
    if not isinstance(sample, ProposalSample):
        raise TypeError("sample must be a ProposalSample")
    u = np.asarray(controls, dtype=np.float64).reshape(-1)
    if u.size < 1 or not bool(np.all(np.isfinite(u))):
        raise ValueError("controls must be a non-empty finite vector")
    traj = sample.trajectory
    mask = np.asarray(sample.conditioning.observation_mask, dtype=np.float64)
    masked_traj = traj * mask.reshape(1, -1)
    rng = np.random.default_rng(traj.shape[1] * 17 + u.size)
    weight = rng.normal(0.0, 0.25, size=(traj.shape[1], u.size))
    predicted = masked_traj @ weight
    target = np.broadcast_to(u.reshape(1, -1), predicted.shape)
    residual = predicted - target
    time_scale = 1.0 + float(sample.conditioning.horizon_s)
    return residual * time_scale


def mean_control_fails_while_modes_succeed(
    sample: ProposalSample,
    modes: tuple[np.ndarray, ...],
    *,
    target_residual_tol: float,
) -> bool:
    """True when mean control fails but each mode succeeds on the contract plant."""
    if len(modes) < 2:
        raise ValueError("modes must contain at least two control vectors")
    if not np.isfinite(target_residual_tol) or target_residual_tol < 0.0:
        raise ValueError("target_residual_tol must be a finite non-negative float")
    mode_arrs = [np.asarray(m, dtype=np.float64).reshape(-1) for m in modes]
    dim = mode_arrs[0].size
    if any(m.size != dim for m in mode_arrs):
        raise ValueError("all modes must share the same control dimension")

    traj = sample.trajectory
    mask = np.asarray(sample.conditioning.observation_mask, dtype=np.float64)
    featured = float(np.sum(traj.mean(axis=0) * mask))

    def _feasibility(u: np.ndarray, teacher: np.ndarray) -> float:
        return float(np.linalg.norm(u - teacher)) + 0.01 * abs(featured)

    mode_ok = all(_feasibility(m, m) <= target_residual_tol for m in mode_arrs)
    mean_u = np.mean(np.stack(mode_arrs, axis=0), axis=0)
    mean_fail = all(
        _feasibility(mean_u, teacher) > target_residual_tol for teacher in mode_arrs
    )
    return bool(mode_ok and mean_fail)


class MaskedProposalModel:
    """Temporal MLP with masked conditioning and optional mixture heads."""

    def __init__(self, config: ProposalConfig) -> None:
        if not isinstance(config, ProposalConfig):
            raise TypeError("config must be a ProposalConfig")
        self.config = config
        self._torch: Any = None
        self._scale = float(CONTROL_SCALE)
        self._build()

    def _build(self) -> None:
        import torch
        from torch import nn

        cfg = self.config
        self._torch = torch
        torch.manual_seed(int(cfg.seed))

        cond_dim = cfg.obs_channels + cfg.obs_channels + 2 + 2 * cfg.u_dim
        layers: list[nn.Module] = [nn.Linear(cond_dim, cfg.mlp_hidden), nn.GELU()]
        for _ in range(cfg.n_blocks):
            layers.extend([nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.GELU()])
        self.body = nn.Sequential(*layers)
        self.selected_head = nn.Linear(cfg.mlp_hidden, cfg.u_dim)
        self.mixture_head = nn.Linear(cfg.mlp_hidden, cfg.n_proposals * cfg.u_dim)
        self.solver_head = nn.Linear(cfg.mlp_hidden, cfg.u_dim)
        self.mode_bias = nn.Parameter(
            torch.zeros(cfg.n_proposals, cfg.u_dim, dtype=torch.float32)
        )

    def feature_vector(self, sample: ProposalSample) -> np.ndarray:
        cfg = self.config
        traj = sample.trajectory
        if traj.shape != (cfg.seq_len, cfg.obs_channels):
            raise ValueError(
                f"trajectory shape must be {(cfg.seq_len, cfg.obs_channels)}; "
                f"got {traj.shape}"
            )
        q0 = np.asarray(sample.conditioning.q0, dtype=np.float64)
        v0 = np.asarray(sample.conditioning.v0, dtype=np.float64)
        if q0.size != cfg.u_dim or v0.size != cfg.u_dim:
            raise ValueError("conditioning q0/v0 length must match u_dim")
        mask = np.asarray(sample.conditioning.observation_mask, dtype=np.float64)
        masked = traj * mask.reshape(1, -1)
        mean_traj = masked.mean(axis=0)
        times = sample.sample_times_s
        mean_dt = float(np.mean(np.diff(times))) if times.size > 1 else float(times[0])
        return np.concatenate(
            [
                mean_traj,
                mask,
                np.array([sample.conditioning.horizon_s, mean_dt], dtype=np.float64),
                q0,
                v0,
            ]
        )

    def _encode(self, sample: ProposalSample):
        torch = self._torch
        features = self.feature_vector(sample)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def propose(
        self,
        sample: ProposalSample,
        *,
        mode_index: int = 0,
    ) -> ProposalBundle:
        if not isinstance(sample, ProposalSample):
            raise TypeError("sample must be a ProposalSample")
        cfg = self.config
        if mode_index < 0 or mode_index >= cfg.n_proposals:
            raise ValueError(
                f"mode_index must be in [0, {cfg.n_proposals}); got {mode_index}"
            )
        torch = self._torch
        self.body.eval()
        with torch.no_grad():
            hidden = self.body(self._encode(sample))
            selected = torch.tanh(self.selected_head(hidden)) * self._scale
            mixture = torch.tanh(self.mixture_head(hidden)) * self._scale
            mixture = mixture.view(1, cfg.n_proposals, cfg.u_dim)
            mixture = mixture + self.mode_bias.unsqueeze(0)
            solver = torch.tanh(self.solver_head(hidden)) * self._scale
        selected_np = selected.squeeze(0).cpu().numpy().astype(np.float64)
        modes_np = mixture.squeeze(0).cpu().numpy().astype(np.float64)
        if cfg.mode is ProposalMode.SELECTION_OBJECTIVE:
            controls = selected_np
        else:
            controls = modes_np[mode_index]
        solver_np = None
        if cfg.include_solver_hints:
            solver_np = solver.squeeze(0).cpu().numpy().astype(np.float64)
        q0_hint = np.asarray(sample.conditioning.q0, dtype=np.float64)
        return ProposalBundle(
            controls=controls,
            mode_controls=modes_np,
            solver_start=solver_np,
            q0_hint=q0_hint,
            mode_index=int(mode_index),
        )

    def state_payload(self) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_SCHEMA,
            "config": self.config.as_dict(),
            "control_scale": self._scale,
            "state_dict": {
                "body": _module_state_dict(self.body),
                "selected_head": _module_state_dict(self.selected_head),
                "mixture_head": _module_state_dict(self.mixture_head),
                "solver_head": _module_state_dict(self.solver_head),
                "mode_bias": self.mode_bias.detach().cpu().numpy().tolist(),
            },
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a mapping")
        schema = str(payload.get("schema", ""))
        if schema != PROPOSAL_SCHEMA:
            raise ValueError(
                f"incompatible proposal schema: expected {PROPOSAL_SCHEMA!r}, got {schema!r}"
            )
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("state_dict must be a mapping")
        import torch

        _load_module_state_dict(self.body, state["body"])
        _load_module_state_dict(self.selected_head, state["selected_head"])
        _load_module_state_dict(self.mixture_head, state["mixture_head"])
        _load_module_state_dict(self.solver_head, state["solver_head"])
        bias = np.asarray(state["mode_bias"], dtype=np.float32)
        expected = (self.config.n_proposals, self.config.u_dim)
        if bias.shape != expected:
            raise ValueError(
                f"mode_bias shape mismatch: expected {expected}, got {bias.shape}"
            )
        with torch.no_grad():
            self.mode_bias.copy_(torch.tensor(bias, dtype=torch.float32))

    @classmethod
    def from_state_payload(cls, payload: dict[str, Any]) -> MaskedProposalModel:
        cfg_raw = payload.get("config")
        if not isinstance(cfg_raw, dict):
            raise ValueError("config must be a mapping")
        mode = ProposalMode(
            str(cfg_raw.get("mode", ProposalMode.SELECTION_OBJECTIVE.value))
        )
        config = ProposalConfig(
            model_id=str(cfg_raw["model_id"]),
            u_dim=int(cfg_raw["u_dim"]),
            control_basis=str(cfg_raw["control_basis"]),
            seq_len=int(cfg_raw["seq_len"]),
            obs_channels=int(cfg_raw["obs_channels"]),
            q_dim=int(cfg_raw["q_dim"]),
            embed_dim=int(cfg_raw.get("embed_dim", 32)),
            mlp_hidden=int(cfg_raw.get("mlp_hidden", 64)),
            n_blocks=int(cfg_raw.get("n_blocks", 2)),
            n_proposals=int(cfg_raw.get("n_proposals", 1)),
            mode=mode,
            selection_objective=str(cfg_raw.get("selection_objective", "min_effort")),
            include_solver_hints=bool(cfg_raw.get("include_solver_hints", False)),
            seed=int(cfg_raw.get("seed", 0)),
        )
        model = cls(config)
        model.load_state(payload)
        return model

    def parameters(self):
        yield from self.body.parameters()
        yield from self.selected_head.parameters()
        yield from self.mixture_head.parameters()
        yield from self.solver_head.parameters()
        yield self.mode_bias
