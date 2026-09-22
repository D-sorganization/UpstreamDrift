"""NM-06 masked trajectory-to-control proposals."""

from .checkpoint import load_proposal_checkpoint, save_proposal_checkpoint
from .model import (
    MaskedProposalModel,
    ProposalBundle,
    ProposalSample,
    mean_control_fails_while_modes_succeed,
)
from .refine import ProposalPolishResult, RefinePolishFn, refine_proposal_hybrid
from .train import (
    MaskedProposalTrainConfig,
    MaskedProposalTrainResult,
    train_masked_proposals,
)
from .types import PROPOSAL_SCHEMA, ProposalConfig, ProposalMode

__all__ = [
    "PROPOSAL_SCHEMA",
    "MaskedProposalModel",
    "MaskedProposalTrainConfig",
    "MaskedProposalTrainResult",
    "ProposalBundle",
    "ProposalConfig",
    "ProposalMode",
    "ProposalPolishResult",
    "ProposalSample",
    "RefinePolishFn",
    "load_proposal_checkpoint",
    "mean_control_fails_while_modes_succeed",
    "refine_proposal_hybrid",
    "save_proposal_checkpoint",
    "train_masked_proposals",
]
