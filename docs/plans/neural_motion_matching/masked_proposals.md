# NM-06 Masked Trajectory-to-Control Proposals With Native Refinement

Governing issue: #10621 (epic #10603).

Schema:
eural-masked-proposals/1.0.0

## What Landed

Primary package under src/shared/python/neural_motion/proposals/ (NM-05-adjacent):

- ProposalConfig / ProposalFromTaskSpec bind MaskedTrajectoryTask / TaskDimensions.u_dim (rejects hard-coded 189 as primary)
- MaskedProposalModel emits (B, u_dim) or (B, K, u_dim) controls; optional q0_hint / solver_start
- Masked conditioning via ConditioningSpec.observation_mask
- Deterministic selection-objective path + mixture/multi-proposal ablation
-     rain_masked_proposals with observation + regularization losses (control aux not sole)
- efine_proposal_hybrid DI polish_fn fail-closed contract
- Checkpoint schema
  eural-masked-proposals/1.0.0 with strict model_id/u_dim/control_basis mismatch rejection
- Optional torch: proposal modules and motion_matching.inverse NM-06 exports import without torch; training/model build remain lazy (NM-05 pattern)

Inverse-path reuse anchors (do not rewrite the legacy trainer):

- motion_matching/inverse/masked_proposal.py — variable-dim stem / heads
- motion_matching/inverse/proposal_training.py — observation-rollout training loop
- motion_matching/inverse/collapse.py + asis_time.py — collapse diagnostics and A..G time-domain conversion
- motion_matching/inverse/regressor.py — docstring reuse note only;
  egressor_training.py stays on main

## Evidence

vidence/nm06_masked_proposals_receipt.json

## Limitations

Software-contract fixtures only. No fabricated native training success,
acceleration claim, or motion-matching certification. Native refinement
requires an independent-replay receipt; missing replay fails closed.

## Next Action

NM-07 (#10622): compare forward surrogates and physics-structured alternatives
under the NM-06 proposal + native refinement boundary.
