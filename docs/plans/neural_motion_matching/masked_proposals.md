# NM-06 Masked Trajectory-to-Control Proposals With Native Refinement

Governing issue: #10621 (epic #10603).

Schema: `neural-masked-proposals/1.0.0`

## What Landed

Primary package under `src/shared/python/neural_motion/proposals/` (NM-05-adjacent):

- `ProposalConfig` binds `MaskedTrajectoryTask` / `TaskDimensions.u_dim` (rejects hard-coded 189 as primary)
- `MaskedProposalModel` emits `(B, u_dim)` or `(B, K, u_dim)` controls; optional `q0_hint` / `solver_start`
- Masked conditioning via `ConditioningSpec.observation_mask`
- Deterministic selection-objective path + mixture/multi-proposal ablation
- `train_masked_proposals` with observation + regularization losses (control aux not sole)
- `refine_proposal_hybrid` DI `polish_fn` fail-closed contract
- Checkpoint schema `neural-masked-proposals/1.0.0` with strict model_id/u_dim/control_basis mismatch rejection

Inverse-path reuse (stem / collapse / basis-time) remains under
`motion_matching/inverse/` for temporal architecture continuity.

## Evidence

`evidence/nm06_masked_proposals_receipt.json`

## Limitations

Software-contract fixtures only. No fabricated native training success,
acceleration claim, or motion-matching certification. Native refinement
requires an independent-replay receipt; missing replay fails closed.

## Next Action

NM-07 (#10622): compare forward surrogates and physics-structured alternatives
under the NM-06 proposal + native refinement boundary.
