# NM-06 Masked Trajectory-to-Control Proposals With Native Refinement

Governing issue: #10621 (epic #10603).

Schema: `neural-masked-proposals/1.0.0`

## What Landed

Generalized temporal inverse architecture under the saved model/basis contract:

- `motion_matching/inverse/masked_proposal.py` — masked observations,
  native timestamps/duration, q0/v0/geometry conditioning; selected-teacher
  and mixture heads with variable `control_dim` (not a fixed 189-vector)
- `motion_matching/inverse/proposal_training.py` — observation residual after
  differentiable software-contract rollout + control regularization; never
  coefficient-MSE-alone as the selection criterion
- `motion_matching/inverse/basis_time.py` — canonical A..G letter order gate and
  time-domain torque conversion via `evaluate_polynomial_torque`
- `motion_matching/inverse/collapse.py` — mode-collapse diagnostics; retained
  cVAE plateau evidence (`CVAE_PLATEAU_EVIDENCE`)
- Anchors wired: `regressor.py`, `regressor_training.py`, `cvae.py`,
  `hybrid.py` (`refine_control_proposal`, fail-closed checkpoint contract)

## Evidence

`evidence/nm06_masked_proposals_receipt.json`

## Limitations

Software-contract fixtures only. No fabricated native training success,
acceleration claim, or motion-matching certification. Independent-replay
evidence is required for native refinement acceptance; missing replay fails
closed. Mixture diversity must pass collapse diagnostics.

## Next Action

NM-07 (#10622): compare forward surrogates and physics-structured alternatives
under the NM-06 proposal + native refinement boundary.
