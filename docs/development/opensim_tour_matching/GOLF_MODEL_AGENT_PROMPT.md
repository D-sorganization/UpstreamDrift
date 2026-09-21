# OpenSim Golf Model Agent Assignment

Implement ONE ready child of epic #10394 under #10363. Begin with OG-01
(#10395); then OG-03 (#10397) is the smallest visible improvement. Read
EPIC_GOLF_MODEL.md, HANDOFF.md, the child issue and repository AGENTS.md /
CLAUDE.md. Planning is in PR #10393 on docs/matching-agent-continuation.

## Starting State

Use your own worktree from current origin/main and claim the chosen child via
Repository_Management. Register presence, read inbox and check upstream PRs.
Fetch handoff/10341-opensim-20260918 at 6914383f5 as read-only evidence; never
reuse/reset Claude's \_wt_claude_10341 or assume its native code is merged.
The model is under docs/development/opensim_tour_matching/evidence/os7_moco_g1/
on that source ref. Verify the inspection.json hash before calling it baseline.
Keep the original model and IK/replay files unchanged. HANDOFF.md has the
ControlTower runtime/job paths; inspect live identity before any job action.

## First Bounded Assignment

For OG-01, inventory baseline geometry, model/capture/motion hashes, joint
centers and marker residuals. Reproduce empty Club attached_geometry and
non-unit joint offsets with unchanged arm meshes. Add failing qualification
fixtures while retaining the old baseline as intentionally rejected evidence.
Produce a short diagnosis and exact reproduction commands; no new long fit.

For OG-03 after OG-01, add visible geometry to the EXISTING Club body through
the existing generator/adapter. Write the failing geometry/FK tests first.
Preserve physical properties for this first visual fix. Reuse ClubSpec but
explicitly convert frame conventions; shared spec is head-origin while this
OpenSim body is grip-origin with shaft along -Y. Verify native save/reload and
address/top/finish views. Do not add a second club or fake a grip by camera.

## Scope and Stop Rules

Keep one issue and one PR. At most one short native diagnostic per measured
hypothesis; agree a runtime/iteration bound in the receipt before launching.
No 600-iteration optimization for a geometry-only task. Never stop or duplicate
an owner job. Escalate scaling policy, anatomical joint centers, constraint
rank, muscle paths/strength and unexplained native divergence to expert review.
Do not clamp segment lengths or joint angles merely to improve appearance.

Run the child's pure tests, relevant native test and visual verification.
Native SDK skips are not passes. Follow repo formatting/governance checks.
No extra parser, metrics schema, model registry or UI stack. Use contracts
and adapter boundaries from the epic. Keep torque controls distinct from
muscle excitation/activation/tendon state. Missing capabilities fail explicitly.

Finish with exact commit/PR, changed model/config hashes, test commands and
counts, before/after evidence, unresolved failures and ONE next action.
Update the governing development-log row and canonical/lane handoff. Send a
durable Repository_Management mailbox notice and link the evidence on the
child issue. Never close the parent epic from a child PR.
