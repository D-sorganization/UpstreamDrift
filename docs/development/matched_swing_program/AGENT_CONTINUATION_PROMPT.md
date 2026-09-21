# Copy-Ready Motion-Matching Continuation Prompt

Continue epic #10363 from the reviewed handoffs, not from old chat summaries.
Choose ONE lane: Pinocchio qualification #10381 or OpenSim #10341. Read repo
AGENTS.md/CLAUDE.md, the updated epic, and the lane handoff before writing code.
Use Repository_Management claim/lease/presence/inbox tools with your own session.
Do not take over another agent's worktree or stop/duplicate its running job.

Read:

- [Pinocchio/Crocoddyl](MS31_PINOCCHIO_CROCODDYL_TURNOVER.md).
- [OpenSim](../opensim_tour_matching/HANDOFF.md).
- [Raw Timestamped Evidence](evidence/continuation_20260918/matching-handoff-snapshot.json).
- [Source and Recovery Manifest](evidence/continuation_20260918/matching-source-manifest.json).

First deliverable: an evidence-only status update. Connect to ControlTower,
inspect the existing job and outputs, retrieve newly completed results into a
new evidence directory, verify hashes/coverage, and report all failed/missing
gates. Keep the current result even if rejected. No parameter sweep is required.

For Windows SSH quoting, write an owned LF script, copy it with scp to
`controltower:C:/Users/diete/`, then execute it through
`ssh controltower "wsl -d ControlTower-Runner -- bash /mnt/c/Users/diete/NAME.sh"`.
Use a unique NAME and script containing only the selected read-only inspection
commands. Native paths and process IDs in the snapshot are historical; verify
current state. Check processes, file timestamps and receipts, not just a quiet
log. Use checkpoint-based checks, not tight polling or blocking long sleeps.

Second deliverable: run the existing focused tests on the exact pinned source,
validate candidate/receipt structure, and prepare a narrow source/evidence PR.
Current main and Claude's native branch have competing module implementations;
never blindly overwrite one with the other. Retrieve the pinned branch or
recovery archive first. Escalate integration conflicts instead of guessing.

Only after the job has exited, a valid checkpoint exists and a single measured
hypothesis is documented, perform at most ONE bounded continuation/diagnostic
in a new output directory using the lane's existing driver. Preserve all
inputs/settings and report elapsed time and outcome. If the same failure
persists or physical/derivative/model decisions are required, stop that
experiment and hand the evidence to an expert; do not repeat expensive sweeps.

Acceptance rules: solver success is not physical acceptance; 0.30/0.60 s is not
0.85 s G1; reduced/residual-supported models are not full-body release; no
missing-marker zero scores, clipped horizons or threshold relaxation. Use
`metrics.*.shared` consistently, retain every failed physical gate and record
unverified evidence. The current acceptance implementation exists; #10374
still owns its qualification and schema integration. Follow the live epic
rather than conflicting historical GATES.md tables.

End with: exact source/runtime/model/capture hashes, current job state, raw
artifact paths, solver and replay metrics, physical failures, tests run,
branch/PR, and ONE next action. Update the canonical handoff, lane handoff and
existing governing-issue development-log entry in the same implementation
commit. Send the durable mailbox notice and link the evidence on the lane issue.
Do not claim full-swing completion without all required native receipts.

OpenSim product clarification: read the Anatomical Playback and Muscle Scope
section of the lane handoff. Deliver a visible anatomical skeleton and clearly
labeled motions; the current model has zero muscles. Inventory and escalate
muscle-actuated model qualification rather than labeling joint torques as
muscle activity. Preserve the current diagnostic baseline.
