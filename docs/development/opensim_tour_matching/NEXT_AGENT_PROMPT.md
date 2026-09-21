# Copy-Ready OpenSim Continuation Prompt

Work #10341, Phase A continuation, under epic #10363. Do not restart completed
OS-2b/OS-3b or claim closed epic #10003 is a matched full swing.

Read [HANDOFF.md](HANDOFF.md) and the
[bounded continuation prompt](../matched_swing_program/AGENT_CONTINUATION_PROMPT.md).
Use the exact local/published source and deployed runtime identities recorded
there. The first task is to inspect and collect the existing `os7_g1f` job,
not launch another solver. Preserve its log, output and b-run warm start.

Then run `tests/opensim/test_moco_g1_ladder.py` on the pinned source, validate
requested versus achieved horizon and receipt provenance, and report
collocation/replay differences plus residual support. Existing phase-A model
limitations keep it unqualified for full-body acceptance. If the live attempt
fails, prepare one bounded diagnostic with an explicit hypothesis; escalate
model/contact/initial-state decisions or repeated infeasibility. Phase A does
not wait for MuJoCo; Phase B depends on shared-model/contact work.

Preserve the prior agent's dirty development log and untracked one-iteration
receipt. Work in your own checkout, hold your own lease, update handoff/log
in place, and publish one concrete next action with links to raw evidence.

## Golf Model Improvement Turnover

For the owner-requested anatomy, visible club and address corrections, follow
[EPIC_GOLF_MODEL.md](EPIC_GOLF_MODEL.md) (epic #10394, children #10395–#10403)
and [GOLF_MODEL_AGENT_PROMPT.md](GOLF_MODEL_AGENT_PROMPT.md). Start #10395,
then #10397. This is separate from resuming the prior Moco job. Preserve
that job and its evidence; corrected physical models invalidate old receipts.
