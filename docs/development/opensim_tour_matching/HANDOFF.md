# OpenSim Matching Handoff

Updated: 2026-09-12 UTC. Epic: [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003).
Plan: [EPIC_10003.md](EPIC_10003.md). Status: planning complete; user check-in pending.
No OpenSim implementation or native solver job has started in this workstream.

## Checkout and Ownership

- Worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003`.
- Branch: `docs/10003-opensim-matching-epic`; base `d03983d08`.
- Session: `opensim-epic-20260912`, agent `codex`, issue #10003.
- Owned path: `docs/development/opensim_tour_matching/`.
- Pinocchio worktree: `../UpstreamDrift-pinocchio-native`; read its current
  `docs/development/simscape_tour_matching/NATIVE_PORT_CHECKPOINT_20260911.md`.
- Gemini worktree: `../UpstreamDrift-simscape-tour`; do not modify its runtime.

## Next-Agent Prompt

Start with this read-only checkpoint command in PowerShell:

```powershell
Set-Location C:/Users/diete/Repositories/Worktrees/UpstreamDrift-opensim-10003
git status --short
git log -1 --oneline
Get-Content docs/development/opensim_tour_matching/EPIC_10003.md
```

Before editing, use the central coordinator from Repository_Management:

```powershell
python3 -m scripts.check_agent_claim --repo UpstreamDrift --issue 10003
python3 -m scripts.agent_communicate --repo UpstreamDrift --session opensim-epic-20260912 inbox
```

These inspect the parent planning session; register a fresh session and child
issue lease for implementation instead of impersonating this one.

Implement only OS-0 from the linked epic after the requested user check-in.
Read AGENTS.md, CLAUDE.md, this handoff, current Pinocchio receipts and public
OpenSim/shared interfaces. Check central claims/inbox and create a child issue
for OS-0 before code changes. Work in an isolated topic branch. Revalidate source
and data hashes. Do not infer installation from an old doc or copied package.

First produce a native runtime inventory and a failing qualification test for
missing/incompatible OpenSim. Discover an existing environment or use a dedicated
supported one; preserve all active Pinocchio/MATLAB environments and jobs. Probe
Moco support on a small constrained model before recommending a large solve.
Run existing pure tests and report their skips separately from native evidence.
Inventory actual packaged/runtime model coordinates, efforts, constraints,
markers, gravity, mass properties, contact and passive effects. Audit the source
C3D clock, masks, axes, units and any force plates. Save immutable receipts.

Do not optimize the real swing yet. Finish OS-0's done gate, commit tests/docs
normally and update this handoff with exact environment/reproduction commands.
Then dispatch OS-1 as a separate bounded task. If scientific model/contact choice
is unresolved, present the documented alternatives to the coordinating agent;
continue independent evidence/contract work without inventing an answer.

## Known Evidence and Limits

The two driver C3D copies match SHA-256
`545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba`.
The existing OpenSim fitter uses 25 coordinates by default; do not map native
27-channel coefficients by array position. Existing equivalence tests are not
current-native continuous-replay proof. Native OpenSim has not been exercised.

Local Python 3.13 and ControlTower Windows Python 3.14 returned no OpenSim module.
ControlTower's separate Pinocchio Python 3.12 environment also returned none.
Other installations remain unsearched. SSH `controltower` worked during this
read-only check. No MATLAB or optimization process was launched or interrupted.

## Required Checkpoint Fields

For every next commit, record: issue/claim/session; commit and dirty status;
source/model/capture hashes; host/executable/version; exact commands; red and green
tests with skip counts; run ID/parent candidate; artifact paths and SHA-256;
current best replay-qualified candidate; all fit/physical/coverage gates; live
process identity or terminal exit; next single action; blocker and owner;
push/PR/CI status; any changed assumptions or tolerances. Append dated evidence
and keep the current status at the top accurate.

Never overwrite completed runs or equate a planned command with one executed.
Copy/restore large artifacts by manifest, not guessed latest filenames.
