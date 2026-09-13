# OpenSim Matching Handoff

## OS-0 Execution Complete: Runtime and Evidence Baseline (2026-09-13 UTC)

Work Package OS-0 of Epic #10003 is complete. Baseline runtime environment, packaged OpenSim model topology, and canonical C3D capture were audited and preserved.

### OS-0 Key Audit Results

1. **Environment & Runtime Inventory**:

   - Python 3.13.5 (AMD64) on Windows-11.
   - Core numerical libraries verified: NumPy 2.4.1, SciPy 1.16.0, ezc3d 1.6.3, pytest 9.0.3, matplotlib 3.10.3.
   - **Mandatory Red Gate Verified**: `opensim` and `casadi` modules are confirmed absent from this environment.
   - Red qualification test suite committed: `tests/opensim/test_opensim_os0_qualification.py` fails predictably on missing `opensim` and missing `MocoStudy`/`MocoTrack`.

2. **Model Topology & Structure (`golf_humanoid.osim`)**:

   - Model SHA-256: `ee651d5452e39fef8618470adb430a53658a7a5a51d91af0333e5bf7cbd305be`.
   - 23 Bodies: pelvis, femur_r, tibia_r, patella_r, talus_r, calcn_r, toes_r, femur_l, tibia_l, patella_l, talus_l, calcn_l, toes_l, torso, head, humerus_r, ulna_r, radius_r, hand_r, humerus_l, ulna_l, radius_l, Club.
   - 39 Coordinates, 39 CoordinateActuators.
   - Club Attachment: `Club` is welded to `hand_r` via `WeldJoint`.
   - Closed kinematic chain note: The current OpenSim model has no closed kinematic loop connecting `hand_l` to `Club`; constructing a second-hand coupler/loop constraint or weld constraint is a prerequisite before physical two-hand dynamics matching can proceed.

3. **C3D Tour-Average Capture & Clock Audit (`data/C3D_TA_Driver.c3d`)**:

   - Capture SHA-256: `545405ccdbae87a297d16951487b501d5d76f5a2ab253cfc6d797744184943ba` (identical to MATLAB export copy).
   - Sample Rate: 360.0 Hz; 654 Frames (duration 1.813889 s).
   - 38 Tracked Markers: WaistLeft, WaistRight, WaistLBack, WaistRBack, BackTop, BackLeft, BackRight, LShoulderTop, LShoulderBack, LUArmHigh, LElbowOut, LWristTop, RShoulderBack, RUArmHigh, RElbowOut, RWristTop, Marker_2:2:1..3, Marker_3:3:1..3, HeadTop, HeadFront, HeadSide, etc.
   - Total observed marker points: 17,466 valid observations.
   - Analog & Force Plates: Zero force plate / GRF channels exist in this optical motion capture file.

4. **Immutable Receipt**:
   - Evidence preserved in `docs/development/opensim_tour_matching/evidence/os0_audit_receipt.json`.
   - Execution script: `docs/development/opensim_tour_matching/os0_runtime_audit.py`.

## Current Cross-Engine Evidence

Updated 2026-09-13. The authoritative native checkpoint is commit 955c36207 on
feat/9967-native-simscape-pinocchio; read its docs/development/HANDOFF.md before
acting (the run 16 figures previously quoted here are 65 runs stale).

Native Pinocchio returned81 (0–0.85 s, uninterrupted original-state replay):
whole 26.366 mm, early 11.427 mm, terminal 46.305 mm, clubhead 15.96 mm, pelvis
yaw 13.9 %; still rejected against the 25/35/12 mm gates. Derivatives are
qualified with measured floors (audit 77). MuJoCo and Drake replay the same
physical model to about 1e-6 to 1e-8 m through 0.8 s and act as verifiers.
Full 1.8138888889 s capture matching remains incomplete. Overnight Simscape
1.15 s/1.233 s extensions failed by more than an order of magnitude and are
archived on feat/9921-simscape-tour-matching. No OpenSim equivalence follows
from any of this.

OS-0 consumes the canonical native model specification and, as its reference
input, the returned81 candidate document
`docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81/returned-candidate.json`
(exact degree-six coefficients, absolute clock, original q0/qd0). No OpenSim
native installation or solver job has started. Execute OS-0 through OS-6
sequentially with failing-then-passing tests, hashes, independent replay and an
updated handoff at every checkpoint. Keep the Pinocchio environment isolated.

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
