# Agent Handoff: Proximal–Distal Research Program

Updated: 2026-08-27

## Authority and Working State

- UpstreamDrift owns scientific sources, models, evidence registers, and release
  artifacts. AffineDrift publishes an immutable source-pinned projection. Tools
  owns reusable mechanics and UI-neutral provider contracts.
- Protected UpstreamDrift #9117 merged as
  `80f10e8d22289d89c764fae8bcb7069cf8352c02`; it is the phase/event baseline.
- Active worktree: `UpstreamDrift-worktrees/9123-trajectory-control-authority`.
- Active branch: `research/9123-trajectory-control-authority` for issue #9123.
- This branch is reconciling the completed trajectory-control implementation
  with protected #9117. Do not restore older generated claim or PDF artifacts.
- Tools provider `3dfbd32cc778536269670c055955073853c0f60a` remains the immutable authority
  consumed through `vendor/ud-tools`.

## Protected Baseline (#9117)

- Exact discrete RK4 variational propagation, transverse event-time
  sensitivity, direct perturbation checks, saltation controls, near-grazing
  rejection, equivalent-unit checks, and fail-closed nonperiodicity/Floquet
  eligibility are merged.
- The merged paper is 246 pages and 1,923,372 bytes at SHA-256
  `0e590a556660fcc5656ad4add657bcfb4e90a8c5b15abc98ccf031118d61f050`.
- It contains 1,148 candidate statements, 315 adjudicated claims, 134 numeric
  contracts, 445 verified numeric literals, 2,332 evidence references, and 644
  release artifacts.
- These are local synthetic, finite-window diagnostics. They do not establish
  asymptotic stability, neural timing demand, participant robustness, passive
  negative torque, human strategy, or coaching guidance.

## Active Trajectory-Control Slice (#9123)

- Implemented files include `trajectory_control_authority.py`, its governed
  runner, JSON/NPZ evidence, publication figure, claim registration, and tests.
- Required estimands are exact-step state/input Jacobians, trajectory-varying
  continuous-energy-equivalent Gramians, channel masks, and event-tangent
  projection at the registered transverse delivery event.
- Required falsifiers are zero input, shoulder-plus-wrist additivity, direct
  nonlinear pulse agreement, input-step and integration-step refinement,
  equivalent units, crossing/grazing failure, and a frozen-local countermodel.
- Claims PD-CLAIM-317 and PD-CLAIM-318 are supported only for the declared local
  first-order analytical trajectory. No bounded nonlinear reachability,
  physiological capacity, controller ranking, passive torque, robustness, or
  coaching inference is authorized.
- Reconciliation rule: preserve protected #9117 phase/event sources and
  regenerate every candidate inventory, review registry, numeric contract,
  release manifest, checksum, TeX, and PDF after adding #9123.

## Validation and Publication

Use Python 3.12 with
`PYTHONPATH=C:/Users/diete/AppData/Local/Temp/codex-precommit-wmi;<worktree>/src`.
Run pytest serially (`-n 0`) and web tests with at most two workers.

```powershell
python -m scripts.research.proximal_distal_energy.run_trajectory_control_authority validate
python -m scripts.research.proximal_distal_energy.make_trajectory_control_authority_figure
python -m pytest -n 0 -q tests/research/test_trajectory_control_authority.py tests/research/test_trajectory_control_authority_evidence.py
python -m pytest -n 0 -q tests/research/test_phase_event_stability.py tests/research/test_phase_event_stability_evidence.py
python -m pytest -n 0 -q tests/research/test_proximal_distal_release_bundle.py tests/unit/research/test_proximal_distal_claim_audit.py tests/research/test_publication_quality.py
```

- Validate deterministic scientific outputs under Python 3.11 and 3.12.
- Run Ruff, MyPy, document governance, title capitalization, claim/numeric
  registries, release qualification, PDF inspection, and affected repo gates.
- Render the canonical QMD, web-linearize the PDF, update exact publication
  metrics, and visually inspect the new figure and nearby pages.
- Use the GitHub App setup script immediately before every GitHub operation.
  Do not bypass reviews or protected checks. Preserve squash auto-merge.

## Remaining Program Order

1. Finish #9123 regeneration, validation, protected PR, and remote-main proof.
2. Start #9124 only from protected #9123: bounded nonlinear event-reaching
   feasibility under explicit scenario torque and torque-rate bounds.
3. Keep those bounds computational and model-specific unless independently
   sourced; do not infer physiology, controller superiority, or coaching.
4. Publish the next AffineDrift projection only after the UpstreamDrift source
   commit is protected and immutable.
5. Continue the broader epic #8557 and unresolved human-data boundary #8556.

## Frozen Campaign Boundary (#8800)

- ControlTower contains 93 atomic checkpoints: shaft 48/48 and ground 45/48.
- Preserve frozen source SHA `1bd4d57da7bd257b76b42b3cc19524b283b5f748`, plan
  contract `c5cfba35ecafa96054ef8cc872f2e91a9f7855db0b93cfa491f9b18ee3db80f4`,
  image identity, and the two-worker/two-CPU contract.
- `F:/WSL/ControlTower-SSD/ext4.vhdx` is unreadable (`0x80070570`). Do not retry
  WSL startup, run CHKDSK, repair/mount/copy/mutate the VHDX, restart services,
  recreate the distro, or launch a replacement campaign without explicit user
  approval and a written recoverability plan.
- DeskComputer runner drain remains active; do not re-enable capacity here.
