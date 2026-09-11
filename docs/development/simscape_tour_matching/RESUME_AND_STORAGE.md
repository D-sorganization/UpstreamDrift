# Resume and Storage Review

## Completed Worktree Cleanup and Resumption Plan

The user authorized removal of the 20 reviewed completed worktrees. All 20 directories are now removed; every original branch still resolves to its recorded commit. Their measured logical file size was 13.35 GiB. Final C: free space was about 30.4 GiB, compared with 18.9 GiB at cleanup start; concurrent storage changes mean that difference is not an exact attribution to this cleanup.

Normal removal refused initialized submodules. Before explicit Git removal, parent/submodule state and ignored files were checked and submodule Git histories were archived with CRC/SHA verification. These recovery archives occupy about 7.09 GiB under `C:/Users/diete/Repositories/simscape-tour-checkpoints/completed-worktree-submodule-backups`. Branches, the main clone, active matching worktree, remote runtimes, native data and WSL remain preserved.

Removal exposed borrowed Git-object dependencies between submodules. Verified archives restored the objects; surviving alternate references now point to `C:/Users/diete/Repositories/UpstreamDrift/.git/retained-submodule-objects`, outside removable worktree metadata. Keep this storage: it is an active dependency, not a cache. The object map and alternate-reference repair receipt preserve exact old/new paths. Final verification found zero missing alternate targets, 32 readable surviving worktrees, and one pre-existing unreadable `_wt_claude_model` submodule path retained unchanged. The affected surviving coaching submodule passes Git connectivity checking. No matching data was lost.

[Cleanup Verification](worktree-cleanup-verification-20260910.json) records all removed paths and preserved branch checks; the accompanying cleanup and alternate-reference receipts retain intermediate refusals and recovery actions. Recreate a removed tree with `git worktree add <new-path> <preserved-branch>` and initialize its submodules. Raw submodule Git archives provide additional recovery history; their historical worktree paths must be adjusted when restoring elsewhere.

[Matching Resumption Plan](RESUMPTION_PLAN.md) defines the next bounded assignment: extend original-state 0.6 s to 0.7 s on DeskComputer R2025b, using the existing cubic runner, identity-checked transfer, cold replay and all-27 audit. A lower-cost coding agent can perform this prescribed execution and evidence collection. Geometry, attachment choices, convergence diagnosis and final scientific acceptance still need stronger review. No new fit or agent was launched in this cleanup/plan turn.

## Saved Matching State

Epic #9921 and draft PR #9948 remain unfinished. MATLAB R2025b is required. All source changes preceding this checkpoint are published on `feat/9921-simscape-tour-matching`; use the commit containing this document for the recovery snapshot. The main clone and other tasks have unrelated dirty work; this checkpoint does not claim they are committed.

The two latest fit/cold-audit pipelines completed on DeskComputer with explicit exit codes zero. No MATLAB fitting process was observed during this review. Preserve remote runtime checkouts and experiment folders. Historical running/pending text below the newest AGENT_HANDOFF section is superseded.

| Experiment                    | Interval | Marker RMS   | p95          | Maximum      | Cold Replay                              |
| ----------------------------- | -------- | ------------ | ------------ | ------------ | ---------------------------------------- |
| Original State, Refined Cubic | 0�0.6 s  | 9.062024 mm  | 19.457409 mm | 42.595899 mm | Exact Marker Parity; 27 Efforts Verified |
| Calibrated State, Constant    | 0�0.1 s  | 12.191880 mm | 21.632996 mm | 25.221325 mm | Exact Marker Parity; 27 Efforts Verified |

Both optimizers exhausted their budgets without convergence. These are partial forward-dynamics fits, not accepted full-swing or physiological models. The capture lasts 1.813889 s. Calibrated attachments improved sampled kinematic matching but have a nonzero initial target residual; keep their forward sequence separate.

## Find the Best Current Match

Longest verified forward candidate:

- Local extracted run: `C:/Users/diete/Repositories/simscape-tour-checkpoints/prefix-600ms-cubic-refine-02`.
- DeskComputer original run: `C:/Users/diete/SimscapeTour9921/prefix-600ms-cubic-refine-02`.
- Full controls, initial-state identity and predicted marker arrays: `first_prefix_fit.json` in that run. Normalized Bernstein controls are in `stage.parameters`, with scales, duration and coordinate order in the report. The existing replay routine converts these into native A�G polynomial coefficients.
- Native simulation logs: `final_native_replay.mat` and `qualified_candidate_replay.mat`.
- Comparison chart: [Refined 0.6 s Fit](native_evidence/prefix_600ms_refined_fit.png).
- Readable metrics: [Refined Summary](native_evidence/prefix_600ms_refined_summary.json).
- Provenance: [Refined Archive Receipt](native_evidence/prefix_600ms_refined_bundle_receipt.json).

The chart shows marker residuals, optimization evaluations and torque ranges. Its zero-effort comparison uses a much larger scale than the fitted residual. No full-swing matching animation is claimed.

Complete ZIP archives exist on both machines. SHA256 and every archived file hash are in the receipts. Archive hashes were compared across machines and every ZIP CRC checked. Do not replace these immutable bundles; save subsequent runs under new names.

## Resume on DeskComputer

Use SSH over Tailscale with `ssh -o BatchMode=yes -o StrictHostKeyChecking=yes deskcomputer`. Explicit MATLAB: `C:/Program Files/MATLAB/R2025b/bin/matlab.exe`. Python engine environment: `C:/Users/diete/SimscapeTour9921/python-r2025b/Scripts/python.exe`.

Original-state runtime: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour-runtime`, detached at `0715f95f3`. Calibrated runtime: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-calibrated-runtime`, detached at `f28343726`. Preserve both and the isolated actuator-audit checkout with its archived native patches/model backups.

Each archived run contains `launch.ps1`, `launch-arguments.json`, `cold-replay-launch.ps1`, the exact runner, capture payload, checkpoints and native exit receipts. Use these as the invocation authority. Do not rerun against an existing output folder. For the next original-state experiment, use a new folder, transfer from the refined report, retain its qualified seed, and extend duration modestly (suggested 0.7 s). Every trial starts at the same global t0. Do not change geometry or marker offsets during a fixed-identity torque continuation.

The existing plot can be regenerated locally from the extracted original-state run:

```powershell
python3 docs/development/simscape_tour_matching/native_evidence/reproduction/plot_prefix_100ms.py C:/Users/diete/Repositories/simscape-tour-checkpoints/prefix-600ms-cubic-refine-02 --repo C:/Users/diete/Repositories/Worktrees/UpstreamDrift-simscape-tour
```

Require independent cold replay, q/qd initialization checks and all-27 actuator auditing after the next fit. Record actual nonblank native exit codes, not just the SSH wrapper status. Archive before extending. Goal-service status was observed as blocked from an earlier run; the goal tools cannot set active or edit its objective. The user request remains to finish matching after storage recovery.

## Local Storage Measurements

Measurements on 2026-09-10 are a snapshot; unrelated processes changed Windows free space during inspection (about 7.4�9.2 GiB). No files were deleted in this review.

- Ubuntu VHD: `C:/Users/diete/AppData/Local/wsl/{f6f6af68-637b-4599-bc69-d9310fcc47d3}/ext4.vhdx`.
- VHD file length: 358,843,678,720 bytes = 334.2 GiB = 358.8 GB. No prior measurement establishes its growth delta.
- Normal Ubuntu commands fail with getpwuid/read I/O errors. The WSL system environment successfully reports `/dev/sdd`: 1007 GiB capacity, 293 GiB used, 663 GiB available. These are rounded guest-filesystem figures, not Windows free space.
- Kernel logs contain repeated read I/O errors on that device. The cause is not established. Avoid assuming compaction fixes the filesystem issue.
- 53 registered UpstreamDrift worktrees including the main clone; 52 linked trees total approximately 35.94 GiB of logical file lengths. The main-clone count includes its Git storage and nested worktree, so do not add it as an independent reclaim estimate.
- 20 candidates have clean tracked/untracked status and HEAD exactly equal to a merged PR head; they total approximately 13.35 GiB. [Candidate Inventory](worktree-cleanup-candidates-20260910.json) records exact paths, commits, PRs and ignored-file categories. Ignored contents are predominantly caches, dependencies and bytecode; the context-implementation tree also has generated UI distribution files. Active-task checks and a fresh status check are still required immediately before removal.
- [Full Worktree Inventory](worktree-storage-inventory-20260910.json) also retains dirty/unmerged trees. A failed Git status is treated as not clean.

## Recovery Order

1. Remove only approved completed worktrees after checking no task/process still uses them, verifying unchanged HEAD/clean status and checking ignored artifacts. Use `git worktree remove` on verified absolute paths within `C:/Users/diete/Repositories`; preserve branches for recreation. Git refuses normal removal of initialized submodules. For this cleanup, explicit `git worktree remove --force` is permitted only after clean parent/submodule and artifact checks, exact merged-head verification, and CRC/SHA-verified backups of submodule Git histories. Do not use shell recursive deletion or delete the main clone/shared Git directory. Dirty or uncertain trees stay intact.
2. Recheck Windows free space. The matching evidence and remote R2025b execution do not depend on Ubuntu, so model work can resume through DeskComputer after the local checkpoint is safe.
3. Diagnose Ubuntu's actual I/O problem before further writes. Save critical Linux work to another device/host if accessible. A controlled shutdown/restart interrupts WSL workloads and requires coordination. If filesystem repair is necessary, follow Microsoft's offline VHD procedure after preserving a backup; do not guess the block-device name.
4. After the filesystem is healthy, inspect Linux caches, build outputs and obsolete worktrees, remove approved content, then trim free blocks and compact the detached VHD. The approximately 41 GiB gap between VHD length and rounded guest usage is not a guaranteed reclaim amount. Compaction cannot recover the 293 GiB still occupied by live files.
5. A longer-term option is to move/export the distribution to another disk with adequate capacity. Only C: was available in this review. Do not unregister Ubuntu as a cleanup shortcut.

Microsoft references: [WSL Disk Space and VHD Repair](https://learn.microsoft.com/en-us/windows/wsl/disk-space), [DiskPart Compact Vdisk](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk). Compaction requires a dynamically expanding VHD that is detached or attached read-only; it is distinct from expanding its maximum capacity. No shutdown, repair, compaction or deletion was performed.
