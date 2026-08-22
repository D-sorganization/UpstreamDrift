# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Remote-Main Specification Repair (#8998)

- Remote `main` is `93cc564b3e6bfd30e498fe2066d8c9f280c9f056` from PR
  #8995. Its intended MediaPipe mean optimization and tests remain valid.
- That merge corrupted `SPEC.md`: 5,457 additions inserted the same changelog
  row throughout unrelated prose and tables, expanding the file from 4,317 to
  9,749 lines.
- Worktree: `UpstreamDrift-worktrees/8998-spec-corruption`; branch:
  `fix/8998-spec-corruption`.
- The repair restores only `SPEC.md` from #8995's exact first parent and adds
  one truthful 1.0.574 changelog row. Production and test changes are untouched.
- Next: validate, commit, push a full issue-closing PR, obtain human review,
  shepherd protected CI, and verify the merge on remote `main`.

## Cross-Repository Authority

- Tools `main` is `9d1efb8b4162503badd63dcd95b5e1f06b09c404`; #4635 is
  merged and supplies the provenance-aware ground workspace used by consumers.
- AffineDrift `main` is `60b95283a43c9ebc14462327d988ca5b0bd3c6a6`.
  Its immutable publication projection still pins an earlier UpstreamDrift
  release and must be refreshed only after the scientific campaign is merged.
- UpstreamDrift is the scientific source authority. AffineDrift is a generated,
  revision-pinned publisher; Tools exposes typed consumers, not a second paper.

## Publication Authority

- The current computational candidate is the 235-page proximal-to-distal PDF.
- PDF SHA-256:
  `ce51e6fe4f3d9033bf730c0fe2538c72bf88b1b9707f77a7b6385923a1b5fdcf`.
- It has 194 valid URI links and 246 outline entries; all pages render and were
  inspected. Archival qualification remains fail-closed because the PDF is
  untagged and retains Type 3 and unembedded font resources.

## Active Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` is the intentional source-locked coordinator with 20
  workers. Do not kill workers individually, edit source-hashed files, or start
  a duplicate campaign.
- At 2026-08-22 08:30 PDT, 13 of 19 corners were terminal. Corner 14,
  `ground_translation_damping_scale-low`, retained 63 of 72 atomic ground
  branch checkpoints. Five further ground-only corners remain after it.
- Completed rows and digest-bound branch checkpoints are restartable. Partial
  checkpoints are execution evidence, not release evidence.
- After completion, independently audit the record, then integrate
  `fix/8752-atomic-campaign-checkpoint` (`9f850a67f...`). Execute #8800 next,
  then regenerate claims, figures, the paper, and the AffineDrift projection.

## Pinned Tools Docker Boundary (#8996)

- PR #8993 is open from `fix/8789-docker-tools-boundary`; exact head
  `81867ef2ea13191c3fba0d86be1c6682375fcbf3`.
- It binds modular images to the exact Tools gitlink and content digest, fixes
  isolated PEP 517 hook loading, and advances pip to 26.2.1.
- Docker is unavailable locally, so protected image builds and scans remain
  authoritative. Human review is required; do not create redundant runs.

## Executable Quarantine Ledger (#8766)

- PR #8997 is open from `fix/8766-executable-cluster-ledger`; head before base
  repair is `c2d33b13d718fe94216313c58d515b201baea5b6`.
- The 520-node ledger has an executable 10-cluster ownership map. The checker
  rejects duplicate, unassigned, ambiguous, replacement, or new node IDs and
  CI compares PR state with the fetched base branch.
- #8997 intentionally remains open and conflicted until #8998 repairs `main`.
  Then merge current main normally, resolve handoff/SPEC metadata, and rerun
  protected checks. This tranche organizes debt; it fixes no quarantined test.

## Scientific Boundaries

- #8556 remains externally data-gated: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Never substitute
  synthetic traces for human validation.
- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- #8724, #8443, #8448, #8449, #8450, #8595, #8668, #8684, and #8796 remain
  open. Verify exact acceptance evidence before changing issue state.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Human review is required.
- Never force-push, admin-merge, bypass hooks/checks, add quarantine debt, or
  edit `vendor/ud-tools`.
- Use TDD, DbC, DRY, and LoD. Keep scientific evidence hash-pinned and use
  governed generators rather than hand-editing generated artifacts.
- Use title case for document headings and captions.
- Verify exact PR head, review, checks, merge SHA, remote-main ancestry, and a
  clean worktree before reporting protected completion.

## Focused Validation

```powershell
python3 scripts/check_spec_paths.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 -m pytest -q tests/unit/test_spec_freshness.py
python3 -m pytest -q tests/unit/pose_estimation/test_mediapipe_estimator.py
python3 scripts/ci/check_file_size_budget.py
git diff --check
```

Passing common gates does not close a child issue whose acceptance evidence
remains incomplete.
