# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Verified Remote State

- UpstreamDrift `main` is
  `dcd5db55fce2565073bb31e5e7a2fd9cdf90d143` from PR #8987.
- Tools `main` is `8d251ba0ec6e7978ce366abd0be7baa0554ffd60`.
  Tools #4430 is complete and immutably consumed; Tools #4142 remains open.
- AffineDrift `main` is `60b95283a43c9ebc14462327d988ca5b0bd3c6a6`.
  Its immutable publication projection still pins the earlier 231-page
  UpstreamDrift release and must be refreshed after the scientific campaign.

## Publication Authority

- PR #8987 and issue #8977 reconciled the exact 235-page computational PDF.
- PDF SHA-256:
  `ce51e6fe4f3d9033bf730c0fe2538c72bf88b1b9707f77a7b6385923a1b5fdcf`.
- The PDF is 1,863,127 bytes, has 194 valid URI links and 246 outline entries,
  renders all 235 pages, and is fast-web linearized.
- Computational qualification passes. Archival qualification remains
  fail-closed because the PDF is untagged and contains 112 Type 3 resources
  plus two unembedded resources.
- The complete ordered render set was visually inspected; pages 152–160 were
  additionally inspected at full resolution.

## Active Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` is the intentional source-locked coordinator. It may own
  20 workers; do not kill workers individually or start a second campaign.
- At 2026-08-22 03:35 PDT, 23 of 30 computed pathways were terminal. The
  `ground_translation_stiffness_scale-high` pathway was active with 25 of 72
  atomic branch checkpoints retained. Seven ground pathways remain.
- Completed rows and digest-bound branch checkpoints are restartable. Partial
  checkpoints are execution evidence, not release evidence.
- Do not edit any campaign source or source-hashed test while PID `18404` is
  active. Retained numerical failures are data and must not be discarded.
- After completion, independently audit the final record, then integrate
  `fix/8752-atomic-campaign-checkpoint` (`9f850a67f...`) so future top-level
  checkpoint writes are atomic.
- #8800 follows the campaign because height, body-mass, and joint-limit
  propagation changes the same authority. Then complete #8752 and #8668.

## Full-Suite Truth Gate (#8789)

- Current worktree: `UpstreamDrift-worktrees/8977-publication-record`; branch
  `fix/8789-qapplication-lifecycle`, merged through remote `main` `46469edeb`.
- Both Python 3.11 and 3.12 post-merge lanes reached
  `tests/launchers/test_base.py` and exited 139 at `test_run_launcher`.
- Root cause: `run_launcher` unconditionally constructed a second
  `QApplication` while pytest already owned one. Qt permits only one
  application object per process.
- Reusing an application now also preserves event-loop ownership: only a newly
  constructed application receives `exec()`. A typed resolution seam lets tests
  avoid replacing the SIP-backed Qt class, which crashed Linux xdist teardown.
- Exact head `6c702a5f2` passed both protected Python lanes but its unit gate
  reported 11,727 passed and four failures: both mock-based lifecycle tests
  crashed workers, while two containment tests leaked the global embed registry.
- Local repair evidence: 26 launcher-base tests pass serially and under xdist;
  the exact four-worker CI-marked three-file slice passes 6/2 quarantined. Ruff,
  formatting, suite-marker ratchet, and registry-isolation regressions pass.
- PR #8990 remains a full PR with human review requested. Push the repair after
  hooks, then require exact-head protected checks, review, merge, and a verified
  post-merge `main` run.
- Do not close #8789 with this PR. Docker profile boundaries and the
  removal-only quarantine burn-down remain separate unchecked exit criteria.

## Other Open Scientific Gates

- #8724: normalized independent claim-adjudication contract.
- #8556: governed participant validation remains externally blocked because no
  qualifying synchronized bilateral six-axis grip-wrench dataset is available.
  Never substitute synthetic traces for human validation.
- #8789: full-suite, Docker, and quarantine truth gates.
- #8443, #8448, #8449, #8450, #8595, #8668, #8684, and #8796 remain open in
  the canonical dependency ledger. Verify their actual acceptance evidence
  before changing issue state.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Human review is required.
- Never force-push, admin-merge, bypass hooks/checks, add quarantine debt, or
  edit `vendor/ud-tools`.
- Use TDD, DbC, DRY, and LoD. Keep scientific evidence hash-pinned and use
  governed generators rather than hand-editing generated artifacts.
- Use title case for document headings and captions.
- Verify exact PR head, review decision, checks, merge SHA, remote-main
  ancestry, and a clean worktree before reporting protected completion.

## Focused Validation

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q -n 0 tests/launchers/test_base.py
python -m ruff check src/launchers/base.py tests/launchers/test_base.py
python -m ruff format --check src/launchers/base.py tests/launchers/test_base.py
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/ci/check_file_size_budget.py
```

Passing common gates does not close a child issue whose scientific acceptance
criteria remain unmet.
