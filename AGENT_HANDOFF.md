# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the canonical proximal-to-distal completion authority.

## Remote-Main Specification Repair (#8998)

- PR #8999 is protected-merged at remote-main commit
  `345f4c8cd58bbe368be9225527571b42754f983b`. It retained PR #8995's intended
  MediaPipe optimization while restoring the canonical 4,318-line `SPEC.md`.
- The repair commit is verified as an ancestor of remote `main`; the repeated
  misplaced changelog corruption is no longer present.

## Cross-Repository Authority

- Tools `main` is `81a4a617a64c8d35880416f3b769dad06525afbd`; #4635 is
  merged and supplies the
  provenance-aware ground workspace used by consumers. PR #4646 repairs the
  registered visual-evidence timeout without weakening its assertions and is
  awaiting a clean protected run after the uncertainty campaign releases CPU.
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
- UpstreamDrift is the scientific source authority. AffineDrift is a generated,
  revision-pinned publisher; Tools exposes typed consumers, not a second paper.

## Active Articulated Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- Parent PID `18404` is the intentional source-locked coordinator with 20
  workers. Do not kill workers individually, edit source-hashed files, or start
  a duplicate campaign.
- At 2026-08-22 19:24 PDT, 16 of 19 corners were fully accounted.
  `ground_free_moment_stiffness_scale-high` is the active seventeenth corner;
  its first ordered branch checkpoint had not yet been promoted. Two registered
  ground-only damping corners follow it. The 21-process group remains healthy;
  do not trade scientific identity for a faster restart.
- Completed rows and digest-bound branch checkpoints are restartable. Partial
  checkpoints are execution evidence, not release evidence.
- After completion, independently audit the record, then integrate
  `fix/8752-atomic-campaign-checkpoint` (`9f850a67f...`). Execute #8800 next,
  then regenerate claims, figures, the paper, and the AffineDrift projection.

## Pinned Tools Docker Boundary and Security Follow-Up (#8996)

- PR #8993 is protected-merged at remote-main commit
  `4e672c5b051a8859743f622872f63b86a0cfbb9d`. It binds
  modular images to the exact Tools gitlink and content digest, fixes isolated
  PEP 517 hook loading, and advances pip to 26.2.1.
- The workflows attest only `src/shared`, `src/sidekick`, `src/chat`,
  `src/python/src/utils`, and `src/contracts.py`; the registered digest is
  `30dc761a34ec30eb3bf41d11d2dca1aff90448e71defbe82c32fcd657525fcc3`.
- Protected head `37774ebdd` exposed a collection-order defect between two
  Hatchling test stubs. The local correction gives both stubs the constructor
  contract and isolates UI-build tests from the independently covered Tools
  registration boundary; the combined 70-test ordering regression passes.
- Head `17b2bca63` then correctly failed the suite-marker ratchet on its new
  regression. The test is now explicitly unit-marked; the full ratchet passes
  with no drift.
- That head's canonical-conformance job reached a runner whose NumPy install
  has no package metadata/RECORD and cannot be repaired by the job. Re-evaluate
  once on the consolidated correction head; do not change shared runners.
- Trivy correctly rejected that image for `msgpack` 1.1.2
  (GHSA-6v7p-g79w-8964) and `setuptools` 70.3.0 (CVE-2025-47273). The pending
  correction reasserts fixed `msgpack==1.2.1` and `setuptools==83.0.0` after
  every supported image's final dependency layer and adds a unit-marked
  repository contract. The higher setuptools floor also resolves
  PYSEC-2026-3447, which the first protected follow-up build surfaced after
  the older finding was cleared. PR #9001 merged the correction that removes
  pip and its embedded third-party SBOM from final runtime/test images after
  dependency installation and restores the audited builder venv only in the
  explicit training stage. No scanner waiver or skip-file rule is used. Its
  protected slim-profile build exposed a
  separate builder-context omission: `pyproject.toml` force-includes
  `launch_upstream_drift.py`, but the isolated Docker context copied only the
  legacy launcher. Follow-up PR #9003 copies both launcher modules before
  feature installation and guards their order. Its focused
  Docker/packaging/spec set and mandatory local push gates pass. The protected
  slim image now builds, proving the missing launcher is resolved, and measures
  848 MB. Its size step exposed stale workflow budgets (800/2000 MB) that
  contradicted the canonical `docker/profiles.yaml` values (900/2200 MB). The
  current correction synchronizes those values and adds a regression contract;
  protected image and scanner validation must run again on the new head.
- Docker is unavailable locally, so protected image builds and scans remain
  authoritative. Human review is required; do not create redundant runs.

## Executable Quarantine Ledger (#8766)

- PR #8997 is protected-merged at remote-main commit
  `fe411260cab577dd197b54ad3eab4d9d9ffa4213`.
- The 520-node ledger has an executable 10-cluster ownership map. The checker
  rejects duplicate, unassigned, ambiguous, replacement, or new node IDs and
  CI compares PR state with the fetched base branch.
- CI Standard validates exact coverage and compares PR node IDs with the fetched
  base branch. The existing skip behavior remains unchanged and scoped to
  `UNIT_GATE_QUARANTINE=1`.
- Ten contract tests, Ruff, YAML parsing, the checker, file-size validation, and
  protected CI passed. This tranche organizes debt; it does not claim any
  quarantined test fixed.

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
python3 scripts/ci/check_unit_gate_quarantine.py --baseline-ref origin/main
python3 -m pytest tests/ci/test_unit_gate_quarantine_contract.py -q
python3 -m ruff check scripts/ci/check_unit_gate_quarantine.py tests/ci/test_unit_gate_quarantine_contract.py
python3 -m ruff format --check scripts/ci/check_unit_gate_quarantine.py tests/ci/test_unit_gate_quarantine_contract.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/ci/check_file_size_budget.py
```

Passing common gates does not close a child issue whose acceptance evidence
remains incomplete.
