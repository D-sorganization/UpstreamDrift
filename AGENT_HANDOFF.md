# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

This is current operational state, not history. Epic #8557 is the canonical
proximal--distal completion authority.

## Verified Remote State

- UpstreamDrift `main` is
  `c7904b530fe8921ecdb17362f2100e5c85400af4`, the protected merge of PR
  #8990. The lifecycle head passed 35 checks; the optional-stack job was
  cancelled after merge rather than failing.
- Tools `main` is `9d1efb8b4162503badd63dcd95b5e1f06b09c404`, the protected
  merge of PR #4635, with 25 successful and one skipped check.
- Last verified AffineDrift `main` was
  `60b95283a43c9ebc14462327d988ca5b0bd3c6a6`. Its immutable projection still
  pins an earlier paper and must be refreshed after the scientific campaign.

## Publication Authority

- PR #8987 reconciled the current 235-page computational PDF, SHA-256
  `ce51e6fe4f3d9033bf730c0fe2538c72bf88b1b9707f77a7b6385923a1b5fdcf`.
- The PDF is 1,863,127 bytes, has 194 valid URI links and 246 outline entries,
  renders all 235 pages, and is fast-web linearized. Every page was visually
  inspected; pages 152--160 received full-resolution review.
- Computational qualification passes. Archival qualification remains
  fail-closed because the PDF is untagged and contains 112 Type 3 resources
  plus two unembedded resources.

## Active Source Freeze: Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- PID `18404` is the intentional checkpointed 19-corner coordinator and may own
  up to 20 workers. Do not kill workers, edit source, or start another campaign.
- At 2026-08-22 07:48 PDT, 13 corners were complete; corner 14,
  `ground_translation_damping_scale-low`, had 47 of 72 ground checkpoints.
  The progress tree held 942 checkpoint files. Completed pathways and
  digest-bound ground branches are restartable.
- Retained numerical failures are evidence. Partial checkpoints are execution
  evidence, not release evidence.
- After completion, independently audit the record and integrate
  `fix/8752-atomic-campaign-checkpoint` commit `9f850a67f...`. Then implement
  #8800 height, mass, and joint-limit propagation before closing #8752/#8668.

## Docker Build Boundary (#8996, PR #8993)

- Worktree: `UpstreamDrift-worktrees/8789-docker-tools-boundary`; branch
  `fix/8789-docker-tools-boundary`; full PR #8993 has human review requested.
- Owner-closed phase issue #8789 remains closed. Its unfinished Docker criterion
  is extracted to #8996; PR #8993 closes #8996 only after protected merge. The
  520-node removal-only quarantine burn-down remains owned by #8766.
- The modular workflows fetch exact Tools gitlink
  `aec16af5a1e69c0d5542da5e04a1db1023cceff2` and attest only
  `src/shared`, `src/sidekick`, `src/chat`, `src/python/src/utils`, and
  `src/contracts.py`. Current digest:
  `30dc761a34ec30eb3bf41d11d2dca1aff90448e71defbe82c32fcd657525fcc3`.
- `Dockerfile.modular` copies exactly those roots. The Hatch hook verifies the
  digest when Git metadata is absent and fails closed on incomplete, malformed,
  symlinked, or mismatched provenance. Both Dockerfiles pin pip 26.2.1.
- Protected head `f5858ce2e` exposed one shared installation failure across
  fourteen lanes: PEP 517 does not guarantee the source root on `sys.path`.
  Commit `6d72fec71` loads the adjacent canonical helper by file path. The new
  isolated-import regression and 78 focused Docker/packaging tests pass.
- A real isolated source build passed the formerly failing hook stage; optional
  archive compression was stopped to return CPU to the uncertainty campaign.
  Docker is unavailable locally, so protected CI owns image and Trivy evidence.
- Current `main` was merged normally into the branch after #8990. Resolve only
  this handoff conflict, validate, commit the merge, and push once. Do not rerun
  the obsolete failing head.
- The full document-size gate retains an unrelated baseline failure:
  `_ch06c_spatial_cross_formulation.qmd` is 51,523 bytes against 51,200. Do not
  edit that campaign-adjacent source during the freeze.

## Full-Suite QApplication Repair (#8990)

- `run_launcher` reuses an existing Qt application and calls `exec()` only when
  it owns the newly constructed application.
- Full-suite module replacement had invalidated string-based mocks. The merged
  tests patch the retained function globals and preserve the existing
  `keep_terminal_open=True` child-launch contract.
- PR #8990 merged as `c7904b530fe8921ecdb17362f2100e5c85400af4`; do not reopen or
  duplicate its CI.

## Scientific and Repository Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, calibration, injury, coaching
  strategy, or a universal speed benefit.
- #8556 remains externally data-gated because no governed participant dataset
  with synchronized bilateral six-axis grip wrenches is available. Never
  substitute synthetic traces for human validation.
- Other canonical gates include #8724, #8443, #8448, #8449, #8450, #8595,
  #8668, #8684, and #8796. Verify acceptance evidence before changing state.
- Use TDD, DbC, DRY, and LoD. Never force-push, admin-merge, bypass hooks or
  checks, add quarantine debt, hand-edit governed artifacts, or edit the pinned
  Tools submodule.
- PRs target `main`; use full PRs and require human review. Verify exact head,
  checks, review, merge SHA, remote-main ancestry, and clean trees.

## Current Docker Validation

```bash
python3 -m pytest -q -n 0 \
  tests/unit/test_pinned_tools_provenance.py tests/unit/test_build_hooks.py \
  tests/docker/test_pinned_tools_build_boundary.py \
  tests/docker/test_docker_hardening_7159_7161.py \
  tests/docker/test_docker_integration.py \
  tests/unit/scripts/test_dockerfile_contracts.py
python3 scripts/ci/check_dockerfile_contracts.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 -m ruff check build_hooks.py \
  scripts/packaging/pinned_tools_provenance.py tests/unit/test_build_hooks.py
python3 -m ruff format --check build_hooks.py \
  scripts/packaging/pinned_tools_provenance.py tests/unit/test_build_hooks.py
git diff --check
```

Passing common gates does not close a child issue whose acceptance remains
unmet.
