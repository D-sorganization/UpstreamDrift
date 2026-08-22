# Agent Handoff — UpstreamDrift

Last updated: 2026-08-22

This is current operational state, not a changelog. Git and GitHub retain
history. Epic #8557 is the canonical proximal--distal completion authority.

## Active Source Freeze: Uncertainty Campaign (#8752)

- Worktree: `UpstreamDrift-worktrees/goal-8752-uncertainty`.
- PID `18404` is the live coordinator for the checkpointed 19-corner campaign;
  it may own up to 20 workers. It is not orphaned.
- At 2026-08-22 06:05 PDT, 13 corners were complete and corner 14,
  `ground_translation_damping_scale-low`, was active. The progress record held
  14 corner rows and 896 ground-branch checkpoint files.
- Do not kill individual workers, edit this worktree's source, or start another
  campaign. Completed pathways and digest-bound ground branch checkpoints are
  restartable. Retained numerical failures are evidence, not cleanup targets.
- `fix/8752-atomic-campaign-checkpoint` commit
  `9f850a67f03d6010f463760a73c5e04b2f9133cc` makes the top-level progress write
  atomic. Integrate it only after the running source-pinned campaign finishes.
- After completion: audit the record, integrate atomic writing, implement the
  remaining registered propagation in #8800, regenerate governed evidence and
  publication artifacts, and complete #8752 acceptance without human-data
  promotion.

## Active Docker Boundary Work (#8789)

- Worktree: `UpstreamDrift-worktrees/8789-docker-tools-boundary`; branch
  `fix/8789-docker-tools-boundary`; base
  `46469edebf1d1dea20b1e2090faa0d2c297bb4cf`.
- The existing same-session `claim:codex` lease is valid for session
  `019fe886-6614-70a2-a596-e5b0dea725d0`.
- The modular Docker workflows previously neither fetched nor copied pinned
  Tools. Adding a submodule alone is insufficient because `.git` is excluded
  from the isolated build context.
- `scripts/packaging/pinned_tools_provenance.py` hashes the required Tools roots
  deterministically and rejects missing roots and symlinks. The reusable fetch
  action optionally emits gitlink and source digests. Both modular workflows
  pass them as Docker arguments; the Dockerfile copies only those roots; the
  build hook validates content when Git metadata is unavailable.
- Required roots are `src/shared`, `src/sidekick`, `src/chat`,
  `src/python/src/utils`, and `src/contracts.py`. The pinned gitlink is
  `aec16af5a1e69c0d5542da5e04a1db1023cceff2`; the current source digest is
  `30dc761a34ec30eb3bf41d11d2dca1aff90448e71defbe82c32fcd657525fcc3`.
- `.dockerignore` retains exactly those roots, and `pyproject.toml` includes the
  provenance helper in source builds. Both Dockerfiles now pin pip 26.2.1 to
  clear PYSEC-2026-3721.
- Seventy-eight focused provenance, build-hook, Docker-boundary, hardening,
  integration, and contract tests pass. Ruff and changed-file architecture,
  title, and diff gates pass. Docker is not installed locally, so actual image
  construction and Trivy evidence must come from protected CI.
- The full document-size gate retains an unrelated baseline failure:
  `_ch06c_spatial_cross_formulation.qmd` is 51,523 bytes against 51,200. This
  branch does not touch that source; do not edit it during the campaign freeze.
- Commit `956678ea28ccf36c405a4e236d607a0293f98e46` passed commit and push
  hooks and is published in full PR #8993; human review is requested. Shepherd
  its exact head through protected Docker builds and scans. Do not close #8789:
  its broader quarantine burn-down remains open.
- Protected head `f5858ce2e` exposed one common install failure across fourteen
  lanes: PEP 517 loaded `build_hooks.py` without the repository root on
  `sys.path`, so its package-style helper import failed. The local repair loads
  the adjacent canonical helper by file path. A new isolated-import regression
  and 43 focused tests pass; a real isolated source build passed the formerly
  failing hook stage before optional archive compression was stopped to return
  CPU to the uncertainty campaign. Push one corrected head; do not rerun the
  obsolete failing head.

## Review-Bound Pull Requests

- UpstreamDrift PR #8990, head
  `b3a9ea8ecaa6fe15215381ad08cc239e51b1dad5`, repairs Qt application event-loop
  ownership and containment suite-order leaks. Required checks were still in
  progress at the last inspection; human review is required.
- Tools PR #4635, head
  `22d661a5a21f2e6eb24060abebb54c7ca95962bf`, has green required checks and is
  waiting for human review.
- Never enable auto-merge, admin-merge, bypass protection, or treat a requested
  review as an approval. Verify exact head, review decision, checks, merge SHA,
  and remote-main ancestry before reporting completion.

## Verified Scientific Baseline

- UpstreamDrift PR #8954 is on remote `main` at
  `81cc731d0dd19367b00cd819be5677ab157ce125`; it pins Tools merge
  `1664d806df8a2c7b184d2d3fbcea93b714caaee5` and qualifies the 18-case
  rotating-base consumer contract without copying solver logic.
- Cross-engine repair PR #8960 is on remote `main` at
  `fdf2eb0d1e37db8f5b58109dbbf224a519538170`; records use genuine MuJoCo 3.8.0
  and robotics Pinocchio 3.8.0 rather than the unrelated PyPI package.
- Manufactured-solution repair PR #8961 is on remote `main` at
  `3c75dfdd14404bb897779a6899d85cc21078c4d0`; it uses analytical
  Lagrange--Christoffel, MuJoCo inverse dynamics, Pinocchio RNEA, real
  conservation rollouts, Richardson estimates, and corruption killswitches.
- Native-contact discrepancy PR #8962 is on remote `main` at
  `05a76a2bfcececf0a01df8311d0c4265a0e60e55`; native equality constraints and
  the shared projected compliant-contact formulation remain explicitly
  distinct rather than being claimed as parity.
- The publication remains computationally qualified but not archival-quality:
  tagged-PDF and font findings remain disclosed and fail closed.

## Scientific and Repository Boundaries

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- #8556 remains externally data-gated because no governed participant dataset
  with synchronized bilateral six-axis grip wrenches is available. Never
  substitute synthetic traces for human validation.
- Exact zeros require an analytic identity or explicit degeneracy control.
- Use TDD, DbC, DRY, and LoD. Do not force-push, bypass hooks/checks, add
  quarantine debt, hand-edit governed artifacts, or edit `vendor/ud-tools`.
- Document titles and captions use title case. Research evidence is hash-pinned.

## Current Docker Validation

```bash
python3 -m pytest -q -n 0 \
  tests/unit/test_pinned_tools_provenance.py \
  tests/unit/test_build_hooks.py \
  tests/docker/test_pinned_tools_build_boundary.py \
  tests/docker/test_docker_hardening_7159_7161.py \
  tests/docker/test_docker_integration.py
python3 scripts/ci/check_dockerfile_contracts.py
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 -m ruff check scripts/packaging/pinned_tools_provenance.py \
  tests/unit/test_pinned_tools_provenance.py \
  tests/docker/test_pinned_tools_build_boundary.py tests/unit/test_build_hooks.py
python3 -m ruff format --check scripts/packaging/pinned_tools_provenance.py \
  tests/unit/test_pinned_tools_provenance.py \
  tests/docker/test_pinned_tools_build_boundary.py tests/unit/test_build_hooks.py
git diff --check
```

Passing common gates does not close a child issue whose scientific criteria
remain unmet.
