# Agent Handoff — UpstreamDrift

Last updated: 2026-08-21

This file records current operational state. Git and GitHub retain history.
Epic #8557 is the canonical proximal-to-distal completion authority.

## Protected Delivery in Progress

### Tools Rotating-Base Consumer

- Full PR #8954 targets `main`; branch `feat/4430-rotating-base-consumer`.
- It pins Tools remote-main revision
  `1664d806df8a2c7b184d2d3fbcea93b714caaee5` and verifies the qualified
  18-case rotating-base provider without copying solver or catalog logic.
- The consumer asserts exact source/study/catalog digests, run order, 13 valid
  and five adverse cases, nonanatomical coordinates, unsupported coaching
  inference, and unavailable governed human validation.
- Standard CI, shared consumer contracts, documentation, SPEC freshness, and
  optional-stack checks passed on head `2061fdaed487633d63b9eb82a1f8a71457af9be1`.
- The package job proved its wheel build/content gate and 8m35s artifact upload.
  It then timed out in setup-node's post-job npm cache upload. Commit
  `6bf348dd0` removes that unnecessary cache; measured `npm ci` took seconds.
- Current `origin/main` is merged into this branch. Resolve no further conflict
  by dropping either the launch-monitor conformance contract or this consumer.
- Auto-merge is off. Human approval is required. Never bypass protection or
  create redundant reruns.

### Critical Scientific Corrections

- #8909 is active in worktree `8909-real-pinocchio-parity`. The published
  distributed-grip "parity" used PyPI `pinocchio` 0.1 and silently compared
  MuJoCo with itself. The repair fails closed on engine identity/model build,
  rejects exact-zero degeneracy, and regenerates with Pinocchio 3.8.0.
- #8910 remains open: current manufactured inverse-dynamics, action-reaction,
  and conservation controls are tautological or hardcoded. Do not close #8752
  or promote those controls until independent operators and actual drift pass.
- The #8752 checkpointed uncertainty campaign runs in worktree
  `goal-8752-uncertainty` under parent PID `18404`. It is not orphaned. Do not
  kill its workers, edit campaign sources, or launch a duplicate campaign.

## Publication and Human-Data Boundaries

- UpstreamDrift owns the scientific source/evidence bundle; AffineDrift is the
  immutable pinned public projection; Tools owns reusable provider logic.
- The computational PDF is not an archival or human-validation release.
  Tagging, font, stable-archive/PID, equipment calibration, and governed human
  gates remain distinct.
- #8556 remains externally data-gated: no governed participant dataset with
  synchronized bilateral six-axis grip wrenches is available. Synthetic traces
  must never substitute for human validation.
- Do not infer anatomy, physiology, injury, coaching strategy, or universal
  speed benefit from model-conditional synthetic results.

## Launch-Monitor Contract on Current Main

- Current main includes `launch-monitor-analytics-conformance/1.0.0`: ten
  deterministic, data-free cases with typed available/unavailable outcomes,
  evidence/provenance wrappers, schema, golden fixture, and ADR 0040.
- Its quantized serialization boundary stabilizes hashes without changing the
  underlying analytics. Preserve these files and SPEC entries during merges.

## Repository and Review Rules

- PRs target `main`; use full PRs, never drafts. Auto-merge is prohibited and
  human approval is mandatory.
- Before issue work, run the Repository_Management claim check and post a lease.
- Use TDD, DbC, DRY, and LoD. Never force-push, admin-merge, bypass hooks or
  checks, add quarantine debt, or edit `vendor/ud-tools`.
- Research evidence is hash-pinned. Use governed regeneration; do not hand-edit
  source digests, claims, JSON/NPZ, figures, or release manifests.
- Use title case in document headings/captions and run the title-case audit.

## Consumer Validation

```bash
python3 -m pytest \
  tests/launchers/test_tools_vendor_authority.py::test_tools_pin_targets_qualified_rotating_base_release \
  tests/launchers/test_launcher_model_sources.py::test_all_tools_launchers_resolve_from_pinned_vendor \
  tests/shared_contracts/test_tools_provider_contracts.py::test_rotating_base_provider_retains_complete_qualified_authority \
  --tools-mode vendored -n 0 -q
python3 -m pytest \
  tests/unit/packaging/test_standalone_sidekick_workflows.py::test_package_workflow_preserves_time_for_verified_artifact_upload \
  tests/unit/packaging/test_standalone_sidekick_workflows.py::test_package_workflow_does_not_cache_the_fast_frontend_install \
  -n 0 -q
```

## Release and Repository Gates

```bash
python3 -m scripts.research.proximal_distal_energy.claim_audit validate
python3 -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python3 -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate \
  --source-revision "$(git rev-parse HEAD)" --publication-profile computational
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/check_doc_size_budget.py
python3 scripts/ci/check_architecture_budget.py
python3 -m ruff check .
python3 -m ruff format --check .
```

Passing common gates does not close a child issue whose scientific criteria
remain unmet. Verify exact PR head, review decision, checks, merge SHA, and
remote-main ancestry before reporting protected completion.
