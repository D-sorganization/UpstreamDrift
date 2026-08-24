# Agent Handoff — UpstreamDrift

Last updated: 2026-08-23

This file records current operational state, not history. Git and GitHub retain
history. Epic #8557 is the single proximal-to-distal completion authority.

## Repository Authority

- UpstreamDrift owns scientific sources, models, evidence registers, and the
  release bundle. AffineDrift is an immutable revision-pinned public projection.
  Tools owns reusable consumers; do not copy its implementations here or into
  `vendor/ud-tools`.
- Remote `main` is `3ecd8c2be0ad25da1548a4b948a93fbfa2268179`, the
  protected squash merge of PR #9019.
- The current computational publication is 239 pages with SHA-256
  `be85b7b62bba060a26ce3fea8355aa8b01dcf8c1b1ccf09304450898a4e5e78b`,
  194 URI links, and 247 outline entries. It was fully rendered and inspected.
  It is not archival: it remains untagged and contains Type 3 and unembedded
  font resources.

## Active Headline Structural Campaign (#8800)

- The scientific execution source is the clean, pushed commit
  `1bd4d57da7bd257b76b42b3cc19524b283b5f748` on
  `research/8800-headline-structural-propagation`. It includes the exact
  `shaft._RecordContext` adapter and a regression that rejects variadic mocks.
  Later handoff-only commits do not change or relabel that frozen source.
- The governed seven-corner plan contains 84 planned states: 83 feasible plus
  the declared low-height infeasible state. It requires 14 terminal atlas paths,
  with 48 shaft and 72 ground atomic branch packs per applicable corner. The
  plan contract SHA-256 is
  `c5cfba35ecafa96054ef8cc872f2e91a9f7855db0b93cfa491f9b18ee3db80f4`.
- The corrected campaign is running on ControlTower in detached container
  `upstreamdrift-8800-1bd4d57da`. Source is the clean read-only detached
  worktree `C:\Users\diete\Repositories\UpstreamDrift-worktrees\8800-execution-1bd4d57da`.
  Campaign state is under
  `C:\Users\diete\Campaigns\UpstreamDrift-8800-1bd4d57da`.
- Launch occurred at `2026-08-24T03:46:19Z` in image
  `sha256:b40d91fe2326c5fae288e4a853377fb164aa0a6ba1de62cb28aba15d65500a1e`.
  The run has two model workers, a hard two-CPU cap, 56 GB memory, 112 GB
  memory-plus-swap, and `on-failure:3`. Initial inspection showed running,
  no OOM/restart, about 259 MiB resident memory, and the expected two-CPU load.
- The latest read-only Tailscale/SSH inspection on 2026-08-23 showed the
  container still running with zero restarts, no OOM, 199.28% CPU, 270.6 MiB
  resident memory, 97 PIDs, and 39 durable checkpoint files. The governed
  campaign status remained `running`, with no terminal result or retained
  execution failure yet. Do not treat checkpoint count as scientific progress
  or release evidence; it only establishes durable operational advancement.
- `launch-manifest.json`, atomic checkpoints, campaign status, figures, output,
  separate stdout/stderr logs, and an atomic terminal `exit-code.txt` live in
  the campaign root. They persist if SSH, Codex, or DeskComputer disconnects.
  Absence of `exit-code.txt` means the container has not reached terminal state.
- Monitor over pinned-key SSH from DeskComputer. The duplicate local SSH config
  block points host-key storage at `NUL`; every command must explicitly use
  `-o StrictHostKeyChecking=yes` and
  `-o UserKnownHostsFile=C:/Users/diete/.ssh/known_hosts`. Do not weaken trust.
- The Windows Docker client points to inactive TCP port 2375. Operate Docker
  through WSL distribution `ControlTower-SSD`, where `dockerd` is active. Do not
  alter the Docker service, runner services, or fleet topology for this campaign.
- Do not run in ControlTower's dirty primary `UpstreamDrift` checkout. Do not
  restart either retained failed container or reuse its checkpoints. The failed
  roots `UpstreamDrift-8800-f276e779f` and `UpstreamDrift-8800-8a20df8fe`
  remain provenance evidence only.
- The pinned Linux plan validator passed before launch. The image lacks pytest,
  so the remote regression invocation could not run; 52 focused tests and the
  broader 192-test structural/headline suite passed locally before launch.
- No #8800 scientific result exists until all 14 paths are terminal and the
  common-support, negative-control, retained-failure, result, and publication
  audits pass. Partial checkpoints are not release evidence.

## Completed Constitutive Campaign (#8752)

- The ControlTower campaign completed with exit 0 at source
  `13146cdcece879e7156e06e2dca6626c1a54e045`. Its 1,329-file manifest
  verifies 1,327 atomic checkpoints and terminal evidence. Record SHA-256 is
  `e8a7e53701217e4de875a370f7483172f3cfbfb167416b5133ba269b8fef689b`;
  archive SHA-256 is
  `c54e021eb5ad1e8270ee2a6b473c2cb6d9799583fd41c190b19df9581a6f6d1a`.
- Nonnominal shaft corners span 80--182 matched cells versus nominal 126/384;
  both grip-damping corners are retained failures. All completed ground corners
  remain 0/384, with high grip damping retained as a ground failure.
- This rejects invariance of the shaft comparability set only over the registered
  constitutive bounds. It is not a speed, participant, equipment, or coaching
  result. Claim `PD-CLAIM-314` records that bounded conclusion. #8800 blocks the
  final #8752/#8668 integration and release regeneration.

## Evidence and External Boundaries

- PR #9018 merged normalized claim adjudication at
  `9e220712025564caf0ac5201a0ddcf69dd98299e`. The authority has 303 material
  claims: 283 supported only within declared scope, five inconclusive, and 15
  untested. Model evidence is never human validation.
- PR #9017 merged fail-closed trajectory-processing authority at
  `ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`. #9004 remains open because no
  governed participant trajectory dataset or held-out human outcome is
  registered.
- #8556 remains externally blocked by the absence of governed synchronized
  bilateral six-axis grip-wrench participant data. Synthetic traces and
  paper-level curves cannot substitute.
- #8443, #8448, #8449, #8450, #8595, #8668, #8684, #8796, and #9004 remain
  open. Verify exact acceptance evidence before changing issue state.

## Cross-Repository Path

- Tools PR #4662 merged source-pinned React and PyQt visual baselines at
  `9604773d7576a330602821f88dd964503b698ae0`. Trusted post-main run
  `32689177846` passed lifecycle, accessibility, performance, candidate
  retention, all PyQt renders, and the visual comparator. Tools PR #4663 then
  merged release-runtime portability at
  `eebdddf8c6e366722be40c25278cf34a0392f256`; post-main Release Automation run
  `32690255930` passed analysis, validation, and version-bump stages and opened
  ordinary protected release PR #4664. Do not bypass or silently merge it.
- Tools #4142 remains open for R10--R15 qualification and immutable
  UpstreamDrift consumption. Tools #4430 is complete.
- AffineDrift #3930 remains downstream of a qualified UpstreamDrift release.
  Do not project a moving or partial campaign.
- #8963 architecture debt remains separate from the frozen campaign source.
  Do not regenerate source-bound evidence from a refactored implementation.

## Scientific and Review Rules

- The model ladder is synthetic and model-conditional. It does not establish
  participant mechanics, anatomy, physiology, equipment calibration, injury,
  coaching strategy, or a universal speed benefit.
- Distinguish energy transfer, momentum redistribution, joint work, constraint
  forces, and clubhead speed. Preserve falsifiers, adverse cases,
  identifiability limits, uncertainty, countermodels, and unavailable states.
- PRs target `main`; use full PRs and human review. Never force-push,
  admin-merge, bypass checks, add quarantine debt, or edit `vendor/ud-tools`.
- Use TDD, DbC, DRY, and LoD. Edit canonical sources, regenerate governed
  artifacts, and use title case for headings and captions.

## Next Actions

1. Monitor the detached #8800 container without redundant restarts. Preserve
   every retained failure and retrieve evidence only after terminal exit.
2. Audit all 14 outputs, common support, negative controls, provenance, and
   result/figure contracts; do not infer success from exit code alone.
3. Regenerate the paper, claims, release bundle, and PDF; run scientific,
   publication, link, title-case, file-size, architecture, Ruff, MyPy, Bandit,
   and clean-checkout gates; render and inspect every page.
4. Open and shepherd the protected UpstreamDrift PR with human review. After
   merge, verify the squash commit on remote `main` and post-main evidence.
5. Qualify Tools release PR #4664 without bypass, reconcile its conflicting
   `main` versus GAAI `staging` branch rules, then publish the revision-pinned
   AffineDrift projection and finish Tools consumer qualification.

Passing shared gates does not close a scientific child whose narrower evidence
or governed external-data requirement remains incomplete.
