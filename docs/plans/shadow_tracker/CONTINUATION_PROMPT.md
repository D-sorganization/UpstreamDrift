# Shadow Tracker Full Development Continuation Prompt

Copy the following prompt into the development agent's task. It preserves the
whole product objective while requiring small, reviewable implementation slices.

```text
Continue Shadow Tracker in D-sorganization/UpstreamDrift, epic #10122, toward
the complete video-to-silhouette-to-forward-dynamics product, integrated into
the UpstreamDrift launcher with a high-quality usable interface. Support modern
single/multiview captures and historical footage with explicit ambiguity and
unknown timing. Do not stop at data classes, a mock UI, or a synthetic-only demo.

Read current AGENTS.md, CLAUDE.md, SPEC.md and root AGENT_HANDOFF.md. Then read:
  docs/plans/shadow_tracker/TURNOVER_CURRENT.md
  docs/plans/shadow_tracker/DEVELOPMENT_REVIEW.md
  docs/plans/shadow_tracker/CONTRACT_FREEZE.md
  docs/plans/shadow_tracker/QUALIFICATION_FINDINGS.md
  docs/plans/shadow_tracker/WORK_PACKAGES.md
  docs/plans/shadow_tracker/INTEGRATION.md
  docs/plans/shadow_tracker/VALIDATION.md
  docs/plans/shadow_tracker/CAPTURE_AND_ARCHIVES.md
Inspect current source and GitHub state. Reviewed main is 0ec64e45f. Timing
PR #10253 merged; unknown physical time and incremental decode exist. The restart
review corrects estimated-CFR is_timing_exact. Native PTS and persisted clock
authority remain #10273. Read TURNOVER_CURRENT.md before older review sections.

FIRST ACTION: check ownership and status of renderer PR #10264 / #10232. Its
unit-test-gate failed importing motion_matching.diagnostics, and it has merge
conflicts. Existing uncommitted import corrections are present in the original
checkout; preserve them. The owning agent should verify them, integrate current
main and obtain passing full-unit collection plus focused tests at the new SHA.
Do not create a competing renderer PR or claim this unmerged feature is ready.

NEXT INDEPENDENT TASK: claim #10233 for revision identity and atomic persistence.
Start with failing duplicate-ID/idempotence/parent-ownership tests, then implement
save/reopen and invalidation. Keep changes small for reliable cheaper-agent
execution. Follow with #10273 native timestamp authority, not another FPS proxy.

Then fix initialization no-evidence/club-scoring/velocity acceptance and deliver
a real import -> review -> manual edit -> save/reopen launcher slice using one
service, manifest/entry point/lazy adapter and real UI tests. Fitting remains
unavailable until real continuous rollout and scientific qualification pass.
Regenerated grip evidence is improved but still outside the physical profile;
retain that limitation and complete all ST-07–ST-12 stages, including uncertainty,
modern/archive validation, full UI/API/React/export parity and packaged release.
Do not jump from a Stage 7 prototype directly to release.

Apply TDD, DbC, LoD and DRY throughout production code, adapters and UI. Preserve
behavioral red/green evidence. Reuse existing public camera/canonical-state,
geometry, job/service, launcher-host and theme facilities. Test both sides of
every integration and all failure/unknown states. Keep the GUI responsive with
bounded background work, progress, cancellation and compatible checkpoints.
Profile each stage and freeze evidence-based performance budgets; never trade
away accuracy, physical validity or provenance for speed.

Implement the complete user journey: import and source review; swing/golfer
selection; mask correction and lineage; camera/shape/initial-state hypotheses;
bounded forward fitting; independent continuous replay; overlays, physics and
uncertainty inspection; save/reopen; validated export to existing model tools.
Register the launcher tile, lazy embed adapter and installed entry point
together; provide React/API capability parity through the same service.
Test real launcher behavior, accessibility, visual quality and packaged install.

Do not claim that silhouette agreement uniquely identifies 3D motion, torques
or muscle actions. No fake measured markers, guessed physical time, hidden pose
resets, fabricated confidence, zero errors for missing evidence or mock engine
qualification. Unknown time/scale prohibits qualified SI kinetics. Keep observed,
inferred, simulated and scientifically qualified results visibly distinct.

Finish each PR with applicable CI/CD green at its exact SHA, preserved test and
benchmark receipts, and a precise turnover update. Do not waive failed checks,
silently skip required real engines, or broaden unrelated fixes. If an external
dependency blocks one slice, document it and continue other legitimately
unblocked work. Never mark the full epic complete until its entire stage/gate
ledger, real-data validation, UI journey, performance profile and release/install
evidence pass. Handoff delivery itself is not full-package completion.
```
