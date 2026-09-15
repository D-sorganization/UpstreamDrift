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
  docs/plans/shadow_tracker/DEVELOPMENT_REVIEW.md
  docs/plans/shadow_tracker/CONTRACT_FREEZE.md
  docs/plans/shadow_tracker/QUALIFICATION_FINDINGS.md
  docs/plans/shadow_tracker/WORK_PACKAGES.md
  docs/plans/shadow_tracker/INTEGRATION.md
  docs/plans/shadow_tracker/VALIDATION.md
  docs/plans/shadow_tracker/CAPTURE_AND_ARCHIVES.md
Inspect current code and GitHub state; the review baseline is c3395229, not an
assumption about your checkout. A/B/C (#10137/#10139/#10138) are already merged.

FIRST ACTION: check the claim for #10151 in Repository_Management and lease it
if free. In an isolated topic worktree reproduce the review's malformed-input
and pre-conversion mask-length findings as failing behavioral tests. Repair only
that packet, preserving schemas/hash compatibility and immutable ownership.
Resolve the physical_time_s coercion discrepancy explicitly against the frozen
contract. Run all Shadow Tracker and affected-consumer tests, lint, format,
types, inventory checks and repository CI; publish a focused PR referencing
#10151. Do not reimplement A/B/C from the old pickup prompt.

AFTER THAT: follow the complete delivery sequence in DEVELOPMENT_REVIEW.md.
Before each stage, inspect issue dependencies and create/claim a bounded child
slice with one concrete outcome, allowed files and red/green acceptance tests.
Specialist work #10140/#10141 and #10124 remains prerequisite to qualified
dynamics. Never compensate for incorrect model coordinates with optimizer
freedom or loosen physics tolerances. Image-only work can advance only with an
explicit bounded dependency decision, without closing the blocked parent.

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
