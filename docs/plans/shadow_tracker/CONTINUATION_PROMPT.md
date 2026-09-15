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
  docs/plans/shadow_tracker/PROGRESS_REVIEW_2026_09_15.md
  docs/plans/shadow_tracker/DEVELOPMENT_REVIEW.md
  docs/plans/shadow_tracker/CONTRACT_FREEZE.md
  docs/plans/shadow_tracker/QUALIFICATION_FINDINGS.md
  docs/plans/shadow_tracker/WORK_PACKAGES.md
  docs/plans/shadow_tracker/INTEGRATION.md
  docs/plans/shadow_tracker/VALIDATION.md
  docs/plans/shadow_tracker/CAPTURE_AND_ARCHIVES.md
Inspect current code and GitHub state. Reviewed baseline is b97e159dc. Record
hardening and prototype stages through ST-06 have merged, but their full
acceptance is incomplete. The current progress review supersedes old pickup
instructions and unsupported claims of 100% boundary coverage.

FIRST ACTION: inspect/claim a bounded follow-up under #10127. Write a red test
showing that an arbitrary existing checkpoint must not report segmentation
success; the current provider returns a mask count without running inference.
Make it fail explicitly until real masks are produced. Do not add fake inference.

NEXT: resolve the point-only renderer and conflicting state conventions under
#10128, add independent filled-area/articulation oracles, and repair DTO finite,
shape and ownership invariants. Do not fit a golfer against two landmark pixels.
In a separate bounded image-only slice, finish real decoding under #10168 and
persisted mask revisions under #10127, then expose the real review journey in
the launcher. Scientific #10167 must regenerate valid calibration/IK evidence;
Stage 7 boundary work cannot qualify physics on the historical invalid pose.
Follow all remaining stages and acceptance gates in the current progress review
and DEVELOPMENT_REVIEW.md. Do not redefine partial prototypes as completed stages.

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
