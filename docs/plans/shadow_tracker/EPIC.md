# Shadow Tracker Epic

## Purpose

Enable researchers to investigate how golfers move using accessible video,
including historical greats for whom marker data do not exist. Provide a direct
connection from image evidence to the existing UpstreamDrift dynamics models,
with enough provenance and uncertainty to distinguish observation from inference.

## Scope

- Single-camera and synchronized multiple-camera body/club silhouettes.
- Local footage ingestion and a resumable archive catalog with reviewed sources.
- Subject proportions, visual body envelope, camera, and starting-state fitting.
- Time-dependent controls, ground contact, grip constraints, and forward replay.
- Competing hypotheses, quality classification, overlays, and model exports.
- Modern reference captures and historical degradation benchmarks.
- Shared headless service, followed by PyQt and React integration.
- TDD, DbC, LoD, DRY, and documented agent turnover for every implementation PR.

## Out of Scope for the First Release

Unique recovery of anatomy, muscle recruitment, ground reaction forces, or
intent from silhouette alone; guaranteed accuracy on arbitrary video; real-time
archive fitting; reconstruction of invented frames as measurements; autonomous
mass downloading; replacing existing engines; a second generic video player.
Ball flight is optional corroboration. The first dynamics demonstration ends
before impact; whole-swing acceptance must explicitly qualify impact handling
and follow-through, rather than silently omit them.

## User Outcomes

| Milestone               | User Outcome                                                     | Exit Evidence                                                              |
| ----------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| M0: Foundation          | Another agent can implement without guessing conventions         | This documentation, tracked work packages, source/test homes               |
| M1: Synthetic Proof     | A known simulated swing can be recovered from its outlines       | Camera, mask, starting-state, and forward-replay fixtures                  |
| M2: Modern Capture      | A calibrated real swing can be reconstructed and audited         | Held-out cameras plus independent reference measurements                   |
| M3: Historical Pilot    | An archive clip yields defensible hypotheses or a useful refusal | Reviewed catalog, timing ambiguity, degradation and abstention reports     |
| M4: Product Integration | Results open in existing model workflows                         | Saved-session round trip, PyQt/React parity, cancellation and exports      |
| M5: Qualified Release   | Claims are bounded by independent evidence                       | Locked holdouts, physics audit, scientific review, reproducibility package |

## Success Metrics

Use [Validation](VALIDATION.md) as the gate definition. Report silhouette overlap
and contour distance alongside held-out 3D error, continuous-replay error,
physical constraint violations, sensitivity, uncertainty coverage, abstention,
human correction time, and compute cost. A lower training silhouette loss alone
is never evidence of better motion reconstruction.

## Stories and Dependencies

[Work Packages](WORK_PACKAGES.md) specifies ST-00 through ST-12. The
[Roadmap](roadmap.json) connects them to GitHub. Existing full-body model epic
#10062, canonical-state epic #6772, and Simscape matching epic #9921 provide
integration context; their presence is not proof that every capability is ready.
Do not duplicate or close those issues as part of Shadow Tracker.

## Definition of Done

- Every required child is merged with tests and its acceptance evidence.
- At least one full-body engine passes independent continuous forward replay.
- Other advertised engines pass their own capability/convention gates; unsupported
  engines are reported honestly and never substituted with mocks.
- One modern reference cohort and one rights-reviewed historical pilot pass
  preregistered gate profiles, including failure and abstention behavior.
- End-to-end swing coverage, timing assumptions, contacts, and impact treatment
  are visible in every accepted result.
- Canonical export preserves uncertainty and provenance; inferred markers are
  never reclassified as measured mocap.
- Product tests, scientific qualification, manual governance, and handoffs pass.

M0 completion does not complete this epic or establish scientific validity.
