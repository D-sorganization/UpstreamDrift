# Shadow Tracker Progress Review: September 15, 2026

Historical review: use [Current Turnover](TURNOVER_CURRENT.md) for resolved
findings and the latest correction queue. Preserve this baseline as evidence.

## Authority and Current Status

Reviewed `b97e159dcc1686b4fc36351124996862619d8f35`, including PRs #10169–#10173,
#10179 and #10180. This review supersedes older pickup/status paragraphs; the
original full scope and G0–G7 in [Validation](VALIDATION.md) remain mandatory.
Tracking issue: #10184. Commit/PR presence proves code landed, not stage acceptance.
Turnover publication: [PR #10185](https://github.com/D-sorganization/UpstreamDrift/pull/10185).

Useful progress includes hardened rollout acceptance (#10170), additional
evidence DTOs/protocols (#10171), shot/timing helpers (#10172), mask metrics and
an in-memory provider (#10173), projection/loss helpers (#10179), and candidate
enumeration/ranking (#10180). The current product is **partially implemented**.
The earlier statements “Stages 0–6 complete” and “100% boundary and DbC
enforcement” are contradicted by the reproduced cases below.

ST-01's audit documents an unqualified model; it does not satisfy the original
qualification acceptance. The committed probe still reports about 1.266 m grip
displacement. Follow #10167 for regenerated evidence and canonical mapping.
Do not reopen or close issues solely from this document: inspect their current
state and file/claim a bounded follow-up under the indicated parent.

## Reproduced Findings and Next Tests

| Priority / Parent           | Observed Behavior at This Commit                                                                                                                                                                                                               | First Failing Test and Required Outcome                                                                                                                                                                                                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1 / ST-04 #10127           | `ModelSegmentationProvider.segment` accepts a file containing `b'not a model'` and returns `mask_count=1` for one requested frame. It does not load a model, read images or produce masks.                                                     | An existing arbitrary checkpoint must never report successful segmentation. Until an actual provider runs, return explicit unsupported/unavailable status or raise an actionable exception. Then require concrete mask artifacts, source identity and checkpoint hashes from a real inference test.                     |
| P1 / ST-05 #10128           | `AnalyticSilhouetteRenderer` writes at most one pixel for body and one for club, taking state elements 0:3 and 3:6 as landmarks. A visible test state produced foreground counts `(1, 1)`. It has no body geometry binding.                    | A known projected primitive must occupy its independently calculated filled area; then a qualified articulated body/club binding must change the outline with joint pose. Retain the point helper only as an explicitly named test/reference utility. Do not use point hits as golfer silhouettes.                      |
| P1 / ST-05–06 #10128/#10129 | Initialization emits a seven-element translation/quaternion-like pose, while the renderer interprets elements 3:6 as a clubhead point. There is no shared model state mapping between those meanings.                                          | An integration test passes the same declared canonical state through initialization, model FK and rendering and proves body/club positions and units independently. Reject mismatched conventions; never infer state meaning from tuple length.                                                                         |
| P2 / ST-02, ST-05, ST-06    | `PinholeCameraModel` accepts an all-zero rotation; `InitialHypothesis` accepts a NaN pose and empty velocity; `VisualMorphology.segment_lengths` remains mutable after construction. Auxiliary request/result DTOs have no validation methods. | Type/finite/shape/rotation/ownership matrices at real public boundaries. Reuse existing validated camera/state types instead of growing a third camera algebra. Mutating constructor inputs or returned nested values must not corrupt frozen evidence.                                                                 |
| P2 / ST-04 #10127           | Manual masks are keyed only by frame ID, overwritten on registration, and held only in memory. `segment` checks frame existence but not that masks belong to the requested shot.                                                               | Two shots with the same local frame ID cannot exchange masks; revisions remain addressable; parent identity is checked; save/reopen restores every revision and invalidates downstream cache keys.                                                                                                                      |
| P2 / ST-06 #10129           | `fit_initial_state_multiview` ranks caller-supplied candidates using body IoU only, does not use `loss.is_valid` or club score, and returns zero velocity. Monocular generation uses a fixed score of 0.5.                                     | All-invalid masks must yield no fitted winner; empty/mismatched camera sets fail explicitly; body-equivalent candidates with different club alignment are distinguished. Unknown velocity remains unknown/prior-labeled until a valid physical-time window supports estimation. No fixed score presented as confidence. |
| P2 / ST-03 #10126/#10168    | Ingestion has a `VideoDecoderAdapter` protocol and `SyntheticVideoDecoder`, but no real decoder implementation. Timing/catalog helpers alone cannot import a video.                                                                            | Tiny redistributable real local VFR clip -> exact decoded PTS/timebase, image hashes and frame records; unknown physical time survives. Bound memory, cancellation and resource cleanup. No synthetic fixture presented as real-media integration.                                                                      |

The first five observations were executed in a temporary local probe, without
editing production code. The remaining findings follow directly from the named
functions' source. These are actionable evidence gaps, not a request to rewrite
all modules or erase useful unit tests.

## Ordered Agent Dispatch

1. **Stop false provider success first.** Claim a small follow-up under #10127;
   write the arbitrary-checkpoint red test, then make the provider honest.
   Pair this with request/result boundary validation, not a new neural dependency.
2. **Repair renderer and state semantics before optimization.** Under #10128,
   identify the canonical model/geometry interface and implement filled-mask
   rendering with independent primitive and articulation oracles. Resolve the
   seven-value versus six-landmark state mismatch explicitly. Reuse camera math.
3. **Complete real ingestion and persisted manual evidence.** Use #10168 and
   #10127 for bounded image-only slices. This track need not wait for physics:
   import -> frame review -> mask correction -> save/reopen must work on real media.
4. **Continue scientific qualification separately.** #10167 must regenerate
   affected calibration/IK artifacts and demonstrate a valid initial grip,
   canonical state/velocity parity, contact audits and fresh real-engine replay.
   Stage 7 (#10130) may characterize/extract engine boundaries, but cannot claim
   qualified rollout or unblock fitting on the old invalid initial state.
5. **Complete initialization against the real renderer.** Repair empty/invalid
   evidence handling, body/club scoring, subject-shape binding, ambiguity and
   physical-time-aware velocity. A supplied-candidate ranker is an intermediate
   primitive, not proof of fitted full-body pose/shape recovery.
6. **Ship a truthful launcher evidence-review slice**, then the complete fitter.
   Under #10134 use the existing host, manifest, lazy adapter and installed entry
   point together. UI and API share a service. Exercise the real import/edit/
   persistence journey, keyboard use, cancellation and visual review. Label fitting
   unavailable until continuous rollout/fitting exists. Continue #10130–#10135
   through real-data validation, uncertainty, export, React parity and release.

Every slice needs behavioral red/green evidence, pre/postcondition and ownership
tests, narrow provider boundaries, shared implementation reuse and affected
consumer tests. No closure from a class name, protocol conformance, a mock count,
or tests that assert only the implementation's own output.

## Performance and CI/CD Evidence

Benchmark real decode and mask storage/validation/hash on pinned 480p/1080p/4K
fixtures. The current point renderer allocates Python lists and tuples per pixel;
do not build a production hot loop around that representation without measured
memory/work limits. Cache immutable preparation outside objective evaluation;
keep cancellation and queue bounds under tests. Preserve numerical and hash
parity after optimizations. Use the budget/receipt guidance in
[Development Review](DEVELOPMENT_REVIEW.md).

GitHub access initially returned HTTP 401 during this review, then recovered.
PR #10180 is merged but its check rollup includes a failed
`articulated-manufactured-rolling` check. That observation alone does not identify
the cause. Inspect its exact job log and relevance, plus current main CI/CD,
before claiming all checks pass. Do not suppress companion failures or infer
scientific validation from successful workflow-approval jobs.

The selected local suite passed 195 tests on Python 3.13.5/Windows, using the
pinned Tools provider, with this command:

```bash
python3 -m pytest tests/unit/shadow_tracker tests/integration/shadow_tracker/test_model_probe.py tests/unit/motion_matching/test_full_body_forward_dynamics.py tests/unit/motion_matching/test_full_body_parity.py tests/unit/motion_matching/test_full_body_ik_coordinates.py -m "unit or live_simulation" --no-cov -q --timeout=60
```

That run and the separate review probes have different purposes: passing tests
do not negate the reproduced gaps. They cannot certify unimplemented real-media,
filled-renderer, inference, fitter, GUI or scientific pathways.

## Turnover Rules

Update [Agent Handoff](AGENT_HANDOFF.md), [Ready Tasks](READY_TASKS.md), root
handoff and roadmap together. Keep implementation status, acceptance status and
issue state separate. Preserve historical receipts with their original hashes;
add regenerated evidence at new versioned paths. Never replace failed evidence
with an updated narrative. The full epic stays open until the original product,
scientific, performance and release requirements have direct passing evidence.
