# Independent Bounded Trial Audit

This read-only audit evaluates an existing saved canonical Hermite trajectory. It does not optimize, repair a trajectory, add image observations, or qualify physical time or scientific acceptance.

## Reproduction

Run the script in a clean interpreter from the worktree after the saved fits and implementation are frozen. Supply the existing library explicitly and a new output path outside that library. Repeat `--fit-id` for multiple saved fits, including authored seeds whose `optimizer_ran` is false.

```powershell
python3 docs/development/historical_capture/audit_bounded_trial.py `
  --library-root '<existing-library-directory>' `
  --fit-id tiger-bounded-heel-hypothesis-fit-v8 `
  --fit-id hogan-bounded-heel-hypothesis-fit-v8 `
  --output '<new-external-assessment.json>'
```

The CLI imports MuJoCo before workspace modules. It loads public bound native geometry, reconstructs the original saved source-clock spline and observations, verifies original RMS and saved dense poses, and writes the receipt exclusively. Existing output paths are rejected.

## Claims and Boundaries

The q-bound certificate uses the canonical `HermiteBoundsDomain.encode` conservative Bernstein domain and `assess_spline_bounds` scalar global extrema, including locked coordinates. Authored ranges come only from the exact bound definition and reproduced XML identity. Missing ranges remain explicitly unbounded. No compiled joint limits are claimed.

Nonlinear geometry is sampled at every original capture PTS inside the preserved spline interval, every exact rational adjacent-frame midpoint, and each coordinate's canonical global minimum and maximum time. Public native grip position/rotation and declared contact-sphere heights are cross-checked against the public residual rows with unit weights and scales. This finite sampling is not a continuous nonlinear certificate. Declared ground geometry and pinned heel hypotheses are not measured contacts.

Original frame identities, exact rational PTS, confidence-weighted image RMS, and dense observed-point count remain explicit. Midpoints and extrema add no image objective points. Canonical extrema use the canonical floating source-clock time; source frames and midpoints retain rational container timestamps. Container time never establishes physical capture time.

## Provenance and Failure Handling

The receipt binds fit, model, capture, definition, and XML hashes. Relevant public parent assets are checked before and after assessment; source/runtime execution stamps and the audit script hash must remain unchanged across the audit. The historical producer stamp is preserved separately from the current audit execution stamp. An uncommitted audit script is identified by its exact script hash; committing it later does not change the historical fit producer.

The source-video identity is copied from the verified capture receipt. The raw source MP4 is not opened or rehashed, and `source_video_file_rehashed` is false. Unknown or changed parent identities, duplicate timestamps, saved spline disagreement, invalid native geometry, or source/runtime drift reject the audit. The library and its media are never modified.

Scientific acceptance, physical-time qualification, optimization performed by this auditor, and continuous nonlinear certification remain false even when the coordinate-bound certificate passes. Preserve rejected research outputs and their actual metrics without promoting them to accepted motion capture.
