# Reference Timing and Camera Evidence Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-timing
- Branch: fix/9881-reference-timing
- Baseline commit: a5dfc3542 (state qualification #9884)
- Implementation commit: SELF
- Governing issue: #9881; advanced epic #9863
- Pull request: #9885; #9884 merged at 2c99bc83e

## Changes and Evidence

Twelve adverse timing regressions failed before implementation; the existing
round-trip control passed. Event pairs now require complete, unique, increasing
clocks and interval rates between 0.25 and 4. One anchor aligns its event and
retains the rate about that pivot. Event maps are immutable; paired events
replace the affine offset. The sampler searches original timestamps and touches
only neighboring frames, preserves valid origin points, masks missing endpoints,
and refuses gaps beyond max_gap_s (default 0.25 reference seconds). Exact
samples remain valid. Existing sparse fixtures explicitly allow a 0.5-second
gap to isolate missing-endpoint behavior.

CameraSnapshot and ViewClock preserve actual calibration/distortion and clock
evidence. Shared calibration_for_view retains the existing variant/source
selection while camera_for_view keeps the old ideal-pinhole API. Registration
binds asset geometry/mapping identity, camera identity and clock evidence;
notes are excluded. The dialog checks saved bindings on reopening, saves a
manual alignment snapshot and uses the same scene clock as export. A legacy
calibration flag alone no longer grants calibrated export status.

The combined comparison/backend/native suite passes 46 tests, including actual
camera loading, stale calibration rejection and independent two-camera pixels.
Architecture and DRY/LoD no-growth gates pass with unchanged baselines.
Protected validation remains pending.
Fourteen modified source modules pass mypy. Benchmark report and rerunnable script
are included: 17 joints, 120/1200/12000 source frames; single-frame sampling
medians 0.157/0.093/0.304 ms versus a full-trajectory workload control of
17.934/172.309/1486.035 ms. This excludes load, decode, draw and display; no
whole-app speedup, portable threshold, or Rust rewrite is claimed. Run in an
installed environment or set this worktree's src directory on PYTHONPATH.

## Remaining Work

Local qualification passes; finish protected checks; #9884 must be
integrated without overwriting its remote rebases. #9882 owns strict export
failure behavior and complete renderer parity; #9883 owns spatial/event controls
and visual qualification. No scientific calibration accuracy or manual
publication release claim is made. Fleet adoption #1579 remains independent.

## Coordination

Session capture-product-01a08427-timing-qualification owns #9881. The separate
state worktree remains on #9884. Vendor Tools is unchanged at eab74a901.
Earlier incompatible registration branch 8ad3f5be7 is reference material only.

## Parent Qualification Integration

Merged the state/export-lifetime branch through b2b72a099. Its remote peer
packaging work is preserved separately for #9406; no parameter-budget waiver
is required. The inherited projection/export callers pass seven additional
regression tests. The current change passes 46 focused tests and 14-module
mypy, architecture, Ruff and DRY/LoD no-growth checks.

Main 2c99bc83e integrates state qualification #9884 and concurrent OCP work.
The merge retains timing validation at saved-registration load.

## Protected CI Repair

CI found export_comparison_video one line over its 100-line budget and newer
NumPy typing rejected assigning a general transformed array back to the
shape-inferred allocation variable. Camera conversion is now a separate helper;
the transformed array retains its own variable. Sixteen focused timing/export
regressions pass. Architecture passes with --base-ref origin/main, which checks
the complete PR rather than the default local change scope. Use that explicit
base for remaining product PRs. No limits or type exclusions were relaxed.
The diagnostic benchmark was rerun and its source fingerprint updated after this
non-algorithmic typing correction; host-load variability is not a performance claim.

## Integration With Concurrent #9886

Merged main c6f58aef9 and preserved legacy fingerprint fields, the explicit sampler gap keyword, the uncalibrated-ID guard, and all peer test scenarios. The shared immutable synchronizer retains stricter complete pairing/rate bounds; neighboring-frame sampling and typed geometry/camera/clock evidence remain authoritative. Removed the duplicate #9879 SPEC row, retaining PR #9884. CI had 14,769 passing tests and one duplicate-SPEC failure; the duplicate gate is rechecked after resolution.

Concurrent remote merge 054655131 is preserved. Its explicit single-anchor affine inverse and error wording are retained alongside finite positive gap validation. All peer tests remain included; no remote history is rewritten.
