# Comparison Rendering Qualification Handoff

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: C:/Users/diete/Repositories/Worktrees/UpstreamDrift-reference-rendering
- Branch: fix/9882-comparison-rendering
- Baseline: cab3505a4, timing qualification #9885 integrated
- Implementation commit: SELF
- Issue: #9882; advanced epic #9863
- Pull request: #9888 (draft until parent #9885 merges)

## Implementation

ComparisonRenderer owns one expert decoder per preview/export. The same existing
coaching ClipRendering/\_rendered/export_clip path draws detector pose, saved
coaching shapes, reference, source-coordinate crop, even edge padding and clock.
Motion opacity now blends drawn pixels. Expert videos use a source-pixel
homography with warped coverage, preserving uncovered player pixels and valid
black reference pixels. Missing camera evidence refuses a visible 3-D export.

Exports validate speed, actual recording bounds/dimensions, drawings and linked
expert identity; stage the entire clip; verify every encoded frame; recheck input
hashes and current session camera/clock evidence before exclusive publication.
Failures/cancellation publish neither media nor JSON. Sidecars include the exact
reference geometry/mapping document, camera/clock/registration, drawings, source
selection, dimensions/padding, timestamps and compositor recipe. Source video
remains external and silent output preserves every selected frame. This is not a
cross-file power-loss transaction or a physical calibration accuracy claim.

The preview respects the saved swing selection/crop and uses the same compositor.
An export snapshots current library identity and reloads saved visual sources.
The general coaching exporter now pads odd uncropped dimensions consistently;
its swing sidecar records that padding. No shared Tools child code was changed.

## Evidence

Seven initial rendering regressions failed; invalid infinite export speed reached
an encoder timeout before validation was added. The first complete regression
batch passed 48 tests across comparison, native state, clips, coaching and swing
exports. Three additional cancellation/homography cases and the catalog identity
check pass in the 33-case renderer/state rerun. Pixel tests compare exact frames
before lossy encoding, and a real encoded container is decoded completely. Reader
instrumentation confirms one expert decoder per preview/export owner. Synthetic
camera changes before/during export, corrupt drawings, changed sources, decode
failure and encoded failure publish nothing.

Five source modules pass mypy; Ruff and architecture budgets pass. Three new
LoD occurrences were corrected without changing baselines. Final LoD no-growth
passes (2996 files, 490 baseline occurrences, 60 reductions); DRY no-growth passes
with 666 existing quarantined fingerprints unchanged. Design-manual governance passes (2 QMD sources, 0 calculations), with
publication still blocked-inventory-required. Protected checks remain pending.

## Continuation

Integrate parent #9885, finish final no-growth checks, commit and open a focused PR
for #9882. Then complete #9883 spatial/event controls, recoverable alignment,
unsaved-change guards, responsive visual evidence and atlas updates. Do not close
advanced epic #9863 yet. Fleet adoption #1579 is a separate active main-branch
rollout run 34370327634; its 41-repository dry run passed.

## Coordination

Session capture-product-01a08427-render-qualification owns #9882 and the five
capture-rig source modules plus tests/docs. Other active presences on the same
files are this root agent's preceding qualification branches. #9884 merged at
2c99bc83e; #9885 is ready with auto-merge and cab3505a4 pending protected checks.
Use ordinary topic PRs, hooks and protections. Preserve unrelated agent work.

Parent cab3505a4 is merged with #9884 and concurrent main OCP changes retained.
Only the canonical handoff conflicted; this rendering handoff remains current.

Normal pre-push mypy needs explicit TypeAlias declarations under its import policy;
Image and Camera aliases now declare their role. No runtime behavior changed.

Parent timing CI repair 250c421e4 is integrated. The newer compositor keeps its
own existing camera-snapshot helper instead of the superseded export helper.
NumPy shape typing and refreshed benchmark evidence are retained. Full-PR
architecture passes with --base-ref origin/main; use this explicit comparison.

## Timing Integration

Merged parent #9885 at de430c500, preserving concurrent #9886 compatibility and tests. This export implementation retains its shared renderer and stricter staged publication. Canonical timing evidence is in #9885; rendering tests and preview parity remain required before merge.

The stacked SPEC merge driver restored the old #9879 row. Removed it again, retaining PR #9884; explicit duplicate validation passes. No rendering change.
