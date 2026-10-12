# Necromatcher Raster ROI View Optimization

## Scope and Existing Authority

Issue #11356; continuation PR #11359. This narrow change reuses the public shared `render_surface_layer` boundary in `body_part_viz`. Discovery covered the shared renderer and camera contracts, native/web option consumers, the curated renderer facade and existing shape methods supplement. It adds no camera, model, geometry adapter or solver. The implementation lives in the isolated hypothesis-admission worktree; the published `00b7d9846bd2fb2572b7c84f03cfc6c9699323cb` producer remains unchanged during its active numerical jobs.

## Exact Integer ROI Semantics

The existing integer coordinate grid enumerates one unique rectangular region of interest per triangle. Reading old depth with two coordinate arrays unnecessarily gathers a copy; writing each output repeatedly gathers winning coordinates. Use rectangular workspace views and the same boolean winner mask instead.

Compute the entire winner mask before writing the depth view. Keep the original barycentric operations, reciprocal-depth equation, pixel centers, near-plane clipping, triangle ordering and depth tie tolerance unchanged. Color, coverage, semantic geometry IDs, alpha rounding, immutable `SurfaceLayer` contracts and source-image validation remain unchanged. Tessellation, resolution, camera, saved geometry, opacity and the 600-second worker budget are unchanged.

## Fail-First and Exact Parity Evidence

A public-renderer counting-buffer regression failed on both source-size fixtures before the change: the old depth workspace used coordinate-array gathers/scatters. The corrected renderer uses rectangular views. An analytical overlapping-surface test checks exact nearest depth, BGR color, deterministic equal-depth identity, coverage and uncovered state. The focused ROI/grid/surface suite passes 43 cases.

The pre-change renderer is preserved outside Git at `C:/Users/diete/Repositories/Temp/shape-roi-view-optimization-v1/projective-renderer-old-00b7.py`, SHA256 `aa8027adc8e527d26085ab91e9a12c5175e2fcc341b3de5c4c75783edf584a3a`. Twenty-seven SDK-free synthetic comparisons at 32x24, 320x240 and 1280x720 check exact pixels, mask, depth and geometry IDs across front-facing, perspective, behind-camera, near-plane, off-frame, degenerate, subpixel, sorted-tie and offset cases. They pass without native compilation or historical rendering. TEMP red/green, parity and pinned-type-check logs remain preserved.

## Qualification and Later Measurement

This is an allocation/byte-equivalence change, not a performance guarantee or scientific qualification. The current twelve forearm pilot records remain unchanged: Tiger passes its selected-pose 600-second heuristic, while Hogan control/variant fail. The old six-pose pilot concerns different saved motions and cannot qualify these new fits. No native re-verification, profiling, timing, full-video export or numerical operation was performed for this patch while the original four budget jobs are active.

After all active source/library brackets close and a new source producer is published, root may separately authorize read-only verification of the exact saved current twelve poses. Recompute all original source identities, layer pixels/depth/mask/IDs and alpha compositions against the preserved current artifacts, with whole-source/runtime/library/input before/after preservation even on failure. Record fresh timing independently; retain the old failed gates and do not relax the budget or claim whole-clip throughput from selected poses. Changed fit identity requires another fresh pilot.
