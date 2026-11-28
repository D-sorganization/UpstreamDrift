# Tiger 2000 Held-Out Phase Annotations (#15837 WP-2)

This package freezes the _contract_ for the independent held-out phase annotations
that `#15837` WP-2 requires. It deliberately contains **no annotation values**: the
pinned rendition of `9i7ok4rVkVM` and its decoded frames live on the capture host, not
here, and `#15844` records the window as "preliminary, not independent phase annotation".
Fabricating keypoint coordinates would violate the markerless honesty contract, so the
committed artifact ships `status: "pending"` with zero frames and every gate that
depends on it fails closed.

## What Is Frozen

- `src/config/tiger_2000_heldout_phase_annotations.json`
  (`heldout-phase-annotations/1.0.0`): provenance pins copied from
  `src/config/tiger_2000_scorecard_gates.json`, the five required phases
  (`address`, `top`, `downswing`, `impact`, `finish`), an independence declaration,
  the annotator noise floor, the review trail, and a SHA-256 content hash.
- `src/shared/python/shadow_tracker/heldout_phase_annotations.py`: validation of frame
  digests, the pinned candidate window `[1080, 1500]`, in-frame keypoint coordinates,
  repeat-pass noise-floor measurement, and `scoreability_blockers()`.

A scorecard may only report `passed` when it was computed against a `reviewed` set whose
`noise_floor.status` is `measured`, whose pins agree with the gate config, whose frame
population is exactly the annotation population that was scored, and whose records are not
a copy of the null template. Otherwise it reports `below_gate` with the blocker names.

## Completing the Annotation Pass

Run on the host that holds the pinned frames, then commit the regenerated artifact:

1. Reproduce the frame map from `#15844` and confirm `source_sha256` equals
   `6e2893fbcaa8daa45bb2d5e979eeff12e2f57f5c697962d95eb6c48c04219934`. A different
   rendition requires a new pin and a new annotation set.
2. Annotate `left_wrist`, `club_head` and the shaft angle on at least three frames per
   phase inside the window, citing each frame's `frame_sha256`, and record the phase
   by eye from the full frame (crop `[0, 0, 1280, 720]`, no mirror or rotation).
3. Repeat pass two independently on the same frames and derive the floor:
   `measure_repeat_noise_floor(pass_one, pass_two)`. Do not copy the configured
   `8.0 px / 4.0 deg` placeholders into a `measured` floor.
4. `verify_frame_map(...)`, set `status: "reviewed"`, name the reviewer, rewrite
   `content_sha256` by loading and re-serialising, and keep
   `timebase_status: "unverified"` plus `qualification_floor: "image_observations_only"`.

Presentation timestamps are not physical swing time; this package scores image-space
agreement only.
