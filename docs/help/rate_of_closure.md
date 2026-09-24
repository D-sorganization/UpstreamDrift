---
title: Rate of Closure Impact Explorer
tile_id: rate_of_closure
status: stub
---

# Rate of Closure Impact Explorer

## Purpose

The registry describes this tile as quantifying how a rotating clubhead's
impact-point delivery differs from the delivery of the tracked reference point
- the clubhead centre of mass or its geometric centre. The distinction matters
because a launch monitor or motion-capture system tracks one point on the head
while the ball is struck at another, and a head that is rotating through impact
gives those two points different velocities.

No implementation of this tile could be found in this repository, so this page
does not describe its behaviour. See [Unclear](#unclear) below.

## Inputs

Not determinable from this repository. No input controls, parameters, ranges,
or units could be substantiated, so none are listed here rather than guessed.

The registry declares the following capability tags for the tile, which
indicate its intended scope but are not a description of implemented controls:
`rate_of_closure`, `impact_analysis`, `clubhead_kinematics`, `attack_angle`,
`face_rotation`, `3d_visualization`, `putting_simulation`,
`launch_monitor_analytics`.

## Outputs

Not determinable from this repository. No output quantities or units could be
substantiated.

## Method

Not determinable from this repository. No calculation, model, or module
implementing rate of closure, face rotation, or reference-point correction was
found here, so no method is stated.

One piece of related context is on record. The tile is named in
[ADR-0045](../adr/0045-putting-integration-one-experience-two-preserved-stacks.md)
as the surface hosting the Tools putting stack inside UpstreamDrift - the
"Impact Explorer putting tab". Per that ADR, the Tools putting stack owns the
stroke and impact solve (aim, face, path, attack, offsets, mesh-derived putter
MOI), dispersion Monte Carlo, and putter-fitting counterfactuals, and it uses
the `usga-stimp-roll/1` roll model (mu approximately 0.559/stimp, USGA
stimpmeter geometry at a 1.83 m/s release speed, with Holmes/Penner
speed-dependent hole capture) - the preserved counterpart to the model behind
the [Putting Green](putting_green.md) tile.

## Limitations

- **Any number this tile produces is model output, not measurement**, including
  the corrected impact-point delivery, which by definition is a computed
  correction applied to a tracked point.
- If the tile's putting tab is reached: results from the `usga-stimp-roll/1`
  model must never be compared numerically with results from the
  `ud-legacy-roll/1` model behind the [Putting Green](putting_green.md) tile
  unless both model names are attached. The two preserved models differ by a
  pinned roll-out ratio of approximately 2.854. ADR-0045 makes naming the model
  mandatory, and fail-closed readers refuse an unnamed result.
- Beyond that, what this tile does not do cannot be stated, because what it
  does could not be established. Nothing further is asserted here.
- The tile's registry status is `gui_ready` with maturity `ready`. Neither
  claim could be verified against code in this repository.

## Unclear

The entry point does not exist. Nothing implementing this tile could be found.

- The registry (`src/config/launcher_manifest.json`, generated from
  `src/config/models.yaml`) gives the path as
  `src/rate_of_closure/launch_pyqt6.py`. There is no `src/rate_of_closure`
  directory in this repository, and `git ls-files` matches exactly one tracked
  path containing `rate_of_closure`: `assets/logos/rate_of_closure.svg`, the
  tile's icon.
- Being a `provider: "tools"` tile, its path is expected to resolve against an
  external provider pack rather than this repo
  ([ADR-0004](../adr/0004-launcher-provider-migration.md)). In the Tools
  checkout on this machine there is no `src/rate_of_closure/` either, and
  `git ls-files` there matches no `rate_of_closure/` path.
- A repository-wide search of `src/` for `closure_rate`, `face_closure`, and
  `rate of closure` returned only unrelated uses of the word "closure"
  (energy closure, moment closure, issue-closure workflows) - no clubhead
  face-rotation or reference-point-correction implementation.
- No help, physics, or ADR document in this repository defines the rate of
  closure calculation. ADR-0045 references the tile by name but specifies the
  putting integration, not the closure computation.

Files and locations inspected: `src/config/launcher_manifest.json`,
`src/config/models.yaml`, `src/rate_of_closure/` (absent),
`docs/adr/0045-putting-integration-one-experience-two-preserved-stacks.md`,
`assets/logos/rate_of_closure.svg`, and the Tools provider checkout at
`../Tools/src/`.

## See Also
- [ADR-0045: Putting Integration - One Experience, Two Preserved Physics Stacks](../adr/0045-putting-integration-one-experience-two-preserved-stacks.md)
- [ADR-0004: Launcher Provider Migration](../adr/0004-launcher-provider-migration.md)
- [Putting Green](putting_green.md) - the other preserved putting stack
- [Putting Kinematics and Kinetics - Public-Data Review](../physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md)
- [Simulation Controls](simulation_controls.md)
