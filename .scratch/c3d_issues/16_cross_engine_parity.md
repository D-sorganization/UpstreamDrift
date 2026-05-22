# [Tracking] Cross-engine motion-matching parity — Drake / MuJoCo / Pinocchio / OpenSim / Simscape / Pendulum

## Why

The motion-matching pipeline is now end-to-end on the **target** side: load a `MultiSourceTarget` (club + ball + body) from xlsx / `.mat` / `.c3d`, render it live in the matcher view. The next bottleneck is the **engine** side: each physics engine has its own `motion_matching/` directory with partial, non-uniform plumbing for consuming a target and producing a fitted theta trajectory.

We need every supported engine to:

1. Accept a `MultiSourceTarget` (or `ClubTarget` / `BodyTarget` slice when the engine doesn't yet know about the new types).
2. Run the same `fit_swing` API: `fit_swing(target, opts) -> FitResult`.
3. Emit the same diagnostics: per-frame error timecourse, fit-quality card, leaderboard row.
4. Be discoverable by the engine-discovery logic so the launcher's Motion-Match Preview tile can drive it.

This issue is the umbrella; child issues land per-engine implementation work.

## Current state (audit)

| Engine                                                                                               | motion_matching dir | `fit_swing` API? | Consumes ClubTarget?                   | Consumes BodyTarget? | Notes                                                        |
| ---------------------------------------------------------------------------------------------------- | ------------------- | ---------------- | -------------------------------------- | -------------------- | ------------------------------------------------------------ |
| Drake (`src/engines/physics_engines/drake/python/motion_matching/`)                                  | yes                 | partial          | partial                                | no                   | uses local Excel adapter; needs `MultiSourceTarget` plumbing |
| MuJoCo (`.../mujoco/python/motion_matching/`)                                                        | yes                 | partial          | partial                                | no                   | recent `synthesize.py` work — extend                         |
| Pinocchio (`.../pinocchio/python/motion_matching/`)                                                  | yes                 | partial          | partial (via `club_target_adapter.py`) | no                   | most mature; reference for others                            |
| OpenSim (`.../opensim/python/motion_matching/`)                                                      | yes                 | partial          | partial                                | no                   | prescribed-controller path needs refresh                     |
| MyoSim (`.../myosuite/python/motion_matching/`)                                                      | unknown             | unknown          | unknown                                | unknown              | needs audit                                                  |
| Simscape 3D (MATLAB) (`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/`) | yes                 | yes              | yes                                    | partial              | MATLAB layer — different surface                             |
| Simscape 2D (MATLAB)                                                                                 | partial             | unknown          | unknown                                | unknown              | needs audit                                                  |
| Pendulum (`.../pendulum/python/motion_matching/`?)                                                   | unknown             | unknown          | unknown                                | unknown              | needs audit                                                  |
| Pendulum models                                                                                      | similar             | unknown          | unknown                                | unknown              | needs audit                                                  |

## Children to file (one issue each)

| #   | Title                                                                                                           | Effort |
| --- | --------------------------------------------------------------------------------------------------------------- | ------ |
| A   | feat(motion-matching): canonical engine-side `fit_swing(target, opts) -> FitResult` API + provider registry     | M      |
| B   | feat(drake): adapt motion-matching to `MultiSourceTarget`; expose `fit_swing` provider                          | M      |
| C   | feat(mujoco): adapt motion-matching to `MultiSourceTarget`; expose `fit_swing` provider                         | M      |
| D   | feat(pinocchio): adapt motion-matching to `MultiSourceTarget`; bring forward existing `club_target_adapter`     | M      |
| E   | feat(opensim): adapt motion-matching to `MultiSourceTarget`; refresh prescribed-controller path                 | M      |
| F   | audit + feat(myosim): motion-matching plumbing audit + first-pass implementation                                | M      |
| G   | feat(matlab/simscape-3d): consume `BodyTarget` via .json bridge from Python                                     | M      |
| H   | audit(matlab/simscape-2d): identify motion-matching gaps + file follow-ups                                      | S      |
| I   | audit(pendulum): identify motion-matching gaps + file follow-ups                                                | S      |
| J   | feat(motion-matching): cost-function term that uses BodyTarget marker error (foot-contact, hand position, etc.) | L      |
| K   | feat(motion-matching): leaderboard rows for each engine + "compare against measured C3D" view                   | M      |

## Generic-naming policy

Carries forward from #4475: no vendor / lab / person / study names anywhere in code, docstrings, error messages, or test names. Engine names (Drake / MuJoCo / Pinocchio / OpenSim / MyoSim / Simscape / Pendulum) are obviously fine — those are the physics engines themselves.

## Sequencing

A is the foundation issue (canonical engine-side API contract). It must land first. B–F can run in parallel after A. G depends on the BodyTarget JSON bridge (separate small task). H–I are small audits that can run anytime. J is a follow-up after the providers exist. K is the user-facing capstone.

## Reference

- `src/shared/python/motion_matching/` — target side (now stable on `main`)
- `docs/adr/0018-multi-source-motion-targets.md` — design record
- `docs/user_guide/motion_matching/loading_targets.md` — end-user guide
- Existing per-engine code: `src/engines/physics_engines/<engine>/python/motion_matching/`

## Out of scope

- Real-time launch-monitor data ingest (separate effort).
- Replacing the matplotlib matcher view with a GPU-accelerated viewer.
- Any change to physics-engine internals other than the motion-matching adapter and `fit_swing` API.
