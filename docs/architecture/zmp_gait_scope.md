# Scope audit: ZMP / gait stack relevance to stationary golf swing

Closes the investigation requested in issue #2707.

## TL;DR

The `src/robotics/locomotion/` subtree implements a bipedal walking stack
(ZMP, footstep planner, gait state machine, gait types). A golf swing is a
_stationary_ motion: both feet stay planted for the entire swing. None of
these locomotion modules are imported, constructed, or executed anywhere in
the golf-swing simulation path. They exist only as a library and are
exercised exclusively by their own unit tests.

**Recommendation: archive** (move to `archive/locomotion/`). See
[Recommendation](#recommendation) for rationale and why we are _not_
deleting in this PR.

## Inventory

Files under `src/robotics/locomotion/`:

| File                    |     Lines | Exports                                                                            |
| ----------------------- | --------: | ---------------------------------------------------------------------------------- |
| `__init__.py`           |        63 | Re-exports all public symbols                                                      |
| `footstep_planner.py`   |       610 | `FootstepPlanner`, `Footstep`, step-sequence dataclasses                           |
| `gait_state_machine.py` |       391 | `GaitStateMachine`, `GaitState`                                                    |
| `gait_types.py`         |       193 | `GaitType`, `GaitPhase`, `FootState`, `GaitParameters`, `WALKING_PARAMETERS`, etc. |
| `zmp_computer.py`       |       415 | `ZMPComputer`, `ZMPResult`, `SupportPolygon`, `CapturePoint`                       |
| **Total**               | **1 672** |                                                                                    |

Associated tests:

- `tests/unit/robotics/test_locomotion.py`
- `tests/unit/dbc/test_dbc_zmp_robotics.py`

## Wiring

### Imports (production code)

```
src/robotics/__init__.py
    └─ re-exports FootstepPlanner, GaitStateMachine, ZMPComputer, ...
src/robotics/locomotion/__init__.py
    └─ re-exports the same symbols from submodules
src/robotics/locomotion/footstep_planner.py
    └─ from src.robotics.locomotion.gait_types import GaitParameters
src/robotics/locomotion/gait_state_machine.py
    └─ from src.robotics.locomotion.gait_types import (...)
```

All production imports are **internal to `src/robotics/locomotion/`** or
are re-exports from the `src/robotics/` facade. No module outside
`src/robotics/locomotion/` constructs a `ZMPComputer`, `GaitStateMachine`,
or `FootstepPlanner`.

### Callers in the swing pipeline

A full-tree grep for `ZMPComputer`, `GaitStateMachine`, `FootstepPlanner`,
`compute_zmp`, `plan_footsteps`, `support_polygon` turns up **zero** hits
in:

- `src/engines/` (physics backends)
- `src/pipelines/` (simulation drivers)
- `src/shared/python/` (analysis, biomechanics, GUI)
- Any notebook or script that runs a swing.

The only non-test hit is documentation (`docs/UPSTREAM_DRIFT_USER_MANUAL.md`)
which shows the public API in a usage example — not an actual call site.

### Conclusion on usage

The locomotion package is **dead code with respect to the golf-swing use
case**. Its only runtime consumers are its own tests. The facade re-export
(`from src.robotics import ZMPComputer`) is available to downstream users
but is not exercised by any code path in this repository.

## Physics review (from issue #2707)

Issue #2707 documents eight correctness problems in the current ZMP/gait
implementation even in its nominal walking domain. Summarised:

1. `_estimate_com_acceleration()` returns `np.zeros(3)` — ZMP degenerates
   to CoM-on-ground projection, wrong during any acceleration (downswing
   peak `|a_com| > 5 m/s²`).
2. Support polygon is a hard-coded 30x45 cm rectangle — no API to pass the
   real foot pose or stance width, so golfer stance-width effects (driver
   vs. wedge) are invisible.
3. No weighted double-support formula — a swing is asymmetric double
   support (~65/35 at top, ~20/80 at finish) and is simply undefined in
   the current code.
4. Capture point uses the single-support ICP formula unconditionally; it
   returns garbage in double support.
5. Gait FSM is time-driven, not event-driven; it "advances" through
   walking phases regardless of actual foot contact state.
6. Footstep planner emits steps with no IK reachability check.
7. Yaw extraction from quaternion lacks gimbal-lock handling.
8. The module that _would_ be relevant to a swing — an address-pose
   composer from `ClubSpec` + `TargetShot` — does not exist.

Items 1-7 are bugs in the walking-domain behaviour; item 8 is the
observation that the locomotion stack does not address the swing use case
at all.

## Relevance matrix

| Module             | Relevant to walking | Relevant to stationary swing  | Currently called from swing code |
| ------------------ | ------------------- | ----------------------------- | -------------------------------- |
| `ZMPComputer`      | Yes (buggy)         | No (double-support undefined) | No                               |
| `GaitStateMachine` | Yes (buggy)         | No (no gait phases in swing)  | No                               |
| `FootstepPlanner`  | Yes (buggy)         | No (feet don't move)          | No                               |
| `gait_types`       | Yes                 | No                            | Only by the other three          |

The intersection of "used by swing pipeline" and "correct enough to rely
on" is empty.

## Recommendation

**Archive the locomotion subtree to `archive/locomotion/` in a follow-up
PR.** Reasoning:

- The code is not called by any swing simulation path, so removing it from
  the active source tree does not break the product.
- It is not correct enough, even for walking, to be a credible foundation
  for future work — fixing items 1-7 from #2707 would be a rewrite, not a
  patch.
- A `ClubSpec`-driven `golf_stance` composer (item 8 in #2707) is a new
  module; it does not need or benefit from the current walking code.
- Archiving (rather than deleting) preserves git history in-tree for
  anyone who later wants to salvage the ZMP math or the gait FSM
  skeleton, without paying the import-surface, test-maintenance, and
  CI-time cost on `main`.

### What archiving looks like

1. `git mv src/robotics/locomotion archive/locomotion`
2. Move `tests/unit/robotics/test_locomotion.py` and
   `tests/unit/dbc/test_dbc_zmp_robotics.py` alongside or delete them.
3. Remove the re-exports from `src/robotics/__init__.py`
   (lines 88-98, 175-182).
4. Update `docs/UPSTREAM_DRIFT_USER_MANUAL.md` section referring to the
   locomotion API (around line 9407).
5. Close the eight sub-issues from #2707 as "won't fix — archived" once
   the move lands.

### Why not in this PR

This PR is a **scope-audit / documentation deliverable only**. Deleting
or moving 1 672 lines of code plus its tests and facade exports is a
surgical change that deserves its own review, CI run, and rollback
anchor. Mixing the audit with the move would make both harder to review.

### Alternative: keep with a "future-use" note

If leadership decides the walking stack is strategic for a future
biped-gait product line (outside the golf swing), the alternative is:

- Add a module docstring to `src/robotics/locomotion/__init__.py` stating
  "not used by the golf-swing pipeline; retained for future biped work".
- Gate the eight acceptance criteria in #2707 behind a dedicated epic.
- Exclude the subtree from coverage floors so it does not drag the ratchet.

This path is viable but the author of this audit recommends **archive**
based on current product direction (stationary swing only, no announced
biped roadmap).

## Follow-up tickets

- Surgical-move PR to execute the archive recommendation (file after
  leadership sign-off on this audit).
- New issue: "Address-pose composer (`golf_stance`) from ClubSpec +
  TargetShot" — tracks acceptance criterion 8 from #2707, independent
  of the archive decision.
