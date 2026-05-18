# [Tracking] Multi-source motion targets — full-body C3D + ball-aware club + animated preview

This is the umbrella tracking issue for a 12-issue effort to bring **full-body mocap** and **ball-aware club kinematics** into the canonical motion-matching pipeline, and to surface them in the matcher GUI as an **animated, multi-source preview**.

## Goal

Today the motion-matching pipeline knows about **only** a 6-DOF club trajectory (`ClubTarget`). Body markers in the four C3D files in the repo are parsed but immediately thrown away by the loader, the matcher GUI shows only static event-frame snapshots, and the source-revealing names of the upstream mocap supplier and the swing-data dataset are baked into module names, directory names, and class names.

After this effort:

- A new `BodyTarget` and `ClubBallTarget` sit alongside `ClubTarget` as canonical, frozen, validated targets.
- A `MultiSourceTarget` aggregator lets cost functions and the GUI reason about any combination of club / club+ball / body.
- Loaders dispatch by file extension across `.xlsx`, `.c3d`, and `.mat`.
- The matcher GUI animates the full trajectory with a timeline scrubber, layer toggles, and a source-toggle panel.
- Every code identifier, directory, docstring, and UI string is source-agnostic. Data filenames stay as-is.
- Three duplicate C3D readers collapse to one canonical module.

## Children

| #   | Title                                                                                              |
| --- | -------------------------------------------------------------------------------------------------- |
| 1   | feat(motion-matching): introduce `BodyTarget` canonical contract for full-body marker trajectories |
| 2   | feat(motion-matching): C3D body-marker loader producing `BodyTarget`                               |
| 3   | feat(motion-matching): `.mat` club-target loader (TW/GW × ProV1/Wiffle)                            |
| 4   | feat(motion-matching): `ClubBallTarget` — club kinematics + ball impact boundary condition         |
| 5   | refactor(motion-matching): rename source-revealing identifiers and directories to generic names    |
| 6   | feat(starting-pose-matcher): animated full-trajectory marker preview with timeline scrubber        |
| 7   | feat(starting-pose-matcher): source-toggle UI — choose Club, Club+Ball, Body, or any combination   |
| 8   | feat(motion-matching): body-skeleton segments — connect anatomical markers into a stick figure     |
| 9   | refactor(c3d): consolidate three duplicate C3D readers into one canonical module                   |
| 10  | feat(launcher): expose multi-source motion-target preview as a launcher tile                       |
| 11  | test(motion-matching): integration tests + golden fixtures for body / club / ball pipeline         |
| 12  | docs(motion-matching): ADR + user guide for multi-source motion targets                            |

## Dependency graph

```
1 BodyTarget contract ──────────┬───► 2 C3D body loader ──┐
                                ├───► 8 Body skeleton ────┤
                                │                         │
4 ClubBallTarget ───────────────┤                         ├──► 6 Animated preview
                                │                         │     (matcher GUI)
3 .mat club loader ─────────────┤                         │
                                │                         │
9 Unify C3D readers ────────────┘                         │
                                                          │
                                                          ▼
                                                    7 Source-toggle UI
                                                          │
                                                          ▼
                                                    10 Launcher tile
                                                          │
                                                          ▼
                                                    11 Integration tests
                                                          │
                                                          ▼
                                                    12 ADR + user guide

5 Generic rename ─── runs in parallel; lands as a coordinated PR
```

## Branching policy

GAAI fleet repos target `main`. Use focused topic branches: `feat/body-target-contract`, `feat/c3d-body-loader`, `refactor/generic-source-names`, etc.

## Generic-naming policy (binding for all child issues)

- File data on disk (`.c3d`, `.mat`, `.xlsx`) keeps its existing filename.
- Every other identifier — directory, module, class, function, error message, log message, docstring, UI string — must be source-agnostic. No vendor names, lab names, person names, or study names.
- Compatibility shims with `DeprecationWarning` cover one release for renamed module imports.

## Out of scope

- Real-time launch-monitor (Trackman / FlightScope / GCQuad) data ingest.
- Cost-function changes that use the body or ball boundary terms (separate follow-up after the contracts and the matcher view are stable).
- WebGL / Tauri rendering of the matcher view; this effort stays on matplotlib / PyQt6.
