# Capture-O Video Companion Procedure

Epic: [#11268](https://github.com/D-sorganization/UpstreamDrift/issues/11268).
Parent program: [#11161](https://github.com/D-sorganization/UpstreamDrift/issues/11161)
(owner swing capture `capture-O`). Consumers: Necromatcher #11232/#11235, Tiger
#11226, Hogan #11229. Development log entry: `DL-#11268`.

This runbook is for fleet agents executing the `tier:cli` children locally.
Read the epic, the child issue and `AGENTS.md` first. The issue bodies hold the
red tests and acceptance criteria; this page holds the shared operating rules,
layout and commands so they are not repeated per child.

## What This Project Is

The owner has a video album recorded at the marker capture session behind
`capture-O`. The owner is the only subject with both ordinary video and a
labelled marker capture taken the same day, with the same club, in the same
place. Measuring markerless and Necromatcher reconstructions against that
marker capture gives an error budget the historical-player projects can cite.

## Imperfect-Data Rules

- Salvage first: grade every clip (A, B, C or R with reasons) and use it at
  the comparison level it supports. Degraded results are valid results.
- Video and markers are not synchronized, and a video may not show a captured
  swing. Pairing is inferred with confidence and abstention (COV-6), never
  assumed.
- Presentation time is not physical time unless slow-motion or frame-rate
  evidence says so. With an unknown clock, report phase-normalized time and
  no velocities.
- Freeze the comparison protocol (COV-3) before looking at comparison numbers.
  Never loosen a threshold to obtain a pass.
- Never label inferred quantities as observed, and never synthesize a missing
  observation.

## Comparison Levels

| Level | Requires | Claim |
| --- | --- | --- |
| L0 | Any graded A–C swing | Visual agreement only |
| L1 | Fitted virtual camera | Inside or outside the 13-swing capture envelope |
| L2 | Paired swing and time mapping | Per-frame 2D agreement with that swing |
| L3 | L2 and a 3D source | 3D joint position and angle agreement |

COV-3 ratifies or amends these; its protocol document supersedes this table.

## Privacy

- Owner media, decoded frames, overlays, per-frame numbers and the album
  locator stay under `$CAPTURE_DATA_DIR/capture-O-video/`. They never enter Git,
  issues, pull requests, logs or public screenshots.
- Public text uses neutral ids only: `capture-O`, `cov-NN`, `cov-NN-sK`,
  `subject-O`. No vendor names, personal names or private paths.
- Set `NECROMATCHER_LIBRARY_ROOT` inside the private store before any
  Necromatcher import (COV-8).
- Public summaries need owner approval and contain only aggregate,
  body-height-normalized numbers.

## Private Store Layout

```text
$CAPTURE_DATA_DIR/capture-O-video/
  PRIVATE_SOURCE.md        album locator, consent, retention, owner recollection
  originals/               untouched downloaded bytes, read-only
  originals.sha256
  ffprobe/                 per-file ffprobe JSON
  acquisition_receipt.json COV-1
  review/                  COV-2 review sheet and thumbnails
  swing_windows.json       COV-2
  grades.json              COV-2
  derived/
    camera/                COV-4 cameras and envelopes
    observations/<backend>/<cov-NN-sK>/   COV-5 runner outputs
    pairing/               COV-6 matrix and time mappings
    comparison_2d/         COV-7
    comparison_3d/         COV-9
    simscape/              COV-10
  necromatcher/            COV-8 private library root
  report/                  COV-11 private report and error budget
```

Output directories are never overwritten. Rerun into a new directory and keep
the earlier receipts.

## Environment

Python 3.11+ with the pinned Tools gitlink. Always run headless.

```powershell
git submodule update --init vendor/ud-tools
python3 -m pip install -e '.[historical-capture]'
python3 -m src.shared.python.pose_estimation.mediapipe_models --variant full
$env:QT_QPA_PLATFORM='offscreen'; $env:MPLBACKEND='Agg'; $env:MUJOCO_GL='egl'
$env:CAPTURE_DATA_DIR='<private data repository checkout>'
```

Detector weights never download silently. Fetch each backend's weights
explicitly and record their SHA-256 in the receipt.

## Hosts

| Children | Host |
| --- | --- |
| COV-1 | Fleet machine able to open the owner's album; cloud sandboxes cannot |
| COV-2, COV-4, COV-6, COV-7, COV-11 | Any fleet machine with the private store |
| COV-5 monocular 3D, COV-8, COV-9 | GPU fleet machine preferred |
| COV-10 | Windows host with MATLAB R2025b at its explicit path |

For COV-10 launch `C:/Program Files/MATLAB/R2025b/bin/matlab.exe` explicitly
and follow `docs/development/simscape_tour_matching/REMOTE_EXECUTION.md`. A
missing runtime or license is a blocked outcome, not a pass.

## Reuse Map

| Need | Existing authority |
| --- | --- |
| Capture id resolution | `src/motion_capture/capture_registry.py` (#11162) |
| Swing metrics | `src/shared/python/swing_comparison/` (#11164) |
| Decoding, frame identity | `src/shared/python/shadow_tracker/historical_capture.py`, `source_records.py` |
| Runner | `scripts/historical_capture.py`, extended with `--estimator` in COV-5 |
| Detectors | `src/shared/python/pose_estimation/registry.py`, `src/tools/hmr2_sidecar/` |
| Canonical observations | `src/shared/python/motion_pipeline/contracts.py` and `sources/` adapters |
| Projection into a camera | `src/motion_capture/reference/registration.py` |
| Time mapping | `src/motion_capture/reference/synchronization.py` |
| Keypoint offsets | `src/shared/python/pose_estimation/keypoint_offsets.py` |
| Metrics | `pose_estimation/validation_metrics.py`, `shadow_tracker/evaluation.py` |
| Necromatcher | `src/shared/python/workspace/necromatcher*.py` public services |
| Simscape replay and overlays | `motion_matching/simscape_replay_harness.py`, `src/motion_capture/simscape_c3d_video_overlay.py` |

Extend these with tests. Do not add a second parser, projector, runner, metric
module or ledger.

## Per-Child Workflow

1. Check and post the issue lease, then create an isolated worktree from
   current `main`.
2. Write the issue's red tests, run them and keep the failing output.
3. Implement the minimum, refactor and rerun the focused tests.
4. Run Ruff check, Ruff format check, the file-size budget and the
   error-handling ratchet.
5. Run the real-data step on the fleet host; keep outputs private.
6. Update `docs/development/HANDOFF.md`, the `DL-#11268` entry and one SPEC.md
   change-log row keyed by the pull request.
7. Open a draft pull request with `Refs #11268`; comment aggregate results on
   the child issue. If data or a host is unavailable, leave a status comment
   and keep the issue open.

## Completion

The epic's checklist defines done. The owner closes the epic after reviewing
the COV-11 error budget. Work that needs a new synchronized capture goes to the
existing deferred-validation authority, never into a silent close.
