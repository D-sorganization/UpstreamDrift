# Historical Player Capture Procedure

## Scope and Authority

Track Tiger 2000 in #11226, Ben Hogan in #11229, and the shared runner in #11230.
Reuse Shadow Tracker source/frame identity and the existing MediaPipe estimator.
The current runner exports image observations. It does not recover club landmarks,
calibrate cameras, verify subject identity, fit anatomical 3D motion, or qualify dynamics.
A detector can select a bystander; inspect every selected swing and its missingness.

## Acquire and Preserve Sources

Keep original media outside Git. Preserve filename, source link, original SHA-256,
uploader description, event/year evidence, rights, and film lineage in source-catalog.json.
Do not infer recording year from upload date. The supplied Tiger compilation claims
2000 in its title and description; event-by-event verification remains required.
Use one source identity per content hash. Keep crops/re-encodes of an original in
one film lineage and one train/validation/test partition.

The requested Tiger source was downloaded with yt-dlp 2026.8.19:

```powershell
python3 -m yt_dlp --no-playlist --js-runtimes node --write-info-json --write-description -f 299 -o "$env:USERPROFILE/Downloads/historical-capture/tiger_2000/%(id)s.%(ext)s" 'https://www.youtube.com/watch?v=t_J6Vik3Tss'
```

Format 299 is 1080p50 video only; format 140 was downloaded separately and both
were remuxed without transcoding into t_J6Vik3Tss-with-audio.mp4. Analyze the
original video-only bytes identified in the catalog; muxing changes file identity.

## Install and Run

Use Python 3.11+ and the exact Tools gitlink. Install the historical-capture extra
in the project environment. The library never silently downloads detector weights.

```powershell
git submodule update --init vendor/ud-tools
python3 -m pip install -e '.[historical-capture]'
python3 -m src.shared.python.pose_estimation.mediapipe_models --variant full
python3 -m scripts.historical_capture SOURCE.mp4 OUTPUT_DIRECTORY --subject ben-hogan --start 110 --end 135
```

For an uninstalled source checkout, add the repository src directory to PYTHONPATH
so the canonical bunkershot3d package resolves. OUTPUT_DIRECTORY must be new;
failed partial directories have no completion receipt and must not be treated as
successful. Never overwrite evidence while changing code, settings, or models.

Intervals are half-open presentation seconds. PyAV seeks to the preceding
keyframe and streams frames, retaining exact rational container PTS. Physical
clock remains unknown. The detector's millisecond clock is only an inference API
input. Slow motion, interpolated frames and repeated stills need separate review.

Each run contains lossless original decoded PNGs, observations.jsonl and
receipt.json. Frame hashes bind decoded BGR pixels and decoder provenance;
receipt hashes bind source bytes, observation bytes, detector weights and runner
implementation. Model-relative Z and detector-computed joint angles are excluded.
Normalized XY and per-landmark visibility are preserved, including offscreen
coordinates and unknown visibility. Missing detections are empty observations.
PNG compression level 1 reduces CPU cost without changing decoded pixels.

## Review and Continue Reconstruction

1. Inspect an overlay throughout each window, identify every cut, and split it
   into continuous swings. Do not label an unreviewed interval a qualified shot.
2. Review P1–P10 checkpoint frames and adjacent impact frames. Preserve original
   hashes, visibility, annotation revision and view-dependent ambiguity.
3. Resolve original event/year lineage, physical playback scaling and rights.
   Do not derive physical velocities from presentation time without evidence.
4. Fit cameras and body dimensions jointly, using declared anthropometric priors
   and uncertainty. Single-camera depth remains ambiguous.
5. Fit constrained dense kinematics via existing pose/camera/model interfaces;
   review feet, hand-club closure, ROM and alternative depth hypotheses.
6. Export via existing native adapters; obtain MuJoCo, Drake and Pinocchio FK
   comparison receipts. Unsupported coordinates must remain unsupported.
7. Run fresh uninterrupted forward replay and physics gates for dynamics claims.
   Use MATLAB R2025b where the Simscape lane is required.
8. Evaluate lineage-separated held-out swings before any subject adaptation.
9. Integrate eligible artifacts with UpstreamDrift and AffineDrift. Publication
   requires source permission and the applicable scientific/product gates.

## Export a Bound Native Research Overlay

After capture import and native research-fit storage, reuse the workspace facade
to review the actual model against its exact original source frames. Run from
the owned repository root in a clean SDK process; the documented Windows worker
boundary imports MuJoCo before workspace code. For an uninstalled checkout,
configure this repository's `src` on `PYTHONPATH` as above. Set an unused output
directory; export never overwrites an earlier review version.

```python
from pathlib import Path

import mujoco  # Initialize the native SDK in the clean process.
from src.shared.python import workspace

library = workspace.default_necromatcher_library()
destination = Path.home() / "Desktop" / "Tiger Overlay New Version"
manifest = workspace.export_fit_video(
    library,
    "tiger-authored-ground-fit-v6",
    destination,
    selected_frames=(0, 103, 209),
)
```

For Hogan, use `hogan-authored-ground-fit-v6` with selected frames `(0, 549, 749)`
and another new destination. These immutable IDs refer to stored authored-ground
research hypotheses. Preserve rejected research qualification rather than
relabelling export as accepted reconstruction.

The exporter draws on source-sized original PNG backgrounds. Its blue native
body-origin joint tree and attachment markers are not a body mesh or an observed
silhouette. Green detector observations and yellow residuals separate observed
image inference from model projection. Overlays and video encoding do not alter
the original capture archive. The output includes `overlay.mp4`, selected
losslessly verified overlay PNGs and `manifest.json`, with exact original PTS,
frame/asset/output hashes, recorded missing rig data and physical-time
qualification false. Source frames must be contiguous with a uniform rational
presentation rate; irregular or sparse clocks are rejected. Original frame PTS
remain authoritative in the manifest, not the exported MP4's playback labels.

Current deliverables are in
`C:/Users/diete/Desktop/Necromatcher Review 2026-10-01/Tiger Overlay` and
`C:/Users/diete/Desktop/Necromatcher Review 2026-10-01/Hogan Overlay`.
The `Report` subfolder holds the compiled methods report; its editable
standalone source is [Necromatcher Methods](necromatcher-methods.tex). The report
documents source hashes, equations, failed constraints and repeatability; it is
separate from the canonical engineering design manual. API/UI export execution
controls remain under #11246 and report/Desktop delivery under #11247.

## Continue Sequential Constrained Repair

The current uncommitted `workspace.repair_native_motion(binding, iterations=150)`
method uses canonical native `solve_trajectory` with bounded TRF. Initialize the
first solve from the first source pose; subsequent solves use the previous
repaired pose as initialization and weak prior. Reports retain
`initialization=previous_repaired_pose` and
`prior_target=previous_repaired_pose_or_first_source_pose`. Targets are still
world marker positions inferred from the original native fit, not measured 3D
motion. The twelve-test repair regression checks this method/metadata change.
There is no new formal whole-track receipt for this uncommitted change yet;
do not rewrite the earlier independent LM/TRF results or promote the exploratory
sequential Tiger run to scientific acceptance.

After any repair, compare original image residuals, authored bounds, grip and
ground at all source frames and between them. Discrete closure and bounded
samples do not qualify a continuous interpolated motion. Keep prior versions
and actual failed midpoint checks. Timing, historical effort identification,
independent replay, downstream analysis and site integration remain open.

## Validation Evidence

Tests were written and run before implementation: absent module produced a
collection error; the streaming-export test then failed on the absent function.
The first streaming implementation exposed local-URI rejection and decimal
interval-boundary errors. Both were corrected without weakening expectations.
The focused suite passed 35 tests before the final metadata refinements. Synthetic
video tests prove source/frame contracts and lossless image hash round trips;
they are not Hogan/Tiger biomechanical validation. Current exact commands and
results belong in AGENT_HANDOFF.md and the per-run receipts.

## CI Receipt Refactor

PR #11231 separates receipt writing from frame extraction to satisfy the 100-line function budget. The saved captures retain their original implementation hash; they were produced before this refactor. Twelve capture tests passed after the refactor.
