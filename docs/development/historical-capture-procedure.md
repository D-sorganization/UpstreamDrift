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

## Validation Evidence

Tests were written and run before implementation: absent module produced a
collection error; the streaming-export test then failed on the absent function.
The first streaming implementation exposed local-URI rejection and decimal
interval-boundary errors. Both were corrected without weakening expectations.
The focused suite passed 35 tests before the final metadata refinements. Synthetic
video tests prove source/frame contracts and lossless image hash round trips;
they are not Hogan/Tiger biomechanical validation. Current exact commands and
results belong in AGENT_HANDOFF.md and the per-run receipts.
