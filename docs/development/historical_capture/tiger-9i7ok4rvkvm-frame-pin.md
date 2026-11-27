# Tiger Woods Source Frame Pin

This preliminary image-only pin supports #15837 WP-1; it is not a motion fit.
The uploader attributes the source to 2000; its historical provenance is not
independently verified. The selected YouTube video-only rendition is format 136
(`https://www.youtube.com/watch?v=9i7ok4rVkVM`), SHA-256
`6e2893fbcaa8daa45bb2d5e979eeff12e2f57f5c697962d95eb6c48c04219934`.
The raw video remains local. Another rendition requires a new pin.

PyAV 19.0.0 decoded 1,678 frames at 1280 × 720, with PTS timebase 1/30000.
All frames reported progressive; field dominance is therefore not applicable
to this decode, and no deinterlacing was applied. Exact BGR24 consecutive
hash comparison retained 1,672 images. This does not detect near-duplicates or
prove the original film was progressive. The full source-to-output map is a
local JSON artifact, SHA-256
`614864cf8897802b452b211671ba764da6a17475d9d0dff2fcf6f59409bff66a`.
Each source index and PTS remains represented, including duplicates. The
presentation rate and PTS do not establish physical swing timing.

The candidate window is source indices 1080–1500 inclusive, selected from a
30-frame contact sheet as a single profile-view address-through-finish swing.
This is preliminary, not independent phase annotation or verified shot
continuity. Use the full frame, crop `[0, 0, 1280, 720]` in pixel coordinates;
no mirror or rotation is assumed. Intrinsics, lens distortion, camera pose,
perspective and physical scale are unknown and must be fitted or calibrated,
not inferred from this pin.

The candidate model is `sized/tiger_woods/model.xml` and skin is
`sized/tiger_woods/model.visual.json` inside `assets/player_models/player-models.zip`
(archive SHA-256 `b53c8069a9905fcc0b6fa2da87545197425a98ed51b97ef1cba0580788b44216`).
Their respective SHA-256 values are
`c4e88c17c82b9b344e7a0c0f87eb40c79ef21923220417ece157696ccf2bba61` and
`9f224c19c7229c449d8a6237e4844499cff7ddfb44615ab30de924a5c79311b7`.
The model is a stylized candidate, not measured capture-specific morphology.

To reproduce the map with the same source and decoder, run from the repository
root with `src/shared/python` on `PYTHONPATH`:

```python
from pathlib import Path
import json
from shadow_tracker.frame_pin import pin_source
pin = pin_source(Path("path/to/9i7ok4rVkVM.mp4"), output=Path("new-frame-directory"))
Path("frame-map.json").write_text(json.dumps(pin, indent=2) + "\n")
```

`physical_time_verified` remains false and the qualification remains
`image_observations_only`. Human review of the window and field cadence,
and deinterlacing support for interlaced sources remain outstanding before
WP-1 can be called complete. The local output directory contains lossless PNGs
numbered by output index; a failed extraction can leave a partial directory and
must be discarded or inspected before retrying to a new directory.
