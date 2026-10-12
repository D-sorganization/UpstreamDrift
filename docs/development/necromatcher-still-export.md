# Source-Bound Research Still Export

## Purpose and Authority

`workspace.export_fit_stills` exports selected source-sized PNGs through the same
canonical native skeleton, body observations, residuals, optional shaft axis,
translucent model proxies and caption composition as `export_fit_video`. It
renders only selected frames and uses no video codec or frame-rate inference.
This engineering procedure is separate from the canonical QMD design manual;
manual release remains blocked. Display agreement is not historical, anatomical,
camera, surface-occlusion, physical-clock or clinical-ROM qualification.

## Repeatable Invocation

Use a clean process with the configured native SDK import order, reviewed source
producer/runtime pins, and an exclusive output directory. Load reviewed evidence
through the canonical typed parser; the exporter freshly rebinds its capture,
original PNGs, clock and camera against the selected saved fit. No caller-created
DTO alone authenticates original source evidence.

```python
import json
from pathlib import Path

import mujoco  # Windows SDK-first protocol before workspace imports.
from src.shared.python.motion_matching.historical_fit import ShaftAxisEvidence
from src.shared.python.workspace import (
    CaptionOverlayOptions,
    NecromatcherLibrary,
    ShapeOverlayOptions,
    export_fit_stills,
)

library = NecromatcherLibrary(library_root)
evidence = ShaftAxisEvidence.from_record(
    json.loads(Path(reviewed_evidence_path).read_text(encoding="utf-8"))
)
manifest = export_fit_stills(
    library,
    saved_fit_id,
    Path(exclusive_output_directory),
    selected_frames=(0, 150, 209),  # Example: reviewed Tiger source indices.
    shaft_evidence=evidence,
    shape_overlay=ShapeOverlayOptions(0.35),
    caption_overlay=CaptionOverlayOptions(),
)
```

The eight-argument public interface also permits disabling all optional layers.
`selected_frames` must be nonempty and contain unique strict integer indices in
the saved fit. Requested order is retained. The exporter compiles one binding and
opens one verified capture archive for the selected set. Model proxies and shaft
diagnostics reuse that binding and its saved camera. Shapes are display proxies;
0.35 opacity is authored display configuration, not a fitted body parameter.

## Output and Verification

The exclusive directory contains `frame-NNNNNN.png` files and a completion
`manifest.json` with schema `necromatcher/source-overlay-stills/1`. Each PNG is
losslessly read back before publication. The manifest contains original frame
identities and PNG digests, fit/model/capture hashes, native/body/shaft display
diagnostics, optional layer provenance and output hashes. `frame_pts` records
each exact reduced rational container timestamp, including nonuniform source
intervals. There is no MP4, video codec or inferred source frame rate.

Per-frame composition and diagnostic records match selected video frames with
identical layer options. Sparse shaft abstentions stay explicit; observed interior
fragment points are not physical endpoints. Raw shaft diagnostics stay separate
from body metrics. Captions retain research status and unknown physical time.
Whole source/runtime/library before/finally preservation and reviewed producer
admission remain the responsibility of a separately authorized orchestration
wrapper; this exporter does not itself claim those execution-wide brackets.

Fit, capture/model parents and optional shaft binding are revalidated immediately
before exclusive publication. A failed render, source change or PNG readback
does not publish the directory. Existing output directories are never overwritten.
Do not blindly retry a failed historical operation: retain its owned receipt and
inspect actual artifact state first. No optimization or library write occurs.

## Validation Scope

`test_necromatcher_stills.py` verifies codec-free selected-only composition,
strict indices/options, one compiled binding, lossless PNG hashes, nonuniform
rational PTS, exact video/still raster and diagnostic parity (including shaft
abstention, translucent proxies and compact captions), malformed source identities,
fit/capture mutation, readback failure, exclusive output and SDK-free facade import.
These native fixtures are synthetic and establish software behavior only.

Layer NPZ output is intentionally outside this slice: no second renderer or
duplicate surface-layer evaluation was introduced. A still diagnostic does not
require passing a full-video throughput budget and does not qualify full video.
Any subsequent historical visual stage requires reviewed source publication,
fresh complete preservation baselines and separate root authorization.
