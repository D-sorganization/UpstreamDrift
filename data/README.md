# Capture Registry and Datasets

This directory contains public capture assets and the public registry manifest for motion capture and club datasets across UpstreamDrift.

## Overview of the Capture Registry

The capture registry (`data/capture_registry.json`) provides a single, unified source of truth for resolving motion capture files and club workbooks in Python and MATLAB. Instead of hard-coding file paths across tools and physics engines, callers resolve files through neutral capture identifiers.

## Neutral Capture Identifiers

Datasets and workbooks are referenced solely by neutral identifiers:

- `capture-A`: Reference tour-average driver capture (`data/C3D_TA_Driver.c3d`, 360 Hz, 654 frames, public).
- `capture-B`: Reference tour-average iron capture (`data/C3D_TA_Iron.c3d`, 360 Hz, 657 frames, public).
- `capture-O`: Repository owner's driver swing capture (240 Hz, 367 frames, private).
- `club-workbook-main`: Main club data workbook (`e91f760171f5.xlsx`, private).
- `club-workbook-wiffle`: Wiffle ProV1 3D club data workbook (`d61999450688.xlsx`, private).

## Private Data And `CAPTURE_DATA_DIR`

Public captures (`capture-A`, `capture-B`) reside directly in the repository under `data/` and are available in public CI without special configuration.

Private datasets (`capture-O`, `club-workbook-main`, `club-workbook-wiffle`) are not stored in this repository. They come from the repository owner's private data repository. To access private captures, set the `CAPTURE_DATA_DIR` environment variable to point to the local checkout of the private data repository:

```bash
export CAPTURE_DATA_DIR="/path/to/private/data"
```

In PowerShell:

```powershell
$env:CAPTURE_DATA_DIR = "C:\path\to\private\data"
```

When `CAPTURE_DATA_DIR` is unset or private files are not present, test suites automatically skip tests requiring private data rather than failing.

## Resolving Captures in Python and MATLAB

### Python

```python
from src.motion_capture.capture_registry import resolve_capture, require_capture

# Resolve a public capture to its verified local path
path_driver = resolve_capture("capture-A")

# Resolve or skip cleanly in pytest if private data is unavailable
path_wiffle = require_capture("club-workbook-wiffle")
```

### MATLAB

```matlab
% Resolve a capture via resolve_capture.m
driver_path = resolve_capture("capture-A");
```

## Integrity Verification

The resolver verifies SHA-256 digests against `data/capture_registry.json` on resolution to guarantee file integrity across platforms and engines.
