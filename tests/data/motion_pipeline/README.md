# Motion Pipeline — Golden Fixtures

> Part of issue #4571 gap-fill. Each fixture in `golden/` is intentionally tiny
> (<= 50 KB), deterministic, and synthetic. They exist to exercise the
> source-format adapters in `src/shared/python/motion_pipeline/sources/` —
> not to provide biomechanically realistic captures.

## Regenerating

```bash
python3 tests/data/motion_pipeline/_generate.py
```

The generator uses fixed seeds and `numpy.linspace`/`numpy.sin`, so the output
should reproduce identically (modulo float rounding).

## Fixture Inventory

| File | Format | Schema | Frames | Markers/Landmarks | Notes |
|------|--------|--------|-------:|-------------------|-------|
| `sample.bvh` | BioVision Hierarchy | 5-joint chain | 30 | — | 6 root + 4×3 child channels, 60 fps |
| `sample.trc` | OpenSim TRC | 6 markers | 30 | RASI, LASI, RPSI, LPSI, RKNE, LKNE | meters, 60 fps |
| `sample.mot` | OpenSim Coordinates | 5 DoF | 30 | — | inDegrees=yes |
| `sample.sto` | OpenSim States | 5 states | 30 | — | uses `/jointset/*` paths |
| `openpose_keypoints.json` | OpenPose | BODY_25 | 30 | 25 | array-form multi-frame |
| `alphapose.json` | AlphaPose | COCO_17 | 30 | 17 | per-frame detections |
| `hrnet.json` | HRNet | COCO_17 | 30 | 17 | single-person |
| `sample.csv` | Generic CSV | 5 joints | 30 | — | columns: frame, time, x_*/y_*/z_* |
| `mediapipe.json` | MediaPipe Pose | MediaPipe_33 | 30 | 33 | normalized image coords, 1920×1080 |
| `sample.c3d` | C3D (binary) | 6 markers | 30 | — | optional — only generated if `ezc3d` is installed |

All time-series fixtures use `60 Hz` (`dt = 1/60 s`) and 30 frames so the
duration is exactly `0.4833…` seconds. The "swing" signal is
`sin(linspace(0, 2π, 30))` so RMSE checks against synthesized references are
deterministic.

## Tolerance / round-trip expectations

The integration test
[`tests/integration/motion_pipeline/test_loader_golden_roundtrip.py`](../../integration/motion_pipeline/test_loader_golden_roundtrip.py)
loads each fixture through `motion_pipeline.sources.registry.load_any` (added
by PR #4619) and asserts:

1. The returned object is a CIR type (`KeypointSequence`, `MarkerTrajectory`,
   or `JointTrajectory`).
2. Timestamps are monotonically increasing (already enforced by the contracts,
   but verified explicitly).
3. The declared schema / marker-set on the CIR object is consistent with the
   fixture metadata.

If `motion_pipeline.sources.registry` does not yet exist (i.e., PR #4619 has
not landed), the test gracefully skips rather than failing.

## See also

- [`docs/motion_pipeline/formats.md`](../../../docs/motion_pipeline/formats.md)
  — full format-support matrix
- [`docs/adr/0007-motion-pipeline-architecture.md`](../../../docs/adr/0007-motion-pipeline-architecture.md)
  — CIR architecture decisions
