# Model Card: Shadow Tracker Neural Silhouette Segmentation (MMR-13, #11099)

## Model Overview

- **Model Names:** `sam-vit-b-golf` (Primary), `mobilesam-golf` (Edge/Lightweight)
- **Architecture:** Segment Anything Model (ViT-B image encoder + prompt-guided mask decoder) / MobileSAM (TinyViT encoder)
- **Version:** `1.0.0`
- **License:** Apache-2.0
- **Intended Use:** Dual-channel binary silhouette extraction (separate `body` and `club` masks with `valid` pixel masks) for golf swing tracking.

## Checkpoint Pins Are Operator-Supplied (no trusted pins in this repository)

- No SHA-256 weight pin for either model is committed to this repository. The
  hash strings previously claimed here were fabricated placeholder values
  (they did not correspond to any recorded checkpoint) and have been removed.
- Weights are never downloaded by the software. An operator must provision the
  offline checkpoint per deployment and pin it explicitly at use sites:
  `RealSegmentationAdapter(..., expected_sha256="<sha of the file>")` and
  `verify_checkpoint(name, path, expected_sha256=...)` enforce exactly 64
  lowercase-hex pins, fail closed on mismatch (`RuntimeError`), and reject
  missing artifacts (`FileNotFoundError` / typed unavailability) without any
  hidden network traffic.
- Until real weights plus a calibrated body/club postprocess decoder are
  provisioned, `RealSegmentationAdapter` is a **dev-only checkpoint-validation
  harness**: it verifies, loads and executes whatever artifact the operator
  pinned (TorchScript / ONNX), but reports typed `SegmentationUnavailableError`
  rather than fabricating dual masks. No segment-anything accuracy claim is
  made without that executed stage.

## Hardware Budgets and Compute Requirements

| Model            | Parameter Count | Input Resolution   | Min RAM | Min VRAM | Recommended Execution                    |
| :--------------- | :-------------- | :----------------- | :------ | :------- | :--------------------------------------- |
| `sam-vit-b-golf` | 91.0M           | $1024 \times 1024$ | 8 GB    | 4 GB     | CUDA / DirectML (CPU fallback supported) |
| `mobilesam-golf` | 9.66M           | $1024 \times 1024$ | 4 GB    | 2 GB     | Real-time CPU or integrated GPU          |

## Known Failure Modes and Scientific Limitations

1. **No qualified vendor weights in this repository:** without an executed pinned checkpoint the adapter must not run, so a false "model inference" success is impossible; the benchmark correspondingly reports `blocked` rather than numbers.
2. **High-Speed Clubhead Motion Blur:** during late downswing and impact (>120 deg/frame at standard 30-60 fps video), clubheads experience significant motion blur. Any future boundary metrics must be measured against independently recorded gold masks, not adapter self-echoes.
3. **Thin Shaft Visibility:** high-aspect steel or graphite shafts (<3 pixels width in standard HD frames) may experience segmentation dropouts even with a provisioned model.
4. **Occlusions:** overlap with spectators, trees, golf carts, or bags triggers partial/full occlusion flags in `track_occlusion_and_identity()`.
5. **Manual Review Mandate:** automated inference outputs are tagged as unreviewed drafts (`producer_id="model:..."`); final acceptance requires review through `ManualMaskProvider`.