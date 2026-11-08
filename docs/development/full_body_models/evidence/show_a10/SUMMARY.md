# Best-Match Styled Render: `mujoco_driver` (SHOW-A10)

- Model id: `mujoco_driver`
- Source commit: `4bd0ae5099bca6f51c26e7bd0b356d2a6176b0c7`
- Receipt verdict: REJECTED (G3) - recorded as candidate per AC 3
- Address grip roll: lead +15.0°, trail -100.7° (SHOW-A7 15° CCW convention)
- Trunk: 6-DOF lumbar + thoracic segments (SHOW-A8, `full_body_spec_anthro_driver_v2.json`)
- Shading: axial tension/compression segment shading enabled
- Force vectors & torque arcs: enabled

## Outputs (SHA-256)

| File | SHA-256 |
|---|---|
| `down_the_line_address_1280.png` | `c41eae4bb80d04f22f57b596bf84edd4b36a276f8d4589cd71558ec292adabf0` |
| `down_the_line_impact_1280.png` | `b9ff901892714ba7376bfe95db5ed8c44ae044d233dc7beacac50624c52d2199` |
| `down_the_line_top_1280.png` | `f46c19647300c1ac75af0487600e7f1562064192d9a14e8d0e36343f087f4540` |
| `face_on_address_1280.png` | `c82d400c3a5a4d38507110d29620d5af9ccbc10ae62c5d95a967c1dd3405b761` |
| `face_on_impact_1280.png` | `1257191c83b30a4c89a08b624d619e49efb40e2e363ad75f6bc945b961069da9` |
| `face_on_top_1280.png` | `20e9ed5dafe2eea6e0c00111d2905a8d8132bf534aa809f4717d53ca80024ae0` |
| `side_by_side_address_dtl.png` | `76044dfd62e8d80838fe400c45ab93fd8b5b9660293dee5d57f7224df364ae0d` |
| `side_by_side_address_face_on.png` | `6d1c0815c6554e579a6b11c0c95f735d4d49e80d71295db9338380d0e4e6e3cf` |
| `side_by_side_top_dtl.png` | `2ff8de6e474b458adf88bf366c31e021d5feb616a80a89e7ef3af32b4630717a` |
| `side_by_side_top_face_on.png` | `3afcabc498cc85808f67c23d7d763504b47fa9277a0251c6c0bfa7f40b5561b9` |

## Inputs (SHA-256)

| Input | SHA-256 |
|---|---|
| `baseline_commit` | `fa7a9b7075` |
| `capture_sha256` | `cbcb4d84aca0a1f558073f0c2328c3f501c4726aaf699e382cf20c5861a0971d` |
| `show_a9_receipt_sha256` | `d06949d671a20564450998ca1d9ea1406cb75f86530275988f5dc759c98d51ad` |
| `v1_spec_sha256` | `fd72d8be7a6c5c557a10886a7d2717a4eb4619805a24cc1be644215eea21dcb3` |
| `v2_spec_sha256` | `323c2182ed1592552de700c02ab0f297e620660f2ac5a9c7f0f01e6af8f25b8c` |

## Acceptance Comparison vs 11-04 Baseline (`fa7a9b7075`)

| Metric | 11-04 Baseline (fa7a9b7075) | A9 Candidate Refit | Delta |
|---|---|---|---|
| Marker RMS | 50.9 mm | 55.3 mm | +4.4 mm |
| Joint RMS | 2.4 deg | 2.4 deg | +0.0 deg |
| Closure Residual | 62.2 mm | 76.8 mm | +14.6 mm |
| Weight Fraction | 0.04-2.82 BW | 0.00-2.82 BW | -0.04 min BW |
| Peak Joint Torque | 537 N*m | 537 N*m | +0 N*m |
| In-Support Fraction | 0.637 | 0.637 | +0.000 |

## Visual Artifacts

- `face_on_*_1280.png`: Face-on key frames (address, top, impact) with A7 grip clock inset (12 o'clock = 0°, green tick at +15°, red needle lead, blue needle trail).
- `down_the_line_*_1280.png`: Down-the-line key frames (address, top, impact).
- `side_by_side_*`: Side-by-side comparison with 11-04 baseline render showing grip and trunk changes.

## Reproduction Command

```bash
python -m src.tools.native_viewer_export --bundle <A9 bundle> --swing driver --club Driver --engines mujoco --speeds 1,0.5 --segment-shading
```
