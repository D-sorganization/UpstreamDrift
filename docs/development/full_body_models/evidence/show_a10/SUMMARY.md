# Best-Match Styled Render: `mujoco_driver` (SHOW-A10)

- Model id: `mujoco_driver` (candidate receipt `mujoco_driver_candidate_a9`)
- Source commit: `6392e76112 (SHOW-A8 stills) / f304efdcd4 (SHOW-A9 receipt)`
- Receipt verdict: REJECTED (G3) - recorded as candidate per AC 3 (not physically accepted)
- Clock overlay angles (illustrative target, not measured from rendered frames): lead +15.0°, trail -100.73° (SHOW-A7 15° CCW convention)
- Rendered-frame hand roll measurement: unavailable (SHOW-A8 stills with fixed clock overlay)
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

## Source Stills Consumed (SHA-256)

| Source Still | SHA-256 |
|---|---|
| `docs/development/full_body_models/evidence/show_a8/down_the_line_address_1280.png` | `a84d0e3bb6fe2659a5e8cda292916b88b234cc28d3c4e1db3ae5b29b0c30a55b` |
| `docs/development/full_body_models/evidence/show_a8/down_the_line_impact_1280.png` | `04a2cdcf509687630eb9aa8328fe503b6a15e77e1aabbd3f446b53a241a3868c` |
| `docs/development/full_body_models/evidence/show_a8/down_the_line_top_1280.png` | `94e1b10f5f66638449ca0fc0b6f03fa232633e5edfda14386c906fd0d022606a` |
| `docs/development/full_body_models/evidence/show_a8/face_on_address_1280.png` | `8091537d51e275318ca8fc37f71b770f34d4c7e533648e3d933a5455d7dc6a5a` |
| `docs/development/full_body_models/evidence/show_a8/face_on_impact_1280.png` | `378bde6b7e894391c3d2186248740ab30081a081b7690c85f0fe61c3d8ad268d` |
| `docs/development/full_body_models/evidence/show_a8/face_on_top_1280.png` | `5d3e404d19c760a5c8c83d76831980ec7acbe929f2378425b09ed334bcc4cd4d` |
| `docs/development/full_body_models/evidence/trunk_segments/after_v2_segmented_address_dtl.png` | `021db3583098997fdceed7b07f0f6da82223701c782abcf0aeffd1fad3303f93` |
| `docs/development/full_body_models/evidence/trunk_segments/after_v2_segmented_address_face_on.png` | `51410c46304198d61bd897fa193434572308ae5b4a5f58a81edda29428fa4332` |
| `docs/development/full_body_models/evidence/trunk_segments/after_v2_segmented_top_dtl.png` | `baf7798669377cd92b558066eaa3b50a62788a6eb5935b48f9f54085f8edd2af` |
| `docs/development/full_body_models/evidence/trunk_segments/after_v2_segmented_top_face_on.png` | `1a7a5ff8921bfb2af85fc1ce49b69bf781ce005b250943cf6bebd62256a81f1f` |
| `docs/development/full_body_models/evidence/trunk_segments/before_v1_rigid_address_dtl.png` | `5e1fd4598f220359d50a4d38d1f6cbd8c82da7f331f4c1422329db934249ec55` |
| `docs/development/full_body_models/evidence/trunk_segments/before_v1_rigid_address_face_on.png` | `9a4fdbd2d713d4f98db604c40a4000bd457c6bfb48dcfb03994abfa71fd24757` |
| `docs/development/full_body_models/evidence/trunk_segments/before_v1_rigid_top_dtl.png` | `4db0325c100a0113523383f9ae9bb80e4c08f04c6433ecb95cb128c1c9c0e397` |
| `docs/development/full_body_models/evidence/trunk_segments/before_v1_rigid_top_face_on.png` | `01804d31c0fae5ccaffe030788983197911254d3c5510819268d4756a7ca6877` |

## Angle & Measurement Attribution

- **Clock overlay (`+15.0°` lead / `-100.73°` trail):** Illustrative target / modelling choice from SHOW-A7 / SHOW-A9 receipt drawn over SHOW-A8 stills; **not measured from rendered frames**.
- **Rendered-frame hand roll & wrist angles:** `unavailable` (SHOW-A10 reuses pre-rendered SHOW-A8 PNGs).
- **Quantitative refit metrics:** Imported from `show_a9/receipt.json` (`mujoco_driver_candidate_a9`, `REJECTED (G3)`).
- **Video delivery:** `unavailable_in_repo` (no A9 `.npz` bundle or video set generated here; epic `#14906` remains open).

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

- `face_on_*_1280.png`: Face-on key frames (address, top, impact) with A7 grip clock inset (12 o'clock = 0°, green tick at +15°, red needle lead, blue needle trail; copied from `show_a8/`).
- `down_the_line_*_1280.png`: Down-the-line key frames (address, top, impact) copied from `show_a8/`.
- `side_by_side_*`: Side-by-side comparison with 11-04 baseline render showing grip and trunk changes (from `trunk_segments/`).
- `provenance_manifest.json`: Per-file provenance and angle attribution manifest.

## Reproduction Command

```bash
python docs/development/full_body_models/evidence/show_a10/render_show_a10_evidence.py
```
