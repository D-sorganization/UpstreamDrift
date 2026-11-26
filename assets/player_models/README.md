# Tiger Woods and Ben Hogan Player Models

Parent epic: #15814. These are stylized, source-attributed candidates built through
`full_body_skinned_flex_v2`, with the current skinned hands/body and flexible driver.
They are not photogrammetric likenesses or qualified historical motion fits.

| Player | Selected height | Selected body mass | Visual interpretation |
| --- | ---: | ---: | --- |
| Tiger Woods | 1.8542 m | 83.91458845 kg | Red polo, dark trousers and black cap; broader jaw |
| Ben Hogan | 1.7272 m | 68.0388555 kg | Pale polo, grey-brown trousers and flat cap |

Published alternatives and era limitations are retained in each receipt and in
`src/config/player_appearance_presets.json`. Height is calibrated to neutral
head-skin top minus lowest shoe sole, including shoes and excluding hair/cap.
This convention does not establish measured barefoot anatomy. Both models pass
independent native geometry checks within 1 mm and retain the 52-coordinate order.

## Files and Use

`player-models.zip` contains self-contained native MuJoCo `model.xml` files,
full-body `model.json` specifications, appearance sidecars, dimension/source
receipts, review poses, previews, checksums and asset-license provenance.
The `sized/` models use separately constructed player dimensions. Refit captured
motion and torques before using them for biomechanical inference. The
`reference-skins/` models apply the appearances to the current reference physics.

Load a `model.xml` with MuJoCo. In Necromatcher, import it as a native model with
engine `mujoco` and the exact ordered coordinates from its `receipt.json`.
The bundled address poses are hash-bound inspection poses, not captured swings.
They satisfy the recorded grip-closure checks; cuff angles and unequal shoe
heights remain review limitations. Preview grounding moves the lowest sole to
z=0 and does not assert two-foot contact or dynamic balance.

## Reproduce

Initialize the pinned Tools submodule and install repository dependencies first.
From the repository root on Linux:

```bash
PYTHONPATH=.:vendor/ud-tools/src MUJOCO_GL=egl python scripts/export_player_models.py --output output/players
PYTHONPATH=.:vendor/ud-tools/src MUJOCO_GL=egl python scripts/export_player_models.py --output output/skins --base-spec existing_match.json
PYTHONPATH=.:vendor/ud-tools/src MUJOCO_GL=egl python scripts/export_player_models.py --output output/players --render-existing --pose-root output/players
```

The final command requires matching per-player `review_pose.json` files. It checks
both spec and XML hashes before applying the coordinates. A pose from another
geometry revision is rejected. Exported models default to zero joint coordinates.

Source videos are stored privately; they are not included or licensed by this
bundle. See `docs/development/historical_capture/player-reference-acquisition-commands-2026-11-26.md` for acquisition
receipts, source variants and the unverified USGA clip-time mapping. Model asset
provenance and CC0 notices are included separately from video provenance.
