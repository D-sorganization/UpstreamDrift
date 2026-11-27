# Five Historical Golfer Models

Epic #15832 extends the Tiger Woods and Ben Hogan pipeline to five more players.
The archive contains five separately sized candidates and five appearances on
the current reference physics, using `full_body_skinned_flex_v2`, skinned hands
and body, and the flexible driver. These are stylized interpretations.

| Player | Selected Height | Selected Mass | Prior Limitations |
| --- | ---: | ---: | --- |
| Seve Ballesteros | 1.83 m | 80 kg | Undated professional profile |
| Jack Nicklaus | 1.778 m | 83.91458845 kg | Other published height and mass conflict |
| Bobby Jones | 1.7272 m | 74.84274105 kg | Source describes weight as approximate |
| Sam Snead | 1.8034 m | 86.1825503 kg | Other published height and mass conflict |
| Byron Nelson | 1.8542 m | 79.37866475 kg | A 1945 account gives a lower mass |

Full source URLs, alternatives, era caveats and authored appearance choices are
in each `receipt.json` and `src/config/player_appearance_presets.json`. Exact SI
conversion does not make a historical estimate an exact measurement. Height is
calibrated from neutral head skin to lowest shoe sole: it includes shoes and
excludes hair and headwear. Segment anatomy is estimated from the shared priors.
Period facial hair, collars and garment cuts are not reconstructed; the profile
receipts distinguish inspected source features from the simplified interpretation.

## Use the Models

Extract `legends-models.zip`. Load a player's self-contained `model.xml` in
MuJoCo, or import it into Necromatcher with engine `mujoco` and the exact ordered
coordinates in its receipt. Use `model.visual.json` for native-resource binding;
it explicitly embeds appearance without changing the physical definition.
The semantic coordinate order differs from MuJoCo traversal order: map by joint
name when filling native `qpos` and `qvel` buffers.

`sized/` models intentionally change geometry and inertia and need a new motion
fit. `reference-skins/` preserves the current reference physics. To skin an
existing fitted model, export with `--base-spec existing_match.json`.
Native validation results and hashes are in `delivery-verification.json` and
`manifest.json`. The archive includes previews and CC0 asset provenance.

## Reproduce

Initialize the repository's pinned Tools submodule and install dependencies.
On Linux, set `PYTHONPATH=.:vendor/ud-tools/src` and `MUJOCO_GL=egl`, then run:

```bash
python3 scripts/export_player_models.py --player seve_ballesteros jack_nicklaus bobby_jones sam_snead byron_nelson --output output/legends
python3 scripts/export_player_models.py --player sam_snead --output output/skins --base-spec existing_match.json
python3 scripts/export_player_models.py --player all --output output/all_players
```

`--player both` still selects Tiger Woods and Ben Hogan. For hash-bound address
previews, use `--render-existing --pose-root output/legends` with the archived
per-player `review_pose.json` files and unchanged model files. Default exports
use zero joint coordinates. Review poses are authored inspections, not captured
swings; grip closure does not establish cuff alignment, contact or balance.
Preview grounding places the lowest sole at zero, not both feet on the ground.

## Reference Footage

Video files and extracted frames remain in private storage. The bundled source
receipts identify URLs, downloaded variants, hashes and timing limitations.
They do not license redistribution or establish a physical motion clock.
No historical motion training or forward-dynamics qualification is claimed.
