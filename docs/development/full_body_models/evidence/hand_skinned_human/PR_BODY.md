## Skinned Human Hand on the Club (MakeHuman CC0 Rig, Linear Blend Skinning)

Refs #15150 (epic), #15148. Stacked on `cursor/hrg-gb1b-shaft-roll-2882`. Auto-merge is not armed, on the owner's instruction.

### State at `ebacbcd46e` (2026-11-15)

- The hand is MakeHuman's hm08 mesh with MakeHuman's own `default.mhskel` skeleton and `default_weights.mhw` skin weights (CC0, pinned commit, recorded in `assets/hand_cc0/NOTICE` and `PROVENANCE.json`). Two declared changes to the weights and one to the mesh: see "Deviations".
- Posing is joint rotation plus linear blend skinning only. A test rebuilds the mesh from the reported bone transforms and requires agreement to 1e-9 m, so no vertex is pushed.
- The pose follows the grip spec: flat hand beside the club, club under the index finger's first segment and just below the base of the little finger, then MCP, PIP, DIP close in that order until the skin touches. Knuckle flexion drives the ring and little metacarpals so the palm cups.
- The thumb now lies on the grip along its pad, not on its tip.
- `build_grip_hands` (new module `skinned_pair.py`) poses both hands on one club. The trail hand slides toward the club head by the least amount that frees its index, middle and ring fingers from the lead hand, then curls round the grip. Its little finger rests on the lead index finger (overlap grip).
- The forearm ends in a rounded cap, not a flat disc.
- Each hand declares which vertices a renderer should frame on (`frame_focus`: everything except the forearm stub), so the hand itself fills 0.70 of the frame.
- Shaft roll comes from `HandGripSpec.shaft_roll_deg`. Every size, tolerance, limit, gain and colour is in `scripts/config/skinned_human_hand.json`, `scripts/config/hand_visual_quality.json` or `assets/hand_cc0/build_settings.json`.
- Contact physics is untouched (still capsules). This is the visual model.
- Tests: 59 pass, 1 skipped (the source-rebuild check needs a MakeHuman checkout; I ran `build_hand_asset.py --check` by hand and it matches).

### Measured (Legacy Grip Spec, Grip Radius 14 mm)

| Check | Before | Now | Limit (setting) |
| --- | --- | --- | --- |
| Thumb first segment to grip | 12.2 and 14.5 mm | 5.3 mm lead, 5.0 mm trail | 6 mm |
| Fingertip and middle phalanx to grip, single hands | 0.95 mm | 1.0 mm | 1.5 mm |
| Trail index, middle, ring to the grip itself (pair) | ring 6.8 mm | at most 1.0 mm | 1.5 mm |
| Skin inside the grip | 0.3 mm | 0.3 mm | 0.5 mm |
| One hand inside the other (pair, mesh winding number) | 0.0 mm | 0.0 mm | 1.0 mm |
| Forearms apart, seen from the butt | 25 degrees | 26 degrees | 15 to 50 |
| Degenerate triangles | 0 | 0 | 0 |
| Triangles folded into a crease | 5 per hand | 2 per hand (0.04 to 0.08 % of skin area) | 2 and 0.15 % |
| Hand share of the frame (forearm left out) | 0.48 to 0.60 | 0.67 to 0.70 | 0.70 plus or minus 0.12 |
| Left-handed golfer is the exact mirror | yes | yes | 1e-9 m |
| Shaft roll 0 and plus or minus 20 degrees is a rigid turn | yes | yes | 1e-9 m |

### Self-Review (I Opened All 36 Frames; Scores Out Of 5)

| Criterion | Lead | Trail | Both hands | Change |
| --- | --- | --- | --- | --- |
| Separate fingers and thumb | 4 | 4 | 4 | same |
| Cupped palm | 3 | 3 | 3 | same |
| Knuckle line | 4 | 4 | 4 | same |
| Fingers wrap the grip | 4 | 4 | 4 | pair up from 3 |
| No tubes, blobs or torn triangles | 3 | 3 | 3 | down from 4: at 70 % fill the two remaining folds are visible |
| Smooth shading | 4 | 4 | 3 | pair down from 4: facets on the back of the hand in the butt view |
| Correct mirroring | 5 | 5 | 5 | same |

Each hand reads as a human hand holding a golf club, and the pair now reads as two hands gripping one club with the trail fingers closed on it. It does not reach the bar yet. What still falls short:

- **Palm cup (3).** The thumb pad now lies on the grip, but the thenar pad cannot. In a finger grip the thumb's base sits about 66 mm from the grip axis and the first thumb bone is 29 mm long, so the thenar skin stays about 30 mm clear. That space is where the lead thumb sits under the trail hand. Rotation alone cannot close it. The palm therefore still looks flatter than a real one.
- **Two folded triangles per hand (3).** One in the web between the little and ring fingers, one in the thumb-to-index web. They show as a small dark notch at the knuckles in the trail back views and a sliver in the lead thumb-side views. Stronger weight smoothing made more folds, and reducing the little-finger arch removed one fold but flattened the palm.
- **Butt view.** The camera sits close to the butt end, so the grip's end cap covers a large part of the frame and the hand is seen at a steep angle. The view meets the brief but tells little.
- **Facets.** The back of the hand shows flat facets near the knuckles under grazing light in the butt views. The mesh is MakeHuman's base resolution (1,770 vertices per hand).
- **Lead thumb in the pair.** It is hidden under the trail palm, as in a real grip. The trail thumb shows in the back and thumb-side views.

| | Palm | Back | Thumb side | Butt |
| --- | --- | --- | --- | --- |
| Lead | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_palm_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_back_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_thumb_side_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_butt_rollp00.png?raw=true) |
| Trail | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_palm_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_back_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_thumb_side_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_butt_rollp00.png?raw=true) |
| Both | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_palm_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_back_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_thumb_side_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_butt_rollp00.png?raw=true) |
| Both, roll -20 | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_palm_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_back_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_thumb_side_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_butt_rollm20.png?raw=true) |
| Both, roll +20 | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_palm_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_back_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_thumb_side_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/ebacbcd46e57a9ace87178c8b483afcc86abf2a7/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_butt_rollp20.png?raw=true) |

All 36 frames (1024 px, MuJoCo EGL through the shared renderer of #15154) and `render_manifest.json` are in `docs/development/full_body_models/evidence/hand_skinned_human/`.

### Deviations and Things to Check

- **Skin weights at the thumb web are smoothed** (`thumb_web_smoothing`, approved). Zero iterations rebuilds with MakeHuman's untouched weights.
- **Skin weights at the finger middle creases are lightly smoothed** (`finger_crease_smoothing`, new, not yet approved). One ring, two passes at half blend. It took the folds from 5 to 2 per hand. Zero iterations turns it off. Dieter, please say whether to keep it.
- **The forearm cut is closed with a dome** (`forearm_cap`: 4 rings, depth 0.6 of the loop radius). These vertices are not MakeHuman's. Zero rings gives the flat fan back.
- **The trail hand may slide up to 30 mm along the grip** beyond the spec's hand spacing (`pair.max_spacing_adjust_m`) so its fingers reach the grip. Zero keeps the spec's spacing exactly and puts the ring finger back on the lead hand.
- **Thumb joints may bend back** (MCP to -10 degrees, IP to -25) so the pad lies flat. These limits are my values.
- **Wrist bend** (lead 0 and 20 degrees, trail 15 and 10) and **`palm_centre_wrap_deg` (37)** are my values, tuned by eye.
- **Finger-length references are unverified.** The ratios in `hand_visual_quality.json` are recalled from Buryanov and Kotiuch 2010 and were not checked against the paper.
- **Shared posing has no branch of its own**, so `hand_backends/posing.py` and its tests live here.
- **Size.** Several thousand added lines, far over the 500-line gate. It needs `large-pr-approved` or an owner decision.

### Blockers

- **Palm cup and crease folds cannot reach 4 or better under the brief's method.** Both are limits of joint rotation with plain linear blend skinning on this skeleton. Closing them needs a decision from Dieter: allow corrective shapes or dual-quaternion blending, or accept the current look.
- **CI.** `unit-test-gate` still lists `grip_spec/*` and `grip_contact/field_validation.py` as unregistered paths. Those files belong to the base PRs, not this one, so I left them. The Fleet policy gate is left alone as instructed.

## Learnings

- Rig with the source's own skeleton and weights. Cropping a body mesh and inventing weights gave tubes and tears; MakeHuman's rig gave a hand on the first pose.
- Fit finger hinge axes to the palm plane, not to each finger's nearly straight rest shape. The per-finger fit fanned the fingers.
- Greedy proximal-to-distal curl hooks a digit on its tip. Scan the first joint, close the others to contact, and keep the pose whose named segments both touch (`wrap_to_contact`). This also laid the thumb flat once its joints were allowed to bend back a little.
- The thumb is a ramp from its base to the grip. Do not try to press the thenar pad onto the grip; measure the pad of the thumb instead.
- For two hands, decide the spacing first: solve the trail hand alone on the bare grip, slide it by the least amount that frees its gripping fingers from the lead hand, then solve against grip plus lead hand. Curling against the lead hand at the spec's spacing left the ring finger 7 mm off the grip.
- Measure hand-in-hand overlap on the meshes (winding number), not on the capsules used to solve. A 1 mm capsule margin removed the last 1 mm of mesh overlap.
- Light weight smoothing at a crease removes folds; more smoothing adds them and lifts the palm off the grip. Report the count; do not claim zero.
- Frame the camera on the hand, not on hand plus forearm. A bounding-box fill of 0.70 that includes the forearm stub shows the hand at about 0.55.
- Solver speed is forward kinematics. Precomputed hinge matrices and skinning only the vertices a joint moves took a pair from 55 s to 14 s.
- Never run Python from inside `hand_backends/`: its `types.py` shadows the standard library.

### Next Steps

1. Owner decision on corrective shapes or dual-quaternion blending for the two web folds and the palm cup.
2. Owner decision on `finger_crease_smoothing`.
3. Subdivide the mesh once at build time to remove the facets on the back of the hand.
4. Check the finger-length references against the paper.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
