## Skinned Human Hand on the Club (MakeHuman CC0 Rig, Linear Blend Skinning)

Refs #15150 (epic), #15148. Stacked on `cursor/hrg-gb1b-shaft-roll-2882`. Auto-merge is not armed, on the owner's instruction.

### Update at `81c38e59f7` (2026-11-17): Wrists Over the Shaft, Both Wrists Extended

This section is the newest; where it disagrees with the sections below, this one is right. The owner's fifth comment asked for: the lead wrist joint almost centred over the shaft at address and the trail wrist slightly to the trail side; both wrists slightly extended; thumb and forefinger pinched; the grip square to the clubface; the hands joined to the forearms in the MuJoCo model; and said the forearm direction was limiting the grip.

- **Wrists over the shaft.** The lead hand is turned 22 degrees about the shaft and the trail hand 27 degrees (`placement.palm_centre_wrap_deg`: lead 15, trail 10; both were 37). The lead wrist joint is now exactly over the shaft (it was 22 mm to the target side); the trail wrist is 44 mm to the trail side (was 74 mm). New measure `qa.wrist_joint_m` and test `test_the_wrists_sit_over_the_shaft_at_address` (`max_lead_wrist_offset_m` 8 mm, `trail_wrist_offset_range_m` 15 to 50 mm).
- **Both wrists extended.** `wrist.lead.extension_deg` 15 (was 8), `wrist.trail.extension_deg` 20 (was 15); test `test_both_wrists_are_extended_at_address` (`min_wrist_extension_deg` 12). In this model the wrist bend turns the forearm about the wrist joint and leaves the hand where it is, so it does not by itself wrap the hand further.
- **Forearm limit widened, on the owner's word.** `forearm_spread_range_deg` lower bound 15 to 5: with both wrists near the top of the shaft the forearms are 11 degrees apart seen from the butt.
- **Thumb line per hand.** `owner_thumb_azimuth_deg` is now `{lead: 45, trail: 20}`. With the trail hand nearer the top its thumb could not reach 45 degrees (7.4 degrees short); it lies 20 degrees to the target side of the top.
- **Shaft roll is exactly a rigid turn.** `build_grip_hands` solves the pair with the lead hand unrolled and turns the result. Before, a rolled pair was solved again in a turned frame and one skin dent on the trail little finger differed by 0.3 mm.
- **Heel-pad ratchet 7 to 8 mm (a loosening; evidence on #15150).** Turning the lead hand to centre the wrist measures 7.7 mm, against 6.3 mm before. Hand turns from 12 to 22 degrees give 7.0 to 7.7 mm; none recovers 6.3.
- **One test restated.** `test_turning_the_thumb_can_be_switched_off` used the default trail hand, which no longer needs a thumb turn. It now shows the switch on the legacy grip with the knuckle target.

#### Measured at `81c38e59f7`

| Check | Now | At `f4f607338c` | Limit |
| --- | --- | --- | --- |
| Lead wrist joint, sideways from the shaft | 0 mm | 22 mm to the target side | 8 mm |
| Trail wrist joint, sideways from the shaft | 44 mm to the trail side | 74 mm | 15 to 50 mm |
| Wrist extension, lead and trail | 15 and 20 degrees | 8 and 15 | at least 12 |
| Forearms apart, seen from the butt | 11 degrees | 32 | 5 to 50 (was 15 to 50) |
| Thumb joint to forefinger knuckle, lead and trail | 26.9 and 36.5 mm | 31.0 and 39.0 mm | none in this mode (skin touches at about 15) |
| Thumb to its own hand, lead and trail | 1.6 and 2.0 mm | 1.8 and 1.9 mm | 3 mm |
| Lead heel pad to the grip, nearest skin | 7.7 mm | 6.3 mm | 8 mm (was 7) |
| Fingertip to thumb pad, lead index, middle, ring | 8.4, 1.4, 6.5 mm | 4.4, 0.9, 4.5 mm | none |
| Fingertip to thumb pad, trail index, middle | 6.3, 2.4 mm | 0.7, 1.2 mm | none |
| Skin within 0.5 mm of the grip, lead and trail | 26.2 and 18.2 cm2 | 27.1 and 18.8 cm2 | at least 18 cm2 |
| Trail first segments to the grip in the pair, index and middle | 4.6 and 4.7 mm | not recorded | none on first segments |
| Pinched knuckle edges, lead and trail | 3 and 1 | 3 and 1 | 6 |
| Flipped triangles | 0 | 0 | 0 |
| Skin moved by correctives, lead and trail | 30.3 % and 31.7 % | 30.1 % and 32.2 % | 34 % |
| Pair fault | 0.69 mm | 0.66 mm | parts checked at 1.0 and 1.5 mm |

Tests: 153 pass, 1 skipped, **2 expected failures** (see Blockers).

#### Honest Reading

- The wrists are where the owner asked. **The grip is not tighter for it**: the fingertips are further from the thumb pads than before (lead index 8.4 mm, was 4.4), the heel pad is 1.4 mm further off, and the trail hand's contact area is 0.2 cm2 above its floor.
- **Pinch: partly.** The lead thumb joint is 4 mm nearer the forefinger knuckle. The full pinch setting (`thumb.toward_index_target: "knuckle"`) brings the lead to 16 mm, which is skin on skin, but it unseats the trail ring finger in the pair (fault 10 mm). A thumb line of 30 degrees on both hands fails five tests. Not solved.
- The pair solve is brittle around the trail hand's turn: 10 degrees holds (fault 0.69 mm), 0 and 14 degrees do not (4.7 and 7.6 mm).

#### Does the Wrist Work as a Universal Joint?

Yes, as far as a picture needs. `wrist.<side>` has two angles, extension and ulnar deviation, about the wrist joint centre; the forearm turns and the hand stays. There is no twist: that belongs to the forearm, as the owner describes. `qa.wrist_joint_m(result)` gives the joint centre in the club frame, and the forearm bone gives its axis. The mesh still ends in a forearm stub with a rounded cap; cutting it at the wrist is still to do.

The MuJoCo golfer already has the matching joints: forearm rotation about the forearm axis (`LFInput`, `RFInput`) and a two-axis wrist (`LWInputX`/`LWInputY`, `RWInputX`/`RWInputY`) between the forearm and the club.

#### Clubface and Forearm Attachment in the MuJoCo Golfer: Measured, Not Fixed

Measured with a scratch spike (not committed) on `full_body_spec_anthro_driver_v2.json`:

- **The hands are 46 degrees round the shaft from the clubface.** The head is squared by its own constant (`ADDRESS_SQUARE_FACE_ROLL_DEG`, driver -37.92 degrees, `club_assembly.py`), because the fitted wrist constants leave the club body rolled open. The hands are turned separately, from the wrist joint's dorsal axis (`grip_hands.grip_hand_frame`). Nothing ties the top of the grip under the hands to the face. `GripSpec.shaft_roll_deg` exists for this and is 0 on this path; the MuJoCo path never sets it.
- **The hand does not meet the forearm.** The model's lead wrist joint is 25 mm from the shaft axis, level with the middle of the hand. The skinned hand's wrist joint is 60 mm from the axis and 77 mm nearer the butt. Placed on the club, the two are 57 mm apart (75 mm once the hands are squared to the face).

Proposed order of work, for the owner to confirm (it changes the model's kinematics, so the swing has to be fitted again):

1. Make the grip the authority. The hand frame on the club comes from the clubface (top of the grip at right angles to the face), not from the wrist joint.
2. Put each wrist joint on the club where the posed hand has it (`wrist_joint_m`), for both hands, in the spec.
3. Cut the hand mesh at the wrist and draw the model's own forearm to it.
4. Refit address and the swing with the arms reaching those wrists; the forearm rotation and two-axis wrist take up the difference.

#### Self-Review of the Frames at `81c38e59f7` (Harsh; Out Of 5)

I opened the roll 0 frames of the lead hand, the trail hand and the pair, and the pair at roll +20 and -20 (25 frames, five views each). **I did not open the single-hand roll frames** (20); the roll is now a rigid turn by construction.

| Criterion | Lead | Trail | Both | Note |
| --- | --- | --- | --- | --- |
| Reads as a hand on a real golf club | 4 | 4 | 4 | Unchanged |
| Wrist over the shaft at address | 4 | 3 | 4 | New. Golfer view: the lead forearm runs straight back from the top of the grip. The trail wrist is 44 mm to the side, at the wide end of "slightly" |
| Wrists extended | 3 | 3 | 3 | New. The forearms tip back by 15 and 20 degrees; hard to read in these views, and the hand does not wrap further for it |
| Heel pad on top of the grip (Hogan) | 3 | n/a | 3 | Covers the grip, hovers 7.7 mm (was 6.3) |
| Grip looks tight, fingers under the thumb pad | 2 | 2 | 3 | Trail down from 3: its fingertips are now 6 mm from the thumb pad |
| Thumb and forefinger pinched | 2 | 2 | 2 | Lead a little closer; the notch is still open on both. The trail thumb is long and straight down the top of the grip |
| Fingers square to the shaft | 2 | 3 | 2 | Unchanged |
| Skin free of creases and folds | 4 | 4 | 4 | No flipped triangles |
| Hands locked together | n/a | n/a | 4 | Holds in all three rolls |

### Earlier Update at `f4f607338c` (2026-11-16): Butt Proud of the Heel, Golfer's-Eye View

- **The butt of the club stands proud of the lead heel.** The lead heel pad hung 18 mm past the butt cap, so it had no grip to press on. `owner_lead_below_cap_edge_m` is 85 mm (was 60) and the evidence radius follows the taper there (10.9 mm, was 11.3). New test `test_the_butt_of_the_club_stands_proud_of_the_lead_heel` (`min_butt_beyond_lead_heel_m`, 2 mm).
- **Ring-finger knuckle limit.** `curl.limits_deg` has `mcp.ring` at 115 degrees beside `mcp.little`: on the slimmer grip the ring finger otherwise bridged 2.5 mm off. Remove the key to switch it off.
- **Palm pad reach 13 mm** (`correctives.palm_pad.reach_m`, was 12). At 14 mm the palm-cup test fails (the pad then does the cup's work) and from 14.5 mm the skin moves more than the 8 mm limit, so it stops at 13. Heel-pad ratchet `max_lead_heel_pad_gap_m` tightened from 9 to 7 mm.
- **A fifth evidence view, `golfer`**: looking down on the top of the grip with the butt toward the viewer, the angle of the Hogan photographs (`evidence_settings.json`, `golfer_view`). 45 frames.
- **One test restated.** `test_lead_and_trail_are_opposite_hands_on_opposite_sides` took the mean of every vertex, forearm and wrapped fingers included; with the palm across the top of the shaft the lead mean came to +0.05 mm. It now measures the palm's own skin (lead -19 mm, trail +34 mm), which is what its comment always said. No limit was changed.

#### Measured at `f4f607338c`

| Check | Now | At `4e8598b464` | Limit |
| --- | --- | --- | --- |
| Lead heel pad to the grip, nearest skin | 6.3 mm | 7.1 mm | 7 mm (ratchet) |
| Lead heel skin past the butt of the club | none (cap edge 5 mm beyond) | 18 mm past | at least 2 mm inside |
| Skin within 0.5 mm of the grip, lead and trail | 27.1 and 18.8 cm2 | 26.7 cm2 lead | at least 18 cm2 |
| Lead ring and little first segments to the grip | 2.9 and 7.3 mm | 3.8 and 6.7 mm | none on first segments |
| Fingertip to thumb pad, lead index, middle, ring, little | 4.4, 0.9, 4.5, 17.3 mm | not recorded | none |
| Fingertip to thumb pad, trail index, middle, ring, little | 0.7, 1.2, 15.3, 30.7 mm | not recorded | none |
| Pinched knuckle edges, lead and trail | 3 and 1 | 3 and 1 | 6 |
| Flipped triangles | 0 | 0 | 0 |
| Skin moved by correctives, lead and trail | 30.1 % and 32.2 % | 30.7 % and 31.5 % | 34 % |
| Largest corrective shift | 6.9 mm | 6.3 mm | 8 mm |
| Pair fault | 0.66 mm | 1.1 mm | parts checked at 1.0 and 1.5 mm |
| One hand inside the other | 0.0 mm | 0.0 mm | 1.0 mm |
| Forearms apart | 32 degrees | 32 | 15 to 50 |
| Lead fingers off square to the shaft, middle and end segments | 16 to 49 degrees | 15 to 50 | none (soft rule) |

Tests: 151 pass, 1 skipped, **2 expected failures** (see Blockers). The trail hand's contact area (18.8 cm2) is close to its 18 cm2 floor.

#### Tried and Rejected: A Firmer Squeeze (Owner, Fourth Comment)

The owner asked for a tighter grip, with the fingers pressing up under the meaty first segment of the lead thumb. The direct setting is `squeeze.finger_depth_m` (how far the finger pads press into the grip). None of it is committed:

| `finger_depth_m` | Result |
| --- | --- |
| 2.5 mm (current) | Lead fingertips 4.4, 0.9, 4.5 mm from the thumb pad (index, middle, ring) |
| 3.0 mm | Lead ring fingertip moves away, to 8.9 mm; little finger middle segment lifts to 1.4 mm |
| 3.3 mm | Skin 4.1 mm inside the grip |
| 4.0 mm | Index and middle under 1 mm from the pad, but 12 tests fail: finger bones inside the grip (limit 1.5 mm clear), trail ring and little fingers 4.3 mm apart, forearms 10 degrees closer |

Squeeze depth is not the lever: the fingers are already on the grip, so pressing harder pushes bone into the club instead of closing the hand. Raising the end-joint limit (`dip` 85 to 100 degrees) moves the lead index 0.9 mm and nothing else. **Open, see Blockers**: what closes the opening is the thumb's base coming down onto the fingertips, which is the same pose decision as the thumb snug to the hand (owner decision 1).

#### Honest Reading

- In the golfer's-eye view the base of the lead palm now covers the top of the grip and the butt shows beyond it, as in the Hogan photographs. The heel pad's nearest skin is still 6 mm off the grip: it is over the grip, not pressed on.
- **The grip does not yet look tight.** In the lead palm view the grip shows between the fingers and the palm, and the fingertips stop short of the thumb's pad.
- **Fingers square to the shaft is not improved.** A uniform sideways swing of the fingers was tried and rejected: the finger roots splay and cross.
- The forearm stubs end in a rounded cap and look odd in the pair frames; they are there to show direction only.

#### Self-Review of the Frames at `f4f607338c` (Harsh; Out Of 5)

I opened the roll 0 frames of the lead hand, the trail hand and the pair, and the pair at roll +20 and -20 (25 frames, five views each). **I did not open the single-hand roll frames** (20); the roll is tested as a rigid turn.

| Criterion | Lead | Trail | Both | Note |
| --- | --- | --- | --- | --- |
| Reads as a hand on a real golf club | 4 | 4 | 4 | Unchanged |
| Heel pad on top of the grip (Hogan) | 3 | n/a | 3 | The palm base covers the grip in the golfer and butt views and the cap stands proud. Still hovering 6 mm, so not a 4 |
| Butt of the club beyond the hand | 4 | n/a | 4 | New; cap and 5 mm of grip show past the heel |
| Grip looks tight, fingers under the thumb pad | 2 | 3 | 3 | New criterion from the owner's fourth comment. Lead palm view: the grip shows between fingers and palm, fingertips short of the pad |
| Thumb snug to the hand | 2 | 2 | 3 | Unchanged: long straight thumbs, open notch; owner decision 1 |
| Fingers square to the shaft | 2 | 3 | 2 | Unchanged: lead fingers visibly diagonal in the palm view |
| Lead little finger on the grip | 3 | n/a | 3 | First segment 7.3 mm off; splayed from the others in the palm view |
| Skin free of creases and folds | 4 | 4 | 4 | No flipped triangles |
| Hands locked together | n/a | n/a | 4 | Trail ring and little fingers lie on the lead forefinger in all three rolls |

#### MuJoCo Full-Swing Render: Spike, Not yet Wired

The owner asked to see this grip on the club in a full swing. A scratch spike (not committed) put the pair from `build_grip_hands` on the club body of the MuJoCo full-body golfer in place of the capsule hands and rendered a still. The mesh path works unchanged (`grip_hands.grip_hand_meshes` to `club_chain_meshes` to the MuJoCo appearance layer): the golfer's hands are rigid on the club, so one pose serves the whole swing. Missing before a swing video:

1. A switchable backend in `grip_hands` (capsule hands stay the default) with tests, and a cache: the pair takes seconds to solve.
2. One grip geometry. The club assembly's grip is 11.36 mm at the lead hand and `golf_grip.json` gives 10.92 mm; the two also place the lead hand 58 mm apart along the grip.
3. The wrist. The skinned hands carry a forearm stub pointing where the solved wrist put it; the golfer's forearms point elsewhere, so the stub must be cut and the hand turned to the model's wrist.
4. A swing to replay. The reviewed MuJoCo bundle is fitted to a private capture; the public replay file has not been checked as an input.

This is outside the two branches this session may push to, so it is not started.

#### Forces and Pressures Later

The skinned hand is visual only. Forces and pressures stay on the existing contact model (`src/shared/python/grip_contact/`: pad layout, Hunt-Crossley pad contact, force decomposition). Each skinned result already reports `bone_transforms` and `contact_gaps_m`. To show pressure on this skin later: both models must read one grip spec and one grip geometry (item 2 above), and each contact pad needs a named skin region (the rig's dominant-bone vertex sets already give one per segment) so a pad's force can be painted on its region.

### Earlier Update at `4e8598b464` (2026-11-16): Club Under the Lead Heel Pad (65 mm)

The owner answered decision 2 below with "Yes, go deeper".

- **Lead club line 65 mm below the little-finger knuckle** (`owner_lead_heel_pad_depth_m`, was 45). The contract `pinky_palm_depth_range_m` now ends at 70 mm, not 45; the reason and the approval are recorded beside it in `hand_visual_quality.json`. This is the only limit that was widened.
- **A joint limit for one finger.** `curl.limits_deg` takes `joint.finger` keys beside the joint-wide ones. The default sets `mcp.little` to 115 degrees: at the deep line the lead little finger otherwise stops at 95 degrees and bridges 5 mm off the grip. The joint-wide limit stays 95, so the other fingers and the trail hand pose as before. Remove the key to switch it off.
- **Creases at the little knuckle.** The deep line pinched 16 skin edges there (limit 6). `correctives.crease_relax` now runs 96 passes over 3 rings (was 32 over 2) and `fold_relax` 32 passes (was 16). To keep the correctives local, `lift_out.bulge_rings` went from 3 to 2: correctives move 30.7 % of the lead skin and 31.5 % of the trail skin (limit 34 %, unchanged).
- **Heel-pad ratchet tightened**: `max_lead_heel_pad_gap_m` 18 mm to 9 mm.
- **One test restated, not loosened in a quality limit.** `test_the_palm_cups_so_the_thumb_base_lies_on_the_grip` asserted that without the palm cup the thumb's first segment is more than 5 mm off the grip. On the lead hand that is no longer true (2.6 mm uncupped, 0.3 mm cupped), because the deep line already brings the thumb base down. It now asserts the cup brings the segment at least 2 mm closer (`min_palm_cup_first_segment_approach_m`, new); the 5 mm limit on the cupped hand and the 10 mm thenar approach are unchanged.

#### Measured at `4e8598b464`

| Check | Now | At `5344c9496d` | Limit |
| --- | --- | --- | --- |
| Lead heel pad to the grip, nearest skin | 7.1 mm | 17.8 mm | 9 mm (ratchet) |
| Lead heel pad, tenth percentile of its skin | 17.6 mm | not recorded | none |
| Lead heel pad azimuth | 5 degrees from the top | 5 | 30 |
| Skin within 0.5 mm of the grip, lead hand | 26.7 cm2 | 23.8 cm2 | at least 18 cm2 |
| Lead little finger middle segment to the grip | under 1 mm | 1.4 mm | 1.5 mm |
| Lead little finger first segment to the grip | 6.7 mm | 6.3 mm | none on first segments |
| Pinched knuckle edges, lead and trail | 3 and 1 | 3 and 2 | 6 |
| Flipped triangles | 0 | 0 | 0 |
| Pair fault | 1.1 mm (little finger middle segment) | 0.87 mm | parts checked at 1.0 and 1.5 mm |
| One hand inside the other | 0.0 mm | 0.0 mm | 1.0 mm |
| Forearms apart | 32 degrees | 33 | 15 to 50 |
| Lead fingers off square to the shaft, middle and end segments | 15 to 50 degrees | 17 to 43 | none (soft rule) |

Tests: 150 pass, 1 skipped, **2 expected failures** (see Blockers).

#### Honest Reading

- The palm now lies across the top of the shaft and the heel pad's nearest skin is 7 mm from the grip. It still does not touch, and most of the pad is further off (tenth percentile 17.6 mm). Hogan's heel pad is pressed on. The remaining gap needs a palm fold along the club line in the pose, as described below.
- **The fingers got worse against the "more perpendicular" request**: the diagonal club line is what tilts them, and the lead middle and ring fingers are now up to 50 degrees off square. These two owner requests pull against each other in this rig.
- The trail hand's fault rose from 0.87 to 1.1 mm.

#### Self-Review of the Frames at `4e8598b464` (Harsh; Out Of 5)

I opened the roll 0 frames of the lead hand, the trail hand and the pair, and the pair at roll +20 and -20 (20 frames). **I did not open the single-hand roll frames** (16); the roll is tested as a rigid turn.

| Criterion | Lead | Trail | Both | Note |
| --- | --- | --- | --- | --- |
| Reads as a hand on a real golf club | 4 | 4 | 4 | Unchanged |
| Heel pad on top of the grip (Hogan) | 3 | n/a | 3 | Up from 2. In the butt view the heel of the hand now covers the grip and the cap sits against the pad; in the palm view the palm crosses the shaft. The pad still hovers; it is not pressed on |
| Thumb base full on top | 4 | 4 | 4 | Unchanged |
| Thumb snug to the hand | 2 | 2 | 3 | Unchanged: long straight thumbs, open notch; owner decision 1 |
| Fingers square to the shaft | 2 | 3 | 2 | Lead down from 3: the lead fingers run visibly diagonal, most in the palm view |
| Lead little finger on the grip | 3 | n/a | 3 | Its end segments are on the grip; the first segment stands 6.7 mm off and the finger looks splayed from the others in the palm view |
| Skin free of creases and folds | 4 | 4 | 4 | No flipped triangles; the little knuckle is sharply bent (115 degrees) but smooth |
| Hands locked together | n/a | n/a | 4 | Trail ring and little fingers lie on the lead forefinger in all three rolls |

### Earlier Update at `5344c9496d` (2026-11-16): Real Golf Grip, Hogan Review, Heel Pad, Thumb Base

**The "State" list and the "Measured" table below it were taken at `d577971830` on a 14 mm cylinder and have not been re-measured line by line**; where they disagree with this section, this section is right. The images further down are from `81c38e59f7`.

The owner supplied Ben Hogan reference photographs and three rounds of comments on 2026-11-16. What changed:

- **A real golf grip (#15185).** `scripts/config/golf_grip.json` and `golf_grip.py`: men's standard, 0.900 in at 2 in below the cap edge and 0.780 in at 6 in, with a butt cap. The hands are posed on that taper (`grip.shape`: `"tapered"` or `"cylinder"`), the quality measures are taper-aware and the renderer (#15154) draws it. The grip radius under the lead hand is 11.3 mm, down from the 14 mm cylinder.
- **Bug fixed: the thumb was laid at the wrong turn.** `_lay_thumb` searched the turn but returned the pose of the last turn tried, not the chosen one.
- **Pair settle.** On the thin grip the settle flipped between "ring finger on the grip" and "ring finger against the lead forefinger". `build_grip_hands` now tries `pair.spacing_passes` settles, ends each with a rigid slide of at most `pair.final_slide_max_m`, and keeps the best (`pair.spacing_pick`: `"best"` or `"last"`).
- **Lead heel pad further over the grip (owner, third comment).** The lead club line is 45 mm below the little-finger knuckle (`owner_lead_heel_pad_depth_m`; it was just below the knuckle, then 35 mm). The lead wrist is 8 degrees extended so the forearms keep their spread.
- **Fuller thumb base (owner, third comment).** New `rest_shape.py` and setting `mesh.fullness`: pads of tissue added to the rest mesh before subdivision. The default adds 4 mm on the back of the thumb's base. An empty list switches it off. This changes MakeHuman's mesh; it is a declared deviation.

#### Review Against the Hogan Photographs

| What the photographs show | Before (`ac40d96949`) | Now | Verdict |
| --- | --- | --- | --- |
| A tapered rubber grip with a butt cap | 14 mm cylinder, no cap | Standard tapered grip with cap | Done |
| Club from the crook of the forefinger to under the heel pad; heel pad on top of the grip | Heel 26 mm off the grip | Heel 17.8 mm off the grip, over the top (azimuth 5 degrees) | **Open, partly done** |
| No air between hand and club | Thenar pad 5.5 mm off | Thumb base on the grip (0.6 mm) | Better; the heel gap above remains |
| Thumb short, pressed against the hand, closed V | Thumb joint 39 mm from the forefinger knuckle | 32.7 mm lead, 40.8 mm trail | **Open** (owner decision below) |
| Top of the thumb's base full, not hollow | Hollow | 4 mm of tissue added | Done, to be judged by the owner |
| Fingers nearly square to the shaft | Middle and end segments 32 to 40 degrees off square | Lead 17 to 43, trail 19 to 37 degrees off square | **Open, not improved** |
| Hands locked, trail lifeline over the lead thumb | Overlap grip | Kept; ring finger 0.9 mm and little finger 0.4 mm from the lead forefinger | Holds |

#### Measured at `5344c9496d` (Tapered Grip, 11.3 mm at the Lead Hand)

| Check | Value | Limit (setting) |
| --- | --- | --- |
| Lead heel pad to the grip, nearest skin | 17.8 mm (22.0 at 35 mm depth; 26 before) | 18 mm (`max_lead_heel_pad_gap_m`, new; set just above what is reached, as a ratchet) |
| Lead heel pad, azimuth of that skin | 5 degrees from the top | 30 (`max_lead_heel_pad_azimuth_deg`, new) |
| Skin wholly within 0.5 mm of the grip, single hands | 23.8 cm2 lead, 19.1 cm2 trail | at least 18 cm2 |
| Pair fault (worst of: trail fingers off the grip, ring finger off the lead forefinger) | 0.87 mm | none; the tests check its parts at 1.0 and 1.5 mm |
| One hand inside the other | 0.0 mm | 1.0 mm |
| Forearms apart, seen from the butt | 33 degrees | 15 to 50 |
| Lead little finger first segment to the grip | 6.3 mm (1.8 at 35 mm depth) | none on first segments |
| Lead little finger middle segment to the grip | 1.4 mm | 1.5 mm |
| Thumb end joint to own forefinger knuckle | 32.7 mm lead, 40.8 mm trail | checked in the knuckle mode only |

Tests: 149 pass, 1 skipped, **2 expected failures** (see Blockers).

#### What I Tried for the Heel Pad and Did Not Ship

- **Deeper club line.** At 55 mm the heel is 12.5 mm off, at 65 mm 7.4 mm, and the frames look clearly closer to Hogan. But the existing contract `pinky_palm_depth_range_m` stops at 45 mm, the lead little finger comes 9 to 12 mm off the grip, and the trail hand unseats (pair fault 4.4 mm). I did not widen the contract. **Owner decision needed:** may the club lie deeper than 45 mm below the little-finger knuckle?
- **Stronger metacarpal arch** (ring and little, up to gain 1.0 and 50 degrees). It makes the heel worse: the arched knuckles land first and the rigid seat lifts the heel (17.1 mm at 55 mm depth, contact area down to 17 cm2).
- **Longer palm-pad reach** (20 to 35 mm). The nearest heel vertex comes to 4 mm but the mass of the pad does not move (tenth percentile 28 mm), the frames look the same, the shift would pass the 8 mm corrective limit, and above 20 mm it unseats the trail hand.
- **Turning the lead hand round the shaft** does not help: the palm is a flat plate tangent to the grip, so the heel hovers the same distance at any turn.

The cause is structural: this rig's palm does not fold along the club line. A real palm wraps about 55 mm of skin round a 22 mm grip between the finger bases and the heel pad. Getting the heel fully down needs a palm fold in the pose (a hinge along the club line carrying the heel side of the wrist bone), not a setting.

#### Owner Decisions Open

1. **Thumbs at 45 degrees or snug on top?** At 45 degrees the thumb cannot also lie against the forefinger (the forefinger knuckle is on the other side of the grip). The knuckle mode closes the lead thumb to 23 mm and matches the "change of grip" photograph, but the trail hand then loses the grip (8.7 cm2) until it is retuned. An A/B image was sent to the owner.
2. ~~Club deeper than 45 mm under the lead heel pad?~~ Answered 2026-11-16: yes. Done at 65 mm, see the newest section.
3. **Forefinger knuckle straighter** conflicts with `min_index_knuckle_flexion_deg` (15) and unseats the trail ring finger; unchanged.

#### Not Verified, or Weak

- The red phase of the tapered-grip tests was observed for the solver and the measures; for the renderer change in #15154 it was not.
- The new heel-pad limit is set 0.2 mm above the value reached. It records progress and will fail on any regression; it is not evidence that 18 mm is good. Hogan's heel pad is on the grip.
- A lead-hand turn of `palm_centre_wrap_deg` to -10 closed the thumb to 24.6 mm but made a strong, un-Hogan grip and failed four tests; reverted.

#### Self-Review of the Frames at `5344c9496d` (Harsh; Out Of 5)

I opened the roll 0 frames of the lead hand, the trail hand and the pair (12 frames). **I did not re-open the 24 roll frames this round**; the roll is tested as a rigid turn.

| Criterion | Lead | Trail | Both | Note |
| --- | --- | --- | --- | --- |
| Reads as a hand on a real golf club | 4 | 4 | 4 | The tapered grip and cap do most of this |
| Heel pad on top of the grip (Hogan) | 2 | n/a | 2 | In the butt view the heel now covers part of the grip; in the palm view daylight still shows between heel and grip |
| Thumb base full on top | 4 | 4 | 4 | The hollow between thumb knuckle and wrist is filled; the outline is slightly angular at the wrist end |
| Thumb snug to the hand | 2 | 2 | 3 | Long straight thumbs with an open notch; owner decision 1 |
| Fingers square to the shaft | 2 | 3 | 3 | Lead fingers visibly diagonal in the palm view |
| Hands locked together | n/a | n/a | 4 | The trail forearm stub ends close to the frame and looks odd in the thumb-side view |

It does not reach the bar. The lead heel pad and the finger angle are the two things a golfer would pick out first.

### State at `d577971830` (2026-11-16)

- The hand is MakeHuman's hm08 mesh with MakeHuman's own `default.mhskel` skeleton and `default_weights.mhw` skin weights (CC0, pinned commit, recorded in `assets/hand_cc0/NOTICE` and `PROVENANCE.json`). Two declared changes to the weights and one to the mesh: see "Deviations".
- Posing is joint rotation plus linear blend skinning. Four owner-approved corrective shapes then move a little skin (`correctives.py`): fold relaxation around inside-out triangles, a palm pad that settles onto the grip, laying skin pressed up to 4 mm into the grip flat on its surface (`lift_out`, now with a bulge), and (new) relaxing creases that skinning pinched between the knuckles. All are switchable. With all off, a test rebuilds the mesh from the reported bone transforms and requires agreement to 1e-12 m; with them on, the solved joint angles are identical.
- **The skin is Loop-subdivided once** (`subdivision.py`, setting `mesh.subdivision_levels`, 0 switches it off): 7,074 vertices and 14,144 triangles per hand, up from 1,770 and 3,536. Skin weights go through the same operator. The grip is solved on the original vertices of the subdivided skin, so the solve costs the same and a solved contact is a contact of the drawn skin.
- **New: nail edges stay sharp.** Edges on the fingertip bones that fold by 40 degrees or more in MakeHuman's mesh are kept as creases through subdivision (`mesh.creases`; an empty bone list switches it off).
- **New: the hand squeezes the grip (#15179, #15180).** The joints used to stop at first touch, so each finger segment touched at one spot and the grip looked loose. The digits now close `squeeze.finger_depth_m` (2.5 mm) and the palm seats `squeeze.palm_depth_m` (1.5 mm) into the grip, standing for soft pad compression. The `press_flat` corrective then lays the pressed skin on the surface and bulges the skin beside it a little. Zero depths give the old pose. The drawn skin never enters the grip.
- The pose follows the grip spec: flat hand beside the club, club under the index finger's first segment and just below the base of the little finger, then MCP, PIP, DIP close in that order until the skin touches. Knuckle flexion drives the ring and little metacarpals so the palm cups.
- **New: the palm cups (#15181).** The thumb's base joint sat 66 mm from the grip axis, so the thumb reached the grip as a ramp. The base joint is now carried along an arc about a line parallel to the club, toward the line where the thumb is to lie, before the thumb is aimed (`thumb.palm_cup_max_deg`, zero switches it off). The thumb lies along the grip on both hands.
- **New: the forefinger wraps with the other fingers (#15182).** A flat hand seated on the grip has the forefinger's first segment lying on it, so that knuckle could not flex and the finger hooked from its middle joint. The hand is now seated with the knuckle already flexed (`preshape.index_knuckle_flexion_deg`, 15 degrees; zero or `GripSpec.trigger_finger` switches it off), so the club lies in the crook of the finger. Knuckle flexion went from 10 to 19 degrees and the forefinger lies 0.7 mm from the middle finger.
- **New: the fingers touch side to side (#15184).** After the curl, the ring finger swings sideways at its knuckle until its skin meets the middle finger, the little finger meets the ring finger and the index finger meets the middle finger. The index finger keeps a slightly larger gap (`side_close.index_gap_m`, 3 mm; the others `gap_m`, 1.5 mm). Zero `side_close.max_deg` switches it off. The swings are reported as `<finger>_side` joint angles, so supplying the solved angles rebuilds the same pose.
- **New: overlap grip, hands closed up (#15183).** `build_grip_hands` (`skinned_pair.py`) poses both hands on one club. The trail hand is now posed round the lead hand's actual skin, not capsules (`pair.obstacle`: `"skin"` or `"capsules"`). It slides toward the lead hand until its ring finger is against the lead forefinger (`pair.max_spacing_close_m`, `pair.finger_gap_m`), and its little finger swings sideways into the groove between the lead forefinger and middle finger (`pair.overlap_max_swing_deg`, `pair.overlap_bias_m`; zero switches the overlap off). Where the lead forefinger tip stood through a coarse triangle of the trail palm, the trail skin gives way (`pair.yield_max_depth_m`).
- **New: each thumb lies on its own side of the shaft (#15181, owner request 2026-11-16).** The lead thumb lies about 45 degrees round the grip on the side away from the target, the trail thumb about 45 degrees on the target side. That is the spec's `thumb_azimuth_deg` set to 45 on both hands (`owner_thumb_azimuth_deg` in `hand_visual_quality.json`; the evidence and the tests use it). Pressing a thumb onto the grip carries its pad sideways (the trail thumb landed 23 to 31 degrees short), so the thumb is now turned until the middle of its end bone lies within `thumb.azimuth_tolerance_deg` (4) of the spec's line (`thumb.toward_index_target = "azimuth"`, an even scan from zero that stops at the first pose close enough and otherwise keeps the closest it tried, capped by `thumb.toward_index_max_deg`). The palm cups further for it (`thumb.palm_cup_max_deg` 90, was 70), so the thumb's first segment lies on the grip; the trail hand of a pair cups only `pair.palm_cup_max_deg` (70), because its palm lies on the lead thumb. The two earlier behaviours are kept as settings: `"knuckle"` (the tight V the owner accepted before: thumb end joint by the forefinger knuckle) and `"fingertip"`. Zero `toward_index_max_deg` switches the turn off. The turn is reported as the `thumb_side` joint angle. **Not delivered yet:** the thumb and forefinger pinched together at 45 degrees, and a straighter forefinger knuckle; see "Deviations".
- The forearm ends in a rounded cap, not a flat disc.
- Each hand declares which vertices a renderer should frame on (`frame_focus`: everything except the forearm stub), so the hand itself fills 0.70 of the frame.
- Shaft roll comes from `HandGripSpec.shaft_roll_deg`. Every size, tolerance, limit, gain and colour is in `scripts/config/skinned_human_hand.json`, `scripts/config/hand_visual_quality.json` or `assets/hand_cc0/build_settings.json`.
- Contact physics is untouched (still capsules). This is the visual model.
- Tests: 124 pass, 1 skipped (the source-rebuild check needs a MakeHuman checkout; the asset file is unchanged in this round).

### Earlier Measurements, at `d577971830` (Legacy Grip Spec, 14 mm Cylinder)

| Check | Before | Now | Limit (setting) |
| --- | --- | --- | --- |
| Thumb first segment to grip | 12.2 and 14.5 mm | 5.1 mm lead, 4.9 mm trail | 6 mm |
| Thumb pad round the grip from the top (middle of the end bone; plus is away from the target) | lead 10, trail 0 (legacy spec) | lead +44 degrees, trail -49 degrees; trail -48 in the pair | 45 plus or minus 5 (new) |
| Thumb first segment, median gap of the skin facing the grip | 15.0 mm lead, 16.4 mm trail | 2.2 mm lead, 0.0 mm trail on single hands; **5.5 mm for the trail hand in the pair** | 5 mm, checked on single hands |
| Thumb base pad, median gap of the skin facing the grip | 35.7 mm lead, 37.0 mm trail | 21.3 mm lead, 18.1 mm trail | at least 10 mm closer than without the cup (new) |
| Finger to its neighbour, skin to skin (single hand) | index-middle 8.7 mm, middle-ring 4.6 mm, ring-little 2.1 mm | 0.7 mm, 0.3 mm, 0.9 mm (swings: index 14, ring 12, little 15 degrees) | 2 mm; index 5 mm (new) |
| Forefinger knuckle flexion | 10 degrees (middle finger 62) | 19 degrees | at least 15 (new) |
| Forefinger middle and end segments to the middle finger | 3.2 and 2.3 mm | at most 0.7 mm | 4 mm each (new) |
| Trail fingers to the lead index finger (pair) | ring 1.8 mm, little 0.2 mm | ring 0.5 mm, little 0.9 mm (1.07 before `pair.overlap_bias_m` went from 3.0 to 2.5 mm) | ring 1 mm (tightened from 3), little 1 mm |
| Trail little finger to the lead middle finger (pair; the groove) | 18 mm | 0.7 mm | 1.5 mm to each of the two fingers (new) |
| Hand spacing along the grip (pair) | 76.2 mm (spec) plus slide | 74.9 mm (not measured again this round) | spec minus 30 mm to spec plus 30 mm |
| Thumb end joint to its own forefinger knuckle, joint centre to joint centre | 32.7 mm lead, 38.0 mm trail | 40 mm lead, 47 mm trail with the thumbs at 45 degrees; 24.8 and 22.1 mm with `toward_index_target = "knuckle"` | 26 mm lead, 23 mm trail, checked in the knuckle mode only |
| Trail thumb to its own forefinger tip, with `toward_index_target = "fingertip"` | 19.0 mm | 1.4 mm | 2.5 mm |
| Thumb first free segment to the hand beside it | not measured | 1.7 mm lead, 1.6 mm trail (2.1 mm trail in the pair) | 3 mm |
| Each finger segment to the neighbouring finger (single hand) | not measured | 0.5 to 3.2 mm on every segment | none yet; the test checks the least distance per finger |
| Fingertip and middle phalanx to grip, single hands | 0.95 mm | 1.0 mm | 1.5 mm |
| Trail index, middle, ring to the grip itself (pair) | ring 6.8 mm | at most 1.0 mm | 1.5 mm |
| Skin inside the grip | 0.3 mm | 0.0 mm (0.9 mm before the lift-out corrective, see Learnings) | 0.5 mm |
| One hand inside the other (pair, mesh winding number) | 0.59 mm | 0.00 mm (0.9 to 1.6 mm reported before the trail skin was given a 0.3 mm clearance; see Deviations) | 1.0 mm |
| Forearms apart, seen from the butt | 28 degrees | 18 degrees | 15 to 50 |
| Degenerate triangles | 0 | 0 | 0 |
| Triangles folded into a crease | 2 per hand | 0 per hand, single and pair | 0 (tightened from 2) |
| Palm skin within 1.5 mm of the grip | 0 vertices | 95 vertices of the finer mesh, palm gap 0.3 mm (was 1.1 mm) | at least 8 |
| Farthest a corrective moves any vertex | 0 | 4.8 mm lead, 6.0 mm trail | 8 mm |
| Skin within 0.5 mm of the grip (pair) | 10.2 cm2 lead, 11.1 cm2 trail | 28.0 cm2 lead, 24.5 cm2 trail | none; the test uses the triangle measure below |
| Triangles wholly within 0.5 mm of the grip, single hands | under two thirds of now | 18.6 cm2 lead, 20.2 cm2 trail (17.9 and 17.7, under the limit, before the cup went from 70 to 90 degrees) | at least 18 cm2 and 1.5 times the unsqueezed hand |
| Angle of the grip covered by skin within 2 mm, per 10 mm slice | 100 to 150 degrees | 150 to 250 degrees | none yet |
| Finger and thumb bone axis to the grip surface | 4.8 mm | 1.75 mm lead (thumb tip), 2.7 mm trail | at least 1.5 mm (new) |
| Vertices any corrective moves | 7 % | 28.6 % lead, 28.2 % trail | 32 % (widened from 12 %: pressed pads move by design) |
| Mean triangle edge | 5.6 mm | 2.7 mm | 4 mm (new) |
| Edges pinched more than 45 degrees past rest, palm and first finger segments | 21 lead, 18 trail | 3 lead, 3 trail (the tighter knuckles pinch more web skin) | 6 |
| Hand share of the frame (forearm left out) | 0.48 to 0.60 | 0.67 to 0.70 | 0.70 plus or minus 0.12 |
| Left-handed golfer is the exact mirror | yes | yes | 1e-9 m |
| Shaft roll 0 and plus or minus 20 degrees is a rigid turn | yes | yes | 1e-9 m |

### Earlier Self-Review, at `d577971830` (Scores Out Of 5)

| Criterion | Lead | Trail | Both hands | Change |
| --- | --- | --- | --- | --- |
| Separate fingers and thumb | 4 | 4 | 4 | same |
| Cupped palm | 3 | 3 | 3 | same; accepted by the owner for this PR (2026-11-15) |
| Knuckle line | 4 | 4 | 4 | same |
| Fingers wrap the grip | 4 | 4 | 4 | same score, visibly tighter: the pads lie flat on the grip. The thumb now lies along the grip (#15181). The fingers of each hand now lie against each other, the index finger a little apart at the tip (#15184). The forefinger now wraps with them (#15182). Not 5: its knuckle flexes 19 degrees against 62 for the middle finger; the trail hand now lies against the lead forefinger, but the trail lifeline is not yet on the lead thumb (#15183) |
| No tubes, blobs or torn triangles | 4 | 4 | 4 | same: no folded triangle on the finer mesh either |
| Smooth shading | 5 | 5 | 4 | trail up from 4: the knuckle creases are relaxed; the pair keeps faint creases where the fingers interlock |
| Correct mirroring | 5 | 5 | 5 | same |

Each hand reads as a human hand holding a golf club, and the pair now reads as two hands gripping one club with the trail fingers closed on it. It does not reach the bar yet. What still falls short:

- **Palm cup (3).** The palm pad corrective closes the heel of the hand onto the grip (12 palm vertices in contact, up from none), which shows in the butt views. From the palm view the palm still reads flatter than a real one. The thenar pad stays about 30 mm clear of the grip: in a finger grip the thumb's base sits about 66 mm from the grip axis. A stronger pad setting (25 mm reach, four passes) did close it, but it swelled the thumb and creased the palm, so I did not ship it.
- **Thumb.** With the palm cup the whole thumb lies along the grip: end pad on it, first segment 2.5 to 3.7 mm off. Harsh points: the skin between the wrist and the thumb is stretched, so the thumb base looks drawn out in the palm and butt views, and the base pad itself is still about 20 mm from the grip.
- **Thumb-to-index hollow.** What I called a notch between the knuckles is the hollow where the index finger leaves the hand behind the thumb. The pinched creases in it are relaxed; a small dark fold remains at its bottom in the lead thumb-side view.
- **Nail rims.** The nails have crisp rims again. The thumb nail's rim is slightly uneven where the source mesh has few edges.
- **Inner finger creases.** The creases inside the middle finger joints are still sharp (up to 119 degrees past rest). They face the grip and are hidden, and relaxing them would move skin that is in contact, so they are left out of the crease corrective.
- **Butt view.** Fixed in #15154: the view is now tilted to the palm side, so it shows the fingers wrapping and the heel of the hand on the grip. The grip end still crosses part of the frame.
- **Thumbs at 45 degrees (new).** Both thumbs are on the sides the owner asked for and both now lie on the grip along their whole length. Harsh points: both thumbs are long and dead straight (a "long thumb"), where a golfer's thumb is usually drawn back a little; in each thumb-side view an open notch shows between the thumb and the forefinger knuckle; in the pair's back view the lead thumb pad shows above the trail hand, which is right for this thumb position but makes the hands look less locked together than the earlier V. The pinch is visibly looser than the V the owner accepted: the joints are 40 and 47 mm apart, and only the skin at the web is closed (about 1.7 mm). In the pair the trail thumb's first segment stands 5.5 mm off the grip.
- **Overlapped little finger.** It reaches the groove and touches both lead fingers, but it looks thin and a little clawed (side swing 30 degrees, at its limit).
- **Lead thumb in the pair.** It is hidden under the trail palm, as in a real grip. The trail thumb shows in the back and thumb-side views.

| | Palm | Back | Thumb side | Butt | Golfer |
| --- | --- | --- | --- | --- | --- |
| Lead | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_palm_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_back_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_thumb_side_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_butt_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_lead_golfer_rollp00.png?raw=true) |
| Trail | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_palm_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_back_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_thumb_side_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_butt_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_trail_golfer_rollp00.png?raw=true) |
| Both | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_palm_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_back_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_thumb_side_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_butt_rollp00.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_golfer_rollp00.png?raw=true) |
| Both, roll -20 | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_palm_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_back_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_thumb_side_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_butt_rollm20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_golfer_rollm20.png?raw=true) |
| Both, roll +20 | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_palm_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_back_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_thumb_side_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_butt_rollp20.png?raw=true) | ![](https://github.com/D-sorganization/UpstreamDrift/blob/81c38e59f76c8a3acff8fbcfa5dab42c51a41689/docs/development/full_body_models/evidence/hand_skinned_human/skinned_both_golfer_rollp20.png?raw=true) |

All 36 frames (1024 px, MuJoCo EGL through the shared renderer of #15154) and `render_manifest.json` are in `docs/development/full_body_models/evidence/hand_skinned_human/`.

### Deviations and Things to Check

- **Corrective shapes move skin after skinning** (`correctives.fold_relax`, `palm_pad`, `lift_out` and, new, `crease_relax`; approved 2026-11-15). Together they move at most 6.6 mm (palm pad). Lift-out moves finger skin by at most 0.9 mm, and only onto the grip surface. Crease relaxation acts only on skin led by the palm bones and the first finger segments. Zero iterations, zero reach and zero depth switch them off and give pure linear blend skinning back. The reported `contact_gaps_m` are measured on the corrected skin.
- **Fold relaxation may now run up to 16 passes** (was 8). It stops as soon as no triangle is inside out; one triangle needed more than 8 after the crease corrective.
- **The skin is subdivided at load, not in the asset file.** I had said build time; doing it at load keeps `hand_cc0.npz` and its provenance hash untouched and makes it a plain setting. Subdivision pulls the surface in by 0.5 mm on average (1.7 % of volume); the hand is scaled after subdividing, so length and breadth still match the spec.
- **A test contract changed.** The test used to require that no finger skin moves after skinning. It now requires that finger skin moves by at most the lift-out depth and ends on the grip surface.
- **A second test contract changed (this round).** The test that compares the solve on the coarse and on the subdivided skin required every joint angle to agree within 5 degrees. The thumb's turn (`thumb_side`) is now a searched value, and on the trail hand the search answers 15 degrees apart on the two skins (-15 and -30) while the thumb pad ends at -43 and -49 degrees. The test now leaves `thumb_side` out of the joint comparison and requires both skins to put the thumb within 5 degrees of its line. Every other joint is still held to 5 degrees. Please check that this is acceptable.
- **The first-segment limit is checked on single hands only.** In the pair the trail thumb's first segment is 5.5 mm off the grip (limit 5). I did not widen the limit; the pair value is reported here and is open.
- **Skin weights at the thumb web are smoothed** (`thumb_web_smoothing`, approved). Zero iterations rebuilds with MakeHuman's untouched weights.
- **Skin weights at the finger middle creases are lightly smoothed** (`finger_crease_smoothing`, kept with the owner's go-ahead on 2026-11-15). One ring, two passes at half blend. It took the folds from 5 to 2 per hand before the correctives. Zero iterations turns it off.
- **The forearm cut is closed with a dome** (`forearm_cap`: 4 rings, depth 0.6 of the loop radius). These vertices are not MakeHuman's. Zero rings gives the flat fan back.
- **The trail hand may slide up to 30 mm along the grip** either way from the spec's hand spacing (`pair.max_spacing_adjust_m` away, `pair.max_spacing_close_m` toward the lead hand). It now ends 0.9 mm closer than the spec. Zero for both keeps the spec's spacing exactly.
- **The evidence no longer uses the legacy spec's thumb azimuths** (10 and 0 degrees). It uses 45 on both hands, as the owner asked on 2026-11-16. `GripSpec.neutral()` on `main` has 22.5 for the lead thumb and 45 for the trail thumb; I did not change it, the owner may want it at 45.
- **Thumb and forefinger are not pinched at 45 degrees.** The owner asked for the thumbs at 45 degrees and for thumb and fingers kept pinched together. With the hand placed as the spec says, the forefinger knuckle is 51 degrees round the grip on the other side from the lead thumb, and 91 degrees round on the trail hand, so a thumb on the 45 degree line is 40 to 47 mm from the knuckle, joint to joint. The skin at the web is closed (1.7 and 1.6 mm), the joints are not together. Turning the whole hand toward the thumb (`placement.palm_contact_azimuth_offset_deg` 55 to 25) brought them to 31 to 38 mm and changes where the club lies in the fingers; I did not ship it. This needs the owner's call: which gives way, the 45 degrees or the hand placement.
- **The forefinger knuckle is not straightened.** The owner asked for the joint between palm and forefinger to be straighter. It flexes 19 degrees now; the seat setting `preshape.index_knuckle_flexion_deg` takes it to 15 (at 8), 13 (at 5) or 10 degrees (at 0). Single hands hold, but in the pair the trail ring finger then comes off the grip (middle segment 2.4 to 4.1 mm, limit 1.5), because its fingertip lands on the straighter lead forefinger; at 8 the subdivided and coarse solves also disagree by 7.7 degrees (limit 5). More spacing passes or a wider finger gap fix one number and break another (overlap 1.05 to 1.43 mm, limit 1.0). Measured again after the yield clearance removed the overlap (seat angle 0, knuckle 10 degrees): contact area rises to 22 and 29 cm2 on single hands, but the single trail thumb then misses its line (-61 degrees), and in the pair the trail ring finger is 4.1 mm off the grip, or, with 2 mm of room to the lead hand, the trail little finger's first segment is 2.8 mm off and the ring finger 1.8 mm from the lead forefinger (limit 1). It is left at 15 until the trail hand can be settled against a straighter lead forefinger.
- **In the knuckle mode neither thumb reaches its knuckle.** Lead: past 35 degrees the trail ring finger no longer wraps (middle segment 1.6 mm off at 41 degrees, 2.1 mm at 45, limit 1.5). Trail: past about 80 degrees the skin of the thumb web folds (one flipped triangle and nine pinched edges at 90 degrees, limits zero and six). I did not loosen either limit. The owner accepted that V as tight enough on 2026-11-16, before asking for the thumbs at 45 degrees.
- **The trail hand is no longer slid clear of the lead thumb.** The spacing step used to slide the trail hand along the grip until its fingers stood clear of the whole lead hand. A fingertip resting against the side of the lead thumb can never be cleared that way, because the thumb runs along the grip, so the hand was pushed 30 mm down the shaft. The step now settles against the lead hand without its thumb; the fingers are still posed around the thumb.
- **The thumb's middle joint may bend back** (MCP to -10 degrees) so the pad lies flat; the end joint no longer does (IP from 0). These limits are my values.
- **The palm cup moves the thumb's base joint** by up to 90 degrees of arc (was 70) about a line through the middle of the palm; the trail hand of a pair by up to 70 (`pair.palm_cup_max_deg`). A real hand gets there by folding the whole palm; this model moves only the thumb ray and lets skinning stretch the skin between. Visual only, switchable. With 70 on single hands and the thumbs at 45 degrees, two limits were missed by a little (contact area 17.9 and 17.7 cm2 against 18; trail thumb first segment 5.2 mm against 5). I did not loosen them; the deeper cup meets them. Above 73 degrees the trail hand of the pair is erratic (thumb at -31 to -69 degrees), so it stays at 70.
- **Where the hands all but touch, the trail skin stands 0.3 mm off the lead hand** (`pair.yield_clearance_m`, zero switches it off). The lead thumb lay in a hollow of the trail palm: 0.05 mm above the nearest trail vertex, yet inside the skin between vertices, which the overlap measure reports as up to 1.6 mm.
- **Wrist bend** (lead 0 and 20 degrees, trail 15 and 10) and **`palm_centre_wrap_deg` (37)** are my values, tuned by eye.
- **Finger-length references are unverified.** The ratios in `hand_visual_quality.json` are recalled from Buryanov and Kotiuch 2010 and were not checked against the paper.
- **Shared posing has no branch of its own**, so `hand_backends/posing.py` and its tests live here.
- **Size.** Several thousand added lines, far over the 500-line gate. It needs `large-pr-approved` or an owner decision.

### Blockers

- **CI.** After the merge of `main`, `code-quality` failed on two type errors (the scipy stubs that `main` now uses) and `unit-test-gate (1/2)` on a stale divergence inventory (four new hand modules). Both are fixed in this round: `code-quality` passes on `796707e4f5`; `unit-test-gate` was still running when this was written. The Fleet policy gate passes and is left alone as instructed.
- **Coarse and fine skin solve apart (#15181).** `test_the_skin_is_subdivided_and_the_setting_switches_it_off` is a strict expected failure on both hands: on the 0.900 in grip the two solves differ by more than the 5 degree limit (lead `ring_prox` 6.3, trail `thumb_prox` 8.4, because the thumb's turn search finds its 45 degree line on one skin only). Sweeping heel depth and `toward_index_steps` only moved the failure. The limit is unchanged. This is a real open defect, not a tolerance change.
- **Tighter grip, fingers under the lead thumb pad (owner, fourth comment).** Not done. Pressing the finger pads deeper puts finger bone inside the grip (table in the newest section). It needs the thumb's base brought down onto the fingertips, which is owner decision 1 (thumb at 45 degrees against thumb snug to the hand).
- **Clubface and forearm attachment in the MuJoCo golfer.** Measured (hands 46 degrees round the shaft from the face; wrists 57 mm from the model's wrist joints), not fixed. The fix moves the wrist joints on the club, so the swing must be refitted; order of work proposed in the newest section, waiting for the owner.
- **Thumb and forefinger pinch.** Partly. The full pinch unseats the trail hand in the pair.
- **MuJoCo full-swing render.** Spiked only; the wiring lives in `grip_hands` and the club assembly, outside the two branches this session may push to.
- **Fleet claim.** The lease on #15148 ran out at 2026-11-16 18:14Z. Re-claiming returned 409 (`claim:claude` label present, no active lease); the label still marks the issue as this session's.

## Learnings

- A flat palm plate tangent to a 22 mm grip cannot put the heel pad on it: a point 25 mm past the tangent line hovers 16 mm off. Turning the hand, arching the metacarpals and pulling pad skin do not change that; only moving the tangent line (club deeper in the palm) or folding the palm does.
- Arching the ring and little metacarpals lifts the heel, because the arched knuckles land first and the seat is rigid.
- A nearest-vertex measure can improve by 18 mm while the picture does not change. Judge a pad by the frames and by a percentile, not by its closest vertex.
- Missing tissue belongs in the rest shape, not in a pose corrective: it then needs no pose logic, does not count against the corrective limits and is there in every pose.
- A distance to a cone is not radius minus cone radius; divide by the slant, or a "move to the surface" step overshoots.
- On a thinner grip the two-hand settle has two near-equal answers and flips between them. Try several, score them, keep the best.
- The lead wrist angle only moves the forearm, so it can restore the forearm spread without touching the pose.

- Where the wrist sits round the shaft is set by turning the whole hand about the shaft. That turn is also what a clubface roll is: decide which one is the authority before wiring a hand to a club.
- A thumb line fixed on the club moves against the hand when the hand is turned. Re-check every thumb measure after a hand turn.
- If a result must be the same after a rigid turn, solve once and turn the result. Solving again in a turned frame let one skin dent flip on rounding (0.3 mm).
- A wait loop that greps the process list for a name in its own command line never ends.
- Pressing the finger pads deeper does not tighten a grip whose fingers are already on the club: the bones go into the grip and neighbouring fingers move apart. A tighter look has to come from the pose (thumb base onto the fingertips), not from squeeze depth.
- A hand placed so its heel hangs past the butt cap can never have the heel pressed on. Check where the hand is along the grip before working on the gap.
- A mean over every vertex is not a measure of where the palm is; wrapped fingers and the forearm cancel it. Measure the named skin.
- Reverting an experiment by hand and re-rendering gave byte-identical frames, so the evidence render is reproducible.
- When a limit is missed by a little after a requested change, look for what the change took away before touching the limit. Contact area fell 1 cm2 because the thumb's base was now farther from its line; a deeper cup gave it back with margin.
- A turn search over a pose that jumps must return the closest pose it tried, not the last. The last try sat on the wrong side of the jump, 20 degrees off.
- Stop a search at "close enough" (4 degrees). It ends away from the jump, so tiny numeric differences no longer pick different poses, and it saves solves.
- A point can be inside a mesh and above its nearest vertex (a hollow between vertices). A depth measured from the nearest vertex misses it; give the dent a small clearance.
- An overlap measure that reports the distance to the nearest surface sample overstates by up to half an edge: 1.6 mm reported was 0.05 mm real. Find the vertices before believing the number.
- Pressing a thumb onto the grip moves it sideways. Aimed at 45 degrees, the trail thumb's pad landed at 14 to 22. Measure where the pad ends up and turn the thumb until that is on the line.
- A fixed-point correction of that turn overshot to the limit (the pad moves more than one degree per degree of turn near the web). Bisection on the signed miss settled in 8 steps.
- Two requirements can be geometrically exclusive, and it is cheaper to measure that than to tune toward it. Thumb on the 45 degree line and thumb end joint by the forefinger knuckle differ by 39 to 45 mm with this hand placement; no thumb setting closes that.
- A straighter forefinger on the lead hand is a different obstacle for the trail hand. Changing one seat angle by 7 degrees unseated the trail ring finger by 2 to 4 mm. Check the pair before the single hand looks finished.
- Keep the earlier behaviour as a mode when the owner changes direction. The knuckle V and the fingertip pinch are still one setting away and still tested.
- A new dependency version on `main` can break typing on a branch that did not change (scipy stubs: `tree.query` returns a union). Wrap the result in `np.asarray`.
- Pose the second hand round the first hand's real skin, not capsules. Capsules either stand the hands apart or let them overlap; the skin gave ring finger 0.4 mm from the lead forefinger with 0.00 mm overlap.
- A finger that holds the grip must aim for the grip alone. When contact with the other hand counted as success, the trail ring finger rode up on the lead forefinger (2 to 10 mm off the grip).
- Fit the sideways closing on a pose that sees only the grip. Fitted with the other hand in the way, the swings were wrong and the result jumped as the spacing changed.
- All vertices of a coarse mesh can be clear while a fingertip stands through one of its triangles. Let the skin give way at the nearest vertices (3 mm at most) and measure overlap on the mesh.
- The thumb cannot meet the fingertip by aiming alone: it is a straight chain 92 mm long from its base. Turn it toward the fingertip and stop it on the finger's skin.
- Search a pose setting with a plain curl, not the wrap scan. The wrap scan jumps between answers, so the gap against the turn was jagged and each try cost 2 s; the plain curl was steady at 0.1 s and could be bisected.
- A search needs to know when a pose is not a pose. A thumb that started inside the other hand never moved and still counted as valid; test for "never pressed down" and "off the grip".
- What suits one hand can break the pair. The lead thumb turned across the grip put the trail thumb on top of the lead hand, 16 mm off the grip. Check the pair render after every single-hand change.
- Ask what a search can and cannot change. Sliding along the grip cannot clear an obstacle that runs along the grip; the slide went to its far limit instead. Leave such an obstacle out of that search.
- Lay a moved digit with the same routine as an unmoved one. The quick curl left the turned lead thumb's base 3 mm higher than the wrap search does, and that alone displaced the other hand.
- Aim at joints, measure joints. "Thumb end joint by the forefinger knuckle" is a distance between two joint centres; a skin-to-skin gap was already 1 mm at the web and said nothing.
- The skin sets the limit before the skeleton does: the trail thumb could turn 90 degrees, but its web folded past 80.

- A finger whose first segment already lies on the obstacle cannot flex at the knuckle. Flex it before seating the hand (15 degrees), so the obstacle sits in the crook; more than 20 degrees lifts the palm off (first-segment gaps of 3 to 14 mm at 35 to 60 degrees).
- Measure the gap between fingers from the middle and end segments only. The web at the bases is always close, so a whole-finger minimum barely changes with the swing and the answer flipped between the coarse and the fine mesh (4 against 11 degrees of swing).
- Take the direction of a sideways swing from joint positions, not from a skin-gap probe: on a curled finger the probe picked the wrong way and sent the little finger 20 degrees outward.
- Different correctives may need different obstacles. Slimmed collision shapes let the fingers of two hands touch, but a palm pad that settles onto slimmed shapes is drawn into the other hand (1.5 mm). The pad now uses full-size shapes.
- After `git checkout <file>` to drop an experiment, check what else was uncommitted in that file. I lost and re-applied a feature that way.

- A capsule of full skin radius is wider than the skin almost everywhere, so a hand solved against capsules stands off the other hand. Shrink the capsules and check overlap on the meshes: 0.75 of the radius plus 0.5 mm closed the hands with 0.00 mm overlap; 0.70 without margin overlapped 1.2 mm.
- Sliding the trail hand toward the lead hand does not close a gap the collision shapes cause; it only lifts the ring finger off the grip (3.6 mm at 9 mm of slide).
- A least-distance measure says fingers touch when they touch at one point. Check each segment too, and say in the review that the rendered dark lines between fingers are grooves 1 to 3 mm wide, not daylight.

- Fingers that curl in parallel planes do not touch: their bases are a knuckle width apart. Close them with a sideways swing at the knuckle, measured skin to skin, in the order ring, little, index with the middle finger fixed.
- Do the sideways closing between the greedy curl and the final wrap. The wrap then re-seats every pad on the grip with the swing in place, and grip contact stays as it was.
- Anything recomputed inside forward kinematics multiplies by tens of thousands of calls. A cross product per finger per call doubled the solve time (6 s to 13 s); storing the turn matrix when the swing is set brought it back.
- A pose term that is not in the reported joint angles breaks "given angles rebuild the pose". Report it as a named angle.

- Rig with the source's own skeleton and weights. Cropping a body mesh and inventing weights gave tubes and tears; MakeHuman's rig gave a hand on the first pose.
- Fit finger hinge axes to the palm plane, not to each finger's nearly straight rest shape. The per-finger fit fanned the fingers.
- Greedy proximal-to-distal curl hooks a digit on its tip. Scan the first joint, close the others to contact, and keep the pose whose named segments both touch (`wrap_to_contact`). This also laid the thumb flat once its joints were allowed to bend back a little.
- The thumb is a ramp when its base joint is far from the grip, and no thumb joint limit fixes that: a flat palm is tangent to the grip, so the base joint ends up 66 mm from the axis. Move the base joint (palm cup), then aim.
- Carry the thumb base without turning it. Turning the thumb ray with the cup swings the tip round the grip faster than the base, and the aim (limited swing) cannot bring it back: the thumb landed 90 to 140 degrees off its azimuth.
- Choose the cup angle by where the base joint ends up relative to the thumb's line, not by first contact of the base pad: the pad never touches before the joint has passed the line.
- A joint that may bend backwards gives the wrap search two answers that are equally good, and it flips between them on small mesh changes. Allowing the thumb end joint only forward made the coarse and subdivided solves agree.
- For two hands, decide the spacing first: solve the trail hand alone on the bare grip, slide it by the least amount that frees its gripping fingers from the lead hand, then solve against grip plus lead hand. Curling against the lead hand at the spec's spacing left the ring finger 7 mm off the grip.
- Measure hand-in-hand overlap on the meshes (winding number), not on the capsules used to solve. A 1 mm capsule margin removed the last 1 mm of mesh overlap.
- Light weight smoothing at a crease removes folds; more smoothing adds them and lifts the palm off the grip.
- The folds that weights cannot remove go with a local Laplacian relaxation after skinning: two passes at half step over two edge rings were enough. Measure folds on the final mesh, not on a re-skinned copy, or the check cannot see the corrective.
- A contact pad corrective must be mild. Twelve millimetres of reach with a smoothed displacement field seats the heel pad; 25 mm swells the thumb. Smooth the displacement, not the positions, and repeat the pass, because smoothing spreads each pass short of contact.
- Zero folded triangles is not the same as no visible crease. A tight but valid crease still catches the light; report both.
- Frame the camera on the hand, not on hand plus forearm. A bounding-box fill of 0.70 that includes the forearm stub shows the hand at about 0.55.
- Subdividing the skin once removes facets that no shading setting could. Push the weights through the same operator and the finer mesh skins exactly like the coarse one.
- Solve the grip on the original vertices of the subdivided skin, not on the unsubdivided mesh. Subdivision shrinks the surface by about half a millimetre, so a contact solved on the old surface is a gap on the new one.
- A coarse mesh hides penetration. Two touching vertices 5 mm apart span a chord that dips 0.3 mm into a 14 mm grip; the finer mesh put vertices there and measured 0.9 mm. Lift that skin to the surface as flesh would compress, and cap the depth so a real posing fault still shows.
- A brute-force winding number over a 4x mesh is 16x the work (20 s per pair). A ray-parity test along two axes preselects the candidates and the exact winding number runs on those only: 0.6 s, same answer on every point checked.
- Measuring breadth in a band across the knuckles depends on the hand's length, so fit the breadth on the scaled hand, not before scaling.
- A view straight down the shaft shows the end cap, not the hand. Tilt it to the palm side.
- Smoothing a mesh blurs the features that were sharp on purpose. Tag the edges that are sharp in the source (the nails) as creases and subdivide with crease rules: midpoint on the edge, 3/4 and 1/8 along it at a vertex.
- Zero folded triangles still leaves pinched creases. Measure the fold of each edge against its fold at rest and relax the ones that grew too much; an absolute angle limit would flatten the nails.
- Keep a visual corrective off skin that is in contact. The sharpest creases are inside the finger joints against the grip; relaxing them would undo the contact the solver found, and nobody sees them.
- Two relaxers in a row can leave a fold the first one made. Run the fold check last and give it room to finish.
- A solver that stops at first touch gives point contact, and point contact reads as a loose grip however good the hand looks. Close a pad-compression depth past first touch and flatten the pressed skin; contact area nearly tripled.
- Shrink the obstacle for the solver rather than adding a compression term to the stop rule. Every joint then closes the same depth with no new logic, and the lead-hand capsules are left at full size so the hands still do not overlap.
- A tighter pose folds more skin. Each time the joints close further, re-check folded triangles and pinched edges and widen the relaxers' reach (rings) before adding passes.
- Relax creases before folds, never after: crease relaxation last re-folded two triangles.
- A contact-area measure that counts skin inside the obstacle cannot tell pressing from penetrating. Pair it with the penetration check.
- Solver speed is forward kinematics. Precomputed hinge matrices and skinning only the vertices a joint moves took a pair from 55 s to 14 s.
- Never run Python from inside `hand_backends/`: its `types.py` shadows the standard library.
- A joint limit shared by all fingers couples the hands: raising the knuckle limit for the lead little finger changed the trail hand's solve and its corrective footprint. Limits that one finger needs belong to that finger (`joint.finger` keys).
- The 34 % locality limit on correctives is a budget shared by four correctives. Wider crease smoothing was paid for by a narrower press bulge, not by raising the limit.
- A deeper club line and squarer fingers pull against each other in this rig: the diagonal line that puts the heel pad over the grip is what tilts the fingers.

### Next Steps

Filed from the grip review (2026-11-15), in order of impact:

1. #15181 thumb: palm cup done; thumbs on their own sides at 44 and 49 degrees. Still open: thumb and forefinger pinched together at 45 degrees (needs the owner's call above, or a hand placement that brings the forefinger knuckle round to the thumb); the lead thumb reads straight; in the pair the trail thumb's first segment is 5.5 mm off the grip and the pair is erratic above 73 degrees of cup.
2. #15182 forefinger: wraps (knuckle 19 degrees). The owner now asks for this knuckle straighter on both hands. Measured and not shipped: 15, 13 and 10 degrees are reachable alone but unseat the trail ring finger in the pair. Next: settle the trail hand against the straighter lead forefinger (re-fit its sideways closing and spacing), then lower the seat angle.
3. #15183 join the two hands: ring finger against the lead forefinger, little finger in the groove, hands 1.3 mm inside the spec spacing. Still open: trail lifeline on the lead thumb; the little finger looks clawed; the pair is touchy to the lead thumb's turn.
4. #15184 close the fingers side to side: done within each hand. Each segment is within 0.5 to 3.2 mm of its neighbour.
5. #15185 tapered grip with a butt cap.
6. #15186 surface realism.

Open in this round: the bulge beside a pressed pad is faint (the skin there faces the grip, so little of its normal points sideways); the lead little fingertip shows a ragged nail edge where the creased nail rim is pressed flat; bone thickness is not modelled, only the bone axis.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
