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


Earlier updates and measurements: [`PR_BODY_EARLIER.md`](PR_BODY_EARLIER.md). Learnings, blockers and next steps: [`PR_BODY_LEARNINGS.md`](PR_BODY_LEARNINGS.md).
