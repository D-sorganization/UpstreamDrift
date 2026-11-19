# Skinned Human Hand: Handoff

Epic #15150, issue #15148, PR #15149, branch `cursor/skinned-human-hand-backend-751f`.

State, scores, frames, deviations, learnings and next steps are in
[`../evidence/hand_skinned_human/PR_BODY.md`](../evidence/hand_skinned_human/PR_BODY.md),
which is also the text for the PR description.

## Where Things Are

| Path | What it is |
| --- | --- |
| `assets/hand_cc0/` | MakeHuman right-hand extract (`hand_cc0.npz`), builder, NOTICE, PROVENANCE |
| `src/shared/python/model_appearance/hand_backends/skinning.py` | Asset type, hierarchical linear blend skinning, triangle checks |
| `.../hand_backends/posing.py` | Shared curl-to-contact posing for any backend (`PosableHand`) |
| `.../hand_backends/skinned_human.py` | This backend: `build_hand_visual`, `solve_grip_joint_angles`, settings loader |
| `.../hand_backends/skinned_pair.py` | Both hands on one club: `build_grip_hands` (trail hand closes up to the lead hand along the grip, then curls round the grip and the lead hand's skin; overlap grip) |
| `.../hand_backends/correctives.py` | Switchable corrective shapes after skinning: fold relaxation, the palm pad, laying skin pressed into the grip flat on its surface with a small bulge beside it (`press_flat`), and relaxing creases pinched at the knuckles (owner-approved 2026-11-15). The solver closes the joints `squeeze.finger_depth_m` past first touch (#15179, #15180) |
| `.../hand_backends/subdivision.py` | Loop subdivision of the skin mesh and its weights (`mesh.subdivision_levels`, 0 switches it off), with the nail edges kept sharp (`mesh.creases`); the grip is solved on the original vertices of the subdivided skin |
| `.../hand_backends/golf_grip.py`, `scripts/config/golf_grip.json` | The real grip: standard taper and butt cap (`grip.shape`: `tapered` or `cylinder`) |
| `.../hand_backends/rest_shape.py` | Tissue added to the rest mesh before subdivision (`mesh.fullness`; an empty list switches it off). Default: 4 mm on the back of the thumb's base |
| `.../hand_backends/qa.py` | Contact, penetration, fold, anthropometry, pair and forearm metrics |
| `scripts/config/skinned_human_hand.json` | Every pose setting |
| `scripts/config/hand_visual_quality.json` | Every test tolerance and reference |
| `tests/unit/model_appearance/hand_backends/` | 54 tests |
| `docs/development/full_body_models/evidence/hand_skinned_human/` | Evidence script, 36 frames, manifest |

Settings are loaded and validated in `hand_backends/skinned_settings.py`
(split out of `skinned_human.py` to keep it under the file-size limit).

## Commands (Run From the Repository Root)

```bash
PYTHONPATH=. python -m pytest tests/unit/model_appearance/hand_backends -o addopts="" -q
```

```bash
PYTHONPATH=. python assets/hand_cc0/build_hand_asset.py <makehuman checkout> --check
```

```bash
HAND_RENDERER_ROOT=<checkout of cursor/hrg-shared-hand-renderer-cec5> MUJOCO_GL=egl PYTHONPATH=. python docs/development/full_body_models/evidence/hand_skinned_human/render_skinned_human_evidence.py
```

The evidence script needs the shared renderer of PR #15154, which is not on
this branch; `HAND_RENDERER_ROOT` points at a checkout of it. Do not run
Python from inside `hand_backends/`: its `types.py` shadows the standard
library.

## How the Pose Is Built

1. Size the asset to the spec's hand length and breadth.
2. Lay the flat right hand beside the club (`club_axis_in_hand`,
   `flat_hand_rotation`) and seat it rigidly on the grip.
   The forefinger knuckle is flexed first (`preshape.index_knuckle_flexion_deg`,
   zero or `GripSpec.trigger_finger` switches it off), so the club lies in
   the crook of that finger and the finger can wrap.
3. Curl the fingers to contact, set the metacarpal arch from knuckle flexion,
   re-seat, then run the wrap search.
   Before the final wrap the ring, little and index fingers swing sideways
   at the knuckle until their middle and end segments meet the neighbour (`side_close`, zero
   `max_deg` switches it off; the swings are reported as `<finger>_side`
   joint angles).
4. Cup the palm: carry the thumb base toward the line where the thumb is to
   lie (`thumb.palm_cup_*`, zero switches it off; the trail hand of a pair
   cups only `pair.palm_cup_max_deg`, because its palm lies on the lead
   thumb). Then lay the thumb (`_lay_thumb`): aim it along the shaft, press it
   onto the grip, and turn it to its target (`thumb.toward_index_target`).
   `"azimuth"` (the default, owner request 2026-11-16) turns it until the
   middle of its end bone is within `thumb.azimuth_tolerance_deg` of the
   spec's `thumb_azimuth_deg`, because pressing carries the thumb sideways;
   the search is `skinned_targets.scanned_turn` and keeps the closest pose
   it tried, since the pose jumps where the thumb lifts. `"knuckle"` brings
   the thumb's end joint within `thumb.knuckle_distance_m` of its own
   forefinger's knuckle (a tight V; use `toward_index_max_deg` 35 lead and 75
   trail there: the pair limits the lead, the thumb web's skin the trail).
   `"fingertip"` sends the thumb's skin to the forefinger tip.
   `thumb.toward_index_max_deg` caps the turn per hand, zero switches it off;
   the turn is reported as the `thumb_side` angle. The evidence and the tests
   use 45 degrees for both thumbs (`owner_thumb_azimuth_deg` in
   `hand_visual_quality.json`): lead thumb on the away-from-target side,
   trail thumb on the target side. In a pair the trail hand's spacing settles
   against the lead hand without its thumb (`_LeadObstacle.hand(thumb=False)`).
5. Bend the wrist, mirror for a left hand, apply the shaft roll.

For a pair, the lead hand is solved first; its skin (`posing.skin_distance`)
is the trail hand's obstacle (`pair.obstacle`, `"capsules"` gives the old
stand-in). The trail hand is solved once at the spec's spacing, slid until its
gripping fingers are `pair.finger_gap_m` off the lead hand, and solved again.
In an overlap grip its little finger swings into the groove between the lead
forefinger and middle finger. Fingers named in `pair.grip_fingers` aim for the
grip alone; the lead hand is only in their way. Last, the trail skin gives way
where the lead hand stands through it, and stands `pair.yield_clearance_m`
off it where the two all but touch (`correctives.yield_to_points`).
Targets of a hand live in `skinned_targets.py`.

## Open After the Grip Request (2026-11-16)

- Thumb and forefinger are not pinched with the thumbs at 45 degrees: the
  joints are 40 mm (lead) and 47 mm (trail) apart, only the web skin is
  closed. The forefinger knuckle is 51 and 91 degrees round the grip from
  the thumb's line, so no thumb setting closes it.
- The forefinger knuckle is not straightened. `preshape.index_knuckle_flexion_deg`
  below 15 holds on single hands and unseats the trail ring finger in the pair.
- In the pair the trail thumb's first segment is 5.5 mm off the grip, and the
  trail hand is erratic above 73 degrees of `pair.palm_cup_max_deg`.
- The lead heel pad is 7.1 mm off the grip with the club 65 mm below the
  little-finger knuckle (`owner_lead_heel_pad_depth_m`; the owner approved
  going past the old 45 mm bound of `pinky_palm_depth_range_m` on
  2026-11-16). The lead little finger needs `curl.limits_deg` key
  `mcp.little` (115 degrees) to reach the grip there; `joint.finger` keys
  override the joint-wide limit for one finger. The deep line creases the
  skin at the little knuckle, so `crease_relax` runs 96 passes over 3 rings
  and `lift_out.bulge_rings` is 2 to keep the correctives under 34 % of the
  skin. Still open: the bulk of the heel pad (tenth percentile 17.6 mm) does
  not touch; the palm needs a fold along the club line in the pose. A
  stronger metacarpal arch makes it worse; less arch flips triangles.
- Wrists over the shaft (2026-11-17): `placement.palm_centre_wrap_deg` lead 15
  and trail 10 put the lead wrist joint on the shaft plane and the trail
  wrist 44 mm to the trail side (`qa.wrist_joint_m`). Trail 0 and 14 break
  the pair (fault 4.7 and 7.6 mm). Thumb line is per hand
  (`owner_thumb_azimuth_deg` lead 45, trail 40 since #15278; it was 20); 30 on both fails five tests.
  The trail forefinger is a trigger finger (`side_close.trigger_spread_deg`,
  30; owner 2026-11-18): exempt from the fingers-touch limits, brought back
  until its tip seats (12 on a lone hand), thumb tip 8 mm past its middle joint.
  The heel pad went from 6.3 to 7.7 mm for it (ratchet 8 mm, evidence on
  #15150). `build_grip_hands` solves unrolled and turns the result.
  No thumb joint bends back past straight (owner 2026-11-19):
  `thumb.mcp_limits_deg` starts at 0 (was -10; lead thumb sat at -9.3), floor
  `min_thumb_joint_flexion_deg` measured on the posed bones.
- Trail palm 13 mm off the grip: the seated hand is backed straight off the
  lead thumb. Seating it tilted over the thumb closes it to 1 to 3 mm but the
  ring finger then rides on the lead forefinger below about 95 mm hand
  spacing (now 80). Not adopted; owner decision on spacing, see #15150.
- MuJoCo golfer, measured by spike: hands 46 degrees round the shaft from the
  clubface (`ADDRESS_SQUARE_FACE_ROLL_DEG` squares the head only;
  `grip_hand_frame` turns the hands from the wrist's dorsal axis); skinned
  wrist joints 57 mm from the model's wrist followers. Fix order is in the
  PR body and needs the owner's go-ahead: it moves the wrist joints.
- At `f4f607338c` the lead hand sits 85 mm below the cap edge
  (`owner_lead_below_cap_edge_m`) so the butt stands 5 mm proud of the heel;
  the heel pad's nearest skin is 6.3 mm off the grip (ratchet 7 mm).
  `correctives.palm_pad.reach_m` stops at 13 mm: 14 mm fails the palm-cup
  test and 14.5 mm exceeds the 8 mm corrective shift limit. The evidence
  has a fifth view, `golfer` (`evidence_settings.json`).
- A tighter grip with the fingertips under the lead thumb's pad is open.
  `squeeze.finger_depth_m` is not the lever: 3 mm moves the ring fingertip
  away, 3.3 mm puts skin 4 mm inside the grip, 4 mm puts finger bone inside
  the grip and fails 12 tests. It needs the thumb's base brought down onto
  the fingertips (the thumb 45 degrees against snug decision).
- MuJoCo full swing: a scratch spike showed the pair loads on the golfer's
  club through `grip_hands.grip_hand_meshes`. Not wired: needs a switchable
  backend in `grip_hands`, one grip geometry (club assembly 11.36 mm against
  `golf_grip.json` 10.92 mm, hand 58 mm apart along the grip), the forearm
  stub cut and turned to the model's wrist, a cache, and a swing bundle.
- The fingers are 15 to 50 degrees off square to the shaft on their middle
  and end segments, worse on the lead hand since the club line went deeper
  (the diagonal line is what tilts them). The Hogan photographs show them
  nearly square.
- The coarse-skin and fine-skin solves differ by more than 5 degrees on the
  thin grip (lead `ring_prox`, trail `thumb_prox`): a strict expected
  failure tied to #15181.
- Do not wait on a background test run with `pgrep -f` of a pattern that is
  also in the waiting command: it matches itself and never returns.

## Knuckle Sideways Range (#15278)

Owner review 2026-11-19: the lead forefinger had a crooked kink; fingers bend
in flexion and extension, cannot rotate and have limited lateral motion.

- Audit: each finger's three hinge axes are parallel and square to its bones
  to about 2 degrees. The fault is the fit's sideways swing at the knuckle,
  15 to 20 degrees on knuckles bent 90 to 115 degrees.
- `knuckle_sideways.py` and `side_close.limit_by_flexion`: the swing is held
  inside `knuckle_max_deg`, falling to `flexed_share` of that by
  `flexed_at_deg`. Off by default; a strict expected failure in
  `test_knuckle_sideways.py` records the gap.
- With it on, the lead hand alone has no leaning fingers
  (`evidence/trail_hand/lead_knuckle_spread_before_after.png`) but its little
  finger is 3.7 mm off the grip. On the pair at the owner's 52.7 degree cross
  angle the trail ring finger rides the lead forefinger, 4.5 mm off the grip.
- Trail cross angle 25 to 35 degrees seats it (fault under 1 mm, hands 2 mm
  closer, ring finger 18 mm from the lead forefinger instead of 26), but the
  forearm leaves the shaft nearly square; 30 degrees of ulnar deviation does
  not bring it back. 58 to 70 degrees is worse (fault 9 to 15 mm).
- Tried and dropped: turning the finger's bending plane about an axis fixed in
  the palm instead of leaning it. On a bent knuckle that twists the finger
  about its own length; the trail ring finger stopped flexing at its knuckle
  and stood 10.7 mm off. Also: more spacing passes (2.8 mm fault at 83 mm);
  a 1 mm forefinger gap; lead forefinger knuckle preshape of 40 to 85 degrees.
- Disproved: seating the trail palm closer. Letting it press 3, 5 or 8 mm
  into the lead hand's skin (owner approved 5 mm) leaves the ring finger
  further off (5.6, 6.4, 7.9 mm) and unseats the trail thumb.
  `pair.palm_capsule_radius_scale` does nothing while `pair.obstacle` is
  `skin`.
- Disproved: trail cross angle 35 degrees with 25 degrees of ulnar deviation
  and the limit on. It seats every finger, but 15 tests fail: the trail heel
  leaves the lead thumb (18.6 mm against 2.5), the little finger leaves its
  groove (4.5 mm), the thumb tip passes the forefinger joint by 17.8 mm.
- A part limit has no sweet spot: `flexed_share` 0.33, 0.45, 0.55 leave the
  trail ring finger 3.6, 2.6, 1.9 mm off (limit 1.5).
- The lead hand's cross angle changes nothing; `owner_lead_heel_pad_depth_m`
  sets its seat. Its middle and ring fingers wind 46 to 49 degrees off square
  to the shaft, the trail hand's 40 to 44; the squarer trail hand that seats
  winds 25. The lead forefinger's 20 degree swing is on a knuckle bent only
  21 degrees, which the limit allows: its crook is not the swing alone.
- Open: with anatomical knuckles the palm-on-thumb seat and the ring finger
  beside the lead forefinger do not both hold. Untried: a shallower lead heel
  pad (squarer lead fingers), which is an owner setting.
