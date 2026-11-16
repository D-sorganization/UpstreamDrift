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
4. Cup the palm: carry the thumb base toward the line where the thumb is to lie (`thumb.palm_cup_*`, zero switches it off), then lay the thumb (`_lay_thumb`): aim it along the shaft and press it onto the grip, turned toward its own forefinger tip by the least angle that brings the two within `thumb.index_gap_m` (`thumb.toward_index_max_deg` per hand, zero switches it off; reported as the `thumb_side` angle).
5. Bend the wrist, mirror for a left hand, apply the shaft roll.

For a pair, the lead hand is solved first; its skin (`posing.skin_distance`)
is the trail hand's obstacle (`pair.obstacle`, `"capsules"` gives the old
stand-in). The trail hand is solved once at the spec's spacing, slid until its
gripping fingers are `pair.finger_gap_m` off the lead hand, and solved again.
In an overlap grip its little finger swings into the groove between the lead
forefinger and middle finger. Fingers named in `pair.grip_fingers` aim for the
grip alone; the lead hand is only in their way. Last, the trail skin gives way
where a lead fingertip stands through it (`correctives.yield_to_points`).
Targets of a hand live in `skinned_targets.py`.
