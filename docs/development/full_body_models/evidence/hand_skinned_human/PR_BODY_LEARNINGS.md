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
