# Grip Hand Penetration Repair Evidence

PR #15142, issue #15141. The [editable calculation reference](penetration_reference.tex)
defines the signed triangle diagnostic, frame, units, limits, assumptions and
reproduction. It is a separate appearance research reference, not the engineering
design manual. No compiled reference artifact is claimed.

The previous vertex-proximity guard passed while the middle proximal shell
penetrated the dorsum by 6.271 mm. The signed triangle guard fails on both hands
before the correction (maximum deep fraction 14.76%, limit 11%). Ending the
rounded palm and dorsum ten degrees before the MCP azimuth removes the deep
overlap at the tested geometry. Physical contact capsules and grip-pose defaults
are unchanged. The depth and fraction tolerances are unchanged.

The new analytic tests cover deep interior, exterior, face contact, edge distance,
reversed winding, an open-surface refusal and a concave notch. Both hands are
checked at 12.7, 14.0 and 14.5 mm grip radii. Vertex sampling is not a bound on
triangle-interior intersection or intersection volume.

## Static Review Views

These 1200 by 1200 PNGs use three camera views at each visual hand roll and
include the club grip. The saved canvas dimensions and equal-axis half-widths
are in [the render receipt](render_stats_after.json). The visual roll offsets
are not qualified physical shaft-roll states. Earlier `before_*` stills remain
historical evidence at their original resolution and camera setup.

| Visual Roll | Palm | Back | Side |
| --- | --- | --- | --- |
| -20 Degrees | [View](acceptance_palm_roll_-20.png) | [View](acceptance_back_roll_-20.png) | [View](acceptance_side_roll_-20.png) |
| 0 Degrees | [View](acceptance_palm_roll_+0.png) | [View](acceptance_back_roll_+0.png) | [View](acceptance_side_roll_+0.png) |
| +20 Degrees | [View](acceptance_palm_roll_+20.png) | [View](acceptance_back_roll_+20.png) | [View](acceptance_side_roll_+20.png) |

## Remaining Acceptance Gates

The capsule construction still has visible component seams and does not establish
anatomical quality. Human visual approval is outstanding, including the wider
#15150 quality bar and its physical-roll requirements. These static renders do
not qualify contact, native engines, dynamics, or the full hand-model epic.
Independent exact-head review and protected CI are still required.

The broader contact suite has 27 failures at both the unchanged source head and
the repair environment, with identical failing test IDs. Those include native
Pinocchio availability/test-double issues, frozen provenance receipts and the
driver closure tolerance. They remain failures; this evidence does not claim a
green native contact matrix.
