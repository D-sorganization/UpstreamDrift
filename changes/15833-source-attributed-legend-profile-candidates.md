---
issue: 15833
summary: "Add source-attributed appearance candidates for five golf legends"
dl_state: "in_review"
next_step: "Merge the PR; final delivery #15836 adds Snead's straw_hat after #15838 lands."
branch: "pr/15833-legends-profiles"
---

# Source-Attributed Legend Appearance Candidates

Adds catalog profiles for Seve Ballesteros, Jack Nicklaus, Bobby Jones, Sam
Snead and Byron Nelson on the existing ellipsoid-body and CC0-head pipeline.
Each profile includes published stature/mass priors, linked reference footage,
distinct illustrative appearance materials and bounded face morphs. A public
`player_ids()` accessor exposes configured profile IDs to callers.

Measurements are source-attributed priors, not exact measurements for a
specific frame or career year. Conflicting values and unclear dates are
preserved in source notes and limitations. Bobby Jones's mass is expressly
approximate and comes from a secondary encyclopedia. Outfits, skin/hair swatches
and face shapes are stylized interpretations; no precise likeness, historical
motion, body scan or physical qualification is claimed.

Sam Snead's current appearance-v1 entry leaves headwear off to remain valid
against main. Final delivery issue #15836 applies the shared `straw_hat` /
`hat_straw` visual after implementation #15838 lands.

Focused unit coverage validates all five entries, their source catalog,
appearance documents, dimension inputs and public ID ordering. Native player
dimension verification is handled separately from this lightweight contract
suite.
