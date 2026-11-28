---
issue: 15956
summary: "Apply source-indexed, bounded height/mass frame priors to the four contemporary golfer candidates while keeping visual body build independent."
dl_state: "in_review"
next_step: "Independent exact-head review; no source changes to fitted reference-skin physics."
branch: "feat/15956-modern-frame-priors"
---
- Rory McIlroy, Jon Rahm, Dustin Johnson and Sergio García select the PGA TOUR profile source matching each catalog height and mass. Their dates are unspecified, so the values remain roster priors rather than frame-specific measurements.
- The sized candidate uses the bounded equal-density/geometrically-similar estimate from #15947. It is a starting assumption, not measured girth or capture-era body shape; visual `body_build` stays independent. Raw/clamped prior values and the selected source are available in the sized receipt.
- Added source-index, era, distinct-inertia, fixed-mass/COM, native mass-matrix and fitted-reference-skin parity checks.
- Fleet briefing timed out; work proceeded under the existing RM #2851 scope after parent authorization to fail open.
