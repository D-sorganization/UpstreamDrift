# Sampled Iron Ground Contact

This diagnostic slice of issue 15523 observes ground-first iron strikes on a
recorded sole path. Import `analyze_iron_ground_contact` from
`src.tools.iron_ground_contact`. It accepts timestamps in
seconds, signed sole heights in metres, and ball contact time in seconds on the
same clock. Heights must already be relative to the local turf surface, with
positive above turf. The caller owns sole geometry, frame conversion, terrain
sampling and ball-contact detection.

## Method and Contract

Between adjacent samples the sole height is linear. A sign change has zero
crossing fraction `abs(z0) / (abs(z0) + abs(z1))`; the implementation rescales
heights before adding to avoid overflow. Negative segments define penetrating
contact intervals. Tangency at zero without penetration is excluded. First
contact is the first interval's start. Pre-ball contact duration sums each
interval intersected with the observed time before the ball. Airborne gaps
contribute nothing. Any positive pre-ball duration classifies the shot as fat;
entry exactly at ball time does not. Maximum depth is the greatest observed
penetration over the entire trace, including after ball contact.

Records need at least two finite paired samples, strictly increasing times,
finite total duration and ball time inside the record. A record starting below
turf is rejected because its ground-entry time is unknown. A record ending in
turf is allowed: duration is bounded by the observed record. Earlier contact
outside the record cannot be inferred. Outputs are immutable and input sequences
are not changed. Invalid records raise `ValueError`.

## Limits and Validation

This is an engine-neutral diagnostic, not a force or launch model. It does not
change the existing divot model or swing-to-flight pipeline, apply energy or
spin losses, estimate cut volume, or claim measured physical acceptance.
Sampling can miss excursions between samples; no smoothing or tolerance hides
that uncertainty. Sole-height accuracy and sufficiently fine sampling remain
caller responsibilities. No empirical coefficients or external sources are used.

Acceptance tests use analytical piecewise-linear traces: interpolated entry,
ball-first and simultaneous entry, tangency, multiple contacts, completed and
truncated contact, large finite heights and invalid records. Reproduce with:

```sh
python3 -m pytest tests/unit/physics/test_iron_ground_contact_15523.py -m unit -q
python3 -m ruff check src/tools/iron_ground_contact.py tests/unit/physics/test_iron_ground_contact_15523.py
python3 -m ruff format --check src/tools/iron_ground_contact.py tests/unit/physics/test_iron_ground_contact_15523.py
```

The supplementary editable LaTeX calculation note is `iron_ground_contact.tex`;
it is separate from the governed engineering design manual.
