# F07 Generic Shoulder Triangle-Lamina Turnover

Issue #12211 adds a **new** candidate plant, not an edit or accepted repair
of the original derived 557-muscle source. The exact original source and
author-right CEINMS mesh SHA-256s are fixed in
`tour_matching/generic_shoulder_lamina.py`; the reviewed entry point rejects
other bytes. The missing BUET author mesh bytes are not inferred. Right mesh
placement is a declared identity-frame hypothesis and left local vertices are
z-reflected with reversed winding and reflected normals. Native frames,
muscle paths, joints, wraps, force laws and scalar body masses remain in the
source XML. Any other semantic XML edit rejects.

The provider is Tools #5503 `shared.python.humanoid_character_builder.mesh`
`triangle_lamina`, with source hashes for both its kernel and inertia result
representation. The Tools gitlink remains unchanged while #5503 is unmerged;
the reproducible local command accepts an explicit development Tools root.
No lamina tensor formula is copied into UpstreamDrift. The exact mesh is
open/nonmanifold, so the solid-volume route stays failed. The zero-thickness
triangle-multiset recipe is a separate fixed-mass mechanical hypothesis.

TDD began with a missing-module RED. Synthetic four-body tests now cover
hash-mismatched source/mesh, altered source frame/scale, undeclared mass XML
edits, and mirrored COM/full tensor at the same mass. Actual OpenSim 4.6
execution against the final three-reduction XML SHA
`453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d`
produced candidate SHA
`5f2cdd22013c2700f95d625a4c42eb488ed6c5267721cd0d2acde38544754326`.
The fresh 40-body physical audit passes; the original fails exactly four
bilateral clavicle/scapula bodies. Three sampled native poses (initialized
pelvis tilt, SC_x offsets ±0.01) show 0 maximum body-transform and QErr/UErr
difference. The immutable local receipt is
`docs/development/feedback_controls_planning/generic_shoulder_lamina_12211_v4/candidate-receipt.json`
in the fleet workspace, SHA-256
`562186fe895c0f107756e78001e486534d1f379b8dac2da6d289e01d81ec82cf`.
It binds runtime, model, provider, generated mesh and sampled pose identities.
The final Tools kernel source SHA-256 is
`d058b059644c88b382d49348ca8c96b50d4176cf13d8f10789e013598b1462de`;
the inertia representation dependency SHA-256 is
`60a45fe471c2e1cbfb2ae80d9a6b3760b62b07f0fd5327a4d5e652a49f467e10`.
These are conditional diagnostic results; visual meshes for other bodies were
not present during the OpenSim reload and do not enter this mass derivation.

Reproduction uses
`scripts/opensim/derive_generic_shoulder_lamina.py --source <exact-final-557.osim> --right-clavicle <exact-clavicle.vtp> --right-scapula <exact-scapula.vtp> --output <new-directory> --development-tools-root <exact-Tools-5503-checkout>`
with the OpenSim 4.6 Python environment and its native `CASADIPATH`. Once
Tools #5503 is merged and pinned, omit the development-root argument and
revalidate the provider source hashes. Keep the PR draft/unarmed while Tools
and #12207 remain prerequisites.

Remaining science gates include actual source-region/mesh ownership, full
native force/work/acceleration chart evidence, passive muscle/tendon laws,
contact and grip, independent head/marker geometry, private full-horizon
capture matching and fresh excitation replay. The original 557 source stays
physically blocked. All 17 variants and six ecosystems remain in the
denominator; no physiological or publication acceptance is asserted.

## Handoff Status (2026-10-10)

The actual OpenSim 4.6 focused suite passed 9 tests with no skips against
the final Tools provider source hash above. The source-bound v4 receipt and
generated XML hash are preserved in fleet planning. The generic candidate is
diagnostic and must remain a draft, unarmed PR until Tools #5503 is merged,
pinned, and revalidated; #12207 is also an ancestry prerequisite. No private
capture or production fit was run for this candidate.

The earlier reduction PR #12181 is still open with a ledger-freshness unit
failure and a real branch conflict. Those are not corrected by #12211 and
must be reconciled separately before treating its ancestry as accepted.
This branch must not be presented as a replacement for the original 557
source's failed physical-body audit or as completed full-swing matching.

## Provider Source Revalidation (2026-10-10)

The Tools provider source was narrowed for an explicit tuple type annotation in
`triangle_lamina.py`; the prior source commit remains preserved in Tools
history. The updated provider commit is
`920e6fd633e796cf311cfd16f294fd45f5ad8cb1`, with kernel SHA-256
`3fd45937bed011c90b4d4e3fdf8bcf932a8bfd9a0816441ee708cc93cd4080ee`.
The inertia representation dependency is unchanged at SHA-256
`60a45fe471c2e1cbfb2ae80d9a6b3760b62b07f0fd5327a4d5e652a49f467e10`.

Against that exact Tools checkout, the OpenSim 4.6 focused suite passed all
nine tests. A new native derivation used the reviewed source and mesh inputs,
producing the same candidate model SHA-256
`5f2cdd22013c2700f95d625a4c42eb488ed6c5267721cd0d2acde38544754326`.
Its 40-body physical audit and three-pose native comparison passed with zero
maximum transform and constraint-error difference. The source model remains
physically rejected for the same four bilateral clavicle/scapula bodies. The
new immutable receipt is
`docs/development/feedback_controls_planning/generic_shoulder_lamina_12211_v5/candidate-receipt.json`
in the fleet workspace, SHA-256
`74b205b34f81df5ee2f476a9aa44f7e3d1e15493e1ed67a9c55f6eefbf41554a`.
The earlier v4 receipt remains unchanged as the historical source-hash record.

This revalidation used the external development Tools root because Tools #5503
has not yet merged; the consumer gitlink therefore remains unchanged. Repeat
the native suite and receipt derivation against the merged Tools pin before
arming or merging the consumer PR. All scientific limitations and open gates
above remain in force.
