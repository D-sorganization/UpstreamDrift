# WSCG 2024 Legacy Evidence Audit

## Purpose and Evidentiary Status

This audit records what the two archived WSCG 2024 presentation files do and do
not establish. Both decks are project-originated model evidence. They motivate
and illustrate a mechanism; they are neither an independent replication nor a
human-subject validation dataset. Publication claims should therefore retain
the qualifiers _model-derived_, _planar_, and _plausible_.

Every slide was rendered and inspected individually. Slide text, notes, native
chart structure, embedded media, and package relationships were also examined.
The decks contain no substantive speaker notes or bibliographic source blocks.

## Source Identity

| Source                                                                                                       | Slides | SHA-256                                                            | Evidentiary Role                                                      |
| ------------------------------------------------------------------------------------------------------------ | -----: | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `sources/wscg_2024/Charts.pptx`                                                                              |      1 | `17dff3b767ef432d76aacaa7be2ce24339d24755b00da293ced4f4ef62a307a5` | Cached lead- and trail-hand BASE and counterfactual force histories   |
| `sources/wscg_2024/WeT21_Evaluation_of_the_Effect_of_Momentum_on_Interaction_Forces_in_a_Linked_System.pptx` |     11 | `b1d53150a9669744842ac5bb4200522fbd5aa4697092a0f6e88bcd4da687ea0a` | Conference argument, model description, and four static result panels |

The user-supplied backup files and the repository copies are byte-for-byte
identical at these hashes. The existing
`sources/wscg_2024/SOURCE_REGISTER.md` remains the canonical file inventory;
this document adds the scientific claim and provenance audit.

## Slide-by-Slide Evidence Audit

| Slide | Content and Claim                                                                                                                                                                                                | Publication Interpretation                                                                                                                                                                                        |
| ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|     1 | Title, author, and WSCG/ISEA 2024 context                                                                                                                                                                        | Establishes authorship and presentation context only.                                                                                                                                                             |
|     2 | Asks how momentum and other passive effects influence joint forces, and whether familiar force patterns can arise from them                                                                                      | These are research questions, not findings.                                                                                                                                                                       |
|     3 | Argues that constrained linked segments cannot all continue on independent free trajectories and states that zero joint torque does not imply zero interaction force                                             | A sound mechanism hypothesis: constraint reactions may persist without applied joint torque. The slide does not derive or quantify the claim.                                                                     |
|     4 | Classifies joint torque as active and momentum, gravity, and shaft flex as passive                                                                                                                               | This is the deck's operational partition, not a unique physical decomposition.                                                                                                                                    |
|     5 | Defines the zero-torque counterfactual (ZTCF) by disabling joint torques at varied cut times and collecting the value at each cut                                                                                | The archived curve is a stitched matched-state, pointwise counterfactual. It is not one forward zero-torque trajectory.                                                                                           |
|     6 | Describes a joint-torque-driven, planar, two-hand model with a flexible-beam shaft whose kinematics were “roughly matched” to a kinematic sequence                                                               | The model is illustrative rather than calibrated. The slide links the [Two-Hand Golf Swing Model](https://www.mathworks.com/matlabcentral/fileexchange/169071-two-hand-golf-swing-model) and embeds an animation. |
|     7 | Compares BASE and ZTCF net-force vectors around the modeled swing; the vector fields are visually similar                                                                                                        | Supports qualitative similarity of modeled net force. The static image supplies no machine-readable values or uncertainty.                                                                                        |
|     8 | Compares force along the hand path for BASE, ZTCF, and DELTA; BASE and ZTCF largely overlap                                                                                                                      | Supports a qualitative claim that the passive counterfactual accounts for much of this modeled component. Approximate values read from the image must not be reported as measurements.                            |
|     9 | Compares BASE and ZTCF equivalent-couple vectors and shows a late-swing reversal                                                                                                                                 | Supports the existence of a reversal in this simulation. “MP” is not defined on the slide and must not be expanded without model-source evidence.                                                                 |
|    10 | Shows opposing left- and right-hand local-force components for BASE and ZTCF                                                                                                                                     | Supports the separated-contact-force mechanism for a couple. It does not establish independent muscular hand effort or measured grip force.                                                                       |
|    11 | Concludes that passive components contribute to joint forces and club couple, net club force is similar to its passive contribution, and near-impact couple reversal can plausibly arise from interaction forces | The conclusion is explicitly a plausibility claim in a two-hand model. It does not isolate momentum from gravity and shaft flex.                                                                                  |

## Supporting Chart Deck Audit

`Charts.pptx` contains one native scatter chart with time in seconds and force in
newtons. Its visible legend identifies eight series:

- `LeadHandAxial`, `LeadHandNormal`, `TrailHeadAxial`, and `TrailHandNormal`;
- `LeadHandCFAxial`, `LeadHandCFNormal`, `TrailHandCFAxial`, and
  `TrailHandCFNormal`.

`TrailHeadAxial` is preserved exactly as stored and may be a legacy naming
error; it must not be silently corrected. Package inspection finds additional
cached difference and wrist-torque series. The full cache is exported in
`data/wscg_2024_hand_force_series.csv` and enumerated in
`data/wscg_2024_source_provenance.json`.

The chart retains an unavailable external-workbook relationship to:

`Reviewer Comments/ZeroTorqueCFCompilation - Review Comments Version.xlsx`

The absolute relationship originally points into an author's Dropbox folder.
The workbook is not embedded in the deck. Consequently, the repository's
machine-readable evidence is the chart's cached OOXML values, not a live link
to the originating workbook. The single slide has no title, caption, source
note, uncertainty statement, or data-processing description.

## Operational Definitions and Equations

Slide 4 presents the bookkeeping identities

\[
E*{\mathrm{total}}=E*{\mathrm{active}}+E*{\mathrm{passive}},
\qquad
E*{\mathrm{passive}}=E*{\mathrm{total}}-E*{\mathrm{active}}.
\]

Here, “effect” is schematic. The deck does not define it as energy, work,
power, force, or torque, so the symbol \(E\) above denotes a generic effect and
must not be interpreted as mechanical energy. In the archived analysis, ZTCF is
the state-matched response computed immediately after applied joint torques are
disabled. DELTA is the BASE-minus-ZTCF residual under the archived convention.

This construction answers an instantaneous attribution question: _what force
or moment does the model produce at the same state when its joint-torque inputs
are set to zero?_ It does not answer the forward question: _where will the
system evolve after torque is removed?_ Forward counterfactual persistence
requires a stated cut time, integration horizon, solver, and retained-physics
inventory.

## Mechanism Supported by the Decks

The decks support the following bounded chain of reasoning:

1. Kinematic constraints require internal reaction forces when connected
   bodies would otherwise follow incompatible inertial paths.
2. Those reactions can remain nonzero in a matched state even when the applied
   joint torques are set to zero.
3. Two spatially separated hand forces can have a small resultant force while
   producing a substantial moment about a club reference point.
4. In the archived planar simulation, the ZTCF net-force and equivalent-couple
   patterns resemble the BASE patterns, and the equivalent couple reverses late
   in the modeled downswing.
5. The result therefore demonstrates a plausible passive contribution to late
   negative club torque in this model. It does not demonstrate that passive
   interaction is the sole cause, that momentum alone is responsible, or that
   the same magnitude occurs in golfers.

The useful geometric relation is the planar wrench identity

\[
M_O=\sum_i (\mathbf r_i-\mathbf r_O)\times\mathbf F_i,
\]

which makes the mechanism explicit: force magnitude, grip separation, force
direction, and reference point jointly determine the net moment. Equal and
opposite hand forces can cancel translationally while adding rotationally.

## Claim Boundaries and Limitations

- The two-hand result is generated by a planar, joint-torque-driven model with a
  flexible-beam shaft; it is not a full three-dimensional golfer-club model.
- “Roughly matched” kinematics do not constitute parameter identification,
  subject-specific calibration, or out-of-sample validation.
- The passive bucket combines momentum-related, gravitational, shaft-flex, and
  any other retained model terms. The decks do not independently ablate them.
- Pointwise ZTCF samples come from multiple torque-cut runs and cannot be read
  as one dynamically continuous trajectory.
- Slides 7–10 are static JPEGs, not native charts. They contain no uncertainty,
  sensitivity, or machine-readable provenance within the presentation.
- The simulated left- and right-hand forces are constraint/contact solutions,
  not measured hand forces and not direct estimates of muscular intent.
- The decks do not close force-power, segment-energy, or whole-system-energy
  balances and do not establish that a force pattern increases clubhead speed.
- The decks provide no cross-engine reproduction, experimental comparison, or
  statistical uncertainty analysis.
- The Newton/golfer composite on slide 3 has no source or license note and must
  not be reused in a publication artifact.

## Reproducibility Crosswalk

| Legacy Evidence                            | Repository Reproduction or Audit                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Source identity and hashes                 | `sources/wscg_2024/SOURCE_REGISTER.md`                                                    |
| Native chart-cache extraction              | `scripts/research/proximal_distal_energy/extract_wscg_charts.py`                          |
| Portable cached series                     | `data/wscg_2024_hand_force_series.csv` and `data/wscg_two_hand_raw/`                      |
| Extraction metadata and series inventory   | `data/wscg_2024_source_provenance.json`                                                   |
| Frame-explicit two-hand wrench calculation | `scripts/research/proximal_distal_energy/two_hand_wrench.py`                              |
| Deterministic analysis runner              | `scripts/research/proximal_distal_energy/run_two_hand_wscg_analysis.py`                   |
| Machine-readable results                   | `data/two_hand_wscg_analysis.json` and `data/two_hand_wscg_analysis.npz`                  |
| Publication-quality reproductions          | `figures/fig_wscg_registered_hand_forces.svg` and the `figures/fig_two_hand_*.svg` series |
| Scientific interpretation and limitations  | `chapters/_ch05_two_hand_wrench.qmd`                                                      |

## Publication Use Rules

Use the regenerated vector figures and machine-readable exports for the report.
Do not reuse the slide-3 composite or the slide-7 through slide-10 screenshots
unless their source and license are separately established. Cite the decks as
project-originated conference material and cite the executable reproduction for
numerical claims. Preserve BASE, ZTCF, DELTA, coordinate-frame, and reference-
point definitions adjacent to every derived force or moment result.

## Required Follow-On Evidence

The next defensible steps are to isolate gravity, damping, and shaft-flex terms;
compare pointwise attribution with forward torque-cut trajectories; test grip
separation and force-orientation sensitivity; reproduce the wrench in an
independent dynamics engine; and compare preregistered observables with measured
club and hand data. Until those gates are passed, the decks support mechanism
plausibility, not a universal causal or coaching prescription.
