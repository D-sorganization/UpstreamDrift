# Terminology and Conventions

## Purpose

This contract keeps the monograph, evidence records, figures, and companion
website scientifically consistent. A term may be narrowed for a particular
model tier, but it must not silently acquire an anatomical or causal meaning.

## Normative Terms

| Term                        | Required Meaning                                                                                                                                                    | Do Not Substitute                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Proximal-to-distal sequence | An ordering of declared segment or joint events                                                                                                                     | Proof of energy transfer, causation, or optimality                                  |
| Energy transfer             | A signed power integral across a named interface, actuator, constraint, or external pathway                                                                         | Peak-speed order or visual sequencing alone                                         |
| Interaction force           | A force transmitted between modeled bodies or subsystems, with body, point, frame, and sign declared                                                                | Muscle force, effort, or intent                                                     |
| Reaction                    | A constraint or contact wrench required by the declared equations and constraints                                                                                   | A uniquely passive or biological force                                              |
| Drift                       | The modeled acceleration contribution when the declared control coordinates are set to zero at the same state while retained loads and constraints remain           | Force-free, muscle-free, passive, or uncontrolled human motion                      |
| Control contribution        | The modeled contribution associated with the declared input coordinates in a stated decomposition                                                                   | Neural command, effort, or voluntary intent unless measured                         |
| Negative torque             | A signed generalized moment under the declared coordinate convention                                                                                                | Negative power, energy absorption, or deceleration without the velocity sign        |
| Negative power              | A negative force--velocity or moment--angular-velocity scalar pairing                                                                                               | Negative torque by itself                                                           |
| Preload                     | A nonzero declared transmission, force, strain, activation, or stiffness state before an event                                                                      | An unspecified feeling of connection                                                |
| Slack                       | Use only with the measured state: contact gap, tendon slack length, series strain, low tangent stiffness, low activation, delayed force rise, or grip-pressure loss | A single universal biological state                                                 |
| Persistent direction        | A channel retains its modeled sign through a transition                                                                                                             | Proof that an anatomical muscle remains active                                      |
| Role reversal               | One or more declared modeled channels change sign                                                                                                                   | Proof of a particular arm, wrist, or scapular strategy                              |
| Passive after killswitch    | The branch receives no further value from the removed modeled input after the intervention                                                                          | No stored energy, no prior actuation, no gravity, no damping, or no muscle activity |
| Model support               | A preregistered prediction passes at its declared model tier                                                                                                        | Human validation or a coaching recommendation                                       |
| Coriolis term               | A cross-speed monomial in the declared coordinate-specific Christoffel split                                                                                        | A new external force, muscle force, or coordinate-invariant physical source         |
| Squared-speed term          | A squared generalized-speed monomial in the declared coordinate-specific Christoffel split; describe centripetal/centrifugal frame meaning explicitly               | A second force to add to both centripetal and centrifugal descriptions              |
| Force-only endpoint map     | A virtual-work equivalent force with rank and generalized residual reported                                                                                         | A complete measured grip wrench when the task Jacobian is deficient                 |

## Frames, Wrenches, and Power

- Frames are right-handed unless explicitly stated otherwise.
- Spatial wrenches are force-first: $[\mathbf F^T\;\mathbf M^T]^T$.
- Spatial twists are linear-first: $[\mathbf v^T\;\boldsymbol\omega^T]^T$.
- Every moment names its reference point; every vector names its expression
  frame when more than one frame is in use.
- Point transport must transform the couple and point velocity together.
- Proper coordinate rotations must preserve dot products, cross products,
  wrench--twist power, and Jacobian virtual work within the registered numerical
  tolerance.
- Joint-coordinate signs follow the model definition and must not be inferred
  from screen direction or anatomical colloquialism.

## Evidence Labels

Use exactly one of these statuses for a principal claim: **supported**,
**contradicted**, **inconclusive**, or **untested**. Always append the model or
measurement tier. Synthetic, archived, readiness, and measured-human evidence
must remain distinguishable in prose, captions, tables, and metadata.

## Editorial Test

Before publication, ask: could a reader replace a modeled quantity with a
muscle, intention, technique, or universal physical cause without noticing a
change in meaning? If yes, name the coordinate, interface, state, intervention,
and evidence tier more precisely.
