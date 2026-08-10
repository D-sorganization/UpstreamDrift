# WSCG 2024 Source Register

## Scope and Evidentiary Role

These files preserve the project-originated presentation evidence used to
motivate and design the interaction-force analyses. They are primary records of
the author's prior modeling work, not independent validation and not a substitute
for peer-reviewed evidence. The manuscript distinguishes quantities read from
these presentations from quantities reproduced by the repository's executable
models.

## Registered Sources

| File                                                                                       | Context                                              | SHA-256                                                            | Use in This Study                                                                                                                                              |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WeT21_Evaluation_of_the_Effect_of_Momentum_on_Interaction_Forces_in_a_Linked_System.pptx` | Dieter Olson, WSCG/ISEA 2024 presentation, 11 slides | `B1D53150A9669744842AC5BB4200522FBD5AA4697092A0F6E88BCD4DA687EA0A` | Defines the BASE/ZTCF/DELTA framing, the torque-killswitch construction, the two-hand flexible-shaft example, and the late-swing equivalent-couple hypothesis. |
| `Charts.pptx`                                                                              | Supporting chart and six swing-pose images, 1 slide  | `17DFF3B767EF432D76AACAA7BE2CE24339D24755B00DA293CED4F4EF62A307A5` | Preserves cached lead/trail hand-force, counterfactual-force, difference, and wrist-torque series used for provenance checks and later two-hand reproduction.  |

## Interpretation Boundaries

- `BASE` denotes a reported commanded simulation.
- `ZTCF` denotes the reported zero-torque-counterfactual construction. The deck
  obtains a trajectory by applying torque kill switches at successive matched
  states. The present study separately names instantaneous and forward-integrated
  variants because they answer different questions.
- `DELTA = BASE - ZTCF` is an operational decomposition, not a direct measurement
  of muscle force or a proof of biological causality.
- The presentation's late negative equivalent couple is treated as a plausible
  model mechanism to reproduce, perturb, and falsify. It is not presented as a
  universal property of human swings.
- The linked model remains available from the
  [MathWorks File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/169071-two-hand-golf-swing-model).

## Reproducibility

Run the source extractor from the repository root:

```powershell
python -m scripts.research.proximal_distal_energy.extract_wscg_charts
```

The extractor verifies both binary hashes before reading the embedded chart
caches and writes a long-form CSV plus a machine-readable provenance record to
the article `data` directory. The original presentations are never rewritten.
