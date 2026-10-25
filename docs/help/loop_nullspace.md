---
title: Loop Null-Space Exploration
tile_id: loop_nullspace
status: active
---

# Loop Null-Space Exploration

## Purpose

Explore the **closed-loop null space** of a two-handed golf swing: every torque and hand-load pattern consistent with one measured or reconstructed motion, under declared hypotheses. The tool maps feasible families and reports what is **invariant** versus **family-dependent**. It does **not** pick a single "true" hand split.

Where it lives: the Grip Wrench Plots window (desktop, tile `grip_wrench_plots`) and the web Loop Null-Space panel on Analysis Tools. The desktop tabs are described in [Grip Wrench Plots](grip_wrench_plots.md).

## What You See

| Display                     | Meaning                                                                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Invariant**               | Same for every admissible allocation, given the model (not a measurement). Shown as a value with a model-provenance badge.                                  |
| **Band (family-dependent)** | Envelope of values still allowed after the declared hypotheses.                                                                                             |
| **Dashed representative**   | One allocation chosen by a named preset. It is drawn only as an overlay on its band and is labelled with that preset. It is not ground truth.               |
| **Solid segment**           | Sign class decided over the whole band (positive, negative or zero).                                                                                        |
| **Shaded segment**          | Sign class **indeterminate** over the static envelope.                                                                                                      |
| **Hatched segment**         | **Unbounded** direction on that frame (no finite max or min).                                                                                               |
| **Withheld**                | A policy or evidence gate blocked the number. No number is drawn; the reason is shown instead.                                                              |
| **UNVALIDATED badge**       | The quantity has not passed its validity gate (`ValidationStatus`, `src/shared/python/loop_nullspace/evidence.py`). This is the default for every quantity. |

The segment classes come from the plot model (`DisplayClass`: `DETERMINED_SIGN`, `INDETERMINATE`, `UNBOUNDED` in `src/shared/python/loop_nullspace/plot_model.py`). The shading and hatching are drawn by `src/tools/grip_wrench_plots/nsx_draw.py`. The sign classes themselves (`SignClass`: `POSITIVE`, `NEGATIVE`, `ZERO`, `INDETERMINATE`) are computed in `src/shared/python/loop_nullspace/envelopes.py`.

Every value shown has passed through one display gate, `display_decision` in `src/shared/python/loop_nullspace/output_policy_integration.py`. Its actions are `SHOW_VALUE`, `SHOW_BAND`, `SHOW_REPRESENTATIVE_ON_BAND`, `SHOW_REASON` and `WITHHOLD` (`DisplayAction`). Neither user interface computes its own gate.

## Invariant and Family-Dependent Quantities

Each quantity carries an `AllocationDependence` value (`src/shared/python/loop_nullspace/evidence.py`):

- **`INVARIANT`**: the same for every admissible allocation. The data and the model fix it. A `QuantityRecord` with this value cannot carry allocation-prior hypotheses. It can still carry conditioning chips such as "conditioned on" a declared input.
- **`FAMILY_DEPENDENT`**: it changes with the choice among admissible allocations. Its band is the envelope over that family. It must name the hypotheses that define the family.
- **`UNAVAILABLE`**: it cannot be computed for this run, for example an engine that supplies no derivatives or no grip data. It has a reason and no number. It is never shown as zero.

The net club wrench is the main invariant. `net_club_wrench` in `src/shared/python/loop_nullspace/grip_split.py` computes it from bare-club Newton-Euler. Joint-torque envelopes, closing-hand loads and the split coordinates are family-dependent.

## Why the Hand Split Is Not Measured

The two hands together must produce the club's net wrench, which the motion fixes. How that load divides between the left and right hand is not fixed. The module `grip_split.py` says so directly: it "chooses no split". The split coordinates are the six columns named by `SPLIT_COLUMN_NAMES`, which span the kernel of the grasp map. Their values for a real swing are family-dependent.

The hand split is not measured: no sensor in the NSX pipeline observes it. No grip-force sensor or instrumented grip is part of the snapshot. The existing grip-wrench view does draw a left/right split, but it comes from the solver (`split method` values `constraint_multiplier`, `efc_force`, `allocation`, `bushing`, see `src/shared/python/biomechanics/grip_wrench.py`). That split is a model output, not a measurement. The anthropometric NSX model carries only the closing-hand wrench (the multiplier `lambda`). The lead-hand grip point needed for the couple split is unavailable, so the lead-hand split is not reported (see `docs/research/loop_nullspace/contracts_trajectory.md`, NSX-60). The grip-split envelopes are therefore bands under declared hypotheses, not estimates of what the golfer's hands did.

If a hand-split band looks narrow, read the hypotheses that bound it before reading the band. A narrow band under a strong preset is a statement about that preset, not about the golfer.

## Presets

A **preset** is a named set of hypotheses in `src/config/loop_nullspace_registry.json`. Resolve one with `Registry.preset(name)` (`src/shared/python/loop_nullspace/registry.py`). The analysis refuses an unknown preset with `ValueError` (`_resolve_preset` in `src/shared/python/loop_nullspace/facade.py`). The desktop preset selector lists the registry's presets.

| Preset                  | Hypotheses it declares                                                        | What changes when you switch to it                                              |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `envelope_only`         | `H-TORQUE-CAPACITY`, `H-FRICTION-CONE`, `H-GRIP-PRELOAD`                      | Only the feasible set. No representative is chosen.                             |
| `envelope_grf_observed` | The `envelope_only` set plus `H-GRF-OBSERVATION`                              | Adds a band on the observed ground reaction. Needs an observation in the input. |
| `capacity_min_effort`   | `H-TORQUE-CAPACITY`, `H-EFFORT-CAPACITY`, `H-FRICTION-CONE`, `H-GRIP-PRELOAD` | The representative minimises capacity-weighted squared torque.                  |
| `smooth_capacity`       | The `capacity_min_effort` set plus `H-SMOOTH-TORQUE-RATE`                     | The representative also penalises squared torque rate.                          |

Points to keep in mind:

- The torque capacities behind `H-TORQUE-CAPACITY` are **population priors**, not measurements of the athlete. The registry records them as priors until a sourced table is adopted.
- A preset changes the family. Compare presets side by side. Do not treat any one of them as the default truth.
- A representative shown under a preset is labelled with that preset. A representative whose preset is not in the registry is withheld (`display_decision`, NSX-41).
- Some capacity slots are marked ASSUMED. A result that rests on one carries a flag rather than a clean badge.

## Reading a Certificate

A certificate is a record that a solver or verifier produced with its own evidence, which a checker can re-verify. It does not mean the physics is right. Three kinds appear in the pipeline:

- **Infeasibility certificate.** When a set is empty, the LP solver returns a Farkas vector `y` (`y^T A <= 0`, `y^T b > 0`). The test suite checks it rather than trusting it (NSX-20, `solve_lp` in `src/shared/python/loop_nullspace/solvers.py`). Conflicts are reported as the minimal set of hypotheses that cannot hold together (`FeasibleSet.conflict_set`, `src/shared/python/loop_nullspace/feasible_set.py`). An infeasible frame is a statement about the declared priors, for example capacities the recorded effort exceeds.
- **Probe certificate.** A `LeafCertificate` (`src/shared/python/loop_nullspace/probe_certificates.py`) bounds one predicate over the feasible set. Its verdict is `HOLDS`, `FAILS` or `UNDECIDED` (`Verdict`). `verify_leaf` checks the certificate with numpy against the system data. It is exact up to the reported residuals, not a rational-arithmetic proof. A value within tolerance of the threshold is `UNDECIDED`, never forced to a verdict.
- **Strategy certificate.** Each variability strategy arm carries a `certificate` field in the report payload (`strategy_evaluators.py`). Its value is `global-convex` (the optimum of a convex problem), `local` (a local stationary point of a non-convex one) or `flat-invariant` (the objective is flat under the declared model). The arm certificate says how the optimum was obtained. It does not say the strategy works in a golfer.

A certificate describes the declared model and its hypotheses. It is not validation against unseen data. Validation is a separate badge, and until that badge reads VALIDATED the quantity stays UNVALIDATED.

## What Restriction Sensitivity Is Not

The restriction sensitivity report (every output carries `RESTRICTION_LABEL`, which reads `REACTION_RESTRICTION_SENSITIVITY`; see `src/shared/python/loop_nullspace/restriction.py` and `restriction_report.py`) asks what happens when the hand-club joint is allowed to transmit fewer closure components. It compares three `ClosureKind` values on the **same welded kinematics**:

- `WELD6` transmits all six components.
- `SHAFT_FREE5` cannot transmit the twist moment about the shaft.
- `BALL3` transmits the three forces only.

Dropping a component can only shrink the feasible set, so restricted envelopes lie inside the `WELD6` envelope. It is not any of the following:

- It is **not** a different closure model. A grip that behaves differently also changes the kinematics, and that case has to be re-derived. It is not evidence about the true grip mechanics.
- It is **not** a measurement of the hands. Restricted quantities are the same affine functionals on the embedded multiplier.
- It is **not** proof of a model choice. A trajectory recorded under a weld can be dynamically inconsistent under a looser joint. Such frames are reported as `BALANCE_INCONSISTENT` (with the scaled residual) or `INFEASIBLE` (with the conflict set). The balance tolerance and the priors are never relaxed to make them disappear.

## Non-Claims (Read Before Coaching)

- Motion capture fixes generalized loads and the net club wrench. It does **not** fix how load splits between hands, feet, or joints inside the loop.
- Joint torque capacities are **population priors**, not the athlete in front of you.
- Static envelopes are **per-frame** over-approximations. They can be wider than a smooth trajectory would allow.
- A **certificate** or a **restriction sensitivity** result describes the declared model and hypotheses. It is not validation against unseen data until the validation badge says so.
- A ranking of strategies or allocations is a ranking under the declared assumptions. It is not coaching advice.
- No single hand split is the golfer's. No stiffness or co-contraction is inferred from net joint torques.

## Related

- [Grip Wrench Plots](grip_wrench_plots.md) - the desktop tool that hosts the NSX tabs, including the Variability & Consistency tab.
- [Variability Sensitivity Interpretation Guide](loop_nullspace_variability.md) - disturbance and strategy panels.
