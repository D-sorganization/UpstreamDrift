---
title: Variability Sensitivity Interpretation Guide
tile_id: grip_wrench_plots
status: active
---

# Variability Sensitivity Interpretation Guide

## Purpose

This guide explains how to read the Variability & Consistency results of the closed-loop null-space analysis (NSX-99, NSX-100): the Kernel Heatmap, Variance Budget, Strategy Pareto and Face Dispersion panels. It states what a minimum-variance allocation and a strategy ranking mean, and what they do not mean. For the controls see [Grip Wrench Plots](grip_wrench_plots.md); the general loop null-space user help is tracked in NSX-45 (#12577), which is still open.

## Current Status

- **Demonstration numbers only.** The only sensitivity provider in the repository is a pendulum test fixture with invented amplitudes. No real-engine provider is wired, so no variability number from a real golfer exists.
- **No coaching prescription is validated.** Nothing on these panels is advice for a golfer. Every claim type is blocked or deferred in the [publication gate](../research/loop_nullspace/publication_gate.md) (section Variability Sensitivity Claims), and the external experiments that could unblock them are preregistered but not run (`docs/development/planning/DV-12554.md`).
- **The strategy catalogue is partial** (NSX-95, #12724). Implemented: S1 null-space redistribution, S2 co-contraction and impedance, S3 distal stabilization, S5 grip and hand-split preload, S7 delayed feedback, S8 speed-accuracy. Not implemented, so listed as not attempted with a reason and no numbers: S4, S6, S9, S11 and S12. S10 (shaft-flex phase matching) is unavailable, because the rigid club has no shaft flexibility.
- **Real-capture variability is deferred** (NSX-97, #12726). The UCM and TNC falsification code exists and is tested on synthetic fixtures only. The report on a real capture has not been produced.

## Reading the Evidence Badges

Every number on these panels is a typed quantity with six evidence axes, and the panel shows them as badges and chips. The panel does no math and invents no label; it shows what the plot model decided.

| Badge or chip      | Meaning                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INVARIANT`        | Identical for every admissible allocation under the declared model. It is not a measurement.                                                             |
| `FAMILY_DEPENDENT` | Depends on which allocation in the family is chosen. It is shown only with the hypotheses that fix it and is withheld when one is unverified or unknown. |
| `UNAVAILABLE`      | Cannot be computed (a missing derivative capability, an unimplemented strategy, an engine limit). It carries a reason and never a number, never zero.    |
| `UNVALIDATED`      | No external validation has passed. Every variability quantity carries it today.                                                                          |
| A hypothesis chip  | A named declared assumption the value is conditional on, for example a noise model or an impedance. It is an assumption, not a result.                   |
| Conditioning chip  | The orthogonal axis recording the noise-model hypotheses that variance values are conditioned on.                                                        |

## Why Values Are Withheld

A value is withheld, with its reason shown instead of a number, when the display rule (ADR-0054 D7, Option A) refuses it: a `FAMILY_DEPENDENT` quantity with an unverified or unknown hypothesis, a quantity with no family band (one allocation is not a band), a linear result whose validity gate fails, or an unavailable engine capability. A withheld value is information about the evidence, not a gap to fill. Do not read it as zero or as small.

## What a Minimum-Variance Allocation Means

It does mean: among the load allocations that give the same net motion under the declared model, this one gives the smallest predicted face-angle variance under the declared noise amplitudes and mechanisms, to first order around the nominal swing.

It does not mean:

- that any golfer uses this allocation, or that golfers minimize variance;
- that the amplitudes are real: they are declared hypotheses, typed by the user or fixed by a fixture;
- that it holds for large perturbations, saturation, slip or a missed or grazing impact (the linear result is checked by a nonlinearity gate and paired Monte Carlo);
- that it is the true hand split, the true muscle force, or evidence of co-contraction or stiffness (net joint torques cannot identify those);
- that it is safe, comfortable or achievable by a person.

With no mechanism declared and additive noise only, the objective is flat and every allocation has the same variance; the analysis then reports that, and it must not be read as a preferred allocation.

## What a Strategy Ranking Is Not

A ranking orders strategies by the ratio of their face-angle variance to a baseline of their own model representation, under the declared disturbance, priors and the linear model at the nominal motion. It compares mechanisms inside a model. It is **not coaching advice**: it is not a prediction of any golfer's best strategy, it has not been tested by an intervention, and absolute variances are not compared across representations. A different declared prior can reverse the order (the stability table reports reversals). A strategy that reaches its effect only through a mechanism you did not declare shows no effect by construction. Matched-speed and matched-effort comparisons exclude arms rather than relaxing them.

## Reading the Panels

- **Kernel Heatmap:** the sensitivity of the face angle to each joint disturbance over time. Not a map of where a golfer is variable.
- **Variance Budget:** how the modelled variance divides among the declared input classes. Shares follow the declared amplitudes, so they change when the amplitudes change.
- **Strategy Pareto:** variance against effort or speed for the implemented strategies. Points that were withheld or not attempted are not drawn.
- **Face Dispersion:** the modelled standard deviation, with linear and Monte Carlo values side by side. It is not a measured dispersion.

## Limitations

Linear first-order model; declared, not measured, noise and impedance; a rigid club; a partial strategy catalogue; no validated model of a real golfer. Scientific qualification stays in the design-manual governance pathway, where release is blocked.

## See Also

- [Grip Wrench Plots](grip_wrench_plots.md)
- [Publication Gate](../research/loop_nullspace/publication_gate.md)
