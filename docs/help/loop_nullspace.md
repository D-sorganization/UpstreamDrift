---
title: Loop Null-Space Exploration
tile_id: loop_nullspace
status: active
---

# Loop Null-Space Exploration

## Purpose

Explore the **closed-loop null space** of a two-handed golf swing: every torque and hand-load pattern consistent with one measured or reconstructed motion, under declared hypotheses. The tool maps feasible families and reports what is **invariant** versus **prior-conditioned**. It does **not** pick a single “true” hand split.

## What You See

| Display | Meaning |
| ------- | ------- |
| **Invariant** | Same for every admissible allocation, given the model (not a measurement). |
| **Band (family-dependent)** | Envelope of values still allowed after hypotheses; a dashed **representative** line is one chosen prior, not ground truth. |
| **Shaded segment** | Sign class **indeterminate** over the static envelope. |
| **Hatched segment** | **Unbounded** direction on that frame (no finite max or min). |
| **Withheld** | Policy or evidence gate blocked the number; read the reason chip. |

## Non-Claims (Read Before Coaching)

- Motion capture fixes generalized loads and the net club wrench; it does **not** fix how load splits between hands, feet, or joints inside the loop.
- Joint torque capacities are **population priors**, not the athlete in front of you.
- Static envelopes are **per-frame** over-approximations; they can be wider than a smooth trajectory would allow.
- A **certificate** or **restriction sensitivity** panel describes the declared model and hypotheses; it is not validation against unseen data until the validation badge says so.

## Presets and Registry

**Presets** bundle hypothesis sets (capacity weights, friction, grip preload, and similar). Changing the preset changes the family; compare presets side by side, do not treat one as default truth.

## Related

- [Grip Wrench Plots](grip_wrench_plots.md) — tabbed NSX surfaces in the desktop tool.
- [Variability Sensitivity Interpretation Guide](loop_nullspace_variability.md) — disturbance and strategy panels.
