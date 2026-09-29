# Control-Affine Terminology Contract

This document is the UpstreamDrift implementation profile for the normative
[AffineDrift notation authority](https://github.com/D-sorganization/AffineDrift/blob/main/NOTATION.md).
It fixes the vocabulary used by code, evidence schemas, figures, and the
proximal-to-distal publication.

## Declared Effective Plant

For

$$
\dot x=f(x)+G(x)u,
$$

`f(x)` is the complete autonomous vector field of the declared effective
plant. It includes every retained state-dependent inertial, gravity,
Coriolis/centrifugal, passive elastic/damping, shaft, contact, constraint, and
internal-state effect. `G(x)` is the input map, `u` is the declared applied
generalized-control channel, and `g(q)` is reserved for gravity generalized
force. “Gravity plus Coriolis drift” is valid only for a model whose declared
plant contains no other autonomous term.

## Counterfactual Objects

- A **pointwise ZTCF sample** evaluates `f(x)` at one achieved state after
  setting only `u=0`.
- A **stitched pointwise ZTCF trace** repeats that sample along an achieved
  history; it is not a forward trajectory.
- A **forward ZTCF trajectory** integrates the zero-control plant from one
  declared initial state.
- An **achieved-state branched ZTCF trajectory** follows the achieved history
  to a branch event and then integrates with `u=0`.

ZTCF does not mean no muscle activation, no EMG, or a flaccid system. Those are
physiological claims that require separate state and actuator definitions.

The **zero-velocity counterfactual (ZVCF)** is the instantaneous acceleration
or constrained reaction at fixed configuration and internal state with both
generalized velocity and `u` set to zero. It is not a torque, a state, or a
releasable forward trajectory. The distinct evaluation that sets velocity to
zero while preserving `u` is named **zero-velocity control-preserved
evaluation** and is never abbreviated ZVCF.

## Drift-Control Ratio

A drift-control ratio (DCR) compares drift with bounded admissible control
capacity in the same declared acceleration or task-projected metric:

$$
\mathrm{DCR}_{W,\mathcal U}(x)=
\frac{\lVert Wf(x)\rVert}
{\sup_{u\in\mathcal U(x)}\lVert WG(x)u\rVert+\epsilon}.
$$

`W`, the admissible set `U(x)`, norm, units, and epsilon must be reported.
A mixed-unit full-state norm is invalid. A ratio using realized input rather
than admissible capacity is a **realized drift-to-input ratio (DIR)**, not DCR.

## Versioned Implementation Consequences

Analysis schema `3.0.0` assigns `zvcf_acceleration` the canonical zero-control
meaning and preserves the prior result as
`zero_velocity_control_preserved_acceleration`. Evidence generated under an
older schema must be relabeled or regenerated; it must not silently acquire
the new meaning.
