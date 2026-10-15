# Loop Null-Space Exploration User Guide

This guide explains how to read and interpret the **Closed-Loop Null-Space Exploration (NSX)** surfaces in both the desktop **Grip Wrench Plots** tool (`src/tools/grip_wrench_plots/gui.py`) and the web **Analysis Tools** page (`ui/src/components/analysis/LoopNullSpacePanel.tsx`), governed by [ADR-0054](../adr/0054-closed-loop-null-space-exploration.md) and the mathematical specification in [`docs/research/loop_nullspace/loop_nullspace.tex`](../research/loop_nullspace/loop_nullspace.tex) (Epic #12593).

---

## 1. Invariant vs Family-Dependent Quantities

In a two-handed golf swing, both hands grip the club handle, forming a closed kinematic loop through the torso, lead arm, trail arm, and club. Even when full-body kinematics $(q, v, a)$ and bare-club inertial parameters are known, the equations of motion form an underdetermined linear equality system $A x = b$ over active joint torques $\tau$, closure constraint wrenches $\lambda$, and ground-reaction forces $f_g$.

Every quantity displayed in NSX carries a mandatory `allocation_dependence` badge and six orthogonal evidence axes (`allocation_dependence`, `model_provenance`, `input_provenance`, `conventions`, `hypotheses`, and `validation_status`):

| Classification     | Meaning                                                                                                                                                                                                                                                                                                                                                    | Display Policy                                                                                                                                                                                                                                                                                                                                      |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INVARIANT`        | Algebraically constant across **every** solution of the equality system $A x = b$ on an admitted trajectory (for example, net bare-club wrench $W_{\text{net}}^{\text{bare}}$, summed loop actuator power $P_{\text{net}}$, floating-base external wrench, metric-free tangent projection $N_c^\top \tau$, and total constrained ZTCF/ZVCF accelerations). | Rendered as a single deterministic curve with a `MODEL_CONDITIONED_INVARIANT` badge naming the physics engine, model SHA-256, bare-club partition, and mathematical conventions (`C-FRAME-GRIP`, `C-METRIC-*`). Note that `INVARIANT` means _invariant to null-space load sharing under the model_, **not** empirically measured by a force sensor. |
| `FAMILY_DEPENDENT` | Varies across the affine null space $x(z) = x_p + N z$ of $A x = b$ (for example, individual upper-limb joint torques, per-hand grip wrenches $W_L, W_R$, internal grip-split coordinates $s_1 \dots s_6$, and per-joint counterfactual attributions).                                                                                                     | **Band-first display only.** Must be shown as a feasible `[lower, upper]` envelope band conditioned on explicit physiological hypotheses (`H-*`). A single representative curve may **only** be drawn overlaid on top of its envelope band, never alone.                                                                                            |
| `UNAVAILABLE`      | Cannot be evaluated at the given sample or run because a kinematic admission gate failed, a solver extra is absent, or a direction is unbounded.                                                                                                                                                                                                           | Rendered as a gap or hatched unavailable state with its explicit `reason` string—**never** drawn as zero.                                                                                                                                                                                                                                           |

---

## 2. Why the Hand Split Is Not Measured From Kinematics Alone

At the canonical grip midpoint frame (`C-FRAME-GRIP`, with $\hat{u}$ along the shaft toward the clubhead, $\hat{v}$ along the orthogonal clubface normal projection, and $\hat{w} = \hat{u} \times \hat{v}$), the lead-hand wrench $W_L \in \mathbb{R}^6$ and trail-hand wrench $W_R \in \mathbb{R}^6$ sum to the net hand-on-bare-club wrench via the $6 \times 12$ grasp matrix $G$:

$$W_{\text{net}}^{\text{bare}} = G \begin{bmatrix} W_L \\ W_R \end{bmatrix}.$$

Because $G$ maps 12 hand-wrench components to 6 net-wrench components on the rigid bare club ($0.313\,\mathrm{kg}$ in `full_body_spec_anthro_driver.json`, after partitioning out the $1.1082\,\mathrm{kg}$ welded hand solids `LHand`, `LHandStandoff`, and `RHand`), $G$ has an exact **6-dimensional null space** ($\dim \ker G = 6$).

Any internal grip-split vector $s = [s_1, s_2, s_3, s_4, s_5, s_6]^\top \in \mathbb{R}^6$ pushed through the kernel basis $K \in \mathbb{R}^{12 \times 6}$ ($G K = 0$) changes the individual hand loads without changing $W_{\text{net}}^{\text{bare}}$ or the club's motion:

- $s_1 = \tfrac{1}{2}(F_{R,u} - F_{L,u})$ ($\mathrm{N}$): axial internal tension/compression along the shaft axis $\hat{u}$.
- $s_2, s_3 = \tfrac{1}{2}(F_{R,v/w} - F_{L,v/w})$ ($\mathrm{N}$): transverse shear battles paired with compensating wrist moments $\mp h s_{2,3}$ that cancel the force couple of moment arm $2h$.
- $s_4, s_5, s_6 = \tfrac{1}{2}(\tau_R - \tau_L)_{u,v,w}$ ($\mathrm{N\cdot m}$): pure differential free wrist-moment battles (axial twist $s_4$ and bending couples $s_5, s_6$).

Consequently, **kinematics and inverse dynamics alone never measure how hard one hand pulls or twists against the other**. Resolving or bounding $s_1 \dots s_6$ requires either declared physiological inequalities (producing feasible envelopes) or an instrumented split-handle grip transducer ([DV-12554](../development/planning/DV-12554.md)).

---

## 3. What Presets and Hypotheses Mean

The preset selector in the PyQt6 and React NSX surfaces loads verified combinations of physiological hypotheses and mathematical conventions from [`src/config/loop_nullspace_registry.json`](../../src/config/loop_nullspace_registry.json):

| Preset Name           | Label                                                        | Active Hypotheses (`H-*`)                                                                                  | What It Computes                                                                                                                                                                                        |
| :-------------------- | :----------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `envelope_only`       | Static Feasible Envelopes Only (No Representative Selection) | `H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`                                                            | Computes per-frame upper and lower feasible bounds (`[lower, upper]`) and Chebyshev center/radius under joint torque capacity boxes ($                                                                  | \tau_i | \le \tau*{\max,i}$) and grip Coulomb friction cones ($\lVert f*{\text{tan}} \rVert*2 \le \mu*{\text{grip}} f\_{\text{normal}}$) without selecting a single trajectory. |
| `capacity_min_effort` | Capacity-Normalized Minimum Quadratic Effort Representative  | `H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, `H-EFFORT-CAPACITY-QUADRATIC`                             | Adds a strictly convex prior-conditioned representative trajectory minimizing $\sum_i (\tau_i / \tau_{\max,i})^2$ overlaid inside the feasible envelope band.                                           |
| `smooth_capacity`     | Rate-Smoothed Capacity-Normalized Effort Representative      | `H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, `H-EFFORT-CAPACITY-QUADRATIC`, `H-TORQUE-RATE-SMOOTHNESS` | Couples consecutive admitted frames with a finite-difference torque-rate smoothness penalty $\sum_t \lVert (\tau_{t+1} - \tau_t) / \Delta t \rVert_W^2$ alongside capacity-normalized quadratic effort. |

Remember:

- **Hypotheses (`H-*`)** are falsifiable physiological assumptions (such as joint torque bounds or friction cones) that restrict or weight the feasible set.
- **Conventions (`C-*`)** are coordinate frames (`C-FRAME-GRIP`) or norm scalings (`C-METRIC-CAPACITY`, `C-METRIC-IDENTITY`, `C-METRIC-MINV`) chosen by the analyst for reporting.

---

## 4. How to Read Envelope Bands, `INDETERMINATE` Sign Shading, and Binding-Hypothesis Annotations

Each envelope panel (`Envelopes` and `Hand Split`) displays the feasible range `[lower(t), upper(t)]` across time $t$ segmented by sign determinacy:

1. **`DETERMINED_SIGN` Segments (`POSITIVE` or `NEGATIVE`):**
   - Both the lower and upper bounds share the same strict sign ($q_{\min}(t) > \epsilon_{\text{zero}}$ or $q_{\max}(t) < -\epsilon_{\text{zero}}$).
   - Every physically admissible allocation satisfying the active hypotheses agrees on the direction of the torque or hand-split component during this window.
2. **`INDETERMINATE` Sign Shading:**
   - The feasible interval straddles zero ($q_{\min}(t) < -\epsilon_{\text{zero}}$ and $q_{\max}(t) > \epsilon_{\text{zero}}$).
   - Shaded with an amber/neutral sign-indeterminate band to warn the analyst that **both positive and negative values (for example, either lead-hand pull or trail-hand pull) remain feasible** under the declared hypotheses.
3. **`UNBOUNDED` Hatched Segments:**
   - If the active hypothesis set leaves a direction unbounded, the segment is marked `UNBOUNDED` and rendered as hatched/unavailable rather than clamped to an arbitrary cutoff.
4. **Binding-Hypothesis Annotations:**
   - Each envelope segment lists its `binding_hypotheses`—the registry hypothesis IDs (`H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, etc.) whose dual multipliers are active at the upper or lower boundary. This tells you _which physical constraint_ is limiting the envelope at that moment in the swing.
5. **Overlaid Representative Curve:**
   - When `capacity_min_effort` or `smooth_capacity` is selected, the optimal representative trajectory is drawn as a dashed/solid line **inside** the `[lower, upper]` band. By contract, a representative curve is never rendered without its underlying envelope band.

---

## 5. What a Certified Affine Probe (`HOLDS` / `FAILS` / `UNDECIDED`) Means

An affine probe (`NSX-25`) tests whether a linear inequality predicate $c^\top x(t) \ge d$ holds over a swing time window across the **entire** feasible set $\mathcal{F}$:

- **`HOLDS`:** The worst-case minimum of $c^\top x(t) - d$ over all admissible trajectories in $\mathcal{F}$ is non-negative. The statement is guaranteed to hold for every allocation consistent with the declared hypotheses, independent of objective weights.
- **`FAILS` (with Witness):** There exists at least one feasible allocation in $\mathcal{F}$ that violates the predicate; the solver returns the explicit counterexample witness trajectory.
- **`UNDECIDED` (with Reason):** The probe could not be decided because a frame in the window failed kinematic admission or the solver reported numerical/capability unavailability.

---

## 6. What Reaction-Restriction Sensitivity (`WELD6` to `SHAFT_FREE5` to `BALL3`) Is and Is Not

The `REACTION_RESTRICTION_SENSITIVITY` diagnostic (`NSX-26`) evaluates the **same fixed kinematic trajectory** $(q_t, v_t, a_t)$ while restricting which closure wrench components $\lambda$ the closing hand is allowed to transmit in `C-FRAME-GRIP`:

$$\texttt{WELD6} \; (\text{3 forces + 3 moments}) \;\supset\; \texttt{SHAFT\_FREE5} \; (\text{drops axial shaft twist } t_u) \;\supset\; \texttt{BALL3} \; (\text{3 forces only, zero free moments}).$$

- **What It Is:** A mathematical sensitivity diagnostic showing how much the feasible joint-torque envelope shrinks ($\mathcal{F}(\texttt{BALL3}) \subseteq \mathcal{F}(\texttt{SHAFT\_FREE5}) \subseteq \mathcal{F}(\texttt{WELD6})$) when free wrist moments at the grip are restricted on a fixed trajectory.
- **What It Is Not:** It is **not** a forward simulation of how a golfer would swing with a physical hinge or spherical ball-joint grip. A real mechanical release of wrist moments would alter the forward kinematic trajectory $q(t)$ itself.

---

## 7. Explicit Scientific Non-Claims

When interpreting NSX plots or exported reports, keep the five binding non-claims from [`loop_nullspace.tex`](../research/loop_nullspace/loop_nullspace.tex) in mind:

1. **No unique hand split from kinematics:** Inverse dynamics determines $W_{\text{net}}^{\text{bare}}$, never a unique split $s \in \mathbb{R}^6$ without prior hypotheses.
2. **No stiffness or co-contraction inference:** Net joint torques $\tau$ do not measure antagonist muscle co-contraction or joint impedance.
3. **No inertial identification beyond MOSAIC:** NSX performs zero inertial parameter estimation.
4. **No aerodynamic resolution of loop identifiability:** External aerodynamic drag on the clubhead or shaft affects $W_{\text{net}}^{\text{bare}}$ but does not shrink $\ker G$.
5. **No empirical validation claimed:** All outputs carry `validation_status = UNVALIDATED` until compared against held-out dual-handle instrumented grip and synchronized sEMG measurements under [DV-12554](../development/planning/DV-12554.md).
