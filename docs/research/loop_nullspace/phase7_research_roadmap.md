# Phase 7 Research Roadmap Report: Distributional Sampling, Inverse Optimal Control, True Closure Alternatives, and Muscle-Level Redundancy

**Parent Epic:** [UpstreamDrift#12593](https://github.com/D-sorganization/UpstreamDrift/issues/12593) (`NSX: Closed-Loop Null-Space Exploration`)
**Covered Research Children:**

- **NSX-70 (#12588):** Hit-and-Run Sampling of the Feasible Set on a Low-Rank Temporal Basis
- **NSX-71 (#12589):** Inverse Optimal Control of Allocation Weights With Held-Out Swings
- **NSX-72 (#12590):** True Closure Alternatives (Re-Derived Kinematics per Grip Model)
- **NSX-73 (#12591):** Muscle-Level View (Myofullbody Redundancy, Co-Contraction, and Impedance)

---

## 1. Executive Summary and Scope

Phases 1 through 6 of the Closed-Loop Null-Space Exploration (NSX) architecture establish the deterministic mathematical stack: exact $6 \times 6$ grasp-map split coordinates (`NSX-16`), allocation-invariant quantities with validity gates (`NSX-17`), static and trajectory-coupled convex feasible envelopes (`NSX-21`, `NSX-22`), prior-conditioned conic allocation (`NSX-23`), $\epsilon$-constraint Pareto fronts (`NSX-25`), and fixed-kinematics reaction-restriction sensitivity (`NSX-26`).

Phase 7 (`NSX-70` through `NSX-73`) addresses four methodological extensions identified during technical review (`src/shared/python/loop_nullspace/research_spikes.py`):

1. **Distributional Trajectory Exploration (`NSX-70`):** Sampling smooth interior trajectories $z(t)$ across the feasible polytope without relying on a single objective function.
2. **Pre-Registered Inverse Optimal Control (`NSX-71`):** Estimating convex combination weights $w \in \Delta^{p-1}$ across candidate physiological objectives and evaluating out-of-sample generalization on strictly held-out swings.
3. **True Kinematic Closure Alternatives (`NSX-72`):** Re-solving closed-chain inverse kinematics independently for each grip joint model (`WELD6`, `SHAFT_FREE5`, `BALL3`) rather than restricting multiplier rows on a fixed welded trajectory.
4. **Muscle-Level Redundancy and Impedance (`NSX-73`):** Formalizing the actuator-to-muscle null space $\ker R(q)$ to demonstrate why net joint torques $\tau$ cannot identify antagonist co-contraction or short-range joint rotational stiffness.

---

## 2. NSX-70 (#12588): Hit-and-Run Sampling on a Low-Rank Temporal Basis

### 2.1 Mathematical Formulation

At each admitted frame $t \in \{0, \dots, T-1\}$, the equality-reduced null-space coordinate $z(t) \in \mathbb{R}^{\nu_{\mathrm{eq}}}$ is constrained by the linear inequality polytope
$$A_{\mathrm{ub},t} z(t) \le b_{\mathrm{ub},t}.$$
Sampling $z(t)$ independently frame-by-frame produces temporally discontinuous white-noise trajectories that violate torque-rate limits. Instead, `hit_and_run_temporal_sample` parameterizes $z(t)$ on a low-rank orthonormal discrete cosine/Chebyshev temporal basis $\Phi \in \mathbb{R}^{T \times R}$ ($1 \le R \le T$, $\Phi^\top \Phi = I_R$):
$$z(t) = \sum_{r=0}^{R-1} \Phi_{t,r} c_r, \qquad c = \begin{bmatrix} c_0^\top & \dots & c_{R-1}^\top \end{bmatrix}^\top \in \mathbb{R}^{R \nu_{\mathrm{eq}}}.$$
Substituting this basis expansion into all $T$ frame polytopes yields the lifted trajectory polytope:
$$\mathcal{P}_{\Phi} = \left\{ c \in \mathbb{R}^{R \nu_{\mathrm{eq}}} : A_{\mathrm{lift}} c \le b_{\mathrm{lift}} \right\}, \qquad A_{\mathrm{lift},t} = A_{\mathrm{ub},t} \left(\Phi_{t,:} \otimes I_{\nu_{\mathrm{eq}}}\right).$$

### 2.2 Warm-Start Chebyshev Initialization and Mixing Diagnostics

1. **Chebyshev Center Initialization:** The sampler solves the linear program $\max_{c, r \ge 0} r$ subject to $a_i^\top c + r \|a_i\|_2 \le b_i$ (`CIT-BOYD-VANDENBERGHE-2004`, §8.5.1) to obtain a strictly interior starting point $c^{(0)} = c_{\mathrm{cheb}}$ with positive inscribed radius $r_{\mathrm{cheb}} > 0$.
2. **Chord Line Search:** At step $k$, a unit direction $d^{(k)} \in \mathbb{S}^{R \nu_{\mathrm{eq}} - 1}$ is drawn uniformly on the sphere. The exact chord intersection interval $[\alpha_{\min}, \alpha_{\max}]$ with $\partial \mathcal{P}_{\Phi}$ is computed in closed form from the positive and negative ratios of $(b_{\mathrm{lift}} - A_{\mathrm{lift}} c^{(k)})_i / (A_{\mathrm{lift}} d^{(k)})_i$, and $c^{(k+1)} = c^{(k)} + \alpha d^{(k)}$ is drawn uniformly on $[\alpha_{\min}, \alpha_{\max}]$.
3. **Mixing Diagnostics and Envelope Comparison:** `TemporalHitAndRunResult` reports the lag-1 autocorrelation $\hat{\rho}_1$, effective sample size $\mathrm{ESS} = N (1 - \hat{\rho}_1) / (1 + \hat{\rho}_1)$, and pointwise sample bounds $[z_{\min}^{\mathrm{sample}}(t), z_{\max}^{\mathrm{sample}}(t)]$ compared against the exact framewise LP envelope $[z_{\min}^{\mathrm{LP}}(t), z_{\max}^{\mathrm{LP}}(t)]$. Because $\mathcal{P}_{\Phi}$ is a smooth subspace restriction of the Cartesian product $\prod_t \mathcal{F}_t$, the sampled range is guaranteed to satisfy $[z_{\min}^{\mathrm{sample}}(t), z_{\max}^{\mathrm{sample}}(t)] \subseteq [z_{\min}^{\mathrm{LP}}(t), z_{\max}^{\mathrm{LP}}(t)]$ with zero constraint violations.

---

## 3. NSX-71 (#12589): Inverse Optimal Control of Allocation Weights With Held-Out Swings

### 3.1 Pre-Registered Split and Simplex KKT Formulation

Given a family of $p$ strictly convex candidate objective terms $J_k(z) = \sum_{t=0}^{T-1} \left(\tfrac{1}{2} z_t^\top H_{t,k} z_t + g_{t,k}^\top z_t\right)$ (such as capacity-normalized quadratic effort `H-EFFORT-CAPACITY-QUADRATIC`, torque-rate smoothness `H-TORQUE-RATE-SMOOTHNESS`, and grip moment regularization), Inverse Optimal Control (IOC) seeks convex combination weights $w \in \Delta^{p-1} = \{w \in \mathbb{R}^p : w \ge 0, \mathbf{1}^\top w = 1\}$ that best explain observed interior trajectories $z^{\mathrm{obs}}_{1:T}$.

For an interior trajectory on the training split $\mathcal{D}_{\mathrm{train}}$, first-order KKT stationarity requires:
$$\Phi_t(z_t^{\mathrm{obs}}) w = 0, \qquad \big[\Phi_t(z_t^{\mathrm{obs}})\big]_{:, k} = H_{t,k} z_t^{\mathrm{obs}} + g_{t,k}.$$
`fit_ioc_allocation_weights` stacks $\Phi_{\mathrm{train}}$ across all frames and trials in $\mathcal{D}_{\mathrm{train}}$ and solves the constrained quadratic program:
$$\hat{w} = \arg\min_{w \in \Delta^{p-1}} \frac{1}{2} w^\top \left(\Phi_{\mathrm{train}}^\top \Phi_{\mathrm{train}}\right) w.$$
The fitted weight vector $\hat{w}$ is then frozen and evaluated without refitting on the disjoint held-out swing set $\mathcal{D}_{\mathrm{test}}$ (`held_out_kkt_residual` and `held_out_trajectory_rmse`).

### 3.2 Mandatory Scientific Non-Claim

Per Rule R13 and `DV-12554`, fitting allocation weights to inverse-kinematics (IK) or simulation-derived trajectories only tests algorithmic identifiability; it **never** validates physical hand loads (`IOCFitResult.validates_physical_hand_loads = False`, `non_claim = "Weights fitted to IK-derived data do not validate physical hand loads"`). Physical calibration of $\hat{w}$ requires dual 6-axis instrumented grip measurements on $\mathcal{D}_{\mathrm{train}}$ and locked pre-registered testing on $\mathcal{D}_{\mathrm{test}}$.

---

## 4. NSX-72 (#12590): True Closure Alternatives With Re-Derived Kinematics per Grip Model

### 4.1 Distinction From NSX-26 Reaction-Restriction Sensitivity

In `NSX-26` (`REACTION_RESTRICTION_SENSITIVITY`), a single kinematic trajectory $(q_t, v_t, a_t)$ admitted under the 6-DOF welded grip (`WELD6`) is held fixed while rows of the closure Jacobian $J_c$ are progressively dropped (`WELD6` $\to$ `SHAFT_FREE5` $\to$ `BALL3`). That diagnostic isolates the algebraic effect of releasing reaction moments on a fixed trajectory.

By contrast, `NSX-72` (`TRUE_CLOSURE_ALTERNATIVE`, implemented in `compare_true_closure_models`) treats each grip model $m \in \{\texttt{WELD6}, \texttt{SHAFT\_FREE5}, \texttt{BALL3}\}$ as a distinct mechanical system with its own closure manifold $\Phi_m(q) = 0$:

- **`WELD6` ($m_c = 6$):** Enforces 3 translational and 3 rotational closure equations between the hands and grip handle.
- **`SHAFT_FREE5` ($m_c = 5$):** Releases the axial roll closure constraint about the shaft axis $\hat{u}$, allowing the closing hand to rotate relative to the shaft during the closed-loop IK solve.
- **`BALL3` ($m_c = 3$):** Enforces only the 3 translational coincidence constraints at the grip attachment point, allowing all 3 relative wrist/grip orientation angles to follow minimum-postural-effort kinematics.

### 4.2 Per-Model Kinematic Admission

Each re-derived trajectory $(q_m(t), v_m(t), a_m(t))$ is independently verified against its own model-specific admission gates ($\|\Phi_m(q_m)\|_{\mathrm{scaled}} \le 10^{-6}$, $\|J_m v_m\|_{\mathrm{scaled}} \le 10^{-5}$, $\|J_m a_m + \dot{J}_m v_m\|_{\mathrm{scaled}} \le 10^{-4}$). Because $q_{\mathrm{BALL3}}(t) \neq q_{\mathrm{WELD6}}(t)$, the generalized load $b_m(t) = M(q_m)a_m + h(q_m, v_m)$ changes across models, so the subset monotonicity $\mathcal{F}(\texttt{BALL3}) \subseteq \mathcal{F}(\texttt{WELD6})$ of `NSX-26` applies only to fixed-trajectory reaction restriction and not to true kinematic closure alternatives.

---

## 5. NSX-73 (#12591): Muscle-Level View (Myofullbody Redundancy, Co-Contraction, and Impedance)

### 5.1 Two-Tier Redundancy Architecture

Closed-Loop Null-Space Exploration (`NSX-15` through `NSX-26`) resolves the **skeletal loop null space** mapping generalized loads $b$ to net actuated joint torques $\tau \in \mathbb{R}^{n_a}$:
$$S^\top \tau + J_c^\top \lambda + J_g^\top f_g = M(q)a + h(q, v).$$
However, in musculoskeletal models (`myofullbody_swing` / `hill_muscle.py`) with $n_m > n_a$ muscle-tendon units, a second **actuator-to-muscle null space** exists at every fixed joint torque vector $\tau$:
$$R(q) F_m(a) = \tau, \qquad F_{m,i}(a_i) = f_{\max,i} a_i, \qquad a \in [0, 1]^{n_m},$$
where $R(q) \in \mathbb{R}^{n_a \times n_m}$ is the muscle moment-arm matrix (`analyze_muscle_redundancy_layer`). Whenever $\operatorname{rank}(R(q)) = n_a < n_m$, the muscle null space $\ker R(q)$ has dimension $\nu_{\mathrm{muscle}} = n_m - n_a > 0$.

### 5.2 Co-Contraction Continuum and Short-Range Joint Stiffness

Adding any muscle force perturbation $\Delta F_m \in \ker R(q)$ (corresponding to simultaneous agonist and antagonist activation) leaves the net joint torque $\tau = R(q)(F_m + \Delta F_m)$ **algebraically unchanged**, while directly increasing:

1. **Mean Muscle Activation / Co-Contraction Index:** $\mathrm{CCI}(a) = \frac{1}{n_m}\sum_{i=1}^{n_m} a_i \in [\mathrm{CCI}_{\min}(\tau), \mathrm{CCI}_{\max}(\tau)]$.
2. **Short-Range Joint Rotational Stiffness (`K_J`):** Since short-range muscle-tendon stiffness scales linearly with active muscle force, $k_{m,i}(a_i) = (\gamma / \ell_{0,i}) f_{\max,i} a_i$, the resulting joint stiffness matrix
   $$K_J(a) = R(q) \operatorname{diag}\!\big(k_m(a)\big) R(q)^\top \in \mathbb{S}_{++}^{n_a}$$
   varies across a wide positive-definite interval at identical net joint torque $\tau$. This proves analytically (Rule R13 / Review A4) that joint-level inverse dynamics and null-space torque allocation can never infer joint stiffness or co-contraction without an explicit musculoskeletal model and EMG/stiffness observations.
