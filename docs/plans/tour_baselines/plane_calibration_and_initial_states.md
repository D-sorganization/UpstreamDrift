# Calibrate Swing Planes, Fixed Geometry and Feasible Initial States (TB-03 #10588)

Parent Epic: #10584 | Program: #10363 | Task: TB-03 (#10588)

## 1. Executive Summary

This module establishes the authoritative mathematical formulations and numerical contracts for:

1. **Rigid 3D-to-2D Swing Plane Calibration**: Fitting one single $SE(3)$ coordinate basis across a declared capture window from valid weighted observations, forbidding frame-by-frame plane re-fitting.
2. **Geometric Projection Residuals**: Lower-bound geometric diagnostics reporting physical RMSE and maximum deviation while preserving original 3D residuals.
3. **Fixed Geometry Calibration & Identifiability Ranking**: Bounded link length calibration ($L_1 \in [0.4, 0.9]\text{ m}$, $L_2 \in [0.7, 1.3]\text{ m}$), frozen non-identifiable mass/inertia priors, and Fisher sensitivity rank diagnostics.
4. **Initial States & Forward Kinematics Verification**: Mapping $t_0$ observations to generalized coordinates $q_0 = (\theta_1, \theta_2)$ and velocities $v_0 = (\omega_1, \omega_2)$ with gap crossing rejection and verified round-trip forward kinematics ($FK(q_0) = p_{\text{observed}}(t_0)$).
5. **Prescribed Moving-Hub Power Tracking**: Distinct handling of fixed-pivot versus moving hubs, recording external base motion $\mathbf{r}_{\text{hub}}(t)$, velocity $\dot{\mathbf{r}}_{\text{hub}}(t)$, and power contribution $P_{\text{hub}}(t) = \mathbf{F}_{\text{hub}}(t) \cdot \dot{\mathbf{r}}_{\text{hub}}(t)$.

---

## 2. Mathematical Formulations

### 2.1 Rigid Swing Plane Estimation

Given valid weighted marker observations $P \in \mathbb{R}^{N \times 3}$ and weights $w \in \mathbb{R}^N$ with $\sum w_i = 1$:
$$p_0 = \sum_{i=1}^N w_i p_i$$
$$X_i = \sqrt{w_i}(p_i - p_0)$$
Singular Value Decomposition on centered points:
$$X = U \Sigma V^T$$
The principal in-plane axes $u, v$ and normal $n = u \times v$ satisfy:
$$\det([u, v, n]) = +1, \quad [u, v, n] \in SO(3)$$
Rigid coordinate transformation from world to plane:
$$T_{w \to p} = \begin{bmatrix} R & -R p_0 \\ \mathbf{0}^T & 1 \end{bmatrix}, \quad R = [u, v, n]^T$$

### 2.2 Effective In-Plane Gravity

Given world gravitational vector $\mathbf{g}_{\text{world}} = [0, -g, 0]^T$ (Y-up) or $[0, 0, -g]^T$ (Z-up):
$$\mathbf{g}_{\text{plane}} = R \mathbf{g}_{\text{world}}$$
The effective projected in-plane gravity magnitude equals $g \cos(\beta)$, where $\beta$ is the tilt from vertical, matching the analytical double pendulum dynamics.

### 2.3 Forward Kinematics & Coordinate Mapping

For the planar double pendulum:

- $\theta_1$: angle of upper segment relative to downward vertical.
- $\theta_2$: relative wrist angle between lower and upper segments.
  $$\mathbf{p}_{\text{grip}} = \mathbf{p}_{\text{pivot}} + \begin{bmatrix} L_1 \sin\theta_1 \\ -L_1 \cos\theta_1 \end{bmatrix}$$
  $$\mathbf{p}_{\text{head}} = \mathbf{p}_{\text{grip}} + \begin{bmatrix} L_2 \sin(\theta_1 + \theta_2) \\ -L_2 \cos(\theta_1 + \theta_2) \end{bmatrix}$$

Inverse kinematics at $t_0$ maps $(p_{\text{pivot}}, p_{\text{grip}}, p_{\text{head}})$ to $(\theta_1, \theta_2)$ with proper unwrapping:
$$\theta_1 = \text{atan2}(u_{1,x}, -u_{1,y})$$
$$\theta_2 = (\text{atan2}(u_{2,x}, -u_{2,y}) - \theta_1 + \pi) \pmod{2\pi} - \pi$$
Velocities $(\omega_1, \omega_2)$ are evaluated across adjacent valid frames without crossing unmeasured gaps.

### 2.4 Moving Hub Tracking

When the shoulder pivot translates externally, the work done on the mechanism is evaluated by integrating power:
$$P_{\text{hub}}(t) = \mathbf{F}_{\text{hub}}(t) \cdot \mathbf{v}_{\text{hub}}(t)$$
$$W_{\text{hub}} = \int_{0}^T P_{\text{hub}}(t) dt$$
Externally driven hubs are never categorized as unforced or free baselines.
