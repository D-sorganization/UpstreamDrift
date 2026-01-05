# Scientific Python Project Review (Prompt B)

**Repo:** Golf_Modeling_Suite
**Date:** 2026-01-04
**Reviewer:** Automated Principal Agent

## 1. Executive Summary

The project demonstrates high competence with the MuJoCo API but fails in general physics implementation outside the engine's black box. The "Kinematic Force Analysis" module attempts to derive forces manually (Centrifugal, Apparent) and makes fundamental geometric errors.

*   **Overall Assessment**:
    *   **Math**: Inverse Dynamics via MuJoCo is correct. Manual derivations are questionable.
    *   **Physics**: **CRITICAL ERROR** in Centripetal Acceleration calculation. It treats the robot as a point mass orbiting the origin (0,0,0), ignoring the articulated nature of the kinematic chain.
    *   **Numerics**: Reliance on Finite Differences ($10^{-6}$) is noisy and slow compared to analytical Jacobians available in MuJoCo.

*   **Top Risks**:
    1.  **Physics Hallucinations**: `compute_centripetal_acceleration` returns nonsense values for articulated golfers. Users relying on this for stress analysis will have invalid data.
    2.  **Derivative Noise**: Using finite differences (`epsilon=1e-6`) to compute `C` matrix introduces discretization noise that can propagate into control instability.
    3.  **Frame Confusion**: Apparent forces are calculated, but the reference frame (Golfer COM vs World) is implicit and potentially mixed.

## 2. Scorecard

| Category | Score | Notes |
| :--- | :--- | :--- |
| **A. Scientific Correctness** | 4 | **Major**: Centripetal math is simplifyingly wrong. |
| **B. Numerical Stability** | 6 | Finite difference works but is suboptimal. |
| **C. Architecture** | 5 | Simulation state leakage. |
| **D. Code Quality** | 8 | Clean code, just wrong math. |
| **E. Testing** | 6 | Tests confirm the code runs, but not that the physics is true. |
| **F. Performance** | 5 | Python loops for vector math. |
| **G. DevEx** | 8 | Good setup. |

## 3. Findings Table

| ID | Severity | Category | Location | Symptom | Fix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B-001** | **Blocker** | Physics | `kinematic_forces.py`:585 | Invalid Mechanics | `a_c = v**2 / radius` assumes circular motion about (0,0). | Use $a = \omega \times (\omega \times r)$ with Jacobian-derived $\omega$. |
| **B-002** | **Major** | Numerics | `kinematic_forces.py`:276 | Discretization Noise | Finite difference for Coriolis Matrix. | Use analytical RNE properties or `mj_deriv`. |
| **B-003** | **Minor** | Optimization | `inverse_dynamics.py`:469 | Inefficient Linear Algebra | `lstsq(jacp.T, torques)` is correct but slow. | Precompute pseudo-inverse if dimensions allows. |

## 4. Remediation Plan

*   **Immediate**: Delete or mark `compute_centripetal_acceleration` as "Experimental/Broken".
*   **Short-term**: Rewrite `compute_centripetal_acceleration` to use proper spatial acceleration logic: $a_{total} = J \ddot{q} + \dot{J} \dot{q}$. The "centripetal/coriolis" part is $\dot{J} \dot{q}$.
*   **Long-term**: Move all heavy kinematic analysis into a C++ plugin or specialized MuJoCo engine extension to avoid Python loop overhead.

## 5. Non-Obvious Improvements

1.  **Lie Algebra**: Use spatial arithmetic (screw theory) for force composition rather than decoupling "Centrifugal" and "Coriolis" explicitly, which is frame-dependent and artifact-prone.
2.  **Energy Verification**: The `compute_kinematic_power` check is excellent. Add an assertion that `Total Power == d/dt(Total Energy)`.
3.  **Effective Mass**: The derivation $m_{eff} = (J M^{-1} J^T)^{-1}$ is correct and a highlight of the code. Keep this!
