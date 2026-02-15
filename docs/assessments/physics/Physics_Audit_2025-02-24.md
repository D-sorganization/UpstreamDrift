# Physics Audit Report - 2025-02-24

**Auditor**: Jules, Physics Auditor
**Date**: 2025-02-24
**Focus Area**: Physics, Biomechanics, Equipment, and Statistical Models

## Executive Summary

-   **Overall Physics Fidelity Score**: 6/10
-   **Critical Issues Count**: 2 (GRF Fallback, Patent Infringement)
-   **Confidence in Results**: Moderate. While the framework is sound and "Pragmatic" modules show promise, core legacy implementations (ball flight, impact) rely on simplified or hardcoded models that limit high-fidelity simulation.

The codebase is in a transitional state. Newer modules like `aerodynamics.py` demonstrate sophisticated, citation-backed physics with toggleable complexity. However, older core modules like `ball_flight_physics.py` and `impact_model.py` rely on hardcoded coefficients and 1D approximations. A critical physical error exists in the ground reaction force fallback logic, and a potential patent infringement risk was identified in the kinematic sequence scoring.

## Findings by Category

### 1. Mathematical Correctness

-   **File**: `src/shared/python/physics/ground_reaction_forces.py` (Line ~370, `extract_grf_from_contacts`)
-   **Issue**: **Incorrect GRF Fallback Logic** (Critical)
-   **Description**: When contact forces are unavailable, the function falls back to summing gravity forces: `total_force[2] += abs(np.sum(g))`.
-   **Expected Physics**: Ground reaction forces in a dynamic swing are the sum of static weight AND dynamic inertial forces ($F = ma$). Peak vertical force often exceeds 2x body weight.
-   **Actual Implementation**: Assumes GRF equals static weight, completely ignoring dynamic acceleration.
-   **Impact**: drastically underestimates GRF peaks, rendering biomechanical analysis of power generation invalid.
-   **Recommendation**: Remove this fallback or replace it with an inverse dynamics calculation if kinematic data is available. See `ISSUE_P002_GRF_FALLBACK.md`.

### 2. Physical Plausibility & Equipment Models

-   **File**: `src/shared/python/physics/ball_flight_physics.py` (Lines 60-70)
-   **Issue**: **Hardcoded Aerodynamic Coefficients** (High)
-   **Description**: `BallProperties` uses hardcoded, uncited coefficients (`cd0=0.21`, `cl1=0.38`, etc.).
-   **Expected Physics**: Drag and lift coefficients are functions of Reynolds number and Spin Ratio. They vary significantly between ball models.
-   **Actual Implementation**: Fixed polynomial model `cd = cd0 + s * ...`.
-   **Impact**: Trajectory accuracy is limited to a "generic" ball. Legal risk if these specific numbers match a competitor's confidential data.
-   **Recommendation**: Migrate fully to `aerodynamics.py` which implements Reynolds correction, and externalize coefficients to configuration files. See `ISSUE_P001_AERODYNAMICS_COEFFICIENTS.md`.

-   **File**: `src/shared/python/physics/impact_model.py` (Line 135)
-   **Issue**: **Simplified Effective Mass** (Medium)
-   **Description**: `_compute_effective_club_mass` uses a scalar approximation for off-center hits: `1 / (1/m + r^2/I)`.
-   **Expected Physics**: Effective mass is a tensor quantity depending on the full inertia tensor and the impact normal vector.
-   **Actual Implementation**: 1D approximation assuming impact along a principal axis.
-   **Impact**: Inaccurate ball speed on toe/heel hits, affecting "forgiveness" metrics.
-   **Recommendation**: Implement full 3D rigid body impact physics using the inertia tensor. See `ISSUE_P005_IMPACT_EFFECTIVE_MASS.md`.

-   **File**: `src/shared/python/physics/flexible_shaft.py`
-   **Issue**: **Missing Torsional Dynamics** (High)
-   **Description**: `BeamElement` only models bending (4 DOF). Torsion (twisting) is absent.
-   **Expected Physics**: Shaft torque is critical for clubface closure rate and dynamic lie angle.
-   **Actual Implementation**: Bending only.
-   **Impact**: Cannot model "torque" (torsional stiffness) effects, a key shaft specification.
-   **Recommendation**: Add torsional degree of freedom to `BeamElement`. See `ISSUE_P003_SHAFT_TORSION.md`.

### 3. Biomechanics Accuracy & Legal

-   **File**: `src/shared/python/analysis/pca_analysis.py` (Line 135)
-   **Issue**: **Patent Risk in Efficiency Score** (Critical Legal)
-   **Description**: `efficiency_score = matches / len(expected_order)`.
-   **Expected Physics**: "Efficiency" is a complex biomechanical concept (energy transfer).
-   **Actual Implementation**: Defines efficiency solely as adherence to a specific kinematic sequence (e.g., 1-2-3-4). This specific correlation is often patented (e.g., by TPI or others).
-   **Impact**: Potential patent infringement.
-   **Recommendation**: Rename to `sequence_adherence` and remove claims of "efficiency". See `ISSUE_P004_EFFICIENCY_SCORE_PATENT.md`.

### 4. Statistical Methods

-   **File**: `src/shared/python/biomechanics/kinematic_sequence.py`
-   **Issue**: **Simplistic Peak Detection**
-   **Description**: Finds global maximum of velocity magnitude.
-   **Impact**: Susceptible to noise or false peaks (e.g., backswing or follow-through).
-   **Recommendation**: Implement windowed peak detection triggered by specific swing events (e.g., "start of downswing").

## Validation Recommendations

1.  **Aerodynamics**: Validate `aerodynamics.py` models against **TrackMan** public datasets or **Smits & Smith (1994)** wind tunnel data.
2.  **Shaft**: Compare `flexible_shaft.py` deflection profiles against **Wishon Golf** shaft stiffness profiles (EI curves).
3.  **GRF**: Validate `ground_reaction_forces.py` against standard force plate data (e.g., **AMTI** or **Kistler** sample data).
4.  **Impact**: Compare `impact_model.py` launch conditions against **Science of Golf (Cochran & Stobbs)** empirical formulas.

## Citations Needed

The following implementations need explicit academic citations:

-   **Aerodynamics**:
    -   Smits, A. J., & Smith, D. R. (1994). A new aerodynamic model of a golf ball. *Science and Golf II*, 340-347.
    -   Bearman, P. W., & Harvey, J. K. (1976). Golf ball aerodynamics. *Aeronautical Quarterly*, 27(2), 112-122.
-   **Biomechanics**:
    -   Nesbit, S. M. (2005). A three dimensional kinematic and kinetic study of the golf swing. *Journal of Sports Science & Medicine*, 4(4), 499.
-   **Shaft**:
    -   MacKenzie, S. J., & Sprigings, E. J. (2009). A three-dimensional forward dynamics model of the golf swing. *Sports Engineering*, 11(4), 165-175.
