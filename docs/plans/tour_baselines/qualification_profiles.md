# Qualification Profiles and Attainable-Geometry Baselines

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584)  
Work Package: [#10587](https://github.com/D-sorganization/UpstreamDrift/issues/10587) (TB-02)  
Governing Epic: [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363) (Matched Swing Program)

---

## 1. Principle of Attainable Fixed-Geometry Residuals

A fundamental pitfall in multi-model biomechanical evaluation is attempting to apply a single universal millimeter threshold across disparate physical topologies.

- **Full-Body Multibody Models (30–45 DOFs):** Have sufficient articulation to follow 60+ full-body markers while obeying anatomical joint limits and contact dynamics. The authoritative G1/G2/G3 clinical thresholds (`AcceptanceGates` in `acceptance.py`) govern these models.
- **Planar Driven Pendulums (2 DOFs):** Represent arm and club shaft rotation in a planar swing surface. They physically cannot deform out of plane to match pelvic yaw, spinal tilt, or shoulder retraction. Enforcing a 25 mm full-body threshold would fail every planar fit, while relaxing the full-body threshold to 150 mm would compromise clinical rigor.
- **Constrained Upper-Body Golfers (5 independent DOFs):** Embody a closed kinematic loop through the lead arm, club grip, and trail arm. Lower extremity markers (hips, knees, ankles) are unmodeled and absent.

**The Solution:** We establish **frozen numeric qualification profiles** tailored to each model topology, evaluated strictly over their declared observable landmark sets, without modifying or relaxing the authoritative full-body thresholds.

---

## 2. Frozen Profile Specifications

### 2.1 Authoritative Full-Body Profile (`AuthoritativeFullBodyProfile`)

- **Topology:** `ModelTopology.FULL_BODY_MULTIBODY`
- **Applicable Models:** `full_body_golf_g1`, `full_body_pinocchio`, `full_body_drake`, `full_body_opensim`, `full_body_simscape`, `myosuite_golf`
- **Gates:**
  - G1 (0 to 0.85 s): Whole marker $\text{RMSE} \le 25$ mm, early $\le 12$ mm, terminal $\le 35$ mm, club $\le 60$ mm, pelvis yaw $\le 3^\circ$ (0.0524 rad).
  - G2 (through impact): Whole marker $\text{RMSE} \le 40$ mm, early $\le 15$ mm, terminal $\le 50$ mm, club $\le 75$ mm, pelvis yaw $\le 5^\circ$ (0.0873 rad).
  - G3 (full swing): Whole marker driver $\le 60$ mm, iron $\le 95$ mm, early $\le 20$ mm, terminal $\le 80$ mm, club $\le 100$ mm, pelvis yaw $\le 6^\circ$ (0.1047 rad).
  - Physical: Normal force $\le 3.0\times$ BW, penetration $\le 10$ mm, weld closure $\le 5$ mm.

### 2.2 Planar Driven Pendulum Profile (`PlanarDrivenPendulumProfile`)

- **Topology:** `ModelTopology.PLANAR_DRIVEN_PENDULUM`
- **Applicable Models:** `double_pendulum_golf` (Tools package)
- **Observable Landmarks:** `("Grip", "Marker_2", "Marker_3")`
- **Gates:**
  - Club Marker $\text{RMSE} \le 150$ mm ($0.150$ m).
  - Out-of-Plane Residual $\le 50$ mm ($0.050$ m) to guarantee planar kinematic integrity.
- **Rationale:** The planar double pendulum is an educational baseline illustrating parametric swing dynamics and kinetic sequencing. Attainable fixed-geometry residuals reflect the unmodeled trunk translation and wrist cocking deviations.

### 2.3 Constrained Upper-Body Profile (`UpperBodyGolferProfile`)

- **Topology:** `ModelTopology.CONSTRAINED_UPPER_BODY`
- **Applicable Models:** `upper_body_golfer`
- **Observable Landmarks:** Torso, shoulder, elbow, wrist, and club markers.
- **Gates:**
  - Upper-Body Marker $\text{RMSE} \le 55$ mm ($0.055$ m).
  - Weld Closure Residual $\le 5$ mm ($0.005$ m) enforcing grip weld consistency.
- **Rationale:** Captures the 5-DOF closed kinematic chain of the upper torso and bilateral arms. Lower extremities are excluded from tracking.

### 2.4 Triple Pendulum Reconstruction Profile (`TriplePendulumProfile`)

- **Topology:** `ModelTopology.KINEMATIC_RECONSTRUCTION`
- **Applicable Models:** `triple_pendulum`
- **Observable Landmarks:** Torso axis, lead arm, club shaft.
- **Gates:**
  - Club Marker $\text{RMSE} \le 120$ mm ($0.120$ m).
- **Rationale:** Pure kinematic reconstruction baseline for club path validation.

---

## 3. Definition of "Best Feasible Candidate"

Under the Matched Swing Program, "best" is formally defined as:

> **The best feasible candidate within a declared model class, objective, observation set, evaluation horizon, and computation budget.**

A candidate is never ranked against another candidate across different observation sets or model topologies. The cryptographic `landmark_set_signature` and `model_id` enforce this separation.
