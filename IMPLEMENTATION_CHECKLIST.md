# Golf Modeling Suite Implementation Checklist

## Phase 1: Foundation (Weeks 1-3)
**Goal:** Establish optimization and data processing infrastructure.

- [x] **Week 1: CasADi + Pinocchio Setup**
  - [x] Add dependencies (`casadi` in `pyproject.toml`)
  - [x] Create `shared/python/optimization` module
  - [x] Create 2-link arm URDF for validation
  - [x] Implement CasADi+Pinocchio trajectory optimization example
- [ ] **Week 1: Pyomeca Integration**
  - [ ] Clean noisy pose data (Butterworth filters)
  - [ ] Normalize swing cycles
- [ ] **Week 2: BTK Integration**
  - [ ] Implement C3D file reader
  - [ ] Parse industry-standard mocap data

## Phase 2: Input & Kinematics (Weeks 4-5)
**Goal:** Process video into optimized 3D kinematics.

- [ ] **OpenPose Integration**
  - [ ] Implement `OpenPoseEstimator` wrapper
  - [ ] Extract 2D keypoints from video
- [ ] **CasADi Inverse Kinematics (IK)**
  - [ ] Formulate IK as optimization problem
  - [ ] Solve 2D -> 3D lifting

## Phase 3: Biomechanics (Weeks 6-9)
**Goal:** Integrate muscle physiology and bridge to robotics.

- [ ] **OpenSim Setup**
  - [ ] Create `engines/physics_engines/opensim`
  - [ ] Implement OpenSim wrapper `core.py`
- [ ] **MyoConverter Bridge**
  - [ ] Implement OpenSim -> MuJoCo conversion
  - [ ] Validate muscle force preservation

## Phase 4: Advanced (Weeks 10-12)
**Goal:** High-fidelity muscle simulation.

- [ ] **MyoSim / MyoSuite**
  - [ ] Integrate detailed muscle models
  - [ ] Advanced comparative analysis (Pinocchio vs OpenSim vs MuJoCo)
