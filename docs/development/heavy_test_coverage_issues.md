# Heavy Integration Test Coverage — Proposed Issues

This document identifies gaps in heavy integration test coverage that should
be tracked as GitHub issues. Each section is a self-contained issue.

---

## Issue 1: Add heavy integration tests for PyQt6 GUI components

**Labels:** `testing`, `heavy-integration`, `gui`

### Problem

PyQt6 is a core dependency powering the launcher, model explorer, pendulum
simulator, theme system, signal toolkit, and dashboard. There are no heavy
integration tests verifying that PyQt6 widgets can be instantiated and rendered
in the headless (Xvfb) environment.

### Acceptance Criteria

- [ ] Test that `QApplication` can be created in Xvfb
- [ ] Test that `UpstreamDriftLauncher` main window instantiates and shows (headless)
- [ ] Test that theme system applies dark/light themes without error
- [ ] Test that `PendulumSimulator` main window opens and renders
- [ ] Test that `ModelExplorer` main window opens
- [ ] All tests marked `@pytest.mark.live_simulation`
- [ ] Tests skip gracefully if PyQt6 unavailable

---

## Issue 2: Add heavy integration test for Drake real model loading

**Labels:** `testing`, `heavy-integration`, `drake`

### Problem

The existing `test_phase1_drake_integration.py` uses extensive mocking — it
mocks `pydrake`, `DiagramBuilder`, `AddMultibodyPlantSceneGraph`, `Parser`,
etc. These tests belong in unit tests, not heavy integration. When Drake IS
installed in the heavy Docker image, we should test actual loading.

### Acceptance Criteria

- [ ] Real Drake model load + simulate cycle with a minimal URDF
- [ ] Verify `DrakePhysicsEngine.step()` advances time with real Drake
- [ ] Verify `DrakePhysicsEngine.get_state()` returns real positions
- [ ] Remove or relocate the mock-heavy tests to `tests/unit/`
- [ ] All tests skip if Drake unavailable

---

## Issue 3: Add heavy integration test for ezdxf CAD export

**Labels:** `testing`, `heavy-integration`, `data-io`

### Problem

`ezdxf` is installed in the heavy Docker image and used in
`src/shared/python/data_io/export.py` but has no heavy integration test
verifying it can create and write DXF files.

### Acceptance Criteria

- [ ] Test `ezdxf` can create a new DXF document
- [ ] Test that polylines/entities can be added
- [ ] Test export roundtrip (write + read back)
- [ ] Test project's `export` module DXF path works end-to-end
- [ ] Marked `@pytest.mark.live_simulation`

---

## Issue 4: Add heavy integration test for MediaPipe Tasks API (>= 0.10)

**Labels:** `testing`, `heavy-integration`, `pose-estimation`

### Problem

The existing `test_upstream_contracts.py::TestMediaPipeIntegration` covers the
legacy `mp.solutions.pose` API but only checks importability for the newer
`mp.tasks` API (>= 0.10). As MediaPipe moves to the Tasks API, we need tests
that actually run inference.

### Acceptance Criteria

- [ ] Test `PoseLandmarker` creation with tasks API
- [ ] Test processing a synthetic image through the Tasks pipeline
- [ ] Test project's `mediapipe_gui.py` module instantiation
- [ ] Verify landmark output structure (33 landmarks for body)
- [ ] Test both legacy and tasks API paths

---

## Issue 5: Add heavy integration test for Pinocchio URDF loading

**Labels:** `testing`, `heavy-integration`, `pinocchio`

### Problem

The existing Pinocchio test builds an in-memory model. We should also test
loading a URDF file (the primary usage path) and verify FK, ID, and Jacobian
computations on the loaded model.

### Acceptance Criteria

- [ ] Test `pinocchio.buildModelFromUrdf()` with `simple_arm.urdf`
- [ ] Test FK produces valid SE3 transformations at various configurations
- [ ] Test inverse dynamics consistency (M\*qacc + bias = tau)
- [ ] Test Jacobian computation at end-effector frame
- [ ] Compare results with PendulumPhysicsEngine for a matching model

---

## Issue 6: Add heavy integration test for OpenPose / camera pipeline

**Labels:** `testing`, `heavy-integration`, `pose-estimation`

### Problem

`src/shared/python/pose_estimation/openpose_gui.py` exists but has no heavy
integration test. OpenPose integration should be tested when available.

### Acceptance Criteria

- [ ] Test OpenPose module importability
- [ ] Test `openpose_gui.py` instantiation (mocked camera)
- [ ] Skip gracefully when OpenPose is not installed

---

## Issue 7: Add heavy integration test for motion training / IK pipeline

**Labels:** `testing`, `heavy-integration`, `ik`, `motion`

### Problem

The `pinocchio/python/motion_training/` directory has a `dual_hand_ik_solver.py`
and `motion_visualizer.py` but no dedicated heavy integration test exercising
the full pipeline: load model → configure IK tasks → solve → visualize.

### Acceptance Criteria

- [ ] Test `DualHandIKSolver` instantiation with a Pinocchio model
- [ ] Test IK solve for a reachable target pose
- [ ] Test `MotionVisualizer` can record a trajectory (headless)
- [ ] Tests skip if pink/pinocchio/meshcat unavailable

---

## Issue 8: Add heavy integration test for Simscape/C3D data viewer

**Labels:** `testing`, `heavy-integration`, `c3d`, `gui`

### Problem

The `Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py`
has a GUI viewer for C3D data but no heavy integration test verifying it can
load and display C3D files.

### Acceptance Criteria

- [ ] Test C3D viewer module importability
- [ ] Test loading a small synthetic C3D file into the viewer
- [ ] Test rendering in headless (Xvfb) mode
- [ ] Skip if dependencies unavailable

---

## Issue 9: Heavy test parity audit — CI vs Docker vs local

**Labels:** `testing`, `infrastructure`, `parity`

### Problem

There are three entry points for heavy tests with potential parity drift:

1. `.github/workflows/heavy-tests-opt-in.yml`
2. `.github/workflows/heavy-integration-tests.yml`
3. `run_local_heavy_tests.sh` + `Dockerfile.heavy_test`

The older `heavy-integration-tests.yml` workflow installs dependencies
differently (no system apt packages, no `xvfb-run`, different pip list).
It also runs on `d-sorg-fleet-4core` while opt-in runs on `ubuntu-latest`.

### Acceptance Criteria

- [ ] Unify or remove the duplicate `heavy-integration-tests.yml`
- [ ] Ensure all three entry points install identical dependencies
- [ ] Add a CI check that verifies Dockerfile deps match workflow deps
- [ ] Document which workflow is canonical

---

## Issue 10: Add heavy integration test for humanoid character builder mesh pipeline

**Labels:** `testing`, `heavy-integration`, `mesh`, `urdf`

### Problem

The humanoid character builder (`src/shared/python/humanoid_character_builder/`)
uses trimesh, PyVista, and VTK extensively for mesh generation, collision
geometry, and inertia computation. Current tests only verify importability.
We need end-to-end tests of the mesh→collision→inertia→URDF pipeline.

### Acceptance Criteria

- [ ] Test `MeshGenerator` produces valid trimesh output
- [ ] Test `CollisionGenerator` creates convex decomposition
- [ ] Test `InertiaCalculator` computes physically valid inertias from mesh
- [ ] Test `MeshProcessor` pipeline: load→simplify→export
- [ ] Verify generated URDF has valid inertia elements
- [ ] All tests marked `@pytest.mark.live_simulation`

---

## Issue 11: Add heavy integration test for screw theory modules

**Labels:** `testing`, `heavy-integration`, `robotics`

### Problem

The MuJoCo engine has screw theory modules (`adjoint.py`, `exponential.py`)
and rigid body dynamics implementations (ABA, CRBA, RNEA) with no heavy
integration tests verifying numerical correctness against real engines.

### Acceptance Criteria

- [ ] Test screw exponential map matches Pinocchio's SE3 exponential
- [ ] Test CRBA mass matrix matches MuJoCo's mass matrix for same model
- [ ] Test RNEA inverse dynamics matches MuJoCo's inverse dynamics
- [ ] Cross-validate with analytical pendulum solutions

---

## Issue 12: Ensure all heavy_integration tests use `live_simulation` marker

**Labels:** `testing`, `bug`, `marker`

### Problem

(PARTIALLY FIXED) Nine test files in `tests/heavy_integration/` were missing
the `@pytest.mark.live_simulation` marker, meaning they would not run in the
weekly CI despite being in the heavy integration directory. The CI workflow
filters by `-m "live_simulation"`.

### Status

Fixed in this PR by adding `pytestmark = pytest.mark.live_simulation` to all
files. Issue is for tracking and verification.

### Acceptance Criteria

- [x] All test files in `tests/heavy_integration/` have the marker
- [ ] Add a CI lint check that flags new files in `heavy_integration/` missing
      the marker
- [ ] Consider adding the marker automatically in `conftest.py` instead
