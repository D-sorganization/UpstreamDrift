# Golf Modeling Suite - Systematic Implementation Plan

**Created**: 2026-02-06
**Status**: Planning Phase - Issues Created

## Summary

The Golf Modeling Suite needs systematic, proper implementation of all features - no shortcuts or hacks. This document tracks the comprehensive plan.

## ✅ Completed Work

### PR #1128 - Lazy Engine Loading UI
- Implemented lazy loading flow for physics engines
- Users can now selectively load engines on demand
- UI properly shows engine states (idle, loading, loaded, error)
- **Status**: ✅ Merged to main

### PR #1130 - Python 3.10 Backend Compatibility  
- Fixed `datetime.UTC` imports for Python 3.10
- Added lazy loading API endpoints (`/api/engines/{name}/probe`, `/load`)
- Added `configure()` function to engines routes
- **Status**: ✅ Merged to main
- **Note**: Contains temporary hack mapping putting_green → PENDULUM

## 🎯 Systematic Implementation Roadmap

### Phase 1: Physics Engine Installation & Configuration
**Goal**: Get all 5 physics engines properly installed and loading

| Engine | Issue | Priority | Dependencies |
|--------|-------|----------|--------------|
| MuJoCo | #1137 | P0 | mujoco library, GPU drivers |
| Drake | #1138 | P0 | pydrake, VTK, meshcat |
| Pinocchio | #1139 | P0 | Eigen, urdfdom, hpp-fcl |
| OpenSim | #1140 | P1 | OpenSim 4.x binaries |
| MyoSuite | #1141 | P1 | MyoSuite framework |

**Acceptance**: All engines load successfully from UI, basic simulation works for each

### Phase 2: Environment System
**Goal**: Modular, engine-agnostic environment/terrain system

| Component | Issue | Description |
|-----------|-------|-------------|
| Putting Green Engine | #1136 | Register existing putting green implementation |
| Golf Course Environment | #1142 | Full-distance golf course with variable terrain |
| Environment Management | #1145 | Unified system for loading terrains across engines |

**Acceptance**: Can load different environments, work with all physics engines, variable terrain supported

### Phase 3: Full Golf Simulation Capabilities
**Goal**: Complete golf shot simulation from swing to ball landing

**Requirements**:
1. **Golfer Model**: Biomechanical model with muscles, joints, club
2. **Swing Dynamics**: Forward/inverse kinematics, muscle activation
3. **Club-Ball Impact**: Contact physics, energy transfer
4. **Ball Flight**: Aerodynamics (Magnus effect, drag, lift)
5. **Ball Landing & Roll**: Terrain interaction, surface friction
6. **Visualization**: Real-time 3D rendering of full sequence

**Engines Needed**:
- **MuJoCo/Drake**: Rigid body dynamics, contact simulation
- **OpenSim/MyoSuite**: Muscle-skeletal modeling
- **Putting Green**: Specialized green physics
- **Golf Course**: Environment and terrain

### Phase 4: Developer Experience
**Goal**: Easy setup and reproducible environment

| Component | Issue | Description |
|-----------|-------|-------------|
| Docker Environment | #1144 | All physics engines pre-installed |
| Diagnostics Improvement | #1143 | Works in browser mode, copyable output |

## 🐛 Technical Debt to Fix

| Issue | Description | Blocks |
|-------|-------------|--------|
| Putting Green Hack | Remove PENDULUM mapping | #1136 |
| Port Hardcoding | Backend on 8080 but config says 8000 | - |
| Browser Diagnostics | Only works in Tauri mode | #1143 |

## 📋 Implementation Principles

**From this point forward**:
1. ✅ **No Hacks**: Proper implementation every step
2. ✅ **No Shortcuts**: Full feature implementation
3. ✅ **Systematic**: One engine at a time, verify fully before moving on
4. ✅ **Documented**: Every feature properly documented
5. ✅ **Tested**: Integration tests for each capability

## 🔍 Next Immediate Steps

1. **Choose First Engine**: Recommend MuJoCo (most common, good docs)
   - Install MuJoCo library
   - Verify probe/load works
   - Load simple test model
   - Run basic simulation
   - Verify visualization

2. **After MuJoCo Works**: Move to Drake, then Pinocchio, etc.

3. **Putting Green**: Properly register as engine type

4. **Golf Course**: Design and implement environment system

## 📊 Progress Tracking

Track progress through GitHub issues and milestones.

**Milestone 1**: All Engines Loading
**Milestone 2**: Environment System  
**Milestone 3**: Full Golf Shot Simulation
**Milestone 4**: Production Ready (Docker, docs, tests)

---

## Issue Links

- #1136 - Putting Green Integration
- #1137 - MuJoCo Installation
- #1138 - Drake Installation 
- #1139 - Pinocchio Installation
- #1140 - OpenSim Installation
- #1141 - MyoSuite Installation
- #1142 - Golf Course Environment
- #1143 - Diagnostics Fix
- #1144 - Docker Environment
- #1145 - Environment Management System
