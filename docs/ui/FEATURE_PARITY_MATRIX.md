# PyQt6 vs React UI Feature Parity Matrix

**Document Date:** 2026-05-10  
**Last Updated:** 2026-05-10  
**Related Issue:** #4913  
**EPIC:** #4904 (Launcher UI/UX Remediation & Documentation Sync)

## Executive Summary

This document provides a comprehensive feature parity analysis between the PyQt6 desktop launcher and the React/Tauri web application. The analysis reveals significant feature gaps in the React UI across visualization, controls, and model management capabilities.

## Canonical UI Decision

**The PyQt6 desktop launcher is the canonical UI** for the UpstreamDrift platform. The React/Tauri application serves as a complementary web-based interface with limited feature coverage.

### Rationale

1. **Feature Completeness**: PyQt6 implements 100% of platform capabilities
2. **Engine Integration**: Direct integration with all physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite)
3. **Maturity**: Production-ready with full testing coverage
4. **Active Development**: All new features are implemented in PyQt6 first

### React UI Role

The React UI provides:
- Remote access to simulations via browser
- Lightweight visualization for shared sessions
- Educational/demo purposes
- Future web-based collaboration features

---

## Feature Parity Matrix

### Legend
- ✅ = Fully implemented
- ⚠️ = Partially implemented
- ❌ = Not implemented
- 🔄 = In progress

| Feature Category | Feature | PyQt6 | React/Tauri | Priority | Tracking Issue |
|-----------------|---------|-------|-------------|----------|----------------|
| **Launcher** | Model tile grid | ✅ | ❌ | High | - |
| | Drag-and-drop reordering | ✅ | ❌ | Low | - |
| | View modes (Comfortable/Compact/Dense/List) | ✅ | ❌ | Low | - |
| | Zoom slider | ✅ | ❌ | Low | - |
| | Search/filter | ✅ | ❌ | Medium | - |
| | Status chips | ✅ | ❌ | Low | - |
| | Sidebar navigation | ✅ | ❌ | Low | - |
| **Simulation Controls** | Play/Pause/Stop | ✅ | ⚠️ | High | - |
| | Step frame | ✅ | ❌ | Medium | #7 |
| | Speed control (0.1x-5x) | ✅ | ❌ | Medium | #7 |
| | Timestep adjustment | ✅ | ❌ | Low | - |
| | Per-actuator sliders | ✅ | ❌ | High | #2 |
| | Control type selection | ✅ | ❌ | High | #2 |
| | Polynomial generator | ✅ | ❌ | Medium | #2 |
| **Visualization** | 3D scene rendering | ✅ | ⚠️ | High | - |
| | Force/torque overlays | ✅ | ❌ | High | #1 |
| | Contact force visualization | ✅ | ❌ | Medium | #1 |
| | Camera presets | ✅ | ❌ | Medium | #7 |
| | Camera controls (azimuth/elevation) | ✅ | ⚠️ | Medium | - |
| | Trajectory trails | ✅ | ⚠️ | Medium | - |
| | Swing trajectory recording | ✅ | ❌ | Low | - |
| **Model Interaction** | 6-DOF manipulation | ✅ | ❌ | Medium | #6 |
| | Body selection (raycasting) | ✅ | ❌ | Medium | #6 |
| | Pose library | ✅ | ❌ | Low | #6 |
| | IK-based dragging | ✅ | ❌ | Low | #6 |
| **Model Explorer** | URDF tree viewer | ✅ | ❌ | Medium | #3 |
| | Frankenstein Editor | ✅ | ❌ | Medium | #3 |
| | Character Builder | ✅ | ❌ | Medium | #4 |
| | URDF export | ✅ | ❌ | Medium | #3 |
| **Specialized Views** | Putting Green Simulator | ✅ | ❌ | Medium | #5 |
| | C3D Motion Capture Viewer | ✅ | ❌ | Medium | - |
| | Data Explorer | ✅ | ❌ | Low | - |
| | Video Analyzer | ✅ | ❌ | Low | - |
| **AI Features** | Chat panel | ✅ | ⚠️ | Medium | - |
| | Context-aware help | ✅ | ❌ | Low | - |
| **Settings** | Engine runtime config | ✅ | ❌ | High | - |
| | Docker/WSL mode | ✅ | ❌ | High | - |
| | Theme selection | ✅ | ❌ | Low | - |
| | Layout customization | ✅ | ❌ | Low | - |

---

## Detailed Gap Analysis

### High Priority Gaps

#### 1. Force & Torque Vector Overlay (#1)
**Impact:** Critical for physics debugging and analysis  
**Effort:** Medium (requires WebSocket protocol extension)  
**Dependencies:** Backend must stream force/torque data per frame

#### 2. Joint/Actuator Control Sliders (#2)  
**Impact:** Prevents interactive simulation tuning  
**Effort:** Medium  
**Dependencies:** Backend actuator metadata endpoint

#### 3. Engine Runtime Configuration
**Impact:** Users cannot select Docker/WSL mode in web UI  
**Effort:** Low-Medium  
**Dependencies:** None

### Medium Priority Gaps

#### 4. Model Explorer / Frankenstein Editor (#3)
**Impact:** Cannot create/modify URDF models in web UI  
**Effort:** High  
**Dependencies:** URDF parsing, Three.js URDF loader

#### 5. Character Builder (#4)
**Impact:** Cannot generate custom body models  
**Effort:** Medium  
**Dependencies:** Backend character-builder API

#### 6. 6-DOF Model Manipulation (#6)
**Impact:** Cannot interactively pose models  
**Effort:** Medium  
**Dependencies:** TransformControls, state sync API

#### 7. Putting Green Simulator (#5)
**Impact:** Specialized training tool unavailable  
**Effort:** Medium  
**Dependencies:** Backend putting green API integration

#### 8. Full Simulation Control Panel (#7)
**Impact:** Limited runtime control  
**Effort:** Medium  
**Dependencies:** None

### Low Priority Gaps

#### 9. Launcher Features
**Impact:** Web users cannot navigate models efficiently  
**Effort:** Medium  
**Note:** May not be needed if React UI serves different use case

#### 10. Archive Old GUI Versions (#8)
**Impact:** Repository clutter, confusion  
**Effort:** Low  
**Note:** Cleanup task, not feature development

---

## Recommendations

### Immediate Actions (Wave 1)

1. **Document Canonical UI Status**
   - Update React README to clarify canonical status
   - Add deprecation notice for feature-parity expectations
   - Link to this parity matrix

2. **Implement Critical Controls**
   - Force/torque visualization (#1)
   - Joint/actuator sliders (#2)
   - Engine runtime selector

3. **Archive Legacy Code** (#8)
   - Move r0 GUI versions to archive/
   - Clean up backup directories

### Medium-Term (Wave 2)

4. **Model Explorer Foundation** (#3, #4)
   - URDF tree viewer
   - Basic Character Builder

5. **Interaction Layer** (#6)
   - Body selection
   - TransformControls integration

### Long-Term (Wave 3)

6. **Specialized Views** (#5)
   - Putting Green Simulator
   - C3D Viewer integration

7. **Advanced Features**
   - Pose library
   - Frankenstein Editor

---

## Implementation Strategy

### Backend Requirements

For React UI to achieve parity, the following backend APIs are needed:

| Endpoint | Method | Purpose | Priority |
|----------|--------|---------|----------|
| `/api/engines/{name}/actuators` | GET | Get actuator metadata | High |
| `/api/simulation/forces` | WS | Stream force/torque data | High |
| `/api/simulation/set_control` | WS | Set actuator values | High |
| `/api/simulation/set_state` | POST | Apply position changes | Medium |
| `/api/character-builder/generate` | POST | Generate character URDF | Medium |
| `/api/models/{id}/urdf` | GET | Get model URDF | Medium |
| `/api/putting-green/simulate` | POST | Run putting simulation | Low |

### Frontend Architecture

Recommended React component structure:

```
ui/src/
├── components/
│   ├── simulation/
│   │   ├── ControlPanel.tsx (NEW)
│   │   ├── JointControlPanel.tsx (NEW)
│   │   ├── CameraControls.tsx (NEW)
│   │   └── SimulationControls.tsx (EXTEND)
│   ├── visualization/
│   │   ├── ForceOverlay.tsx (NEW)
│   │   ├── TransformGizmo.tsx (NEW)
│   │   └── Scene3D.tsx (EXTEND)
│   └── model-explorer/
│       ├── URDFTreeView.tsx (NEW)
│       ├── FrankensteinEditor.tsx (NEW)
│       └── CharacterBuilder.tsx (NEW)
├── pages/
│   ├── Simulation.tsx (EXTEND)
│   ├── ModelExplorer.tsx (NEW)
│   ├── PuttingGreen.tsx (NEW)
│   └── CharacterBuilder.tsx (NEW)
└── api/
    ├── simulation.ts (EXTEND)
    ├── characterBuilder.ts (NEW)
    └── models.ts (NEW)
```

---

## Success Metrics

| Metric | Current | Target (6mo) | Target (12mo) |
|--------|---------|--------------|---------------|
| Feature Coverage | ~15% | 40% | 70% |
| High-Priority Gaps | 5 | 1 | 0 |
| Active React Users | TBD | TBD | TBD |
| Backend API Coverage | ~30% | 60% | 90% |

---

## Related Documents

- [React UI Parity Issues](../assessments/issues/REACT_UI_PARITY_ISSUES.md)
- [Launcher UI/UX Epic #4904](../development/launcher_parity_assessment.md)
- [React UI README](../../ui/README.md)

---

## Appendix: Assessment Methodology

This parity matrix was generated by:

1. Enumerating all PyQt6 launcher features from source code analysis
2. Comparing against React UI component inventory
3. Consulting existing assessment documents
4. Validating with running instances of both UIs

**Tools Used:**
- Source code grep/search
- Component tree analysis
- Runtime feature verification