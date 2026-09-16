# Native GSPro Model Animation and Autonomous Course Feedback Research

## Executive Summary

- **Epic:** [#10188](https://github.com/D-sorganization/UpstreamDrift/issues/10188)
- **Stage:** **GS-11** (Bonus Research)
- **Issue:** [#10200](https://github.com/D-sorganization/UpstreamDrift/issues/10200)
- **Status:** **Research Completed — Explicitly UNSUPPORTED Boundary Established**

This technical investigation assesses the feasibility of two prospective extensions to UpstreamDrift's golf simulator integration:

1. **Native Model Animation**: Injecting custom biomechanical golfer meshes, skeletal bone rigs, and dynamic motion trajectories directly into the GSPro 3D rendering pipeline.
2. **Autonomous Course Feedback & Aim Control**: Closing the simulation loop by receiving real-time course telemetry (ball landing position, lie, terrain surface, elevation, wind, hazard/penalty states) and programmatically controlling target aim and camera orientation.

Based on comprehensive architectural analysis of installed binaries, vendor documentation, and community SDK tooling, **neither capability is supported by GSPro's authorized public APIs**. UpstreamDrift strictly maintains an honest capability boundary: unverified and closed features are declared `UNSUPPORTED`. Rather than attempting unsupported reverse-engineering or memory scraping, UpstreamDrift establishes the **Synchronized Companion Presentation Architecture** as the authoritative, production-ready solution.

---

## 1. Engine Architecture & Course Designer Analysis

### 1.1 Unity Runtime Architecture

GSPro is built on the Unity game engine (observed runtime metadata: Unity 2018/2019 LTS family). Its internal execution pipeline comprises:

- Core game executable (`C:/GSProV1/Core/GSP/GSPro.exe`).
- Connector subprocess (`C:/GSProV1/Core/GSPC/GSPconnect.exe`).
- Native Unity assemblies (`UnityPlayer.dll`, mono/IL2CPP managed libraries).

While Unity natively supports skeletal animation, blend shapes, and custom mesh deformation, **GSPro does not expose an extension runtime or plugin host** for external process script execution.

### 1.2 Course Designer vs. Runtime Avatar Injection

A common misconception in the golf simulation community is that GSPro's **Course Designer** toolset implies support for arbitrary custom 3D asset injection. Architectural examination confirms this is false:

- **Course Designer Scope**: Uses Unity terrain tools and OpenStreetMap/LiDAR plugins (e.g. OPCD) to compile static 3D course environments into pre-baked Unity AssetBundles.
- **Static Assets Only**: Course files contain terrain meshes, textures, splat maps, vegetation, hazard colliders, and pin/tee placement markers.
- **Absence of Skeletal Hooks**: Course packages cannot register dynamic character skeletons, custom bone transforms, or runtime kinematics listeners.
- **Avatar Lifecycle**: Golfer character rendering inside GSPro is hardcoded to its internal avatar assets and fixed pre-baked swing animations. There is no API seam to substitute or drive the golfer rig from external motion capture or simulation trajectories.

---

## 2. Public Protocol Boundaries: Open Connect V1

GSPro's sole documented, vendor-supported integration interface is **Open Connect v1**:

- **Transport**: Local TCP socket listening on `127.0.0.1:921`.
- **Directionality**: Primarily **unidirectional shot injection** (client to simulator).
- **Framing & Serialization**: CRLF-delimited ASCII JSON envelopes.

### 2.1 Missing Telemetry for Autonomous Golf

Autonomous, closed-loop round simulation requires a bidirectional control loop:

$$ ext{Aim \& Club Selection} \longrightarrow ext{Model Swing} \longrightarrow ext{Launch Impact} \longrightarrow ext{Flight} \longrightarrow ext{Landing Telemetry} \longrightarrow ext{Next Lie Assessment}$$

GSPro Open Connect v1 provides only the launch step. The following telemetry fields are **completely absent** from the public wire protocol:

| Telemetry Requirement            | Status in Open Connect v1 | Consequence for Autonomous Play                                                        |
| :------------------------------- | :------------------------ | :------------------------------------------------------------------------------------- |
| **Ball Landing Coordinates**     | `ABSENT`                  | Controller cannot determine where the ball came to rest in world space $(x, y, z)$.    |
| **Current Lie Condition**        | `ABSENT`                  | Controller cannot detect fairway, rough, heavy rough, sand, or green lie penalties.    |
| **Surface Slope & Grain**        | `ABSENT`                  | Controller cannot calculate launch adjustments for uphill, downhill, or sidehill lies. |
| **Distance & Elevation to Pin**  | `ABSENT`                  | Controller cannot calculate target distance or required club trajectory.               |
| **Dynamic Wind Vector**          | `ABSENT`                  | Controller cannot compensate for crosswind or headwind velocity.                       |
| **Hazard / Out-of-Bounds State** | `ABSENT`                  | Controller cannot detect water hazards, penalty drops, or out-of-bounds re-hits.       |
| **Hole Completion / Score**      | `ABSENT`                  | Controller cannot detect holed putts, conceded putts, or hole transitions.             |

### 2.2 Missing Control Commands

Open Connect v1 specifies no RPC endpoints or command messages for simulator manipulation:

- **Aim Orientation**: No message exists to turn the player's aim angle left or right.
- **Target Selection**: No message exists to set an aiming point or select fairway target markers.
- **Camera Selection**: No message exists to switch between ball-flight tracking, green view, or aerial overhead.
- **Hole / Tee Selection**: No command exists to navigate courses, restart holes, or select practice ranges.

---

## 3. Security, Licensing, & Anti-Cheat Boundaries

In the absence of public APIs, some third-party utilities resort to unsupported, invasive techniques. UpstreamDrift explicitly rules out the following approaches:

### 3.1 Prohibited Integration Methods

1. **Process Memory Scraping & Pointers**:
   - Scraping memory structures via Cheat Engine, ReadProcessMemory, or Mono memory offsets.
   - _Rationale_: Highly fragile across minor game updates; triggers anti-cheat software; violates system stability.
2. **DLL Injection & Binary Hooking**:
   - Injecting dynamic link libraries into `GSPro.exe` or hooking internal Unity functions (e.g. via Harmony, BepInEx, or custom detours).
   - _Rationale_: Explicitly violates the GSPro End User License Agreement (EULA) and terms of service; introduces crash risks into vendor software.
3. **Multiplayer Network Sniffing**:
   - Intercepting private simulator-to-server traffic used for online tournaments or simulator matchmaking.
   - _Rationale_: Threatens online integrity and leaderboard fairness; breaches network security protocols.
4. **Binary Decompilation / Modification**:
   - Modifying game assemblies or patching bytecode to expose private fields.
   - _Rationale_: Legally encumbered and incompatible with UpstreamDrift's open science and commercial compliance policies.

UpstreamDrift will **never** ship or endorse unapproved memory hooks, injected DLLs, or patched executables.

---

## 4. Authoritative Solution: Synchronized Companion Presentation

Rather than attempting unauthorized in-scene injection, UpstreamDrift delivers the **Synchronized Companion Presentation Architecture**.

```
+-----------------------------------------------------------------------------+
|                            UPSTREAMDRIFT PLATFORM                           |
|                                                                             |
|  +-------------------------+                 +---------------------------+  |
|  | Forward Dynamics Model  |                 | 3D Articulated Golfer     |  |
|  | - Simscape / Pinocchio  |                 | Biomechanical Viewport    |  |
|  | - Full Skeletal Rig     |                 | - Rerun / WebGL / PyQt6   |  |
|  | - Impact Dynamics       |                 | - High-FPS Joint Kinematics| |
|  +------------+------------+                 +-------------^-------------+  |
|               |                                            |                |
|               | (contact-qualified                         | (synchronized |
|               |  launch conditions)                        |  swing replay) |
|               v                                            |                |
|  +---------------------------------------------------------+-------------+  |
|  |                    GolfSessionService (Coordinator)                   |  |
|  | - Monotonic Replay Clock Synchronization                              |  |
|  | - Intent Journal Persistence & Audit Trail                            |  |
|  +----------------------------+------------------------------------------+  |
+-------------------------------|---------------------------------------------+
                                | (TCP 127.0.0.1:921)
                                v
+-----------------------------------------------------------------------------+
|                           COMMERCIAL SIMULATOR HOST                         |
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  | GSPro Open Connect Interface                                          |  |
|  | - Receives Ball Speed, Launch Angles, Spin Vectors, Club Kinematics    |  |
|  +-----------------------------------+-----------------------------------+  |
|                                      |                                      |
|                                      v                                      |
|  +-----------------------------------------------------------------------+  |
|  | GSPro 3D World Display (Second Monitor / Composed Simulator Screen)   |  |
|  | - Licensed Course Rendering                                           |  |
|  | - Native Ball Flight Trajectory & Terrain Physics                     |  |
|  | - Green Dynamics, Bounces, and Roll-Out                               |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
```

### 4.1 Topology and Visual Experience

1. **Dual-Monitor Setup**:
   - **Primary Display (Projector / Simulator Bay)**: GSPro renders the immersive course environment, target fairways, and realistic ball flight physics.
   - **Secondary Display (Analysis Console)**: UpstreamDrift renders the high-fidelity biomechanical model, showing joint torques, ground reaction forces, club flex dynamics, and planar swing kinematics.
2. **Monotonic Temporal Synchronization**:
   - UpstreamDrift's `MonotonicReplayClock` and `ReplaySubmissionCoordinator` guarantee that the companion avatar animation reaches impact at the precise instant the ball launch packet is dispatched to GSPro.
   - The user experiences a seamless visual continuation: golfer swings on the analysis screen $ o$ impact occurs $ o$ ball launches instantaneously into the GSPro 3D environment.
3. **Operator-Managed Course Context**:
   - Since GSPro does not export automated course telemetry, course play is managed collaboratively by the human operator.
   - The operator views the lie and pin distance in GSPro, selects the corresponding club/intent in UpstreamDrift, and executes the biomechanical model.

---

## 5. Turnkey Vendor Inquiry Set

Should GSPro or other commercial simulator vendors establish an enterprise developer program or scientific SDK, the following structured question set is prepared for user outreach:

```text
Subject: Technical Inquiry: Scientific Biomechanical Avatar & Telemetry Integration with GSPro

Dear GSPro Developer / Integration Team,

We are developing UpstreamDrift, an open scientific biomechanics and forward dynamics simulation platform for golf swing analysis. We currently integrate with GSPro via the Open Connect v1 interface to transmit launch conditions derived from physics models.

To evaluate potential native integration, we would appreciate technical guidance on the following questions:

1. Dynamic Skeletal Rigging & Avatar Customization:
   - Does GSPro offer, or plan to offer, an API or SDK to import or drive character meshes/rigs at runtime?
   - What bone naming convention, hierarchy, and coordinate framing (e.g. Unity Humanoid Mecanim, glTF 2.0) are required?
   - Can joint rotations/quaternions be streamed dynamically via IPC or network sockets (e.g. 60–120 Hz) during the swing cycle?

2. Temporal Synchronization & Impact Timing:
   - What is the expected latency between socket shot submission and the start of ball flight rendering?
   - Is there an impact trigger event or hardware timestamp mechanism to synchronize external avatar swing animation with the exact ball launch frame?

3. Course & Round Telemetry Feedback:
   - Is there an authorized enterprise interface to receive real-time course telemetry, including ball resting coordinates (X, Y, Z), surface/lie type, distance to pin, elevation delta, and wind vectors?
   - Does GSPro support automated hole completion, hazard, or penalty event broadcasts?

4. Programmatic Control & Aiming:
   - Are there supported RPCs to programmatically adjust aim orientation, target lines, or camera views for research and automated testing pipelines?

5. Developer Licensing & Commercial Terms:
   - What developer license, commercial partner tier, or NDA is required to access native SDK documentation and testing builds?

We appreciate your time and assistance, and look forward to learning about potential partnership opportunities.

Sincerely,
UpstreamDrift Development Team
```

---

## 6. Capability Matrix and Conformance State

| Capability Area                 | Local Reference   | GSPro (Open Connect v1) | Flight Relay  | Native GSPro Avatar (Evaluated) |
| :------------------------------ | :---------------- | :---------------------- | :------------ | :------------------------------ |
| **Ball Launch Injection**       | `SUPPORTED`       | `SUPPORTED`             | `SUPPORTED`   | `SUPPORTED`                     |
| **Club Kinematics**             | `SUPPORTED`       | `SUPPORTED`             | `SUPPORTED`   | `SUPPORTED`                     |
| **Skeletal Mesh Customization** | `SUPPORTED`       | `UNSUPPORTED`           | `UNSUPPORTED` | `UNSUPPORTED`                   |
| **Runtime Joint Streaming**     | `SUPPORTED`       | `UNSUPPORTED`           | `UNSUPPORTED` | `UNSUPPORTED`                   |
| **Course State Telemetry**      | `UNSUPPORTED`     | `UNSUPPORTED`           | `UNSUPPORTED` | `UNSUPPORTED`                   |
| **Programmatic Aim Control**    | `UNSUPPORTED`     | `UNSUPPORTED`           | `UNSUPPORTED` | `UNSUPPORTED`                   |
| **Integrated Flight Return**    | `SUPPORTED` (RK4) | `UNSUPPORTED`           | `UNSUPPORTED` | `UNSUPPORTED`                   |
| **Synchronized Companion View** | `SUPPORTED`       | `SUPPORTED`             | `SUPPORTED`   | N/A                             |

### 6.1 Architectural Verdict

1. **Native Avatar Animation**: Concluded as **`UNSUPPORTED`**. The feature is closed. No vendor API exists.
2. **Autonomous Round Play**: Concluded as **`UNSUPPORTED`**. Open Connect v1 is unidirectional. Operator-assisted play remains the certified workflow.
3. **Software Health**: All capability boundaries are enforced via `UnsupportedCapabilityError` and `assert_capability_supported()`, ensuring zero silent failures or undefined behaviors.
