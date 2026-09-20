# Second Commercial Simulator and Relay Adapter Technical Evaluation

**Epic #10188** | **Child Issue GS-10 (#10199)**  
**Audience:** Lead Integrators, Systems Architects, Operator Liaisons  
**Status:** Canonical Evaluation & Vendor Inquiry Specification

---

## 1. Executive Summary

As part of [Epic #10188](https://github.com/D-sorganization/UpstreamDrift/issues/10188) (Stage GS-10, [#10199](https://github.com/D-sorganization/UpstreamDrift/issues/10199)), this evaluation analyzes candidate secondary commercial simulator destinations—specifically **TruGolf E6 CONNECT**, **Creative Golf 3D**, and the open-source **Flight Relay Protocol** (`flighthook`)—to assess their viability, protocol maturity, licensing requirements, and architectural trade-offs against UpstreamDrift's vendor-independent simulator capability model.

### Key Decisions & Release Boundaries

1. **Local Reference Destination Affirmed:** The built-in `LocalReferenceAdapter` remains the primary, fully qualified alternative to GSPro. It operates offline, requires no subscription or proprietary hardware, provides high-fidelity local RK4 ballistic trajectory calculations, and validates adapter substitution without external vendor dependencies.
2. **Flight Relay Protocol Supported:** An open-standard `FlightRelayAdapter` is implemented in `src.shared.python.golf_simulator.adapters.relay`, conforming to `SimulatorAdapter` and supporting unidirectional JSON event dispatch.
3. **Proprietary Commercial Simulators Declared Unsupported:** Neither E6 CONNECT nor Creative Golf provides an open, unauthenticated local socket API analogous to GSPro's Open Connect v1. Interfacing with them directly requires formal commercial developer partnerships, signed NDAs, and proprietary SDKs. They are formally declared `UNSUPPORTED` in the capability matrix, and direct connection attempts raise `UnsupportedDestinationError` with clear diagnostics.
4. **Turnkey Vendor Inquiry Text Provided:** Precise, non-confidential inquiry templates are prepared below for the user to initiate formal developer relations with TruGolf and Creative Design.

---

## 2. Candidate Simulator Evaluations

### 2.1 TruGolf E6 CONNECT

- **Overview:** Industry-standard commercial golf simulation engine featuring extensive physics modeling and photorealistic course environments.
- **Protocol Architecture:**
  - E6 CONNECT uses a proprietary tracking system interface. Inbound shot injection is handled either through vendor-approved hardware drivers or private DLL/WebSocket interfaces.
  - Unlike GSPro Open Connect, there is no documented, open, unauthenticated loopback TCP port (e.g., port 921) enabling third-party launch monitors to inject shots without a vendor authorization key.
- **License & Entitlement:**
  - Requires a commercial TruGolf Developer License and membership in the TruGolf Tracking Integration Program.
  - Software license tiers (Standard vs. Commercial/Arcade) actively restrict unauthorized third-party device drivers.
- **Delivery & Recovery Semantics:**
  - Proprietary protocol uses synchronous binary or encrypted JSON framing without public documentation on disconnect uncertainty or duplicate injection protection.
- **Feasibility Verdict:** **UNSUPPORTED DIRECTLY**. Model injection can be achieved downstream if the user runs an authorized community relay (e.g. `flighthook` with an authorized E6 license), but direct native socket communication is unavailable without a TruGolf developer partnership.

### 2.2 Creative Golf 3D (Creative Design)

- **Overview:** European golf simulator platform with substantial international course coverage and support for multiple tracking devices.
- **Protocol Architecture:**
  - Creative Golf publishes a hardware compatibility list covering FlightScope, SkyTrak, Foresight, and Garmin.
  - However, hardware device compatibility **does not equal an open developer API**. Device communication is handled via proprietary compiled DLL drivers bundled into the Windows installation directory.
- **License & Entitlement:**
  - No public developer portal or open API specification exists. SDK access requires an OEM/Hardware Partner Agreement directly with Creative Design s.r.o.
- **Feasibility Verdict:** **UNSUPPORTED DIRECTLY**. Absent an OEM agreement, direct socket connection cannot be lawfully or reliably implemented.

### 2.3 TrackMan Performance Studio, Foresight FSX Pro, and Awesome Golf

- **Overview:** Enterprise-tier radar/camera launch monitor ecosystems.
- **Protocol Architecture:** Closed, end-to-end encrypted proprietary telemetry. APIs are strictly restricted to enterprise partners and licensed hardware purchasers.
- **Feasibility Verdict:** **UNSUPPORTED**. Explicitly excluded from core scope.

---

## 3. Community Relay Architecture: Flight Relay Protocol (`flighthook`)

### 3.1 Overview

The open-source `flighthook` community project implements the **Flight Relay Protocol (FRP)**, a vendor-neutral REST and WebSocket specification designed to decouple launch monitors from simulation destinations:

```
+-------------------+       Flight Relay Protocol       +-------------------+       Open Connect v1      +-------------------+
|   UpstreamDrift   | ================================> |    flighthook     | =========================> |  Licensed GSPro   |
| FlightRelayAdapter|   JSON over WebSocket / HTTP      |  (Community Relay)|       TCP 127.0.0.1:921    |    Simulator      |
+-------------------+                                   +-------------------+                            +-------------------+
                                                                  ||
                                                                  || Proprietary Driver
                                                                  \/
                                                        +-------------------+
                                                        |    E6 CONNECT     |
                                                        +-------------------+
```

### 3.2 Trade-Off Analysis: Direct Adapter vs. External Relay

| Evaluation Criteria           | Direct In-Process Python Adapter (GSPro / Local)                                               | External Community Relay (`flighthook`)                                                             |
| :---------------------------- | :--------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| **Dependency Surface**        | Minimal (standard library `asyncio`, zero external runtime deps).                              | Requires external Node.js/Python process, extra dependencies, and process management.               |
| **Failure Observability**     | Immediate: Socket disconnects and timeouts directly captured by `ShotJournal` as `AMBIGUOUS`.  | Mediated: Socket drops between relay and simulator may be hidden or misreported by the relay layer. |
| **Duplicate Shot Prevention** | Strict: Single-producer lock and durable journal recovery prevent silent duplicate injections. | Depends on relay idempotency; risk of blind retries across relay boundaries.                        |
| **Protocol Stability**        | Pinned to characterized vendor profiles (e.g. Open Connect v1 profile).                        | Dependent on evolving third-party community schemas and version drift.                              |
| **License Compatibility**     | MIT / UpstreamDrift core compliant. Zero licensing encumbrance.                                | MIT / Apache community license. Requires dependency isolation.                                      |
| **Destination Flexibility**   | Targets one characterized destination per adapter.                                             | High: Can bridge to multiple downstream targets (GSPro, E6) if supported by the relay.              |

### 3.3 Architectural Conclusion

UpstreamDrift adopts the **hybrid model**:

1. High-assurance direct adapters (`GSProTransport`, `LocalReferenceAdapter`) remain the core, first-class implementation for deterministic biomechanical research.
2. `FlightRelayAdapter` provides an optional, pluggable adapter for operators who choose to run a local `flighthook` instance to bridge to other software.

---

## 4. Capability Matrix Across Evaluated Destinations

| Capability                 | Local Reference   | GSPro (Open Connect v1) | Flight Relay Protocol | E6 CONNECT (Direct) | Creative Golf (Direct) |
| :------------------------- | :---------------- | :---------------------- | :-------------------- | :------------------ | :--------------------- |
| **Shot Input**             | `SUPPORTED`       | `SUPPORTED`             | `SUPPORTED`           | `UNSUPPORTED`       | `UNSUPPORTED`          |
| **Club Kinematics**        | `SUPPORTED`       | `SUPPORTED`             | `SUPPORTED`           | `UNSUPPORTED`       | `UNSUPPORTED`          |
| **Ball Trajectory Return** | `SUPPORTED` (RK4) | `UNSUPPORTED`           | `UNSUPPORTED`         | `UNSUPPORTED`       | `UNSUPPORTED`          |
| **Native Avatar Rig**      | `UNSUPPORTED`     | `UNSUPPORTED`           | `UNSUPPORTED`         | `UNSUPPORTED`       | `UNSUPPORTED`          |
| **Course State Feedback**  | `UNSUPPORTED`     | `UNSUPPORTED`           | `UNSUPPORTED`         | `UNSUPPORTED`       | `UNSUPPORTED`          |
| **Aim Control**            | `UNSUPPORTED`     | `UNSUPPORTED`           | `UNSUPPORTED`         | `UNSUPPORTED`       | `UNSUPPORTED`          |
| **License Required**       | None (Built-in)   | Paid GSPro License      | Open Source           | Commercial Partner  | OEM Agreement          |

---

## 5. Vendor Inquiry Templates for User Transmission

The user may transmit the following formal, non-confidential inquiry templates to TruGolf and Creative Design to request official developer access.

### 5.1 TruGolf (E6 CONNECT) Developer Inquiry

```text
Subject: Technical Inquiry: Developer API / SDK for Robotic & Biomechanical Shot Injection in E6 CONNECT

Dear TruGolf Developer Relations Team,

We are developing UpstreamDrift, an open-source biomechanical simulation and physics research framework for golf swing analysis. Our system calculates forward-dynamics clubhead delivery and impact physics, producing full launch conditions (ball speed, launch angles, 3D spin vector, and club delivery kinematics).

We are seeking information regarding authorized integration with E6 CONNECT:
1. Does TruGolf offer a public or partner Developer SDK / API for programmatic shot injection (e.g. over a local TCP/WebSocket interface or native C++ library)?
2. What are the licensing requirements and partner agreements necessary to qualify an outbound software connector for academic and simulation use?
3. Does the E6 CONNECT API support passing full club delivery parameters (club path, attack angle, face angle, impact location)?
4. Is there a developer sandbox or test harness available for verifying protocol conformance without requiring a full commercial tracking license?

We look forward to learning about your developer program and technical requirements.

Sincerely,
[Name / Organization]
UpstreamDrift Development Team
```

### 5.2 Creative Design (Creative Golf 3D) Developer Inquiry

```text
Subject: Technical Inquiry: Third-Party Launch Monitor & Simulation Interface for Creative Golf 3D

Dear Creative Design Integration Team,

We are researching model-driven golf simulation and forward-dynamics swing modeling within the UpstreamDrift project. We would like to evaluate Creative Golf 3D as an interchangeable visual simulation destination for our model-generated launch conditions.

We would appreciate guidance on the following technical questions:
1. Does Creative Golf 3D support programmatic shot delivery via an open network socket or local API, similar to industry-standard connect interfaces?
2. Are third-party tracking interfaces restricted to compiled OEM device drivers, or is there an application-level API for programmatic integration?
3. What is the formal application procedure to obtain developer documentation and software licensing for testing and protocol characterization?

Thank you for your time and guidance.

Sincerely,
[Name / Organization]
UpstreamDrift Development Team
```

---

## 6. Implementation and Release Conclusion

- **GS-10 Acceptance Satisfied:** Candidate commercial simulators have been rigorously evaluated; SDK and licensing requirements are documented; architectural trade-offs between direct and relay approaches are established; `FlightRelayAdapter` is implemented and verified; and turnkey inquiry text is provided.
- **Operational Status:** `LocalReferenceAdapter` remains the authoritative second destination in production; proprietary simulators remain cleanly declared `UNSUPPORTED` until vendor agreements are established.
