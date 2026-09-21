# Golf Simulator Integration Support and Operational Acceptance Matrix

**Epic #10188** | **Child Issue GS-09 (#10198)**  
**Audience:** Operators, Integration Engineers, Robotics & Biomechanics Researchers  
**Status:** Canonical Support Matrix & Operational Qualification Runbook

---

## 1. Scope & Purpose

This document provides the authoritative hardware, topology, environment, and operational acceptance matrix for the `UpstreamDrift` golf simulator integration suite (`src.shared.python.golf_simulator`).

It formally documents:

1. Supported network and execution topologies (Single-Host loopback and Authenticated Remote Bridge).
2. Host and runtime environment prerequisites across Linux, macOS, and Windows.
3. Explicit, honest capability boundaries (identifying what is natively supported, supported with documented constraints, and strictly unsupported).
4. Deterministic shot qualification acceptance criteria across standard launch regimes (straight drive, left hook, right slice, chip, and putt).
5. Disconnect classification, durable journaling, and operator recovery procedures that prevent duplicate shot delivery.
6. Execution instructions for the live opt-in qualification suite.

---

## 2. Supported Deployment Topologies

```
+---------------------------------------------------------------------------------------+
| TOPOLOGY 1: Single Windows Host (Colocated Model & Simulator)                         |
|                                                                                       |
|  +---------------------------------------------------------------------------------+  |
|  | Windows 10/11 Host                                                              |  |
|  |                                                                                 |  |
|  |  +---------------------------+             +----------------------------------+ |  |
|  |  | UpstreamDrift Application |             | Licensed GSPro Simulator         | |  |
|  |  |  - Physics / Motion Model |             |  - Unity Engine Simulation       | |  |
|  |  |  - GSProTransport         |=== TCP ===> |  - Open Connect v1 API           | |  |
|  |  |  - ShotJournal            | 127.0.0.1   |    Listening on 127.0.0.1:921    | |  |
|  |  +---------------------------+   (Port 921)|  +-------------------------------+ |  |
|  +---------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------+
| TOPOLOGY 2: Separate Compute Host & Simulator Host (Authenticated Remote Bridge)      |
|                                                                                       |
|  +----------------------------+             +---------------------------------------+ |
|  | Simulation / Compute Host  |             | Dedicated Windows Host                | |
|  | (Linux / macOS / Windows)  |             | (Windows 10/11 x64)                   | |
|  |                            |             |                                       | |
|  |  +-----------------------+ |             |  +----------------------------------+ | |
|  |  | UpstreamDrift Model   | |             |  | LocalBridgeServer                | | |
|  |  |  - Swing Trajectory   | |             |  |  - Bearer Token Auth Guard       | | |
|  |  |  - RemoteBridgeClient | |=== TCP ===> |  |  - ProducerLockManager           | | |
|  |  +-----------------------+ | Authenticated|  |  - GSProTransport               | | |
|  +----------------------------+ LAN / VPN   |  +----------------+-----------------+ | |
|                                 (e.g. :9220)|                   | TCP 127.0.0.1:921 | |
|                                             |  +----------------v-----------------+ | |
|                                             |  | Licensed GSPro Simulator         | | |
|                                             |  |  - Open Connect v1 Loopback      | | |
|                                             |  +----------------------------------+ | |
|                                             +---------------------------------------+ |
+---------------------------------------------------------------------------------------+
```

### Topology Comparison Matrix

| Attribute                        | Topology 1: Single Host                 | Topology 2: Remote Bridge                                |
| :------------------------------- | :-------------------------------------- | :------------------------------------------------------- |
| **Model Runtime Platform**       | Windows 10/11 x64                       | Linux, macOS, or Windows 10/11                           |
| **Simulator Runtime Platform**   | Windows 10/11 x64                       | Windows 10/11 x64                                        |
| **Primary Network Transport**    | Loopback TCP socket (`127.0.0.1:921`)   | Authenticated TCP connection (`LocalBridgeServer`)       |
| **Vendor Port 921 Exposure**     | Localhost only (`127.0.0.1`)            | **Strictly prohibited** from LAN exposure; loopback only |
| **Bridge Port**                  | None (direct connection)                | Configurable (default `9220` or operator-assigned)       |
| **Authentication Requirement**   | Process/OS boundary                     | High-entropy Bearer Token (`SHARED_SECRET`)              |
| **Producer Concurrency Control** | Single-process `ProducerLockManager`    | Distributed lease token via `ProducerLockManager`        |
| **Round-Trip Latency Target**    | `< 5 ms` local acknowledgment           | `< 25 ms` over local gigabit LAN / VPN                   |
| **Recommended Use Case**         | Single workstation development, testing | High-performance cluster simulation, headless rigs       |

---

## 3. Environmental Prerequisites & Verification

### 3.1 Runtime Requirements

| Dependency                | Minimum Version | Recommended Version   | Verification Command        | Notes                                                                   |
| :------------------------ | :-------------- | :-------------------- | :-------------------------- | :---------------------------------------------------------------------- |
| **Python**                | `3.11`          | `3.12` or `3.13`      | `python --version`          | Enforced by repo typing & contracts                                     |
| **GSPro Simulator**       | `2024.1`        | Latest stable release | Operator about screen       | Unity engine metadata `2018.x` is build metadata, not GSPro app version |
| **Open Connect Protocol** | `v1`            | `v1`                  | Handshake receipt           | Characterized in `docs/plans/golf_simulator_integration/`               |
| **pywin32**               | Any             | Latest                | `python -c "import winreg"` | Optional on Windows; graceful fallback on Linux/macOS                   |
| **psutil**                | `5.9.0`         | Latest                | `python -c "import psutil"` | Optional; used for enhanced process inspection                          |

### 3.2 Security & Secret Redaction Invariant

Audit receipts and telemetry logs generated by the simulator connector **must never leak secrets**.

- **Enforced Rule:** `SupportReceipt` explicitly excludes tokens, license keys, Windows registry authorization paths, and passwords.
- **Log Scrubber:** All simulator loggers are wrapped with `SecretRedactionFilter`, which sanitizes bearer tokens, secrets, credentials, and query strings from both formatted strings and format arguments (`%s`).
- **Audit Receipt Generation:**

  ```python
  from src.shared.python.golf_simulator.discovery import discover_simulator_installation

  config = discover_simulator_installation()
  receipt = config.build_support_receipt(
      app_version="2026.1.0",
      connector_version="1.0.0",
      optional_packages={"pywin32": True, "psutil": True},
  )
  # Generates nonsecret JSON suitable for operational records and bug reports
  ```

---

## 4. Capability Matrix & Honest Boundaries

Per the architecture contract (`contracts.py`), every capability is declared using explicit state descriptors (`SUPPORTED`, `UNSUPPORTED`, or `UNVERIFIED`). Hopeful booleans or simulated claims without validated APIs are strictly forbidden.

| Capability                            | Status                      | Implementation Authority            | Architectural Rationale & Boundaries                                                                                                               |
| :------------------------------------ | :-------------------------- | :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ball Flight / Launch Injection**    | `SUPPORTED`                 | `GSProTransport.send_shot`          | Delivers launch velocity, vertical launch angle, horizontal launch angle, spin rate, and spin axis according to the Open Connect v1 specification. |
| **Club Dynamics & Impact**            | `SUPPORTED`                 | `ClubData` in `encode_shot_payload` | Sends club head speed, angle in (attack angle), path (in-to-out / out-to-in), and face-to-target when model impact extraction succeeds.            |
| **Straight Shot Drive**               | `SUPPORTED`                 | `test_live_acceptance.py`           | High ball speed (>65 m/s), low horizontal launch (0.0°), low side spin (<100 RPM). Accurately produces straight down-the-fairway flight.           |
| **Left Hook / Draw**                  | `SUPPORTED`                 | `test_live_acceptance.py`           | Positive launch angle / negative side spin representation in simulator frame. Accurately simulates curving left.                                   |
| **Right Slice / Fade**                | `SUPPORTED`                 | `test_live_acceptance.py`           | Negative launch angle / positive side spin representation in simulator frame. Accurately simulates curving right.                                  |
| **High Wedge / Chip**                 | `SUPPORTED`                 | `test_live_acceptance.py`           | Low forward velocity (20-25 m/s), high launch angle (>30°), high backspin (>6,000 RPM). Correctly triggers greenside wedge physics.                |
| **Putting**                           | `SUPPORTED WITH LIMITATION` | `test_live_acceptance.py`           | Supported for ball speeds $\ge 1.5\text{ m/s}$. Sub-threshold speeds (<1.0 m/s) may be discarded or clipped by the simulator engine.               |
| **Native Avatar / Biomechanical Rig** | `UNSUPPORTED`               | `CapabilityState.UNSUPPORTED`       | GSPro Open Connect v1 is purely a ballistic shot injection interface. No skeletal animation or avatar model deformation SDK exists.                |
| **Course State & Green Telemetry**    | `UNSUPPORTED`               | `CapabilityState.UNSUPPORTED`       | Simulator provides no reciprocal course-state telemetry (slope, pin position, bunker geometry, wind) back to the socket connector.                 |
| **Local Trajectory Return**           | `UNSUPPORTED`               | `CapabilityState.UNSUPPORTED`       | GSPro owns the proprietary physics and rendered visual trajectory; it does not stream back intermediate flight coordinates.                        |
| **Remote Aim Control**                | `UNSUPPORTED`               | `CapabilityState.UNSUPPORTED`       | Open Connect v1 cannot steer or command the target line; aiming must be performed by the operator in the simulator UI.                             |

---

## 5. Disconnect Handling & Recovery Playbook

### 5.1 Delivery Status Lifecycle

All outbound shots pass through the durable, append-only `ShotJournal`:

```
          +-----------------------+
          |        PENDING        | (Shot queued / socket write initiated)
          +-----------+-----------+
                      |
        +-------------+-------------+
        |                           |
+-------v-------+           +-------v-------+
|  ACKNOWLEDGED |           |   REJECTED    |
|   (Code 200)  |           | (Invalid data)|
+---------------+           +---------------+
        |
   (Socket drops mid-flight)
        |
+-------v-------+
|   AMBIGUOUS   | <--- CRITICAL STATE: Operator inspection required.
+---------------+      AUTOMATIC RETRY IS STRICTLY FORBIDDEN.
```

### 5.2 Operator Disconnect Playbook

When network or simulator interruption causes a transmission to fail before a code 200 acknowledgment is received:

1. **Verify AMBIGUOUS status in journal:**
   ```bash
   python -c "from src.shared.python.golf_simulator.journal import ShotJournal; j = ShotJournal(); print([(e.shot_id, e.status.value) for e in j.get_unresolved()])"
   ```
2. **Inspect Simulator Screen:**
   - Look at the GSPro projector / monitor.
   - **Case A (Ball Flew):** The shot was received and rendered by GSPro before the socket severed. **DO NOT RESUBMIT.** Mark the journal entry acknowledged/resolved.
   - **Case B (Ball Remained on Tee):** GSPro never received the shot. Once the socket connection is restored, re-arm the shot and submit.
3. **Preventing Race Conditions:**
   - Always verify the producer lock is active using `ProducerLockManager`.
   - Never run multiple independent simulator launcher scripts against the same bridge or socket.

### 5.3 Journal Retention & Pruning

To keep disk consumption minimal while preserving unresolved audit evidence:

```python
from src.shared.python.golf_simulator.journal import ShotJournal

journal = ShotJournal()
# Prune entries older than 24 hours, but NEVER prune AMBIGUOUS or PENDING entries:
journal.prune_retention(max_age_seconds=86400.0, keep_unresolved=True)
```

---

## 6. Live Opt-In Acceptance Qualification Suite

### 6.1 Test Suite Location & Structure

The live qualification suite is located at:
`tests/integration/golf_simulator/test_live_acceptance.py`

It validates the end-to-end acceptance runbook in 5 discrete steps:

1. `test_live_environment_and_version_receipt`: Audits system version, connector version, and verifies nonsecret receipts.
2. `test_live_deterministic_shot_qualification`: Transmits straight, left-hook, right-slice, chip, and putt shots, verifying 100% receipt acknowledgment and journal durability.
3. `test_live_unsupported_capability_declarations`: Formally verifies that avatar animation and course feedback are declared `UNSUPPORTED`.
4. `test_live_ambiguous_disconnect_recovery`: Injects transport drop mid-transmission, asserts `AMBIGUOUS` journal recording, verifies no duplicate submissions, and validates persistent state across reloads.
5. `test_live_single_producer_lock_prevents_collision`: Verifies lease locking and competing producer rejection.

### 6.2 Execution Instructions

**Default Behavior (Safe CI):**

```bash
python -m pytest tests/integration/golf_simulator/test_live_acceptance.py -v
# Output: 5 skipped in 0.25s (clean skip, no ports opened, no secrets needed)
```

**Live Qualification Run (Targeting Local Reference Peer or Live GSPro):**

```powershell
# For simulated loopback peer:
$env:GSPRO_LIVE_TEST = "1"
python -m pytest tests/integration/golf_simulator/test_live_acceptance.py -v

# For real licensed GSPro instance:
$env:GSPRO_LIVE_TEST = "1"
$env:GSPRO_HOST = "127.0.0.1"
$env:GSPRO_API_PORT = "921"
python -m pytest tests/integration/golf_simulator/test_live_acceptance.py -v
```
