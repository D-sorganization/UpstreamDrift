# Patent & Legal Alert: 2026-02-05

**Reviewer:** Jules (Patent Reviewer Agent)
**Status:** NEW FINDINGS

## 1. New Risk Identified: Ball Flight Aerodynamic Coefficients

### Context
During the code audit of `src/shared/python/ball_flight_physics.py`, specific hardcoded values were identified in the `BallProperties` class:

```python
@dataclass(frozen=True)
class BallProperties:
    mass: float = 0.0459
    diameter: float = 0.04267
    cd0: float = 0.21
    cd1: float = 0.05
    cd2: float = 0.02
    cl0: float = 0.00
    cl1: float = 0.38
    cl2: float = 0.08
```

### Risk Assessment
*   **Risk Level:** MEDIUM
*   **Potential Issue:** These specific coefficients (e.g., `cd0=0.21`, `cl1=0.38`) likely correspond to a specific golf ball model (e.g., Pro V1) measured in a specific study, or worse, copied from a competitor's SDK or proprietary whitepaper.
*   **Legal Implication:** Using proprietary data without authorization (or "Fair Use" justification via academic citation) can be grounds for IP claims, especially if we claim "research-grade accuracy" based on these numbers.

### Recommended Actions
1.  **Immediate**: Add a comment citing the source of these numbers (e.g., "Coefficients derived from Smits & Smith (1994)").
2.  **Long-term**: Externalize these values to a JSON configuration file so users can provide their own ball profiles, distancing our codebase from specific hardcoded values.

## 2. Verification of Past Remediation

### Swing DNA (Mizuno)
*   **Status**: **VERIFIED SAFE**
*   **Finding**: Code audit confirmed that the UI no longer uses the term "Swing DNA". The visualization is now labeled "Swing Profile (Radar)".
*   **Action**: No further action needed. Continued monitoring required.
