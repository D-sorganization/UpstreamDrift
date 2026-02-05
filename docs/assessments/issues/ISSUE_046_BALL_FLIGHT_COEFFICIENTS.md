---
title: "Legal Risk: Uncited Ball Flight Aerodynamic Coefficients"
labels: ["legal", "patent-risk", "medium-priority"]
assignee: "physics-team"
status: "open"
---

# Issue Description

The `BallProperties` class in `src/shared/python/ball_flight_physics.py` contains hardcoded aerodynamic coefficients without citation.

## Technical Detail
```python
@dataclass(frozen=True)
class BallProperties:
    # ...
    cd0: float = 0.21
    cd1: float = 0.05
    cd2: float = 0.02
    cl0: float = 0.00
    cl1: float = 0.38
    cl2: float = 0.08
```

## Legal Context
These specific values likely correspond to a specific experimental dataset (e.g., Bearman & Harvey, Smits & Smith) or potentially a competitor's proprietary model (e.g., TrackMan/FlightScope defaults). Using them without citation creates an ambiguity about their provenance. If they are copied from a closed-source SDK or leaked document, it constitutes copyright infringement or trade secret misappropriation.

## Required Actions
1.  **Identify Source**: Determine where these numbers came from.
2.  **Add Citation**: If from public research (e.g., "Aerodynamics of Golf Balls", Smits 1994), add the full citation in the docstring.
3.  **Externalize**: Move these default values to a configuration file (`ball_profiles.json`) to emphasize that they are just *example* values, not a hardcoded "truth" claimed by our software.
