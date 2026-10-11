---
title: Impact Parameters
tile_id: impact_parameters
status: active
---

# Impact Parameters

## Purpose

Show launch-monitor-style delivery numbers for any engine run or matched swing: clubhead speed, attack angle, club path, face angle, face-to-path, dynamic loft, spin loft, low point and smash factor, all relative to an explicit target line. The same card is available in the desktop dock, the web Analysis Tools page and the matched-swing results view.

## Inputs

| Input | Units | Description |
| --- | --- | --- |
| Clubhead series | m, m/s | World-frame face-centre position and velocity with face normal, from the engine's own forward kinematics |
| Target heading | deg | Target line direction, counter-clockwise from the default -Y axis |
| Handedness | right or left | Left-handed mirrors lateral signs only |
| Units | mph or m/s | Display unit for clubhead speed |

## Outputs

| Output | Units | Description |
| --- | --- | --- |
| Speed, attack angle, club path, face angle, face to path | mph or m/s, deg | Positive path is in-to-out, positive face is open, negative attack angle is descending |
| Dynamic loft, spin loft | deg | Face elevation and the 3-D angle between velocity and face normal |
| Low point ahead of ball | m | Positive toward the target |
| Impact location, smash factor | mm, ratio | Shown only when face geometry, ball contact and an impact-model ball speed exist |
| D-plane and path views | diagram | Top view (path, face) and side view (attack angle, loft) |

## Method

The engine adapters (`impact_parameters.adapters`) transport the club body pose to the face centre; `extract_impact_parameters` evaluates the angles at the last pre-contact sample in the target frame. The API route is `GET /api/analysis/impact-parameters?run_id=...&target_dir=x,y`.

## Limitations

A quantity that cannot be computed (face roll unobservable in mocap-matched swings, no ball model, a run with no clubhead series, speed below 1 m/s) is shown as "unavailable" with its reason, never as zero. The Impact Explorer link passes the available values as a query string; whether the Explorer consumes them is owned by the Tools epic (#9546).

## See Also

- [Matched Swing Browser](matched_swing_browser.md)
- [Analysis Tools](analysis_tools.md)
