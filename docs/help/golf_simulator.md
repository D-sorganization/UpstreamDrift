---
title: Golf Simulator
tile_id: golf_simulator
status: active
---

# Golf Simulator

## Purpose

The Golf Simulator Console integrates ball flight projection, terrain interaction, and shot tracing into a unified interactive desktop console. It provides golfers and biomechanists with immediate visual feedback on full shots, trajectory arc, carry and roll distances, and landing dispersion.

## Inputs

The simulator receives ball launch conditions and environmental parameters:

| Input | Units | Description |
| --- | --- | --- |
| Ball Speed | m/s (or mph) | Initial velocity of the golf ball at clubface separation |
| Launch Angle | deg | Initial vertical launch angle relative to the ground plane |
| Launch Direction | deg | Horizontal azimuth angle relative to the target line |
| Backspin | rpm | Spin rate around the horizontal transverse axis |
| Sidespin | rpm | Spin rate around the vertical axis determining fade/draw |
| Wind Speed | m/s | Ambient horizontal wind speed |
| Wind Direction | deg | Direction of wind relative to target azimuth |

## Outputs

Computed ballistic and rollout quantities:

| Output | Units | Description |
| --- | --- | --- |
| Carry Distance | m | Total horizontal distance travelled before first ground contact |
| Total Distance | m | Total distance including bounce and rollout |
| Apex Height | m | Maximum vertical height attained during flight |
| Flight Time | s | Duration from impact to initial ground touch |
| Offline Dispersion | m | Lateral distance from target line at rest |

## Method

Trajectories are computed by integrating aerodynamic lift and drag under standard atmospheric conditions (temperature 20 °C, pressure 1013.25 hPa) using the unified aerodynamic model in `src/shared/python/physics/ball_enhanced_simulator.py`. Spin decay is governed by empirical rotational damping, and post-impact bounce is modeled using coefficient of restitution and tangential ground friction.

## Limitations

Ground interaction assumes uniform fairway or green firmness. Sub-surface soil mechanics and aerodynamic effects of localized gusts or turbulence are omitted.

## See Also

- [Ball Flight Simulator](ball_flight_simulator.md)
- [Shot Tracer](shot_tracer.md)
- [Putting Green](putting_green.md)
