# Segment Kinetic-Energy Globes

The MuJoCo full-body golfer can overlay semi-transparent spheres at each
body centre of mass, scaled and coloured by that segment's instantaneous
kinetic energy (translational plus rotational, in joules). A brighter shell
marks moments when a segment is near its swing peak level — this is a visual
emphasis on magnitude, not a measurement of energy *flow* between segments.

Enable **Segment Kinetic-Energy Globes** in the MuJoCo humanoid golf
visualization tab (alongside joint force vectors, segment force colours and
ground-reaction arrows when those overlays are already active).

## API

Engine-agnostic contracts live in `src/shared/python/segment_kinetic_energy/`.
The MuJoCo implementation is
`mujoco_humanoid_golf.segment_kinetic_energy_provider.MujocoSegmentKineticEnergyProvider`.

```python
from src.shared.python.segment_kinetic_energy import SegmentKineticEnergySeries

series = SegmentKineticEnergySeries.from_trajectory(
    provider, times, qpos_traj, qvel_traj
)
peaks = series.peak_totals()
```

## Proxy, Not Energy Transfer

Peak ordering of segment kinetic energy (or of angular speed) is a **proxy**
for coordination timing. It does not establish inter-segment energy flow.
Robertson–Winter interface power is tracked separately (see issue #15390).

Read `docs/research/proximal_distal_energy_transfer/chapters/_ch02_mechanics.qmd`
(`@sec-energyflow`, `@sec-rw`, and "Kinematic Sequence as a Proxy — And Its
Failure Modes") for the scientific context.