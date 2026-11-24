# Forces and Torques for a New Best Match

A kinematic best match has joint angles but no kinetics. Run the kinetic match
whenever a new best match is produced, so its joint torques, ground forces and
hand-on-club force are available for renders, the ZTCF/ZVCF split and club
fitting (#15685).

## The Command

```bash
MUJOCO_GL=egl python3 -m src.shared.python.motion_matching.kinetic_match \
    <kinematic_bundle.npz> <kinetic_bundle.npz>
```

The output bundle is the input with its efforts filled in. Next to it,
`<kinetic_bundle>.kinetic_receipt.json` records the settings, the peaks, the
root-residual window and a plausibility verdict. Settings live in
`src/config/kinetic_match.json`; pass `--settings <file>` to try others. Keep
private bundles and their outputs outside the repository.

## What It Does

1. Maps the bundle's coordinates to the engine's native order by name. The
   MuJoCo adapter orders coordinates differently from the bundle, and feeding
   bundle order scrambles every result.
2. Low-passes the reference (`cutoff_hz`) and differentiates it.
3. Projects the velocities and accelerations onto the hand-on-club closure,
   mass-weighted, so that the relative hand-club velocity and acceleration are
   zero. Matched angles do not satisfy the weld exactly; without this step the
   closure force absorbs the inconsistency.
4. Allocates joint torques, ground forces and the closure force by the conic
   allocator: minimum effort, with `regularisation_grip` penalising the
   internal loop force, and any root force it cannot close reported as the
   root residual.
5. Writes the torques back as efforts, in the bundle's order.

## One Solution, Not the Solution

The arms and club form a closed loop, so infinitely many force sets produce
the same motion. The result is one solution in that null space, chosen by the
objective and its weights in the settings. It is not unique and it is not
measured. State the objective with any figure that shows these forces.

## Reading the Receipt

| Field | Meaning |
|---|---|
| `verdict` | `plausible` when every peak is within the configured limits |
| `reasons` | Each limit exceeded, with its value |
| `root_residual_window_s` | Time span from the start where the root residual stays under its limit |
| `peaks` | Peak grip force, peak joint torque and its coordinate, peak root residual |
| `max_equilibrium_residual` | Consistency of the allocation; should be near zero |

Render forces only inside the root-residual window, and only from a plausible
result, unless the figure says otherwise.

## Why Not Closed-Loop Tracking

Computed-torque tracking of the matched angles in forward dynamics was tried
first (2026-11-23): full swing, downswing only, root regulation, a pelvis
residual and plain PD on the feet. Every variant diverged or drifted 10 to 30
degrees by impact. Inverse dynamics with allocation needs no integration and
reports where the motion cannot be balanced.

## Known Limits

On the current best driver match the backswing kinetics are physical (grip
force under 100 N, joint torques under 100 N m, zero root residual to 1.11 s).
From 1.11 s, when the feet start leaving the ground, the allocation reports
5 to 14 kN at the grip and 1 to 4 kN m at the lead shoulder, although the club
itself needs only about 400 N. The receipt marks this implausible. The
ground-contact side of the allocation at heel lift is under review on #15685.
