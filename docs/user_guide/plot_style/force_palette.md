# Force Kind Palette

The force and torque overlay pipeline uses a categorical color palette registered
in `src.shared.python.plot_style.force_palette.FORCE_KIND_PALETTE` to render 3D
force arrows and torque arcs.

## Categorical Swatches

Each wrench category is assigned a distinct color from the colorblind-safe Okabe-Ito
palette:

| Wrench Kind      | Hex Code  | Color Name     | Intended Usage                                       |
| ---------------- | --------- | -------------- | ---------------------------------------------------- |
| `joint_actuator` | `#E69F00` | Orange         | Motor and actuator joint torques                     |
| `joint_reaction` | `#CC79A7` | Reddish Purple | Inter-segment constraint reaction forces and moments |
| `contact`        | `#009E73` | Bluish Green   | Foot-ground contact and strike forces                |
| `grip`           | `#56B4E9` | Sky Blue       | Lead and trail golfer grip forces                    |
| `external`       | `#000000` | Black / White  | Wind, aerodynamics, or external disturbance forces   |
| `gravity`        | `#999999` | Gray           | Segment gravitational body forces                    |
| `muscle`         | `#D55E00` | Vermilion      | Muscle-tendon unit line-of-action forces             |

## Reserved Colors

Pure blue (`#0000ff`) and pure red (`#ff0000`) are strictly reserved for axial
segment load visualization (positive tension and negative compression). They must
never be assigned to wrench categories.
