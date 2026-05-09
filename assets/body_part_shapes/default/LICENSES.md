# Default body-part mesh licenses

All meshes in this directory are **procedurally generated** by
[`_generate.py`](./_generate.py) using `trimesh.creation` primitives
(icosphere, cylinder, box). They contain no captured anatomical data,
no vendor-supplied geometry, and no person-identifying content.

| File | License | Source |
| --- | --- | --- |
| `head.stl` | CC0-1.0 | procedural-low-poly (icosphere, scaled) |
| `torso.stl` | CC0-1.0 | procedural-low-poly (box) |
| `upper_arm.stl` | CC0-1.0 | procedural-low-poly (cylinder) |
| `forearm.stl` | CC0-1.0 | procedural-low-poly (cylinder) |
| `hand.stl` | CC0-1.0 | procedural-low-poly (icosphere, scaled) |
| `thigh.stl` | CC0-1.0 | procedural-low-poly (cylinder) |
| `shin.stl` | CC0-1.0 | procedural-low-poly (cylinder) |
| `foot.stl` | CC0-1.0 | procedural-low-poly (box) |

## CC0-1.0

To the extent possible under law, the contributors to this repository
have waived all copyright and related or neighboring rights to these
mesh files under the [Creative Commons CC0 1.0 Universal Public Domain
Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

## Reproducibility

`_generate.py` is deterministic on a given `trimesh` version
(`numpy.random.default_rng(42)`). Re-running it produces byte-identical
STL output on the same platform, allowing the meshes to be regenerated
from source rather than treated as opaque binary blobs.
