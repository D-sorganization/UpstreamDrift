# Driver Head Mesh

`gs3dx_driver_head.stl` is the club head drawn on `GS3DX_Human`'s club
(`gs3dx_build_human`, a massless File Solid). It changes nothing in the
dynamics: the club head's mass and inertia stay those of `GS3DX_Neck`.

## Provenance

Generated from the Tools repository (MIT), package `rate_of_closure`, at
commit `d84796d97` (2026-09-26). It is a generic parametric shape: no brand
geometry is reproduced.

```bash
cd Tools
PYTHONPATH=src python -c "from rate_of_closure.club.library import get_club; from rate_of_closure.club.stl_export import write_clubhead_stl_atomic; write_clubhead_stl_atomic(get_club('Driver 10.5°'), 'gs3dx_driver_head.stl')"
```

## Geometry

| Property                  | Value                                                     |
| ------------------------- | --------------------------------------------------------- |
| Format                    | binary STL, 1,792 triangles, closed and outward-wound     |
| Units                     | mm                                                        |
| Frame                     | x toward the target, y up, z toward the toe (heel at −z)  |
| Size                      | 115 mm front to back, 61 mm tall, 124 mm heel to toe      |
| Enclosed volume           | 570 cm³                                                   |
| Loft                      | 10.5°, leant about the leading edge (built into the mesh) |
| Face                      | 0.30 m bulge, 0.28 m roll                                 |
| Hosel point (m)           | (0.019430, 0.029029, −0.052): heel-crown transition       |
| Face centre (m)           | (0.049897, −0.000469, 0)                                  |
| Face normal at the centre | (0.983255, 0.182236, 0): cos and sin of the loft          |

The hosel point, face centre and normal come from
`rate_of_closure.club.head_profiles.hosel_point`, the lofted first section
and `parametric_head.face_normal_at_offset(spec, 0, 0)`. They are the
defaults of `gs3dx_build_human`'s `club_hosel`, `club_face_centre` and
`club_face_normal` options; regenerate them with the mesh if the spec
changes.
