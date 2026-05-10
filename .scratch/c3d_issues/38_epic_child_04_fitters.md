# feat(body-part-viz): fitters — BetweenTwoMarkers, ClusterKabsch, ProcrustesAnisotropic

Depends on contracts (#1) and primitives (#2).

## Why

Three orthogonal fitter strategies cover the realistic mocap → shape attachments. Each ships as an independent module so adding a fourth (e.g. body-cluster-with-curvature-fitter) doesn't touch the existing three.

## Fitters

### `BetweenTwoMarkersFitter` (in `fitters/between_two.py`)

- For `BindingKind.BETWEEN_TWO` only.
- Per frame:
  - centroid = midpoint(a, b)
  - axis = (b - a) / ‖b - a‖
  - rotation: align shape's local x-axis to `axis`; pick a stable up-vector via Gram-Schmidt against world Z (or world Y if axis is near Z).
  - scale = (‖b - a‖ / rest_length, 1.0, 1.0) — anisotropic in length only.
- valid_mask: NaN-safe; frame invalid iff either marker is NaN.

### `ClusterKabschFitter` (in `fitters/cluster_kabsch.py`)

- For `BindingKind.CLUSTER` (≥ 3 markers).
- Per frame:
  - Compute centroid of cluster markers.
  - Subtract to get centred cluster.
  - Solve Kabsch (SVD) for rigid rotation between cluster and rest cluster.
  - scale = (1, 1, 1) — pure rigid.
- Optional: detect uniform scale via centred-norm ratio if `enable_scale=True`.

### `ProcrustesAnisotropicFitter` (in `fitters/procrustes_anisotropic.py`)

- For `BindingKind.CLUSTER` (≥ 4 markers preferred).
- Per frame:
  - Centroid + Kabsch rotation (as above).
  - Solve anisotropic scale via `scipy.linalg.lstsq` on the centred, rotated cluster vs rest cluster.
- Documented as the most-flexible but least-stable fitter; recommended only when ≥ 4 markers and the user wants anisotropic stretch.

### Helper utilities (`fitters/_kabsch.py`)

- `kabsch_rotation(P, Q) -> rotation_matrix`: pure-NumPy SVD; reflection guard.
- Reused by Cluster + Procrustes fitters.

## Tests

`tests/unit/body_part_viz/fitters/test_between_two.py`:
- 100-frame straight-line trajectory; assert centroids are midpoints.
- Rotated trajectory; assert rotation matrix is orthogonal, det == +1.
- One marker NaN at frames 50–60; assert valid_mask correct.

`tests/unit/body_part_viz/fitters/test_cluster_kabsch.py`:
- 4-marker cluster rotated by known angle; recovered rotation within 1e-9.
- Reflection guard: cluster reflected → rotation has det == +1 (no reflection).

`tests/unit/body_part_viz/fitters/test_procrustes_anisotropic.py`:
- Cluster scaled anisotropically (1.0, 2.0, 0.5); recovered scale within 1e-6.
- DbC: < 4 markers raises ValueError or logs WARNING and falls back to Kabsch.

`tests/unit/body_part_viz/fitters/test_kabsch_helper.py`:
- Pure-NumPy SVD round-trip, including degenerate cases (collinear cluster).

## Acceptance criteria

- [ ] Three fitters implement `ShapeFitter`.
- [ ] All fitters NaN-safe (valid_mask correctly populated).
- [ ] Reflection guard verified.
- [ ] DbC: invalid binding kind raises `TypeError`.
- [ ] Helper module decoupled from fitters (re-usable).
- [ ] ≥ 90% line coverage.

## Files touched

- New: `src/shared/python/body_part_viz/fitters/{between_two,cluster_kabsch,procrustes_anisotropic}.py`
- New: `src/shared/python/body_part_viz/fitters/_kabsch.py`
- Edit: `src/shared/python/body_part_viz/fitters/__init__.py`
- New: `tests/unit/body_part_viz/fitters/test_*.py`
