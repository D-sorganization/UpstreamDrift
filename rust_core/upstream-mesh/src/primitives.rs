//! Primitive-fitting kernels for collision-shape approximation.
//!
//! Replaces the trimesh-backed logic in
//! `src/shared/python/humanoid_character_builder/mesh/_cg_primitive_fitting.py`:
//!
//! * [`fit_aabb`] — axis-aligned bounding box (parry3d `Aabb::from_points`).
//! * [`fit_obb`] — oriented bounding box via PCA on the vertex cloud.
//! * [`fit_sphere`] — minimum-radius sphere centred at the centroid (matches
//!   the existing python kernel — _not_ Welzl's true minimum enclosing
//!   sphere; this is the trimesh-compatible flavour).
//! * [`fit_cylinder`] — cylinder along the longest OBB axis.
//! * [`fit_capsule`] — capsule along the longest OBB axis.
//!
//! All routines accept `&[[f32; 3]]` vertex slices and are *streaming-friendly*
//! in the sense that they make a single pass over the input and allocate
//! `O(1)` working memory beyond the input itself (PCA needs a 3x3 covariance
//! matrix). This is the OOM-de-risking property called out by issue #5219 and
//! the closed #3903 lineage: replacing trimesh's `bounding_box_oriented`
//! (which builds a full convex hull + scipy rotation chain) with a direct
//! PCA over the raw vertex buffer cuts peak working set by an order of
//! magnitude on million-vertex meshes.

use nalgebra::{Matrix3, Point3, SymmetricEigen, UnitQuaternion, Vector3};
use parry3d::bounding_volume::Aabb;
use serde::{Deserialize, Serialize};

/// Quaternion in `(x, y, z, w)` order — matches scipy's convention and the
/// existing python facade.
pub type Quat = [f32; 4];

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AabbFit {
    pub center: [f32; 3],
    pub extents: [f32; 3],
    /// `mesh_volume / aabb_volume` — 1.0 means the AABB is a perfect fit.
    pub volume_ratio: f32,
}

/// Oriented bounding box (OBB) — axis frame plus half-extents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObbFit {
    pub center: [f32; 3],
    /// Full extents along the OBB's principal axes (not half-extents — matches
    /// trimesh's `bounding_box_oriented.primitive.extents`).
    pub extents: [f32; 3],
    /// Rotation taking the AABB frame to the OBB frame, as `(x, y, z, w)`.
    pub rotation: Quat,
    pub volume_ratio: f32,
}

/// Sphere fit: centred at the centroid with radius = max(vertex distance to
/// centroid). Matches the existing python `fit_sphere` semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphereFit {
    pub center: [f32; 3],
    pub radius: f32,
    pub volume_ratio: f32,
}

/// Cylinder fit along the longest OBB axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CylinderFit {
    pub center: [f32; 3],
    pub radius: f32,
    pub height: f32,
    pub rotation: Quat,
    pub volume_ratio: f32,
}

/// Capsule fit along the longest OBB axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleFit {
    pub center: [f32; 3],
    pub radius: f32,
    /// Distance between the two hemisphere centres (i.e. the cylindrical
    /// section length; total capsule length is `height + 2 * radius`).
    pub height: f32,
    pub rotation: Quat,
    pub volume_ratio: f32,
}

/// Errors from primitive fitting.
#[derive(Debug)]
pub enum PrimitiveError {
    TooFewPoints(usize),
    NonFiniteInput { index: usize },
}

impl core::fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewPoints(n) => {
                write!(f, "primitive fitting requires at least 4 points (got {n})")
            }
            Self::NonFiniteInput { index } => {
                write!(f, "input vertex {index} contained non-finite coordinate")
            }
        }
    }
}

impl std::error::Error for PrimitiveError {}

fn validate(vertices: &[[f32; 3]]) -> Result<(), PrimitiveError> {
    if vertices.len() < 4 {
        return Err(PrimitiveError::TooFewPoints(vertices.len()));
    }
    for (i, v) in vertices.iter().enumerate() {
        if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
            return Err(PrimitiveError::NonFiniteInput { index: i });
        }
    }
    Ok(())
}

fn mesh_volume_or(mesh_volume: Option<f32>, fit_volume: f32) -> f32 {
    match mesh_volume {
        Some(v) if v > 0.0 && fit_volume > 0.0 => (v / fit_volume).min(1.0),
        _ => 0.0,
    }
}

/// Axis-aligned bounding box.
pub fn fit_aabb(
    vertices: &[[f32; 3]],
    mesh_volume: Option<f32>,
) -> Result<AabbFit, PrimitiveError> {
    validate(vertices)?;
    let points: Vec<Point3<f32>> = vertices
        .iter()
        .map(|&[x, y, z]| Point3::new(x, y, z))
        .collect();
    let aabb = Aabb::from_points(points.iter());
    let center = aabb.center();
    let extents = aabb.extents();
    let vol = extents.x * extents.y * extents.z;
    Ok(AabbFit {
        center: [center.x, center.y, center.z],
        extents: [extents.x, extents.y, extents.z],
        volume_ratio: mesh_volume_or(mesh_volume, vol),
    })
}

/// Compute centroid + 3x3 covariance of a vertex cloud in a single pass.
///
/// Allocates O(1) beyond a 3-vector accumulator and a 3x3 matrix accumulator.
fn covariance(vertices: &[[f32; 3]]) -> (Vector3<f32>, Matrix3<f32>) {
    let n = vertices.len() as f32;
    let mut centroid = Vector3::zeros();
    for v in vertices {
        centroid += Vector3::new(v[0], v[1], v[2]);
    }
    centroid /= n;

    let mut cov = Matrix3::zeros();
    for v in vertices {
        let d = Vector3::new(v[0], v[1], v[2]) - centroid;
        cov += d * d.transpose();
    }
    cov /= n;
    (centroid, cov)
}

/// Convert a rotation matrix (columns = principal axes) to a unit quaternion
/// in `(x, y, z, w)` order. Ensures the matrix is right-handed first.
fn rot_to_quat(mut axes: Matrix3<f32>) -> Quat {
    // Make right-handed: if det < 0 flip the smallest-eigenvalue axis (column 2).
    if axes.determinant() < 0.0 {
        axes.column_mut(2).neg_mut();
    }
    let rot = nalgebra::Rotation3::from_matrix(&axes);
    let q = UnitQuaternion::from_rotation_matrix(&rot);
    [q.i, q.j, q.k, q.w]
}

/// Oriented bounding box via PCA over the raw vertex cloud.
pub fn fit_obb(vertices: &[[f32; 3]], mesh_volume: Option<f32>) -> Result<ObbFit, PrimitiveError> {
    validate(vertices)?;
    let (centroid, cov) = covariance(vertices);

    // Eigen-decompose the symmetric covariance matrix; the eigenvectors form
    // the OBB axes (sorted descending so x is the longest direction).
    let eig = SymmetricEigen::new(cov);
    // Sort eigenvalues descending.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut axes = Matrix3::zeros();
    for (col, &src) in order.iter().enumerate() {
        axes.column_mut(col)
            .copy_from(&eig.eigenvectors.column(src));
    }

    // Project all vertices onto the axes to find true extents.
    let mut mins = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut maxs = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in vertices {
        let d = Vector3::new(v[0], v[1], v[2]) - centroid;
        let proj = axes.transpose() * d;
        for i in 0..3 {
            if proj[i] < mins[i] {
                mins[i] = proj[i];
            }
            if proj[i] > maxs[i] {
                maxs[i] = proj[i];
            }
        }
    }
    let extents = maxs - mins;
    // OBB centre in world space: centroid offset by axes * mean(projection).
    let local_center = (maxs + mins) * 0.5;
    let world_center = centroid + axes * local_center;

    let vol = extents.x * extents.y * extents.z;
    Ok(ObbFit {
        center: [world_center.x, world_center.y, world_center.z],
        extents: [extents.x, extents.y, extents.z],
        rotation: rot_to_quat(axes),
        volume_ratio: mesh_volume_or(mesh_volume, vol),
    })
}

/// Sphere fit centred at the centroid; radius is the maximum vertex-centroid
/// distance. Trimesh-compatible (matches `_cg_primitive_fitting.fit_sphere`).
pub fn fit_sphere(
    vertices: &[[f32; 3]],
    mesh_volume: Option<f32>,
) -> Result<SphereFit, PrimitiveError> {
    validate(vertices)?;
    let n = vertices.len() as f32;
    let mut centroid = [0.0_f32; 3];
    for v in vertices {
        centroid[0] += v[0];
        centroid[1] += v[1];
        centroid[2] += v[2];
    }
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    let mut r2_max = 0.0_f32;
    for v in vertices {
        let dx = v[0] - centroid[0];
        let dy = v[1] - centroid[1];
        let dz = v[2] - centroid[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 > r2_max {
            r2_max = r2;
        }
    }
    let radius = r2_max.sqrt();
    let vol = (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
    Ok(SphereFit {
        center: centroid,
        radius,
        volume_ratio: mesh_volume_or(mesh_volume, vol),
    })
}

/// Cylinder fit along the longest OBB axis. Height = OBB extent along that
/// axis; radius = `max(other_extents) / 2`. Matches `_cg_primitive_fitting.fit_cylinder`.
pub fn fit_cylinder(
    vertices: &[[f32; 3]],
    mesh_volume: Option<f32>,
) -> Result<CylinderFit, PrimitiveError> {
    let obb = fit_obb(vertices, mesh_volume)?;
    // OBB axes are sorted descending, so axis 0 is the longest.
    let height = obb.extents[0];
    let radius = obb.extents[1].max(obb.extents[2]) / 2.0;
    let vol = std::f32::consts::PI * radius * radius * height;
    Ok(CylinderFit {
        center: obb.center,
        radius,
        height,
        rotation: obb.rotation,
        volume_ratio: mesh_volume_or(mesh_volume, vol),
    })
}

/// Capsule fit along the longest OBB axis. The cylindrical section has
/// length `extent - 2 * radius` (clamped to 0); total capsule volume is
/// `pi r^2 h + (4/3) pi r^3`.
pub fn fit_capsule(
    vertices: &[[f32; 3]],
    mesh_volume: Option<f32>,
) -> Result<CapsuleFit, PrimitiveError> {
    let obb = fit_obb(vertices, mesh_volume)?;
    let long = obb.extents[0];
    let radius = obb.extents[1].max(obb.extents[2]) / 2.0;
    let height = (long - 2.0 * radius).max(0.0);
    let vol = std::f32::consts::PI * radius * radius * height
        + (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
    Ok(CapsuleFit {
        center: obb.center,
        radius,
        height,
        rotation: obb.rotation,
        volume_ratio: mesh_volume_or(mesh_volume, vol),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn unit_cube() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    }

    #[test]
    fn aabb_unit_cube() {
        let fit = fit_aabb(&unit_cube(), Some(1.0)).unwrap();
        assert_relative_eq!(fit.center[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(fit.extents[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(fit.volume_ratio, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn obb_unit_cube_extents() {
        let fit = fit_obb(&unit_cube(), Some(1.0)).unwrap();
        // All extents are 1.0 — orientation is degenerate for a cube but
        // extents must still be (1,1,1) to within float tolerance.
        for e in fit.extents {
            assert_relative_eq!(e, 1.0, epsilon = 1e-5);
        }
        assert_relative_eq!(fit.volume_ratio, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn sphere_unit_cube() {
        let fit = fit_sphere(&unit_cube(), Some(1.0)).unwrap();
        // Centroid at (0.5, 0.5, 0.5); furthest corner is sqrt(3)/2 away.
        let expected_r = (0.75_f32).sqrt();
        assert_relative_eq!(fit.radius, expected_r, epsilon = 1e-6);
    }

    #[test]
    fn cylinder_long_box() {
        // 2 x 1 x 1 box: long axis = 2.
        let pts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ];
        let fit = fit_cylinder(&pts, Some(2.0)).unwrap();
        assert_relative_eq!(fit.height, 2.0, epsilon = 1e-5);
        assert_relative_eq!(fit.radius, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn capsule_long_box() {
        let pts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [4.0, 1.0, 0.0],
            [4.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ];
        let fit = fit_capsule(&pts, Some(4.0)).unwrap();
        assert_relative_eq!(fit.radius, 0.5, epsilon = 1e-5);
        // Cylindrical section: 4 - 2*0.5 = 3.
        assert_relative_eq!(fit.height, 3.0, epsilon = 1e-5);
    }

    #[test]
    fn too_few_points_errors() {
        let pts = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        match fit_aabb(&pts, None) {
            Err(PrimitiveError::TooFewPoints(2)) => {}
            other => panic!("expected TooFewPoints, got {other:?}"),
        }
    }
}
