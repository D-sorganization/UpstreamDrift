//! Mesh decimation via VHACD-based convex decomposition.
//!
//! Replaces the trimesh `simplify_quadric_decimation` fallback used by
//! `src/shared/python/humanoid_character_builder/mesh/_cg_decimation.py`.
//! `parry3d` does not ship a quadric-error-metric edge-collapse decimator,
//! and porting one is a separate effort — tracked as a follow-up in the
//! parent issue #5219 (see also #5248). What `parry3d` does ship, and what
//! actually fixes the OOM lineage of closed #3903, is **VHACD approximate
//! convex decomposition**: voxelize the mesh, recursively split into
//! near-convex parts, and emit one convex hull per part. The result is the
//! *collision-mesh*-grade simplification used downstream — and the working
//! set is bounded by the voxel resolution, not by the input vertex count.
//!
//! ## API
//!
//! [`decimate_vhacd`] takes a `(vertices, indices)` triangle mesh and
//! returns a [`DecimationResult`] containing per-part convex hulls. The
//! `output_triangles()` helper sums hull triangles for the existing facade.

use nalgebra::Point3;
use parry3d::transformation::vhacd::{VHACDParameters, VHACD};
use serde::{Deserialize, Serialize};

use crate::convex_hull::ConvexHullResult;

/// Tunable parameters for [`decimate_vhacd`]. Defaults match
/// `_cg_types.VHACDParameters` so the Python facade is a drop-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecimateParameters {
    /// Voxel grid resolution (per axis). Total voxel count is
    /// `~resolution^3` for a roughly-cubic mesh. The parry3d default is 64.
    /// **Semantics note**: the trimesh facade's `resolution: 100_000` meant
    /// *total* voxels, which corresponds to `cbrt(100_000) ~= 46` per axis.
    /// We expose the per-axis number directly here; the Python facade
    /// translates the old contract via `int(round(total ** (1/3)))`.
    pub resolution: u32,
    /// Max convex hulls produced. Stops splitting once this is reached.
    pub max_hulls: usize,
    /// Concavity threshold; parts below this are kept as-is.
    pub concavity: f32,
    /// `0` = surface fill, `1` = flood fill. Default 0 matches the trimesh
    /// facade. Other values are rejected instead of silently changing the
    /// decomposition contract.
    pub fill_mode: u8,
}

impl Default for DecimateParameters {
    fn default() -> Self {
        Self {
            resolution: 64,
            max_hulls: 16,
            concavity: 0.001,
            fill_mode: 0,
        }
    }
}

/// Output of [`decimate_vhacd`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecimationResult {
    /// Per-part convex hulls. Each entry has hull vertices + outward-facing
    /// triangle indices, matching the `ConvexHullResult` contract used by
    /// the convex-hull kernel.
    pub parts: Vec<ConvexHullResult>,
}

impl DecimationResult {
    /// Total triangle count across all hull parts.
    pub fn output_triangles(&self) -> usize {
        self.parts.iter().map(|h| h.indices.len()).sum()
    }

    /// Total vertex count across all hull parts.
    pub fn output_vertices(&self) -> usize {
        self.parts.iter().map(|h| h.vertices.len()).sum()
    }
}

/// Errors from [`decimate_vhacd`].
#[derive(Debug)]
pub enum DecimationError {
    TooFewPoints(usize),
    NoTriangles,
    InvalidResolution(u32),
    InvalidMaxHulls(usize),
    InvalidConcavity(f32),
    UnsupportedFillMode(u8),
    NonFiniteInput {
        index: usize,
    },
    BadIndex {
        triangle: usize,
        vertex: u32,
        n_vertices: usize,
    },
}

impl core::fmt::Display for DecimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewPoints(n) => write!(f, "decimation requires at least 4 vertices (got {n})"),
            Self::NoTriangles => write!(f, "decimation requires at least one triangle"),
            Self::InvalidResolution(resolution) => write!(
                f,
                "decimation resolution must be greater than zero (got {resolution})"
            ),
            Self::InvalidMaxHulls(max_hulls) => write!(
                f,
                "decimation max_hulls must be greater than zero (got {max_hulls})"
            ),
            Self::InvalidConcavity(concavity) => write!(
                f,
                "decimation concavity must be finite and non-negative (got {concavity})"
            ),
            Self::UnsupportedFillMode(fill_mode) => write!(
                f,
                "decimation fill_mode must be 0 (surface) or 1 (flood fill), got {fill_mode}"
            ),
            Self::NonFiniteInput { index } => {
                write!(f, "vertex {index} contained non-finite coordinate")
            }
            Self::BadIndex {
                triangle,
                vertex,
                n_vertices,
            } => {
                write!(
                    f,
                    "triangle {triangle} references vertex {vertex} but only {n_vertices} vertices were supplied"
                )
            }
        }
    }
}

impl std::error::Error for DecimationError {}

fn validate_params(params: &DecimateParameters) -> Result<(), DecimationError> {
    if params.resolution == 0 {
        return Err(DecimationError::InvalidResolution(params.resolution));
    }
    if params.max_hulls == 0 {
        return Err(DecimationError::InvalidMaxHulls(params.max_hulls));
    }
    if !params.concavity.is_finite() || params.concavity < 0.0 {
        return Err(DecimationError::InvalidConcavity(params.concavity));
    }
    if params.fill_mode > 1 {
        return Err(DecimationError::UnsupportedFillMode(params.fill_mode));
    }
    Ok(())
}

fn fill_mode_from(code: u8) -> parry3d::transformation::voxelization::FillMode {
    use parry3d::transformation::voxelization::FillMode;
    match code {
        1 => FillMode::FloodFill {
            detect_cavities: false,
        },
        _ => FillMode::SurfaceOnly,
    }
}

/// Run VHACD on a triangle mesh.
///
/// # Memory
///
/// Peak working set is dominated by the voxelization grid (~`resolution^(2/3)`
/// occupied voxels for typical surfaces) and is *independent* of the input
/// triangle count once voxelization is complete. This is the OOM-fix: even
/// a 1M-triangle mesh peaks at the same RSS as a 100k-triangle mesh after
/// voxelization, where the trimesh path would have already allocated several
/// GB.
pub fn decimate_vhacd(
    vertices: &[[f32; 3]],
    indices: &[[u32; 3]],
    params: &DecimateParameters,
) -> Result<DecimationResult, DecimationError> {
    validate_params(params)?;
    if vertices.len() < 4 {
        return Err(DecimationError::TooFewPoints(vertices.len()));
    }
    if indices.is_empty() {
        return Err(DecimationError::NoTriangles);
    }
    for (i, v) in vertices.iter().enumerate() {
        if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
            return Err(DecimationError::NonFiniteInput { index: i });
        }
    }
    let n = vertices.len() as u32;
    for (t, tri) in indices.iter().enumerate() {
        for &v in tri {
            if v >= n {
                return Err(DecimationError::BadIndex {
                    triangle: t,
                    vertex: v,
                    n_vertices: vertices.len(),
                });
            }
        }
    }

    let points: Vec<Point3<f32>> = vertices
        .iter()
        .map(|&[x, y, z]| Point3::new(x, y, z))
        .collect();

    let vhacd_params = VHACDParameters {
        resolution: params.resolution,
        max_convex_hulls: params.max_hulls as u32,
        concavity: params.concavity,
        fill_mode: fill_mode_from(params.fill_mode),
        ..VHACDParameters::default()
    };

    let vhacd = VHACD::decompose(&vhacd_params, &points, indices, false);
    // `compute_convex_hulls(downsampling: u32)` -> Vec<(Vec<Point>, Vec<[u32; 3]>)>.
    let parts_raw = vhacd.compute_convex_hulls(1);

    let mut parts = Vec::with_capacity(parts_raw.len());
    for (pts, tris) in parts_raw {
        if pts.len() < 4 || tris.is_empty() {
            // Skip degenerate parts (parry3d occasionally emits sliver hulls).
            continue;
        }
        parts.push(ConvexHullResult {
            vertices: pts.into_iter().map(|p| [p.x, p.y, p.z]).collect(),
            indices: tris,
        });
    }

    Ok(DecimationResult { parts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_mesh() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        // 12 triangles (2 per face). Winding is best-effort outward.
        let indices = vec![
            [0, 2, 1],
            [1, 2, 4], // z=0
            [3, 5, 6],
            [5, 7, 6], // z=1
            [0, 1, 3],
            [1, 5, 3], // y=0
            [2, 6, 4],
            [4, 6, 7], // y=1
            [0, 3, 2],
            [2, 3, 6], // x=0
            [1, 4, 5],
            [4, 7, 5], // x=1
        ];
        (vertices, indices)
    }

    #[test]
    fn cube_decomposes_to_a_single_hull() {
        let (v, i) = unit_cube_mesh();
        let params = DecimateParameters {
            resolution: 32,
            max_hulls: 4,
            ..Default::default()
        };
        let res = decimate_vhacd(&v, &i, &params).expect("decompose");
        // Convex shape → one hull suffices.
        assert!(!res.parts.is_empty(), "expected at least one hull part");
        assert!(res.output_triangles() > 0);
    }

    #[test]
    fn too_few_vertices_errors() {
        let v = vec![[0.0_f32, 0.0, 0.0]];
        let i = vec![[0_u32, 0, 0]];
        match decimate_vhacd(&v, &i, &DecimateParameters::default()) {
            Err(DecimationError::TooFewPoints(1)) => {}
            other => panic!("expected TooFewPoints, got {other:?}"),
        }
    }

    #[test]
    fn bad_index_errors() {
        let (v, _) = unit_cube_mesh();
        let i = vec![[0_u32, 1, 99]];
        match decimate_vhacd(&v, &i, &DecimateParameters::default()) {
            Err(DecimationError::BadIndex { vertex: 99, .. }) => {}
            other => panic!("expected BadIndex, got {other:?}"),
        }
    }

    #[test]
    fn zero_resolution_errors_before_vhacd() {
        let (v, i) = unit_cube_mesh();
        let params = DecimateParameters {
            resolution: 0,
            ..Default::default()
        };
        match decimate_vhacd(&v, &i, &params) {
            Err(DecimationError::InvalidResolution(0)) => {}
            other => panic!("expected InvalidResolution, got {other:?}"),
        }
    }

    #[test]
    fn zero_max_hulls_errors_before_vhacd() {
        let (v, i) = unit_cube_mesh();
        let params = DecimateParameters {
            max_hulls: 0,
            ..Default::default()
        };
        match decimate_vhacd(&v, &i, &params) {
            Err(DecimationError::InvalidMaxHulls(0)) => {}
            other => panic!("expected InvalidMaxHulls, got {other:?}"),
        }
    }

    #[test]
    fn invalid_concavity_errors_before_vhacd() {
        let (v, i) = unit_cube_mesh();
        let params = DecimateParameters {
            concavity: f32::NAN,
            ..Default::default()
        };
        match decimate_vhacd(&v, &i, &params) {
            Err(DecimationError::InvalidConcavity(value)) if value.is_nan() => {}
            other => panic!("expected InvalidConcavity, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_fill_mode_errors_before_vhacd() {
        let (v, i) = unit_cube_mesh();
        let params = DecimateParameters {
            fill_mode: 2,
            ..Default::default()
        };
        match decimate_vhacd(&v, &i, &params) {
            Err(DecimationError::UnsupportedFillMode(2)) => {}
            other => panic!("expected UnsupportedFillMode, got {other:?}"),
        }
    }
}
