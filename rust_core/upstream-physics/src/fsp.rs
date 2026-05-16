//! Functional Swing Plane (FSP) — Rust SVD primitives.
//!
//! Fits a best-fit plane to a set of 3-D marker points via Singular Value
//! Decomposition (SVD), then exposes golf-specific geometric metrics:
//! plane slope, azimuth direction, and signed point-to-plane distance.
//!
//! This module is Phase 1 of the FSP epic (issue #5429 / #5502).
//!
//! # Design by Contract
//! - [`calculate_fsp`]: requires ≥ 3 non-collinear points (finite coordinates).
//! - All functions receiving a [`Plane`] assume it was produced by
//!   [`calculate_fsp`] and therefore has a unit normal; callers must not
//!   construct [`Plane`] manually with an un-normalised normal.
//!
//! # Coordinate convention
//! Z-up: the ground plane is `z = 0`, vertical is `+Z`.

use nalgebra::{DMatrix, SVD};
use serde::{Deserialize, Serialize};

/// Best-fit plane returned by [`calculate_fsp`].
///
/// Both fields use the same coordinate system as the input points.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all))]
pub struct Plane {
    /// Unit normal vector of the best-fit plane.
    pub normal: [f64; 3],
    /// Centroid (mean) of the input point cloud.
    pub centroid: [f64; 3],
}

/// Errors returned by [`calculate_fsp`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FspError {
    /// Fewer than 3 points were supplied.
    InsufficientPoints,
    /// All points are collinear — no unique plane exists.
    /// Detected when the second SVD singular value is < 1e-10.
    DegeneratePoints,
}

impl std::fmt::Display for FspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FspError::InsufficientPoints => {
                write!(f, "At least 3 points are required for plane fitting")
            }
            FspError::DegeneratePoints => {
                write!(f, "Points are collinear — no unique plane can be fitted")
            }
        }
    }
}

impl std::error::Error for FspError {}

/// Threshold below which the second singular value indicates collinearity.
///
/// For a valid planar point set the singular values are [σ₁, σ₂, 0].
/// For collinear points they are [σ₁, 0, 0].  We therefore test sv[1].
const DEGENERACY_THRESHOLD: f64 = 1e-10;

/// Fit a best-fit plane to `points` using SVD.
///
/// # Algorithm
/// 1. Compute the centroid.
/// 2. Centre the points (subtract centroid).
/// 3. Form a 3×N matrix whose columns are the centred points.
/// 4. Run full SVD.  The left singular vector corresponding to the
///    **smallest** singular value is the plane normal.
///
/// # Errors
/// - [`FspError::InsufficientPoints`] — if `points.len() < 3`.
/// - [`FspError::DegeneratePoints`] — if points are collinear (second
///   singular value < [`DEGENERACY_THRESHOLD`]).
///
/// # Postcondition
/// `result.normal` is a unit vector.
pub fn calculate_fsp(points: &[[f64; 3]]) -> Result<Plane, FspError> {
    let n = points.len();

    // DbC precondition: need at least 3 points
    if n < 3 {
        return Err(FspError::InsufficientPoints);
    }

    // Compute centroid
    let centroid = compute_centroid(points);

    // Build 3×N matrix of centred points (rows = x/y/z, cols = points).
    // nalgebra DMatrix::from_vec uses column-major storage, so element
    // at (row, col) is stored at index col*nrows + row.
    let mut data = vec![0.0_f64; 3 * n];
    for (col, p) in points.iter().enumerate() {
        data[col * 3] = p[0] - centroid[0];
        data[col * 3 + 1] = p[1] - centroid[1];
        data[col * 3 + 2] = p[2] - centroid[2];
    }
    let mat = DMatrix::from_vec(3, n, data);

    // SVD — compute_full_u = true so we get left singular vectors (U is 3×3).
    let svd = SVD::new(mat, true, false);

    // Left singular vectors: columns of U, ordered by descending singular value.
    // The plane normal is the column corresponding to the SMALLEST singular value
    // → that is the last column (index 2) of U.
    //
    // Degeneracy check:
    //   - Valid plane:   singular values ≈ [σ₁, σ₂, 0]  (one zero → planar)
    //   - Collinear:     singular values ≈ [σ₁, 0,  0]  (two zeros → line)
    //
    // We test the SECOND-smallest singular value (index 1 in descending order).
    let u = svd.u.ok_or(FspError::DegeneratePoints)?;
    let singular_values = &svd.singular_values;

    if singular_values.len() < 2 || singular_values[1] < DEGENERACY_THRESHOLD {
        return Err(FspError::DegeneratePoints);
    }

    // The plane normal is the last column of U (corresponds to smallest sv).
    let col = u.ncols() - 1;
    let normal = [u[(0, col)], u[(1, col)], u[(2, col)]];

    Ok(Plane { normal, centroid })
}

/// Slope of the FSP in degrees.
///
/// Defined as the angle between the plane normal and the vertical axis
/// `(0, 0, 1)`.  A horizontal plane (z = 0) has slope 0°; a vertical
/// plane has slope 90°.
///
/// Formula: `acos(|normal.z|)` in degrees.
///
/// # Postcondition
/// Returns a value in `[0.0, 90.0]`.
pub fn fsp_slope_deg(plane: &Plane) -> f64 {
    plane.normal[2].abs().acos().to_degrees()
}

/// Azimuth direction of the FSP in degrees.
///
/// Projects `target_line` onto the plane, then measures the signed angle
/// (in degrees) from the `+X` axis to the projected vector, measured
/// counter-clockwise when viewed from the direction of the plane normal.
///
/// Returns a value in `(-180, 180]`.
///
/// # Arguments
/// * `plane` — output of [`calculate_fsp`].
/// * `target_line` — a reference direction vector (need not be a unit
///   vector, but must not be the zero vector).
///
/// # Notes
/// If `target_line` is parallel to the plane normal (i.e. the projection
/// is the zero vector), the function returns `0.0` as a safe fallback.
pub fn fsp_direction_deg(plane: &Plane, target_line: &[f64; 3]) -> f64 {
    let n = plane.normal;

    // Dot product of target_line and normal
    let dot = target_line[0] * n[0] + target_line[1] * n[1] + target_line[2] * n[2];

    // Project target_line onto the plane: proj = target - (target·n)n
    let proj = [
        target_line[0] - dot * n[0],
        target_line[1] - dot * n[1],
        target_line[2] - dot * n[2],
    ];

    let proj_mag = (proj[0] * proj[0] + proj[1] * proj[1] + proj[2] * proj[2]).sqrt();
    if proj_mag < 1e-12 {
        // target_line is parallel to the normal — azimuth undefined, return 0
        return 0.0;
    }

    // Reference direction for azimuth: +X projected onto the plane
    let ref_vec = [1.0_f64, 0.0, 0.0];
    let ref_dot = ref_vec[0] * n[0] + ref_vec[1] * n[1] + ref_vec[2] * n[2];
    let ref_proj = [
        ref_vec[0] - ref_dot * n[0],
        ref_vec[1] - ref_dot * n[1],
        ref_vec[2] - ref_dot * n[2],
    ];
    let ref_mag = (ref_proj[0] * ref_proj[0]
        + ref_proj[1] * ref_proj[1]
        + ref_proj[2] * ref_proj[2])
        .sqrt();

    if ref_mag < 1e-12 {
        // Normal is along X; use Y as reference instead
        return proj[1].atan2(proj[0]).to_degrees();
    }

    let ref_unit = [
        ref_proj[0] / ref_mag,
        ref_proj[1] / ref_mag,
        ref_proj[2] / ref_mag,
    ];
    let proj_unit = [proj[0] / proj_mag, proj[1] / proj_mag, proj[2] / proj_mag];

    // cos θ = ref_unit · proj_unit
    let cos_theta = (ref_unit[0] * proj_unit[0]
        + ref_unit[1] * proj_unit[1]
        + ref_unit[2] * proj_unit[2])
        .clamp(-1.0, 1.0);

    // sin θ via cross product (ref_unit × proj_unit) · normal → sign of rotation
    let cross = [
        ref_unit[1] * proj_unit[2] - ref_unit[2] * proj_unit[1],
        ref_unit[2] * proj_unit[0] - ref_unit[0] * proj_unit[2],
        ref_unit[0] * proj_unit[1] - ref_unit[1] * proj_unit[0],
    ];
    let sign = (cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2]).signum();

    let angle_rad = cos_theta.acos();
    (sign * angle_rad).to_degrees()
}

/// Signed perpendicular distance from `point` to the plane.
///
/// Positive when the point is on the same side as the normal direction,
/// negative on the opposite side.
///
/// Formula: `(point - centroid) · normal`.
pub fn point_to_fsp_distance(point: &[f64; 3], plane: &Plane) -> f64 {
    let dx = point[0] - plane.centroid[0];
    let dy = point[1] - plane.centroid[1];
    let dz = point[2] - plane.centroid[2];
    dx * plane.normal[0] + dy * plane.normal[1] + dz * plane.normal[2]
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn compute_centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in points {
        cx += p[0];
        cy += p[1];
        cz += p[2];
    }
    [cx / n, cy / n, cz / n]
}

// ── PyO3 bindings (feature-gated) ───────────────────────────────────────────

#[cfg(feature = "python")]
pub mod python {
    use super::{FspError, Plane};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    impl From<FspError> for PyErr {
        fn from(e: FspError) -> PyErr {
            PyValueError::new_err(e.to_string())
        }
    }

    /// Calculate the best-fit plane for a list of 3-D points.
    ///
    /// Args:
    ///     points: List of ``[x, y, z]`` coordinates (need at least 3).
    ///
    /// Returns:
    ///     A :class:`Plane` with ``.normal`` and ``.centroid`` attributes.
    ///
    /// Raises:
    ///     ValueError: If fewer than 3 points are given, or if the points
    ///         are collinear.
    #[pyfunction]
    pub fn calculate_fsp(points: Vec<[f64; 3]>) -> PyResult<Plane> {
        super::calculate_fsp(&points).map_err(PyErr::from)
    }

    /// Slope of the FSP in degrees (0° = horizontal, 90° = vertical).
    ///
    /// Args:
    ///     plane: A :class:`Plane` returned by :func:`calculate_fsp`.
    ///
    /// Returns:
    ///     Angle between plane normal and vertical, in degrees ``[0, 90]``.
    #[pyfunction]
    pub fn fsp_slope_deg(plane: &Plane) -> f64 {
        super::fsp_slope_deg(plane)
    }

    /// Azimuth direction of the FSP in degrees.
    ///
    /// Projects *target_line* onto the plane and measures its signed angle
    /// from the ``+X`` axis (counter-clockwise when viewed from the normal
    /// direction).
    ///
    /// Args:
    ///     plane:       A :class:`Plane` returned by :func:`calculate_fsp`.
    ///     target_line: Reference direction ``[x, y, z]``.
    ///
    /// Returns:
    ///     Azimuth angle in degrees ``(-180, 180]``.
    #[pyfunction]
    pub fn fsp_direction_deg(plane: &Plane, target_line: [f64; 3]) -> f64 {
        super::fsp_direction_deg(plane, &target_line)
    }

    /// Signed perpendicular distance from *point* to the plane.
    ///
    /// Positive on the normal side, negative on the opposite side.
    ///
    /// Args:
    ///     point: ``[x, y, z]`` coordinates of the query point.
    ///     plane: A :class:`Plane` returned by :func:`calculate_fsp`.
    ///
    /// Returns:
    ///     Signed distance in the same units as the input points.
    #[pyfunction]
    pub fn point_to_fsp_distance(point: [f64; 3], plane: &Plane) -> f64 {
        super::point_to_fsp_distance(&point, plane)
    }
}

// ── Tests (TDD — written before implementation) ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons
    const TOL: f64 = 1e-6;

    // ── Helper ───────────────────────────────────────────────────────────────

    fn norm(v: &[f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    // ── calculate_fsp ────────────────────────────────────────────────────────

    /// Points on z=0 → normal should be ≈ (0, 0, ±1).
    #[test]
    fn test_horizontal_plane_normal() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).expect("should succeed");

        // Normal must be nearly along Z
        assert!(
            plane.normal[2].abs() > 1.0 - TOL,
            "normal.z should be ≈ ±1, got {:?}",
            plane.normal
        );
        assert!(
            plane.normal[0].abs() < TOL,
            "normal.x should be ≈ 0, got {}",
            plane.normal[0]
        );
        assert!(
            plane.normal[1].abs() < TOL,
            "normal.y should be ≈ 0, got {}",
            plane.normal[1]
        );
    }

    /// Normal returned by calculate_fsp must be a unit vector.
    #[test]
    fn test_normal_is_unit_vector() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.288, 0.816],
        ];
        let plane = calculate_fsp(&points).expect("should succeed");
        let length = norm(&plane.normal);
        assert!(
            (length - 1.0).abs() < TOL,
            "normal must be unit-length, got |n|={length}"
        );
    }

    /// Centroid should equal the mean of the input points.
    #[test]
    fn test_centroid_correctness() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let plane = calculate_fsp(&points).expect("should succeed");
        assert!((plane.centroid[0] - 1.0).abs() < TOL);
        assert!((plane.centroid[1] - 1.0).abs() < TOL);
        assert!((plane.centroid[2] - 0.0).abs() < TOL);
    }

    /// Fewer than 3 points → InsufficientPoints.
    #[test]
    fn test_insufficient_points_single() {
        let result = calculate_fsp(&[[0.0, 0.0, 0.0]]);
        assert!(
            matches!(result, Err(FspError::InsufficientPoints)),
            "expected InsufficientPoints, got {:?}",
            result.err()
        );
    }

    /// Exactly 2 points → InsufficientPoints.
    #[test]
    fn test_insufficient_points_two() {
        let result = calculate_fsp(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        assert!(
            matches!(result, Err(FspError::InsufficientPoints)),
            "expected InsufficientPoints, got {:?}",
            result.err()
        );
    }

    /// Collinear points → DegeneratePoints.
    #[test]
    fn test_degenerate_collinear() {
        let result = calculate_fsp(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        assert!(
            matches!(result, Err(FspError::DegeneratePoints)),
            "expected DegeneratePoints, got {:?}",
            result.err()
        );
    }

    /// Collinear points in 3-D diagonal → DegeneratePoints.
    #[test]
    fn test_degenerate_collinear_diagonal() {
        let result = calculate_fsp(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        assert!(
            matches!(result, Err(FspError::DegeneratePoints)),
            "expected DegeneratePoints, got {:?}",
            result.err()
        );
    }

    // ── fsp_slope_deg ─────────────────────────────────────────────────────────

    /// Horizontal plane (z=0) → slope ≈ 0°.
    #[test]
    fn test_slope_horizontal_plane() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        let slope = fsp_slope_deg(&plane);
        assert!(
            slope.abs() < 0.01,
            "horizontal plane slope should be ≈ 0°, got {slope}"
        );
    }

    /// Vertical plane (x=0, points in YZ) → slope ≈ 90°.
    #[test]
    fn test_slope_vertical_plane() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        let slope = fsp_slope_deg(&plane);
        assert!(
            (slope - 90.0).abs() < 0.01,
            "vertical plane slope should be ≈ 90°, got {slope}"
        );
    }

    // ── point_to_fsp_distance ─────────────────────────────────────────────────

    /// Point at distance 1 above z=0 plane → distance ≈ ±1.0.
    #[test]
    fn test_distance_point_above_horizontal_plane() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        let dist = point_to_fsp_distance(&[0.0, 0.0, 1.0], &plane);
        assert!(
            (dist.abs() - 1.0).abs() < TOL,
            "distance from (0,0,1) to z=0 should be ≈ 1, got {dist}"
        );
    }

    /// Point on the plane → distance ≈ 0.
    #[test]
    fn test_distance_point_on_plane() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        // A point on the z=0 plane
        let dist = point_to_fsp_distance(&[0.5, 0.5, 0.0], &plane);
        assert!(
            dist.abs() < TOL,
            "point on the plane should have distance ≈ 0, got {dist}"
        );
    }

    /// Signed distance: point below z=0 should have opposite sign to point above.
    #[test]
    fn test_distance_sign_convention() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        let above = point_to_fsp_distance(&[0.0, 0.0, 1.0], &plane);
        let below = point_to_fsp_distance(&[0.0, 0.0, -1.0], &plane);
        assert!(
            above * below < 0.0,
            "distances on opposite sides must have opposite signs"
        );
        assert!((above.abs() - below.abs()).abs() < TOL, "magnitudes must match");
    }

    // ── fsp_direction_deg ─────────────────────────────────────────────────────

    /// Target along +X on horizontal plane → azimuth ≈ 0°.
    #[test]
    fn test_direction_along_x() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        let dir = fsp_direction_deg(&plane, &[1.0, 0.0, 0.0]);
        assert!(
            dir.abs() < 0.01,
            "direction along +X should be ≈ 0°, got {dir}"
        );
    }

    /// Target along +Y on horizontal plane → azimuth ≈ ±90°.
    #[test]
    fn test_direction_along_y() {
        let points: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let plane = calculate_fsp(&points).unwrap();
        let dir = fsp_direction_deg(&plane, &[0.0, 1.0, 0.0]);
        assert!(
            (dir.abs() - 90.0).abs() < 0.01,
            "direction along +Y should be ≈ 90°, got {dir}"
        );
    }
}
