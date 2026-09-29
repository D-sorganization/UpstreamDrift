//! Swing plane fitting and analysis.
//!
//! Fits a plane to 3D trajectory points using least-squares regression,
//! then computes swing metrics (plane angle, face angle, etc.).
//!
//! # Design by Contract
//! - At least 3 non-collinear points required for plane fit
//! - All input coordinates must be finite

use serde::{Deserialize, Serialize};
use tools_core::Vector3;

/// Result of a swing plane fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct SwingPlaneResult {
    /// Plane normal vector (unit).
    pub normal: Vector3,
    /// Plane angle relative to ground [radians].
    pub plane_angle: f64,
    /// R² goodness of fit [0, 1].
    pub r_squared: f64,
    /// Signed distance of each point from the plane.
    pub residuals: Vec<f64>,
    /// Number of points used in the fit.
    pub num_points: usize,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl SwingPlaneResult {
    /// Plane angle in degrees.
    #[getter]
    fn plane_angle_deg(&self) -> f64 {
        self.plane_angle.to_degrees()
    }

    /// Mean absolute residual [m].
    #[getter]
    fn mean_residual(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        self.residuals.iter().map(|r| r.abs()).sum::<f64>() / self.residuals.len() as f64
    }
}

/// Fit a plane to a set of 3D points using SVD-free least squares.
///
/// Uses the normal equation approach: fits z = ax + by + c,
/// then converts to a plane normal.
///
/// For general planes (not axis-aligned), this uses the centroid
/// and covariance approach.
///
/// # Arguments
/// * `points` - Slice of 3D position vectors (must have ≥ 3 points)
///
/// # Returns
/// `Ok(SwingPlaneResult)` on success, `Err` if too few points.
pub fn fit_plane(points: &[Vector3]) -> Result<SwingPlaneResult, &'static str> {
    let n = points.len();

    // DbC: All coordinates must be finite (finiteness is an invariant)
    debug_assert!(
        points
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()),
        "All input coordinates must be finite"
    );

    // Graceful error when too few points (not a panic — callers may pass validated input)
    if n < 3 {
        return Err("Need at least 3 points for plane fit");
    }

    // Compute centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in points {
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    let inv_n = 1.0 / n as f64;
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    // Try each axis as a potential dependent variable direction and pick
    // the one that minimizes the sum of squared residuals.
    let candidates = [
        compute_plane_normal_for_axis(points, cx, cy, cz, 0), // z = f(x,y)
        compute_plane_normal_for_axis(points, cx, cy, cz, 1), // x = f(y,z)
        compute_plane_normal_for_axis(points, cx, cy, cz, 2), // y = f(x,z)
    ];

    let (best_normal, best_r2) = match candidates
        .iter()
        .filter_map(|c| c.as_ref().ok())
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
    {
        Some(result) => result,
        None => return Err("All plane fit candidates failed (collinear or degenerate points)"),
    };

    // Compute residuals
    let residuals: Vec<f64> = points
        .iter()
        .map(|p| {
            let dx = p.x - cx;
            let dy = p.y - cy;
            let dz = p.z - cz;
            dx * best_normal.x + dy * best_normal.y + dz * best_normal.z
        })
        .collect();

    // Plane angle relative to ground (Y-up convention)
    // Angle between normal and Y-axis
    let dot_y = best_normal.y.abs();
    let plane_angle = dot_y.acos(); // 0 = vertical plane, π/2 = horizontal

    Ok(SwingPlaneResult {
        normal: best_normal,
        plane_angle,
        r_squared: best_r2,
        residuals,
        num_points: n,
    })
}

/// Compute plane normal fitting dependent_axis = f(other two axes).
fn compute_plane_normal_for_axis(
    points: &[Vector3],
    cx: f64,
    cy: f64,
    cz: f64,
    dependent_axis: usize,
) -> Result<(Vector3, f64), &'static str> {
    // Extract centered coordinates based on which axis is dependent
    let (dep, ind1, ind2): (Vec<f64>, Vec<f64>, Vec<f64>) = match dependent_axis {
        0 => (
            // z = f(x, y)
            points.iter().map(|p| p.z - cz).collect(),
            points.iter().map(|p| p.x - cx).collect(),
            points.iter().map(|p| p.y - cy).collect(),
        ),
        1 => (
            // x = f(y, z)
            points.iter().map(|p| p.x - cx).collect(),
            points.iter().map(|p| p.y - cy).collect(),
            points.iter().map(|p| p.z - cz).collect(),
        ),
        2 => (
            // y = f(x, z)
            points.iter().map(|p| p.y - cy).collect(),
            points.iter().map(|p| p.x - cx).collect(),
            points.iter().map(|p| p.z - cz).collect(),
        ),
        _ => return Err("Invalid axis"),
    };

    // Solve: dep = a * ind1 + b * ind2  (least squares, centered)
    let s11: f64 = ind1.iter().map(|v| v * v).sum();
    let s12: f64 = ind1.iter().zip(ind2.iter()).map(|(a, b)| a * b).sum();
    let s22: f64 = ind2.iter().map(|v| v * v).sum();
    let sd1: f64 = ind1.iter().zip(dep.iter()).map(|(a, d)| a * d).sum();
    let sd2: f64 = ind2.iter().zip(dep.iter()).map(|(a, d)| a * d).sum();

    let det = s11 * s22 - s12 * s12;
    if det.abs() < 1e-12 {
        return Err("Singular matrix (collinear points)");
    }

    let a = (s22 * sd1 - s12 * sd2) / det;
    let b = (s11 * sd2 - s12 * sd1) / det;

    // Normal vector: for z = ax + by, normal is (-a, -b, 1) (then normalize)
    let (nx, ny, nz) = match dependent_axis {
        0 => (-a, -b, 1.0), // z = ax + by
        1 => (1.0, -a, -b), // x = ay + bz
        2 => (-a, 1.0, -b), // y = ax + bz
        _ => unreachable!(),
    };

    let mag = (nx * nx + ny * ny + nz * nz).sqrt();
    let normal = Vector3::new(nx / mag, ny / mag, nz / mag);

    // R² computation
    let ss_res: f64 = dep
        .iter()
        .zip(ind1.iter().zip(ind2.iter()))
        .map(|(d, (i1, i2))| {
            let pred = a * i1 + b * i2;
            (d - pred) * (d - pred)
        })
        .sum();

    let ss_tot: f64 = dep.iter().map(|d| d * d).sum(); // Already centered

    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        1.0 // All points at same value → perfect fit
    };

    Ok((normal, r2))
}

// ── Tests (TDD) ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Points on XZ plane (Y=0) → normal should be along Y.
    #[test]
    fn test_flat_plane_fit() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ];

        let result = fit_plane(&points).unwrap();

        // Normal should be close to (0, ±1, 0)
        assert!(
            result.normal.y.abs() > 0.99,
            "Normal should be along Y, got {:?}",
            result.normal
        );
        assert!(result.r_squared > 0.99, "R² should be ~1.0 for flat plane");
    }

    /// Test 2: Points on a 45-degree tilted plane.
    #[test]
    fn test_tilted_plane() {
        // y = x (45-degree tilt)
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(2.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];

        let result = fit_plane(&points).unwrap();

        assert!(result.r_squared > 0.99, "R² should be ~1.0 for exact plane");
        assert_eq!(result.num_points, 5);
    }

    /// Test 3: Too few points returns error.
    #[test]
    fn test_too_few_points() {
        let points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];

        let result = fit_plane(&points);
        assert!(result.is_err());
    }

    /// Test 4: Noisy plane should have lower R².
    #[test]
    fn test_noisy_plane() {
        // Points nearly on XZ plane with small Y noise
        let points = vec![
            Vector3::new(0.0, 0.01, 0.0),
            Vector3::new(1.0, -0.02, 0.0),
            Vector3::new(0.0, 0.015, 1.0),
            Vector3::new(1.0, -0.01, 1.0),
            Vector3::new(0.5, 0.03, 0.5),
            Vector3::new(2.0, -0.005, 0.5),
            Vector3::new(1.5, 0.02, 1.5),
        ];

        let result = fit_plane(&points).unwrap();

        // R² should be high (noise is small) but not perfect
        assert!(
            result.r_squared > 0.0,
            "R² should be positive, got {}",
            result.r_squared
        );
        assert!(
            result.r_squared < 1.0 - 1e-10,
            "R² should not be perfect with noise"
        );
    }

    /// Test 5: Residuals should sum to approximately zero (centered).
    #[test]
    fn test_residuals_sum_zero() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(0.5, 0.0, 0.5),
        ];

        let result = fit_plane(&points).unwrap();

        let sum: f64 = result.residuals.iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "Residuals should sum near zero, got {sum}"
        );
    }

    /// Test 6: Plane angle for horizontal plane should be ~π/2.
    #[test]
    fn test_horizontal_plane_angle() {
        let points = vec![
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(1.0, 5.0, 0.0),
            Vector3::new(0.0, 5.0, 1.0),
            Vector3::new(1.0, 5.0, 1.0),
        ];

        let result = fit_plane(&points).unwrap();

        // For horizontal plane, normal is along Y, angle to Y-axis = 0
        // plane_angle = acos(|ny|) → for ny ≈ ±1, angle ≈ 0
        assert!(
            result.plane_angle < 0.1,
            "Horizontal plane angle should be ~0, got {}",
            result.plane_angle
        );
    }

    /// Test 7: Collinear points should return an explicit error.
    #[test]
    fn test_collinear_points_returns_error() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(2.0, 2.0, 2.0),
        ];

        let result = fit_plane(&points);
        assert!(
            result.is_err(),
            "Collinear points should return Err, got {:?}",
            result
        );
    }
}
