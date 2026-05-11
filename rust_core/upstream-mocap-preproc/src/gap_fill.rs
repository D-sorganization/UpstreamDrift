//! Gap-fill kernels: linear, cubic-spline, PCA reconstruction.
//!
//! The interface intentionally accepts pre-stacked numeric arrays from Python:
//! `(n_frames, n_points, n_dims)` data plus an `(n_frames, n_points)` boolean
//! occlusion mask. The Python facade is responsible for translating
//! `confidence < 0.5` (keypoints) / `Marker.occluded` (markers) into that
//! mask, and for writing the result back to the contract types.

use ndarray::{Array1, Array2, Array3, ArrayView3, Axis};

use crate::filter::solve_linear_view;

// ── Linear interpolation gap fill ────────────────────────────────────────────

/// Fill gaps in a single 1D series using linear interpolation between the
/// last visible sample before the gap and the first visible sample after.
/// Boundary gaps (no anchor on one side) are left as-is, matching the
/// Python `_linear_interp_*` semantics where `start == 0` or
/// `end >= len(frames)` is a no-op.
fn linear_fill_1d(values: &mut [f64], mask: &mut [bool], max_gap: usize) {
    let n = values.len();
    if n == 0 {
        return;
    }
    let mut i = 0;
    while i < n {
        if !mask[i] {
            i += 1;
            continue;
        }
        // Find the run [i, j] of masked samples.
        let start = i;
        let mut j = i;
        while j < n && mask[j] {
            j += 1;
        }
        let end = j - 1;
        let gap_size = end - start + 1;
        if gap_size > max_gap || start == 0 || end + 1 >= n {
            i = j;
            continue;
        }
        let v_before = values[start - 1];
        let v_after = values[end + 1];
        let denom = (end - start + 2) as f64;
        for k in start..=end {
            let t = (k - start + 1) as f64 / denom;
            values[k] = v_before + t * (v_after - v_before);
            mask[k] = false;
        }
        i = j;
    }
}

/// Linear gap fill operating on `(n_frames, n_points, n_dims)` data and an
/// `(n_frames, n_points)` mask.
pub fn linear_gap_fill(
    data: ArrayView3<f64>,
    mask: ndarray::ArrayView2<bool>,
    max_gap: usize,
) -> (Array3<f64>, Array2<bool>) {
    let (nt, np, nd) = data.dim();
    let mut out = data.to_owned();
    let mut out_mask = mask.to_owned();

    // Run per-point per-dim. Mask is per-point; we duplicate to per-dim
    // working copy so each axis fills independently from the value array.
    let mut series = vec![0.0_f64; nt];
    let mut series_mask = vec![false; nt];
    for p in 0..np {
        for d in 0..nd {
            for t in 0..nt {
                series[t] = out[[t, p, d]];
                series_mask[t] = out_mask[[t, p]];
            }
            linear_fill_1d(&mut series, &mut series_mask, max_gap);
            for t in 0..nt {
                out[[t, p, d]] = series[t];
            }
        }
        // After filling all dims, reconcile the per-point mask: a point's
        // mask is cleared only where every dim got filled. Since all dims
        // share the same mask going in, the after-state is identical, so
        // we can use the last `series_mask`.
        for t in 0..nt {
            out_mask[[t, p]] = series_mask[t];
        }
    }
    (out, out_mask)
}

// ── Cubic-spline gap fill ────────────────────────────────────────────────────

/// Natural cubic-spline interpolation through visible samples of a 1D series.
/// Matches `scipy.interpolate.CubicSpline(bc_type='natural')` evaluated at
/// the masked sample indices. Boundary gaps (no anchor on one side) are
/// left as-is to mirror Python's `_cubic_interp_*` (which currently falls
/// back to linear; we go strictly tighter by using a real spline).
fn cubic_fill_1d(values: &mut [f64], mask: &mut [bool], max_gap: usize) {
    let n = values.len();
    if n == 0 {
        return;
    }
    // Collect visible sample indices and values as floats.
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            xs.push(i as f64);
            ys.push(values[i]);
        }
    }
    if xs.len() < 4 {
        // Not enough anchors for a cubic; defer to linear semantics.
        linear_fill_1d(values, mask, max_gap);
        return;
    }
    // Build natural cubic spline second derivatives at the visible knots.
    let m = xs.len();
    let mut h = vec![0.0_f64; m - 1];
    for i in 0..(m - 1) {
        h[i] = xs[i + 1] - xs[i];
    }
    // Solve tridiagonal for second derivatives (natural BC: M[0]=M[m-1]=0).
    let mut a_diag = vec![0.0_f64; m]; // sub
    let mut b_diag = vec![0.0_f64; m]; // main
    let mut c_diag = vec![0.0_f64; m]; // super
    let mut rhs = vec![0.0_f64; m];
    b_diag[0] = 1.0;
    b_diag[m - 1] = 1.0;
    for i in 1..(m - 1) {
        a_diag[i] = h[i - 1];
        b_diag[i] = 2.0 * (h[i - 1] + h[i]);
        c_diag[i] = h[i];
        rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
    }
    let m_second = solve_tridiag(&a_diag, &b_diag, &c_diag, &rhs);

    // Walk through gap runs as in linear_fill_1d and fill via the spline.
    let mut i = 0;
    while i < n {
        if !mask[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i;
        while j < n && mask[j] {
            j += 1;
        }
        let end = j - 1;
        let gap_size = end - start + 1;
        if gap_size > max_gap || start == 0 || end + 1 >= n {
            i = j;
            continue;
        }
        for k in start..=end {
            let x = k as f64;
            values[k] = evaluate_natural_cubic(&xs, &ys, &m_second, &h, x);
            mask[k] = false;
        }
        i = j;
    }
}

fn solve_tridiag(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0_f64; n];
    let mut dp = vec![0.0_f64; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let denom = b[i] - a[i] * cp[i - 1];
        cp[i] = if i + 1 < n { c[i] / denom } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }
    let mut x = vec![0.0_f64; n];
    x[n - 1] = dp[n - 1];
    for i in (0..(n - 1)).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

fn evaluate_natural_cubic(xs: &[f64], ys: &[f64], m: &[f64], h: &[f64], x: f64) -> f64 {
    // Find segment via binary search.
    let n = xs.len();
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] > x {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let h_i = h[lo];
    let a = (xs[hi] - x) / h_i;
    let b = (x - xs[lo]) / h_i;
    a * ys[lo]
        + b * ys[hi]
        + ((a * a * a - a) * m[lo] + (b * b * b - b) * m[hi]) * (h_i * h_i) / 6.0
}

pub fn cubic_gap_fill(
    data: ArrayView3<f64>,
    mask: ndarray::ArrayView2<bool>,
    max_gap: usize,
) -> (Array3<f64>, Array2<bool>) {
    let (nt, np, nd) = data.dim();
    let mut out = data.to_owned();
    let mut out_mask = mask.to_owned();

    let mut series = vec![0.0_f64; nt];
    let mut series_mask = vec![false; nt];
    for p in 0..np {
        for d in 0..nd {
            for t in 0..nt {
                series[t] = out[[t, p, d]];
                series_mask[t] = out_mask[[t, p]];
            }
            cubic_fill_1d(&mut series, &mut series_mask, max_gap);
            for t in 0..nt {
                out[[t, p, d]] = series[t];
            }
        }
        for t in 0..nt {
            out_mask[[t, p]] = series_mask[t];
        }
    }
    (out, out_mask)
}

// ── PCA gap fill ─────────────────────────────────────────────────────────────

/// PCA reconstruction of occluded marker rows.
///
/// Algorithm mirrors `_pca_reconstruct_markers` in the Python implementation:
/// 1. Flatten `(nt, np, 3)` to `(nt, np*3)`.
/// 2. Identify rows with no occluded markers as the basis.
/// 3. SVD-truncate to rank `min(6, effective_rank)`.
/// 4. For each occluded row, solve least-squares for basis coeffs from the
///    visible coords, then back-fill the occluded entries.
///
/// Returns `(filled_data, filled_mask, success)` where `success[i]` is true
/// when the row was filled by PCA. The Python facade is expected to fall
/// back to linear interpolation for rows where `success` is false.
pub fn pca_gap_fill(
    data: ArrayView3<f64>,
    mask: ndarray::ArrayView2<bool>,
    max_gap: usize,
    rank_override: Option<usize>,
) -> (Array3<f64>, Array2<bool>, Array1<bool>) {
    let (nt, np, nd) = data.dim();
    assert_eq!(nd, 3, "PCA gap fill expects 3D markers");
    let n_cols = np * 3;
    let mut out = data.to_owned();
    let mut out_mask = mask.to_owned();
    let mut pca_success = Array1::<bool>::from_elem(nt, false);

    if nt < 2 {
        return (out, out_mask, pca_success);
    }

    // Build (nt, np*3) matrix and a per-column occluded mask. The Python
    // version inflates the per-marker mask into per-coord (all 3 coords
    // share the same mask), so we match that.
    let mut m = Array2::<f64>::zeros((nt, n_cols));
    let mut col_mask = Array2::<bool>::from_elem((nt, n_cols), false);
    for t in 0..nt {
        for p in 0..np {
            let occ = out_mask[[t, p]];
            for k in 0..3 {
                m[[t, p * 3 + k]] = out[[t, p, k]];
                col_mask[[t, p * 3 + k]] = occ;
            }
        }
    }

    // Identify fully-visible rows.
    let mut visible_rows: Vec<usize> = Vec::new();
    for t in 0..nt {
        let mut any = false;
        for c in 0..n_cols {
            if col_mask[[t, c]] {
                any = true;
                break;
            }
        }
        if !any {
            visible_rows.push(t);
        }
    }
    if visible_rows.len() < 2 {
        return (out, out_mask, pca_success);
    }

    // Build M_visible matrix and column means.
    let n_visible = visible_rows.len();
    let mut mv = Array2::<f64>::zeros((n_visible, n_cols));
    for (row_i, &t) in visible_rows.iter().enumerate() {
        for c in 0..n_cols {
            mv[[row_i, c]] = m[[t, c]];
        }
    }
    let mean: Array1<f64> = mv.mean_axis(Axis(0)).expect("non-empty");
    // Center.
    let mut mc = mv.clone();
    for r in 0..n_visible {
        for c in 0..n_cols {
            mc[[r, c]] -= mean[c];
        }
    }

    // SVD via simple Jacobi-style approach is overkill; use nalgebra.
    use nalgebra::DMatrix;
    let mc_na = DMatrix::<f64>::from_fn(n_visible, n_cols, |r, c| mc[[r, c]]);
    let svd = mc_na.svd(false, true);
    let s = svd.singular_values;
    let vt = svd.v_t.expect("requested v_t=true");

    let s_max = s.iter().copied().fold(0.0_f64, f64::max);
    let eps = (s.len().max(1) as f64) * f64::EPSILON * if s_max > 0.0 { s_max } else { 1.0 };
    let effective_rank = s.iter().filter(|&&v| v > eps).count();
    if effective_rank == 0 {
        return (out, out_mask, pca_success);
    }
    let k = match rank_override {
        Some(r) => r.min(effective_rank),
        None => 6usize.min(effective_rank),
    };

    // V_k: (n_cols, k). V_t is (min(n_visible, n_cols), n_cols). Take first
    // k rows of V_t and transpose.
    let mut v_k = Array2::<f64>::zeros((n_cols, k));
    for i in 0..k {
        for c in 0..n_cols {
            v_k[[c, i]] = vt[(i, c)];
        }
    }

    // Determine which gaps exceed max_gap (skip those).
    let mut skip_frames = vec![false; nt];
    {
        let mut t = 0;
        while t < nt {
            // a "frame in gap" is any frame with any occluded marker.
            let any_occ = (0..np).any(|p| out_mask[[t, p]]);
            if !any_occ {
                t += 1;
                continue;
            }
            let start = t;
            while t < nt && (0..np).any(|p| out_mask[[t, p]]) {
                t += 1;
            }
            let end = t - 1;
            let gap_size = end - start + 1;
            if gap_size > max_gap {
                for k in start..=end {
                    skip_frames[k] = true;
                }
            }
        }
    }

    // Reconstruct each occluded row.
    for t in 0..nt {
        let any_occ = (0..n_cols).any(|c| col_mask[[t, c]]);
        if !any_occ {
            continue;
        }
        if skip_frames[t] {
            continue;
        }
        // Visible col indices.
        let visible_cols: Vec<usize> = (0..n_cols).filter(|&c| !col_mask[[t, c]]).collect();
        if visible_cols.len() < k {
            continue;
        }
        // Solve V_k[visible] * c = M[t, visible] - mean[visible]
        // V_k is (n_cols, k). A: (visible.len(), k).
        let nv = visible_cols.len();
        let mut a_mat = vec![vec![0.0_f64; k]; nv];
        let mut b_vec = vec![0.0_f64; nv];
        for (r, &col) in visible_cols.iter().enumerate() {
            for j in 0..k {
                a_mat[r][j] = v_k[[col, j]];
            }
            b_vec[r] = m[[t, col]] - mean[col];
        }
        // Normal equations: (A^T A) c = A^T b (k <= nv guaranteed).
        let mut ata = vec![vec![0.0_f64; k]; k];
        let mut atb = vec![0.0_f64; k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for r in 0..nv {
                    s += a_mat[r][i] * a_mat[r][j];
                }
                ata[i][j] = s;
            }
            let mut s = 0.0;
            for r in 0..nv {
                s += a_mat[r][i] * b_vec[r];
            }
            atb[i] = s;
        }
        let coeffs = match solve_linear_view(ata, atb) {
            Some(c) => c,
            None => continue,
        };
        // Reconstruct full row: mean + V_k @ coeffs.
        let mut reconstructed = vec![0.0_f64; n_cols];
        for c in 0..n_cols {
            let mut s = mean[c];
            for j in 0..k {
                s += v_k[[c, j]] * coeffs[j];
            }
            reconstructed[c] = s;
        }
        if reconstructed.iter().any(|v| !v.is_finite()) {
            continue;
        }
        // Back-fill occluded entries.
        for p in 0..np {
            if out_mask[[t, p]] {
                out[[t, p, 0]] = reconstructed[p * 3];
                out[[t, p, 1]] = reconstructed[p * 3 + 1];
                out[[t, p, 2]] = reconstructed[p * 3 + 2];
                out_mask[[t, p]] = false;
            }
        }
        pca_success[t] = true;
    }
    (out, out_mask, pca_success)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_fill_recovers_line() {
        let mut x = vec![0.0, 0.0, 0.0, 0.0, 4.0];
        let mut mask = vec![false, true, true, true, false];
        linear_fill_1d(&mut x, &mut mask, 10);
        assert!((x[1] - 1.0).abs() < 1e-12);
        assert!((x[2] - 2.0).abs() < 1e-12);
        assert!((x[3] - 3.0).abs() < 1e-12);
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn linear_fill_respects_max_gap() {
        let mut x = vec![0.0, 0.0, 0.0, 0.0, 4.0];
        let mut mask = vec![false, true, true, true, false];
        linear_fill_1d(&mut x, &mut mask, 2);
        // Gap of size 3 exceeds max_gap=2, so nothing changes.
        assert!(mask[1] && mask[2] && mask[3]);
    }

    #[test]
    fn cubic_fill_polynomial() {
        // y = i^2; cubic spline should reproduce exactly on the gap.
        let mut x: Vec<f64> = (0..10).map(|i| (i as f64).powi(2)).collect();
        let truth = x.clone();
        let mut mask = vec![
            false, false, false, true, true, false, false, false, false, false,
        ];
        // Zero out hidden values to simulate occlusion.
        x[3] = 0.0;
        x[4] = 0.0;
        cubic_fill_1d(&mut x, &mut mask, 5);
        assert!((x[3] - truth[3]).abs() < 1.0);
        assert!((x[4] - truth[4]).abs() < 1.0);
    }
}
