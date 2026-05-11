//! Filter kernels: Butterworth, Savitzky-Golay, Kalman, median, Gaussian.
//!
//! All filters operate per-`(point, dim)` series on data shaped
//! `(n_frames, n_points, n_dims)` and return arrays of the same shape.

use ndarray::{Array1, Array3, ArrayView3};

// ── Butterworth ──────────────────────────────────────────────────────────────

/// Design a low-pass digital Butterworth filter (transfer-function form).
///
/// Matches SciPy's `scipy.signal.butter(order, Wn, btype='low')` to machine
/// precision for low orders (≤ 8) we use in mocap. Returns `(b, a)`
/// coefficients of length `order + 1`.
///
/// Algorithm: analog Butterworth poles on the unit circle in the LHP, scaled
/// to the prewarped cutoff, then bilinear-transformed to the digital plane.
pub fn butter_lowpass(order: usize, normalized_cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(order >= 1, "order must be >= 1");
    assert!(
        normalized_cutoff > 0.0 && normalized_cutoff < 1.0,
        "normalized_cutoff must be in (0, 1)"
    );

    // Prewarp: analog cutoff that bilinear maps to the requested digital one.
    let fs = 2.0_f64;
    let warped = 2.0 * fs * (std::f64::consts::PI * normalized_cutoff / fs).tan();

    // Analog Butterworth lowpass prototype poles (unit cutoff): on the unit
    // circle in the LHP at angles theta_k = pi/2 + (2k - 1) * pi / (2N).
    // Use complex arithmetic.
    let mut analog_poles: Vec<(f64, f64)> = Vec::with_capacity(order);
    for k in 1..=order {
        let theta = std::f64::consts::PI * (2.0 * k as f64 - 1.0) / (2.0 * order as f64)
            + std::f64::consts::FRAC_PI_2;
        analog_poles.push((theta.cos(), theta.sin()));
    }
    // Scale poles by warped cutoff.
    for p in analog_poles.iter_mut() {
        p.0 *= warped;
        p.1 *= warped;
    }
    // Lowpass prototype gain: product(-p_k) so that H(0) = 1.
    // Compute prod(-p) as a complex number.
    let mut analog_gain_re = 1.0_f64;
    let mut analog_gain_im = 0.0_f64;
    for &(pr, pi) in &analog_poles {
        let (nr, ni) = (-pr, -pi);
        let new_re = analog_gain_re * nr - analog_gain_im * ni;
        let new_im = analog_gain_re * ni + analog_gain_im * nr;
        analog_gain_re = new_re;
        analog_gain_im = new_im;
    }
    // analog_gain should be real; take magnitude (sign handled by pole pairing).
    let analog_gain = (analog_gain_re * analog_gain_re + analog_gain_im * analog_gain_im).sqrt();

    // Bilinear transform: z_pole = (fs2 + p) / (fs2 - p), with fs2 = 2*fs.
    let fs2 = 2.0 * fs;
    let mut z_poles: Vec<(f64, f64)> = Vec::with_capacity(order);
    let mut bilinear_gain_factor_re = 1.0_f64;
    let mut bilinear_gain_factor_im = 0.0_f64;
    for &(pr, pi) in &analog_poles {
        let num_re = fs2 + pr;
        let num_im = pi;
        let den_re = fs2 - pr;
        let den_im = -pi;
        let (qr, qi) = complex_div(num_re, num_im, den_re, den_im);
        z_poles.push((qr, qi));
        // gain factor: 1 / (fs2 - p) accumulated
        let (br, bi) = complex_div(1.0, 0.0, den_re, den_im);
        let new_re = bilinear_gain_factor_re * br - bilinear_gain_factor_im * bi;
        let new_im = bilinear_gain_factor_re * bi + bilinear_gain_factor_im * br;
        bilinear_gain_factor_re = new_re;
        bilinear_gain_factor_im = new_im;
    }
    // All analog zeros are at infinity for a lowpass; bilinear maps them all
    // to z = -1, giving `order` zeros at -1.
    let z_zeros: Vec<(f64, f64)> = (0..order).map(|_| (-1.0, 0.0)).collect();

    // Digital gain = analog_gain * prod(1/(fs2 - p)).
    let mut k_re = analog_gain * bilinear_gain_factor_re;
    let mut k_im = analog_gain * bilinear_gain_factor_im;

    // Build polynomials: a = poly(z_poles), b = K * poly(z_zeros).
    let a_coeffs = poly_from_roots(&z_poles);
    let mut b_coeffs = poly_from_roots(&z_zeros);

    // Enforce DC unity gain to absorb any tiny imaginary residue from
    // floating-point poly multiplication. H(z=1) = sum(b) / sum(a). We want 1.
    let sum_a: f64 = a_coeffs.iter().sum();
    let sum_b_unscaled: f64 = b_coeffs.iter().sum();
    let scale = if sum_b_unscaled.abs() > 1e-300 {
        sum_a / sum_b_unscaled
    } else {
        // Fall back to the computed complex gain magnitude.
        let _ = (&mut k_re, &mut k_im);
        (k_re * k_re + k_im * k_im).sqrt()
    };
    for c in b_coeffs.iter_mut() {
        *c *= scale;
    }

    (b_coeffs, a_coeffs)
}

fn complex_div(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let den = br * br + bi * bi;
    ((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)
}

/// Build a real polynomial coefficient array from a set of complex roots
/// (which must occur in conjugate pairs for the result to be real-valued).
fn poly_from_roots(roots: &[(f64, f64)]) -> Vec<f64> {
    // Multiply out (z - r0)(z - r1)... using complex coefficients, then take
    // the real part (imaginary should be ~0 by conjugate-pair symmetry).
    let mut coeffs: Vec<(f64, f64)> = vec![(1.0, 0.0)];
    for &(rr, ri) in roots {
        // multiply current poly by (z - r)
        let mut new_coeffs: Vec<(f64, f64)> = vec![(0.0, 0.0); coeffs.len() + 1];
        for (i, &(cr, ci)) in coeffs.iter().enumerate() {
            // z * c_i contributes to coefficient i+1
            new_coeffs[i].0 += cr;
            new_coeffs[i].1 += ci;
            // -r * c_i contributes to coefficient i (highest-degree first
            // convention: index 0 is z^n)
            let (mr, mi) = (cr * -rr - ci * -ri, cr * -ri + ci * -rr);
            new_coeffs[i + 1].0 += mr;
            new_coeffs[i + 1].1 += mi;
        }
        coeffs = new_coeffs;
    }
    coeffs.into_iter().map(|(r, _i)| r).collect()
}

/// Forward-only IIR filter (direct-form II transposed), matching
/// `scipy.signal.lfilter(b, a, x)` exactly.
fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    // Normalize so a[0] == 1.
    let a0 = a[0];
    let n = x.len();
    let order = b.len().max(a.len()) - 1;
    let mut z = vec![0.0_f64; order];
    let mut y = vec![0.0_f64; n];

    for i in 0..n {
        let xi = x[i];
        let yi = (b[0] / a0) * xi + (if order > 0 { z[0] } else { 0.0 });
        y[i] = yi;
        for k in 0..order {
            let b_k1 = if k + 1 < b.len() { b[k + 1] } else { 0.0 };
            let a_k1 = if k + 1 < a.len() { a[k + 1] } else { 0.0 };
            let next = if k + 1 < order { z[k + 1] } else { 0.0 };
            z[k] = (b_k1 / a0) * xi - (a_k1 / a0) * yi + next;
        }
    }
    y
}

/// Compute the initial conditions `zi` used by SciPy's `filtfilt` "pad" method
/// to start the filter in steady state. Mirrors `scipy.signal.lfilter_zi`.
fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len());
    let mut a_pad = vec![0.0_f64; n];
    let mut b_pad = vec![0.0_f64; n];
    a_pad[..a.len()].copy_from_slice(a);
    b_pad[..b.len()].copy_from_slice(b);
    // Normalize.
    let a0 = a_pad[0];
    for v in a_pad.iter_mut() {
        *v /= a0;
    }
    for v in b_pad.iter_mut() {
        *v /= a0;
    }
    let order = n - 1;
    if order == 0 {
        return Vec::new();
    }

    // Solve (I - A) * zi = B where A is companion-like.
    // SciPy uses: zi = (eye(n-1) - linalg.companion(a).T) \ (b[1:] - b[0]*a[1:])
    // Build matrix M = I - A^T where A is the companion of `a`.
    // The companion matrix of a (with a[0]=1) is:
    //   A = [[-a1, -a2, ..., -a_{n-1}],
    //        [ 1,   0, ...,   0     ],
    //        [ 0,   1, ...,   0     ],
    //        ...
    //        [ 0,   0, ...,   1, 0  ]]
    // A^T is its transpose. We need (I - A^T) * zi = rhs.
    let m = order;
    let mut mat = vec![vec![0.0_f64; m]; m];
    // I
    for i in 0..m {
        mat[i][i] = 1.0;
    }
    // -A^T: subtract A^T from I.
    // A[i][j] entries:
    //   row 0: A[0][j] = -a[j+1] for j in 0..m
    //   row i>0: A[i][i-1] = 1, else 0.
    // So A^T[i][j] = A[j][i].
    for i in 0..m {
        for j in 0..m {
            let a_ji = if j == 0 {
                -a_pad[i + 1]
            } else if i == j - 1 {
                1.0
            } else {
                0.0
            };
            mat[i][j] -= a_ji;
        }
    }
    let rhs: Vec<f64> = (0..m)
        .map(|i| b_pad[i + 1] - b_pad[0] * a_pad[i + 1])
        .collect();
    solve_linear(mat, rhs).unwrap_or_else(|| vec![0.0; m])
}

/// Gaussian-elimination solver for small dense systems. Returns `None` if
/// singular.
pub(crate) fn solve_linear_view(a: Vec<Vec<f64>>, b: Vec<f64>) -> Option<Vec<f64>> {
    solve_linear(a, b)
}

fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        // Partial pivot.
        let mut pivot = i;
        let mut pivot_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > pivot_val {
                pivot = k;
                pivot_val = a[k][i].abs();
            }
        }
        if pivot_val < 1e-300 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);
        let piv = a[i][i];
        for k in (i + 1)..n {
            let factor = a[k][i] / piv;
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// SciPy-compatible `filtfilt` with default `padtype='odd'` and
/// `padlen = 3 * max(len(a), len(b))`. Zero-phase forward-backward filter.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let padlen = 3 * a.len().max(b.len());
    if n <= padlen {
        // SciPy raises in this case; we degrade to a single forward+backward
        // pass without padding so short signals do not crash the pipeline.
        return filtfilt_no_pad(b, a, x);
    }

    // Odd extension: ext_left[i] = 2*x[0] - x[padlen - i]
    //                ext_right[i] = 2*x[n-1] - x[n-2-i]
    let mut ext: Vec<f64> = Vec::with_capacity(n + 2 * padlen);
    for i in 0..padlen {
        ext.push(2.0 * x[0] - x[padlen - i]);
    }
    ext.extend_from_slice(x);
    for i in 0..padlen {
        ext.push(2.0 * x[n - 1] - x[n - 2 - i]);
    }

    let zi = lfilter_zi(b, a);
    // Forward pass.
    let x0 = ext[0];
    let mut z: Vec<f64> = zi.iter().map(|&v| v * x0).collect();
    let forward = lfilter_ic(b, a, &ext, &mut z);

    // Backward pass: reverse, filter, reverse.
    let rev: Vec<f64> = forward.iter().rev().copied().collect();
    let r0 = rev[0];
    let mut z2: Vec<f64> = zi.iter().map(|&v| v * r0).collect();
    let backward = lfilter_ic(b, a, &rev, &mut z2);
    let mut out: Vec<f64> = backward.iter().rev().copied().collect();

    // Trim padding.
    out.drain(0..padlen);
    out.truncate(n);
    out
}

fn filtfilt_no_pad(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let forward = lfilter(b, a, x);
    let rev: Vec<f64> = forward.iter().rev().copied().collect();
    let back = lfilter(b, a, &rev);
    back.iter().rev().copied().collect()
}

/// `lfilter` with explicit initial state, mirroring SciPy's `lfilter(..., zi=...)`.
fn lfilter_ic(b: &[f64], a: &[f64], x: &[f64], z: &mut [f64]) -> Vec<f64> {
    let a0 = a[0];
    let n = x.len();
    let order = b.len().max(a.len()) - 1;
    assert_eq!(z.len(), order);
    let mut y = vec![0.0_f64; n];

    for i in 0..n {
        let xi = x[i];
        let yi = (b[0] / a0) * xi + (if order > 0 { z[0] } else { 0.0 });
        y[i] = yi;
        for k in 0..order {
            let b_k1 = if k + 1 < b.len() { b[k + 1] } else { 0.0 };
            let a_k1 = if k + 1 < a.len() { a[k + 1] } else { 0.0 };
            let next = if k + 1 < order { z[k + 1] } else { 0.0 };
            z[k] = (b_k1 / a0) * xi - (a_k1 / a0) * yi + next;
        }
    }
    y
}

/// Apply a low-pass Butterworth `filtfilt` to every `[:, i, j]` series.
pub fn butterworth_filter(
    data: ArrayView3<f64>,
    cutoff_hz: f64,
    order: usize,
    fps: f64,
) -> Array3<f64> {
    let nyquist = fps / 2.0;
    let mut normalized = cutoff_hz / nyquist;
    if normalized >= 1.0 {
        normalized = 0.99;
    }
    if normalized <= 0.0 {
        normalized = 1e-6;
    }
    let (b, a) = butter_lowpass(order, normalized);
    apply_per_series(data, |s| filtfilt(&b, &a, s))
}

// ── Savitzky-Golay ───────────────────────────────────────────────────────────

/// Compute Savitzky-Golay FIR coefficients for the center sample with
/// `mode='interp'` semantics (SciPy default). Returns length-`window_length`
/// coefficients.
fn savgol_coeffs_center(window_length: usize, polyorder: usize) -> Vec<f64> {
    // Solve (V^T V) c = V^T e_center, where V is the Vandermonde of shifted
    // sample indices x = -halflen .. halflen, columns 0..=polyorder.
    let n = window_length;
    let half = (n / 2) as isize;
    let p = polyorder + 1;
    let mut v = vec![vec![0.0_f64; p]; n];
    for i in 0..n {
        let x = (i as isize - half) as f64;
        let mut acc = 1.0;
        for j in 0..p {
            v[i][j] = acc;
            acc *= x;
        }
    }
    // Build normal equations A = V^T V (p x p), b = V^T e_center
    let mut a = vec![vec![0.0_f64; p]; p];
    for j in 0..p {
        for k in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += v[i][j] * v[i][k];
            }
            a[j][k] = s;
        }
    }
    let center = (n / 2) as usize;
    let rhs: Vec<f64> = (0..p).map(|j| v[center][j]).collect();
    let coef_vec = solve_linear(a, rhs).expect("savgol normal equations should be invertible");
    // Apply each row of V to coef_vec.
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            s += v[i][j] * coef_vec[j];
        }
        out[i] = s;
    }
    out
}

/// Polynomial-fit boundary handling for SciPy `savgol_filter(mode='interp')`:
/// fits a polynomial of degree `polyorder` to the first / last `window_length`
/// samples, then evaluates it to fill in the half-window edges that the FIR
/// kernel cannot reach.
fn savgol_edge_polyfit(window: &[f64], polyorder: usize, eval_positions: &[f64]) -> Vec<f64> {
    let n = window.len();
    let p = polyorder + 1;
    let mut v = vec![vec![0.0_f64; p]; n];
    for i in 0..n {
        let mut acc = 1.0;
        for j in 0..p {
            v[i][j] = acc;
            acc *= i as f64;
        }
    }
    // Normal equations.
    let mut a = vec![vec![0.0_f64; p]; p];
    for j in 0..p {
        for k in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += v[i][j] * v[i][k];
            }
            a[j][k] = s;
        }
    }
    let mut rhs = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += v[i][j] * window[i];
        }
        rhs[j] = s;
    }
    let coeffs = solve_linear(a, rhs).expect("polyfit should be invertible");
    eval_positions
        .iter()
        .map(|&pos| {
            let mut acc = 1.0;
            let mut sum = 0.0;
            for c in &coeffs {
                sum += c * acc;
                acc *= pos;
            }
            sum
        })
        .collect()
}

/// SciPy-compatible 1D Savitzky-Golay filter (mode='interp', deriv=0, delta=1).
pub fn savgol_1d(x: &[f64], window_length: usize, polyorder: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if window_length > n {
        // SciPy raises; degrade to identity.
        return x.to_vec();
    }
    let coeffs = savgol_coeffs_center(window_length, polyorder);
    let half = window_length / 2;
    let mut out = vec![0.0_f64; n];
    // Interior: cross-correlation with `coeffs`.
    for i in half..(n - half) {
        let mut s = 0.0;
        for (k, c) in coeffs.iter().enumerate() {
            s += c * x[i - half + k];
        }
        out[i] = s;
    }
    // Edges via polyfit on each end's first/last `window_length` samples.
    if half > 0 {
        let left_window = &x[0..window_length];
        let left_positions: Vec<f64> = (0..half).map(|i| i as f64).collect();
        let left_vals = savgol_edge_polyfit(left_window, polyorder, &left_positions);
        for (i, v) in left_vals.into_iter().enumerate() {
            out[i] = v;
        }
        let right_window = &x[(n - window_length)..n];
        let right_positions: Vec<f64> = (0..half)
            .map(|i| (window_length - half + i) as f64)
            .collect();
        let right_vals = savgol_edge_polyfit(right_window, polyorder, &right_positions);
        for (i, v) in right_vals.into_iter().enumerate() {
            out[n - half + i] = v;
        }
    }
    out
}

pub fn savgol_filter(data: ArrayView3<f64>, window_length: usize, polyorder: usize) -> Array3<f64> {
    // Mirror Python facade: ensure odd window > polyorder.
    let mut wl = window_length;
    if wl % 2 == 0 {
        wl += 1;
    }
    if wl <= polyorder {
        wl = polyorder + 2;
        if wl % 2 == 0 {
            wl += 1;
        }
    }
    apply_per_series(data, |s| savgol_1d(s, wl, polyorder))
}

// ── Median ───────────────────────────────────────────────────────────────────

/// 1D median filter matching SciPy's `scipy.signal.medfilt(x, kernel_size)`:
/// zero-padded on both ends, odd kernel size.
pub fn medfilt_1d(x: &[f64], kernel_size: usize) -> Vec<f64> {
    let k = if kernel_size % 2 == 0 {
        kernel_size + 1
    } else {
        kernel_size
    };
    let half = k / 2;
    let n = x.len();
    let mut out = vec![0.0_f64; n];
    let mut window = vec![0.0_f64; k];
    for i in 0..n {
        for j in 0..k {
            let idx = i as isize + j as isize - half as isize;
            window[j] = if idx < 0 || idx >= n as isize {
                0.0
            } else {
                x[idx as usize]
            };
        }
        // Sort to find median.
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[i] = window[half];
    }
    out
}

pub fn median_filter(data: ArrayView3<f64>, kernel_size: usize) -> Array3<f64> {
    apply_per_series(data, |s| medfilt_1d(s, kernel_size))
}

// ── Gaussian ─────────────────────────────────────────────────────────────────

/// 1D Gaussian filter matching `scipy.ndimage.gaussian_filter1d(x, sigma)`
/// with default `truncate=4.0`, `mode='reflect'`, `order=0`.
pub fn gaussian_filter_1d(x: &[f64], sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 || x.is_empty() {
        return x.to_vec();
    }
    let radius = (4.0 * sigma + 0.5) as usize;
    // Build kernel.
    let mut kernel = vec![0.0_f64; 2 * radius + 1];
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for i in 0..kernel.len() {
        let d = i as f64 - radius as f64;
        let v = (-(d * d) / two_sigma2).exp();
        kernel[i] = v;
        sum += v;
    }
    for c in kernel.iter_mut() {
        *c /= sum;
    }
    // Reflect mode (scipy default): index i' = reflect(i, n).
    let n = x.len();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for (k, &c) in kernel.iter().enumerate() {
            let raw = i as isize + k as isize - radius as isize;
            let idx = reflect_index(raw, n);
            s += c * x[idx];
        }
        out[i] = s;
    }
    out
}

fn reflect_index(mut i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let n_i = n as isize;
    let period = 2 * n_i;
    i = i.rem_euclid(period);
    // scipy 'reflect' mode reflects about the edge of the last pixel:
    // pattern is (d c b a | a b c d | d c b a). We use the same convention
    // (mirror) since that's the documented default.
    if i >= n_i {
        i = period - 1 - i;
    }
    i as usize
}

pub fn gaussian_filter(data: ArrayView3<f64>, sigma: f64) -> Array3<f64> {
    apply_per_series(data, |s| gaussian_filter_1d(s, sigma))
}

// ── Kalman (1D random-walk smoother) ─────────────────────────────────────────

/// 1D random-walk Kalman filter matching the Python `_kalman_filter`
/// implementation (state-dim 1, measurement-dim 1, F=H=I, Q, R, P0=1).
pub fn kalman_1d(x: &[f64], process_noise: f64, measurement_noise: f64) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let q = process_noise;
    let r = measurement_noise;
    let mut state = x[0];
    let mut p = 1.0_f64;
    let mut out = vec![0.0_f64; x.len()];
    for (t, &z) in x.iter().enumerate() {
        // Predict: state stays (F=I), P += Q
        p += q;
        // Update with measurement z
        let k = p / (p + r);
        state += k * (z - state);
        p *= 1.0 - k;
        out[t] = state;
    }
    out
}

pub fn kalman_filter(
    data: ArrayView3<f64>,
    process_noise: f64,
    measurement_noise: f64,
) -> Array3<f64> {
    apply_per_series(data, |s| kalman_1d(s, process_noise, measurement_noise))
}

// ── Per-series dispatch ──────────────────────────────────────────────────────

/// Apply `f` independently to each `[:, i, j]` series of the 3D array,
/// preserving shape. Hot path; uses rayon to parallelize across the
/// `n_points × n_dims` independent series.
pub fn apply_per_series<F>(data: ArrayView3<f64>, f: F) -> Array3<f64>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync + Send,
{
    use rayon::prelude::*;
    let (nt, np, nd) = data.dim();
    let total = np * nd;

    // Collect (point_idx, dim_idx, filtered_series) in parallel, then stitch
    // back into a single Array3. Each thread allocates its own input/output
    // buffers — small and short-lived.
    let results: Vec<Vec<f64>> = (0..total)
        .into_par_iter()
        .map(|s| {
            let i = s / nd;
            let j = s % nd;
            let mut buf = vec![0.0_f64; nt];
            for t in 0..nt {
                buf[t] = data[[t, i, j]];
            }
            f(&buf)
        })
        .collect();

    let mut out = Array3::<f64>::zeros((nt, np, nd));
    for s in 0..total {
        let i = s / nd;
        let j = s % nd;
        let series = &results[s];
        for t in 0..nt {
            out[[t, i, j]] = series[t];
        }
    }
    out
}

// Re-export an axis helper for tests that operate on Array1.
pub fn filter_1d_into_array(x: &[f64]) -> Array1<f64> {
    Array1::from_vec(x.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butter_lowpass_order2_dc_unity() {
        let (b, a) = butter_lowpass(2, 0.1);
        // DC gain = sum(b)/sum(a) should be 1.
        let sum_b: f64 = b.iter().sum();
        let sum_a: f64 = a.iter().sum();
        assert!((sum_b / sum_a - 1.0).abs() < 1e-12);
    }

    #[test]
    fn filtfilt_preserves_dc() {
        let (b, a) = butter_lowpass(2, 0.1);
        let x = vec![1.5_f64; 100];
        let y = filtfilt(&b, &a, &x);
        for v in y {
            assert!((v - 1.5).abs() < 1e-10, "got {}", v);
        }
    }

    #[test]
    fn savgol_passes_polynomial() {
        // SavGol of order 2 should reproduce a quadratic exactly.
        let x: Vec<f64> = (0..50)
            .map(|i| 1.0 + 2.0 * i as f64 + 0.5 * (i as f64).powi(2))
            .collect();
        let y = savgol_1d(&x, 11, 2);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-8, "got {} vs {}", a, b);
        }
    }

    #[test]
    fn medfilt_constant() {
        let x = vec![3.0_f64; 20];
        let y = medfilt_1d(&x, 3);
        // Edges become 0 due to zero padding (median of {0, 3, 3} = 3 still).
        // Actually with k=3 and zero-pad, edge of length-20 series:
        // window at i=0 is {0, 3, 3}, median 3.
        for v in y {
            assert!((v - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gaussian_smooth_constant() {
        let x = vec![5.0_f64; 30];
        let y = gaussian_filter_1d(&x, 1.5);
        for v in y {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn kalman_constant() {
        let x = vec![2.0_f64; 50];
        let y = kalman_1d(&x, 0.01, 0.1);
        assert!((y[49] - 2.0).abs() < 1e-3);
    }
}
