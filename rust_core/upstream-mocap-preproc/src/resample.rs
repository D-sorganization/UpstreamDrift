//! FPS resampling via linear interpolation, matching `np.interp` semantics.

use ndarray::{Array1, Array3, ArrayView3};

/// Linear interpolation matching `numpy.interp(xp, fp, x)`. Values of `x`
/// outside the range of `xp` clamp to `fp[0]` / `fp[-1]` (numpy default).
/// Assumes `xp` is monotonically non-decreasing.
pub fn interp_linear(xp: &[f64], fp: &[f64], x: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), fp.len(), "xp and fp must have equal length");
    let n = xp.len();
    let mut out = vec![0.0_f64; x.len()];
    if n == 0 {
        return out;
    }
    if n == 1 {
        for v in out.iter_mut() {
            *v = fp[0];
        }
        return out;
    }
    // Use a single forward sweep when `x` is monotonic, otherwise binary
    // search. We binary-search per query for safety; mocap timestamps are
    // monotonic but this kernel doubles as a generic 1D interpolator.
    for (i, &xi) in x.iter().enumerate() {
        if xi <= xp[0] {
            out[i] = fp[0];
            continue;
        }
        if xi >= xp[n - 1] {
            out[i] = fp[n - 1];
            continue;
        }
        let idx = binary_search(xp, xi);
        let x0 = xp[idx];
        let x1 = xp[idx + 1];
        let y0 = fp[idx];
        let y1 = fp[idx + 1];
        let t = (xi - x0) / (x1 - x0);
        out[i] = y0 + t * (y1 - y0);
    }
    out
}

fn binary_search(xp: &[f64], xi: f64) -> usize {
    let mut lo = 0;
    let mut hi = xp.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] > xi {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// Resample a `(n_frames, n_points, n_dims)` array from `source_timestamps`
/// onto `target_timestamps`, linearly interpolating each `[:, i, j]` series.
pub fn resample_fps(
    data: ArrayView3<f64>,
    source_timestamps: &[f64],
    target_timestamps: &[f64],
) -> Array3<f64> {
    let (nt, np, nd) = data.dim();
    assert_eq!(source_timestamps.len(), nt, "timestamps length mismatch");
    let nt_new = target_timestamps.len();
    let mut out = Array3::<f64>::zeros((nt_new, np, nd));

    let mut series = vec![0.0_f64; nt];
    for p in 0..np {
        for d in 0..nd {
            for t in 0..nt {
                series[t] = data[[t, p, d]];
            }
            let resampled = interp_linear(source_timestamps, &series, target_timestamps);
            for t in 0..nt_new {
                out[[t, p, d]] = resampled[t];
            }
        }
    }
    out
}

/// Helper: compute target timestamps for `target_fps` over the duration of
/// `source_timestamps`. Mirrors the Python facade's `np.linspace` call.
pub fn target_timestamps(source: &[f64], target_fps: f64) -> Array1<f64> {
    if source.len() < 2 {
        return Array1::from(source.to_vec());
    }
    let t0 = source[0];
    let t_end = source[source.len() - 1];
    let duration = t_end - t0;
    let n_new = (duration * target_fps) as usize + 1;
    let mut out = Array1::<f64>::zeros(n_new);
    if n_new <= 1 {
        out[0] = t0;
        return out;
    }
    let step = duration / (n_new - 1) as f64;
    for i in 0..n_new {
        out[i] = t0 + step * i as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interp_linear_matches_numpy() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![0.0, 10.0, 20.0, 30.0];
        let x = vec![-1.0, 0.0, 0.5, 1.5, 2.99, 3.0, 4.0];
        let got = interp_linear(&xp, &fp, &x);
        let want = [0.0, 0.0, 5.0, 15.0, 29.9, 30.0, 30.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-12, "got {} want {}", g, w);
        }
    }

    #[test]
    fn resample_constant_signal_stays_constant() {
        let nt = 10;
        let np = 2;
        let nd = 3;
        let data = Array3::<f64>::from_elem((nt, np, nd), 7.5);
        let src: Vec<f64> = (0..nt).map(|i| i as f64 * 0.01).collect();
        let tgt: Vec<f64> = (0..30).map(|i| i as f64 * 0.003).collect();
        let out = resample_fps(data.view(), &src, &tgt);
        for v in out.iter() {
            assert!((v - 7.5).abs() < 1e-12);
        }
    }
}
