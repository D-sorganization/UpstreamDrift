// Numerical kernels prioritise loop clarity over iterator-chain golf;
// the named indices (`i`, `j`, `k`) match standard signal-processing
// references (SciPy `lfilter`, Savitzky–Golay normal equations, cubic-
// spline tridiagonal solve), and rewriting them with `.iter_mut().enumerate()`
// only obscures the intent.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

//! # upstream-mocap-preproc — Mocap preprocessing kernels
//!
//! High-performance Rust implementations of the motion-pipeline preprocessing
//! kernels: low-pass Butterworth, Savitzky-Golay, Kalman, median, Gaussian
//! filters; linear/cubic/PCA gap-fill; FPS resampling.
//!
//! The Rust side operates on `(n_frames, n_points, n_dims)` float64 arrays;
//! the Python facade in `motion_pipeline/preprocessing/{filter,gap_fill,
//! resample}.py` keeps responsibility for converting to/from
//! `KeypointSequence` / `MarkerTrajectory` contracts.
//!
//! Numerical-fidelity goal: match SciPy's `butter`/`filtfilt`,
//! `savgol_filter`, `medfilt`, `gaussian_filter1d`, and `np.interp` outputs
//! to < 1e-9 RMSE on representative mocap captures.

pub mod filter;
pub mod gap_fill;
pub mod resample;

#[cfg(feature = "python")]
mod bindings;
