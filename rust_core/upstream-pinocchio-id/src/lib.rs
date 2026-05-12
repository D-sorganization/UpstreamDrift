// Numerical kernels favour explicit indexed loops to match the Python
// reference in `motion_pipeline/matching/inverse_dyn_pinocchio.py`.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! # upstream-pinocchio-id — Rust outer loop for Pinocchio inverse-dynamics
//!
//! Pinocchio is a C++ library exposed to Python via bindings. The bottleneck
//! in `motion_pipeline/matching/inverse_dyn_pinocchio.py` is *not* the inner
//! `pin.rnea` call itself but the surrounding per-frame Python orchestration:
//! finite-difference `qdot`/`qddot` from `q`, buffer slicing per frame, and
//! result aggregation into `TorqueFrame` lists.
//!
//! This crate moves that outer loop into Rust:
//!
//! - [`finite_diff`] computes central/forward/backward finite differences on
//!   `(n_frames, n_dof)` joint-position arrays exactly matching the Python
//!   reference scheme (non-uniform `dt` aware).
//! - [`buffers`] manages pre-allocated `q`/`qdot`/`qddot`/`tau` storage so the
//!   driver loop allocates nothing per frame.
//! - [`driver`] orchestrates the per-frame loop. It does *not* depend on
//!   Pinocchio's C++ libs: callers provide an `rnea`-shaped closure (in the
//!   PyO3 binding, that closure invokes `pinocchio.rnea` from Python).
//!
//! Numerical-fidelity goal: tau outputs match the pure-Python reference to
//! <1e-9 RMSE on a synthetic 1000-frame trajectory.

pub mod buffers;
pub mod driver;
pub mod finite_diff;

#[cfg(feature = "python")]
mod bindings;
