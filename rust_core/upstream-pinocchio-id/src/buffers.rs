//! Pre-allocated buffer management for the per-frame inverse-dynamics loop.
//!
//! Allocations are the second-largest source of Python-side overhead in the
//! original implementation (after Python loop dispatch). Pre-sizing all
//! buffers once and reusing single-row scratch arrays for each frame avoids
//! per-frame allocator pressure.

use ndarray::Array2;

/// Pre-allocated workspace for the inverse-dynamics driver loop.
pub struct DriverBuffers {
    pub n_frames: usize,
    pub n_dof: usize,
    /// `(n_frames, n_dof)` joint velocities.
    pub qdot: Array2<f64>,
    /// `(n_frames, n_dof)` joint accelerations.
    pub qddot: Array2<f64>,
    /// `(n_frames, n_dof)` output torques.
    pub tau: Array2<f64>,
}

impl DriverBuffers {
    pub fn new(n_frames: usize, n_dof: usize) -> Self {
        Self {
            n_frames,
            n_dof,
            qdot: Array2::<f64>::zeros((n_frames, n_dof)),
            qddot: Array2::<f64>::zeros((n_frames, n_dof)),
            tau: Array2::<f64>::zeros((n_frames, n_dof)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_have_correct_shape() {
        let b = DriverBuffers::new(1000, 7);
        assert_eq!(b.qdot.dim(), (1000, 7));
        assert_eq!(b.qddot.dim(), (1000, 7));
        assert_eq!(b.tau.dim(), (1000, 7));
        assert_eq!(b.qdot[(0, 0)], 0.0);
    }
}
