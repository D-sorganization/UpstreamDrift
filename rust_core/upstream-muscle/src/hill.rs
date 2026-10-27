//! Hill-type muscle curves — pure scalar functions.
//!
//! Direct port of `src/shared/python/biomechanics/hill_muscle.py` so that
//! parameter names and constants match 1:1 with the Python source. These are
//! the building blocks of the contractile element (CE), parallel elastic
//! element (PEE), and series elastic element (SEE/tendon).
//!
//! References:
//! - Hill (1938), "The Heat of Shortening and the Dynamic Constants of Muscle"
//! - Zajac (1989), "Muscle and Tendon: Properties, Models, Scaling..."
//! - Thelen (2003), J. Biomech. Eng., 125(1), pp. 70-77
//!
//! Numerical parity vs the Python source is asserted within 1e-6 in
//! `tests/parity_hill.rs`.
//!
//! Slice 1 of UD#5216: pure functions only, no muscle state, no solver,
//! no batched / parallel API. Those land in subsequent slices.

/// Default width of the active force-length Gaussian curve, dimensionless.
///
/// From Thelen (2003), J. Biomech. Eng., 125(1), pp. 70-77; matches
/// `HillMuscleModel.DEFAULT_FORCE_LENGTH_GAMMA = 0.45` in the Python source.
pub const DEFAULT_FORCE_LENGTH_WIDTH: f64 = 0.45;
pub const DEFAULT_FORCE_LENGTH_GAMMA: f64 = 0.45;

/// Stiffness coefficient of the passive (PEE) exponential spring in
/// `force_length_passive`. Matches `k_passive = 4.0` in the Python source.
pub const K_PASSIVE: f64 = 4.0;

/// Reference tendon strain at F_max (~0.04 per Thelen 2003 / 0.033 Zajac 1989).
pub const DEFAULT_TENDON_STRAIN_AT_FMAX: f64 = 0.04;

/// Hill hyperbola shape parameter for the concentric branch of `f_v`.
/// Matches the literal `0.25` in the Python source's denominator.
pub const HILL_A_CONCENTRIC: f64 = 0.25;

/// Hill clamp on normalized concentric velocity to keep the denominator
/// `1 - v_norm/0.25` strictly positive. Matches `max(v_norm, -0.99)` in the
/// Python source.
pub const V_CONCENTRIC_CLAMP: f64 = -0.99;

/// Eccentric force asymptote multiplier (~1.4 × F_max plateau).
/// Matches the literal `1.4` in the Python source.
pub const F_ECCENTRIC_ASYMPTOTE: f64 = 1.4;

/// Eccentric Hill shape parameter; matches the literal `0.10` in the Python
/// source's `(1 + v_norm * 1.4 / 0.10) / (1 + v_norm / 0.10)`.
pub const ECCENTRIC_SHAPE: f64 = 0.10;

// ── Active force-length ──────────────────────────────────────────────────────

/// Active contractile force-length curve, `f_l(l_norm)`.
///
/// Gaussian-like multiplier on isometric force as a function of normalized
/// fiber length `l_CE / l_opt`. Returns a value in `[0, 1]`.
///
/// Equivalent to `HillMuscleModel.force_length_active` with the default
/// `force_length_width` of [`DEFAULT_FORCE_LENGTH_WIDTH`].
#[inline]
pub fn f_l(l_norm: f64) -> f64 {
    f_l_with_gamma(l_norm, DEFAULT_FORCE_LENGTH_GAMMA)
}

/// Variant of [`f_l`] that takes an explicit Gaussian shape parameter gamma.
///
/// Mirrors the `force_length_gamma` (or `force_length_width`) constructor argument on the
/// Python `HillMuscleModel`.
#[inline]
pub fn f_l_with_gamma(l_norm: f64, gamma: f64) -> f64 {
    if l_norm <= 0.0 {
        return 0.0;
    }
    let d = l_norm - 1.0;
    (-(d * d) / gamma).exp()
}

/// Backward compatible alias for [`f_l_with_gamma`].
#[inline]
pub fn f_l_with_width(l_norm: f64, width: f64) -> f64 {
    f_l_with_gamma(l_norm, width)
}

// ── Passive force-length (PEE) ───────────────────────────────────────────────

/// Passive parallel-elastic force-length curve, `f_p(l_norm)`.
///
/// Returns 0 below optimal length; an exponential spring above. Equivalent
/// to `HillMuscleModel.force_length_passive`.
///
/// Not part of the slice-1 parity surface (`f_l`, `f_v`, `f_t`) but provided
/// so downstream slices can consume it without re-porting.
#[inline]
pub fn f_p(l_norm: f64) -> f64 {
    if l_norm <= 1.0 {
        return 0.0;
    }
    let num = (K_PASSIVE * (l_norm - 1.0)).exp() - 1.0;
    let den = K_PASSIVE.exp() - 1.0;
    num / den
}

// ── Force-velocity ───────────────────────────────────────────────────────────

/// Force-velocity curve, `f_v(v_norm)`.
///
/// `v_norm` is normalized by `v_max * l_opt` (positive = lengthening /
/// eccentric, negative = shortening / concentric). Returns the Hill
/// hyperbola for the concentric branch and the standard eccentric plateau
/// for `v_norm >= 0`.
///
/// Equivalent to `HillMuscleModel.force_velocity`.
#[inline]
pub fn f_v(v_norm: f64) -> f64 {
    if v_norm < 0.0 {
        // Concentric. Clamp to keep the denominator strictly positive (the
        // Python source uses max(v_norm, -0.99) for the same reason).
        let v = if v_norm > V_CONCENTRIC_CLAMP {
            v_norm
        } else {
            V_CONCENTRIC_CLAMP
        };
        (1.0 + v) / (1.0 - v / HILL_A_CONCENTRIC)
    } else {
        // Eccentric.
        (1.0 + v_norm * F_ECCENTRIC_ASYMPTOTE / ECCENTRIC_SHAPE) / (1.0 + v_norm / ECCENTRIC_SHAPE)
    }
}

// ── Tendon (SEE) ─────────────────────────────────────────────────────────────

/// Tendon (series-elastic) force-length curve, `f_t(l_tendon_norm)`.
///
/// `l_tendon_norm = l_tendon / l_slack`. Below slack the tendon carries no
/// force; above slack the cited Thelen (2003) / Zajac (1989) tendon curve
/// applies, reaching F_max at reference strain `DEFAULT_TENDON_STRAIN_AT_FMAX`.
///
/// Equivalent to `HillMuscleModel.tendon_force`.
#[inline]
pub fn f_t(l_tendon_norm: f64) -> f64 {
    f_t_with_reference_strain(l_tendon_norm, DEFAULT_TENDON_STRAIN_AT_FMAX)
}

/// Tendon force with customizable reference strain epsilon_0.
#[inline]
pub fn f_t_with_reference_strain(l_tendon_norm: f64, eps_0: f64) -> f64 {
    if l_tendon_norm <= 1.0 {
        return 0.0;
    }
    let strain = l_tendon_norm - 1.0;
    let f_toe = 0.33;
    let k_toe = 3.0;
    let eps_toe = 0.609 * eps_0;
    let k_lin = 1.75 / eps_0;
    if strain <= eps_toe {
        f_toe * ((k_toe * strain / eps_toe).exp() - 1.0) / (k_toe.exp() - 1.0)
    } else {
        f_toe + k_lin * (strain - eps_toe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn f_l_peaks_at_optimal_length() {
        assert_relative_eq!(f_l(1.0), 1.0, epsilon = 1e-12);
        assert!(f_l(0.5) < 1.0);
        assert!(f_l(1.5) < 1.0);
    }

    #[test]
    fn f_l_at_1_4_matches_thelen_2003() {
        assert_relative_eq!(f_l(1.4), 0.7007827, epsilon = 1e-4);
    }

    #[test]
    fn f_l_is_symmetric_about_optimum() {
        for d in [0.05, 0.1, 0.2, 0.3] {
            assert_relative_eq!(f_l(1.0 - d), f_l(1.0 + d), epsilon = 1e-12);
        }
    }

    #[test]
    fn f_p_zero_below_slack() {
        assert_eq!(f_p(0.5), 0.0);
        assert_eq!(f_p(1.0), 0.0);
    }

    #[test]
    fn f_p_monotonic_above_slack() {
        let a = f_p(1.05);
        let b = f_p(1.10);
        let c = f_p(1.20);
        assert!(a > 0.0 && a < b && b < c);
    }

    #[test]
    fn f_v_isometric_is_one() {
        assert_relative_eq!(f_v(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn f_v_concentric_below_one() {
        // Shortening => below isometric.
        assert!(f_v(-0.1) < 1.0);
        assert!(f_v(-0.5) < f_v(-0.1));
    }

    #[test]
    fn f_v_concentric_clamps_at_0_99() {
        // The Python source clamps v_norm to -0.99; values past the clamp
        // must produce the same result as exactly -0.99.
        let clamped = f_v(V_CONCENTRIC_CLAMP);
        assert_relative_eq!(f_v(-1.5), clamped, epsilon = 1e-12);
        assert_relative_eq!(f_v(-5.0), clamped, epsilon = 1e-12);
    }

    #[test]
    fn f_v_eccentric_above_one() {
        // Lengthening => above isometric, approaching the 1.4 plateau.
        assert!(f_v(0.1) > 1.0);
        assert!(f_v(10.0) < F_ECCENTRIC_ASYMPTOTE + 1e-9);
        assert!(f_v(10.0) > 1.3);
    }

    #[test]
    fn f_t_zero_below_slack() {
        assert_eq!(f_t(0.5), 0.0);
        assert_eq!(f_t(1.0), 0.0);
    }

    #[test]
    fn f_t_at_reference_strain_near_fmax() {
        // At reference strain 1.04, F_T ≈ 1.0 (within 2%)
        assert_relative_eq!(f_t(1.04), 1.01425, epsilon = 1e-4);
        assert!(f_t(1.05) > f_t(1.04));
    }
}
