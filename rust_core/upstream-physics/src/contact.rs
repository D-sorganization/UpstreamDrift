//! Contact dynamics for ball-surface impacts.
//!
//! Implements a coefficient-of-restitution (COR) model with Coulomb friction
//! coupled to scalar backspin about an axis perpendicular to the contact normal
//! and primary tangential slip direction.
//!
//! # Design by Contract
//! - COR must be in [0, 1]
//! - Friction coefficient must be non-negative
//! - Incoming speed must be positive

use serde::{Deserialize, Serialize};
use tools_core::Vector3;

/// Speed below which a ball is considered to be rolling [m/s].
const ROLLING_SPEED_THRESHOLD: f64 = 0.5;

/// Minimum magnitude for a valid surface normal vector.
const MIN_NORMAL_MAGNITUDE: f64 = 1e-10;

/// Minimum tangential speed used to pick a friction axis [m/s].
const MIN_TANGENTIAL_SPEED: f64 = 1e-8;

/// Standard golf ball mass [kg] (45.93 g).
const DEFAULT_BALL_MASS_KG: f64 = 0.04593;

/// Standard golf ball radius [m].
const DEFAULT_BALL_RADIUS_M: f64 = 0.02135;

fn default_ball_mass_kg() -> f64 {
    DEFAULT_BALL_MASS_KG
}

fn default_ball_radius_m() -> f64 {
    DEFAULT_BALL_RADIUS_M
}

/// Solid-sphere moment of inertia factor \(I = \frac{2}{5} m r^2\).
const SPHERE_INERTIA_FACTOR: f64 = 2.0 / 5.0;

/// Parameters for a contact surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ContactParameters {
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    pub cor: f64,
    /// Kinetic friction coefficient.
    pub friction: f64,
    /// Surface normal vector (unit vector pointing away from surface).
    pub normal: Vector3,
    /// Ball mass [kg].
    #[serde(default = "default_ball_mass_kg")]
    pub mass: f64,
    /// Ball radius [m].
    #[serde(default = "default_ball_radius_m")]
    pub radius: f64,
}

impl Default for ContactParameters {
    fn default() -> Self {
        Self {
            cor: 0.78,                           // Typical golf green COR
            friction: 0.4,                       // Typical grass friction
            normal: Vector3::new(0.0, 0.0, 1.0), // Flat ground (Z-up)
            mass: DEFAULT_BALL_MASS_KG,
            radius: DEFAULT_BALL_RADIUS_M,
        }
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl ContactParameters {
    #[new]
    #[pyo3(signature = (cor=0.78, friction=0.4, mass=DEFAULT_BALL_MASS_KG, radius=DEFAULT_BALL_RADIUS_M))]
    fn py_new(cor: f64, friction: f64, mass: f64, radius: f64) -> pyo3::PyResult<Self> {
        if !(0.0..=1.0).contains(&cor) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "COR must be in [0, 1], got {cor}"
            )));
        }
        if friction < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Friction must be non-negative, got {friction}"
            )));
        }
        if !mass.is_finite() || mass <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mass must be positive and finite, got {mass}"
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "radius must be positive and finite, got {radius}"
            )));
        }
        Ok(Self {
            cor,
            friction,
            normal: Vector3::new(0.0, 0.0, 1.0),
            mass,
            radius,
        })
    }
}

/// Result of a contact calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct ContactResult {
    /// Post-impact velocity.
    pub velocity: Vector3,
    /// Post-impact spin rate [rad/s] about the spin axis used for coupling.
    pub spin_rate: f64,
    /// Energy lost during impact [J] (translational plus rotational).
    pub energy_lost: f64,
    /// Whether the ball is rolling (speed below threshold).
    pub is_rolling: bool,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl ContactResult {
    #[getter]
    fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }
}

fn sphere_moment_of_inertia(mass: f64, radius: f64) -> f64 {
    SPHERE_INERTIA_FACTOR * mass * radius * radius
}

/// Effective mass for tangential impulse at the contact point of a solid sphere.
fn tangential_effective_mass(mass: f64, radius: f64, inertia: f64) -> f64 {
    1.0 / (1.0 / mass + (radius * radius) / inertia)
}

fn reference_tangent(normal: &Vector3) -> Vector3 {
    let mut reference = Vector3::new(1.0, 0.0, 0.0);
    if reference.dot(normal).abs() > 0.9 {
        reference = Vector3::new(0.0, 0.0, 1.0);
    }
    let projected = reference - *normal * reference.dot(normal);
    let mag = projected.magnitude();
    if mag < MIN_NORMAL_MAGNITUDE {
        return Vector3::new(0.0, 0.0, 0.0);
    }
    projected.scale(1.0 / mag)
}

fn primary_tangent(v_tangential: &Vector3, normal: &Vector3) -> Vector3 {
    let tangential_speed = v_tangential.magnitude();
    if tangential_speed >= MIN_TANGENTIAL_SPEED {
        return v_tangential.scale(1.0 / tangential_speed);
    }
    reference_tangent(normal)
}

fn spin_axis(normal: &Vector3, tangent: &Vector3) -> Vector3 {
    let axis = normal.cross(tangent);
    let mag = axis.magnitude();
    if mag < MIN_NORMAL_MAGNITUDE {
        return Vector3::new(0.0, 0.0, 0.0);
    }
    axis.scale(1.0 / mag)
}

fn translational_rotational_energy(mass: f64, velocity: &Vector3, inertia: f64, spin: f64) -> f64 {
    let speed = velocity.magnitude();
    0.5 * mass * speed * speed + 0.5 * inertia * spin * spin
}

fn unchanged_impact(velocity: &Vector3, spin_rate: f64) -> ContactResult {
    let speed = velocity.magnitude();
    ContactResult {
        velocity: *velocity,
        spin_rate,
        energy_lost: 0.0,
        is_rolling: speed < ROLLING_SPEED_THRESHOLD,
    }
}

struct ImpactState {
    velocity: Vector3,
    spin_rate: f64,
    energy: f64,
}

struct FrictionImpactContext<'a> {
    v_normal_out: &'a Vector3,
    v_tangential: &'a Vector3,
    t_hat: &'a Vector3,
    spin_dir: &'a Vector3,
    r_contact: &'a Vector3,
    spin_rate: f64,
    mass: f64,
    inertia: f64,
}

fn apply_friction_impulse(ctx: &FrictionImpactContext<'_>, friction_impulse: f64) -> ImpactState {
    let v_out =
        *ctx.v_normal_out + (*ctx.v_tangential + *ctx.t_hat * (friction_impulse / ctx.mass));
    let torque = ctx.r_contact.cross(&(*ctx.t_hat * friction_impulse));
    let spin_out = ctx.spin_rate + torque.dot(ctx.spin_dir) / ctx.inertia;
    let energy = translational_rotational_energy(ctx.mass, &v_out, ctx.inertia, spin_out);
    ImpactState {
        velocity: v_out,
        spin_rate: spin_out,
        energy,
    }
}

fn dissipative_friction_impulse(
    candidate: f64,
    ctx: &FrictionImpactContext<'_>,
    energy_in: f64,
) -> f64 {
    if candidate.abs() < MIN_TANGENTIAL_SPEED {
        return 0.0;
    }
    let full = apply_friction_impulse(ctx, candidate);
    if full.energy <= energy_in + 1e-12 {
        return candidate;
    }

    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..48 {
        let mid = 0.5 * (low + high);
        let trial = apply_friction_impulse(ctx, candidate * mid);
        if trial.energy <= energy_in + 1e-12 {
            low = mid;
        } else {
            high = mid;
        }
    }
    candidate * low
}

/// Calculate the post-impact state of a ball hitting a surface.
///
/// Uses COR for the normal direction, Coulomb friction capped by the normal
/// impulse, and couples tangential slip with scalar spin about
/// `normalize(normal × tangential)`.
///
/// # Arguments
/// * `velocity` - Incoming velocity vector [m/s]
/// * `spin_rate` - Incoming spin rate [rad/s] about the coupled spin axis
/// * `params` - Surface contact parameters (includes ball mass and radius)
///
/// # Contracts (DbC)
/// - Precondition: `params.cor` in [0, 1]
/// - Precondition: `params.friction` >= 0
/// - Postcondition: no impulse when `velocity · normal >= 0` (receding contact)
/// - Postcondition: `energy_lost` >= 0
pub fn calculate_impact(
    velocity: &Vector3,
    spin_rate: f64,
    params: &ContactParameters,
) -> ContactResult {
    let speed = velocity.magnitude();

    debug_assert!(
        params.cor >= 0.0 && params.cor <= 1.0,
        "COR must be in [0, 1]"
    );
    debug_assert!(params.friction >= 0.0, "Friction must be non-negative");
    debug_assert!(params.mass > 0.0, "mass must be positive");
    debug_assert!(params.radius > 0.0, "radius must be positive");

    if speed < ROLLING_SPEED_THRESHOLD {
        return ContactResult {
            velocity: *velocity,
            spin_rate: 0.0,
            energy_lost: 0.0,
            is_rolling: true,
        };
    }

    let normal_mag = params.normal.magnitude();
    if normal_mag < MIN_NORMAL_MAGNITUDE {
        return unchanged_impact(velocity, spin_rate);
    }
    let n = params.normal.scale(1.0 / normal_mag);

    let v_dot_n = velocity.dot(&n);
    if v_dot_n >= 0.0 {
        return unchanged_impact(velocity, spin_rate);
    }

    let inertia = sphere_moment_of_inertia(params.mass, params.radius);
    let energy_in = translational_rotational_energy(params.mass, velocity, inertia, spin_rate);

    let v_normal = n * v_dot_n;
    let v_tangential = *velocity - v_normal;
    let t_hat = primary_tangent(&v_tangential, &n);
    let spin_dir = spin_axis(&n, &t_hat);

    let r_contact = n * (-params.radius);
    let spin_vector = spin_dir * spin_rate;
    let contact_velocity = spin_vector.cross(&r_contact);
    let slip_scalar = v_tangential.dot(&t_hat) + contact_velocity.dot(&t_hat);

    let normal_impulse = params.mass * (1.0 + params.cor) * (-v_dot_n);
    let max_friction = params.friction * normal_impulse;
    let effective_mass = tangential_effective_mass(params.mass, params.radius, inertia);
    let candidate_friction = if slip_scalar.abs() < MIN_TANGENTIAL_SPEED {
        0.0
    } else {
        let desired = -slip_scalar * effective_mass;
        desired.clamp(-max_friction, max_friction)
    };

    let v_normal_out = v_normal * (-params.cor);
    let friction_ctx = FrictionImpactContext {
        v_normal_out: &v_normal_out,
        v_tangential: &v_tangential,
        t_hat: &t_hat,
        spin_dir: &spin_dir,
        r_contact: &r_contact,
        spin_rate,
        mass: params.mass,
        inertia,
    };
    let friction_impulse =
        dissipative_friction_impulse(candidate_friction, &friction_ctx, energy_in);
    let post = apply_friction_impulse(&friction_ctx, friction_impulse);
    let v_out = post.velocity;
    let spin_out = post.spin_rate;
    let energy_out = post.energy;
    let energy_lost = energy_in - energy_out;

    debug_assert!(
        energy_lost >= -1e-10,
        "DbC postcondition: energy_lost must be non-negative, got {energy_lost}"
    );

    let speed_out = v_out.magnitude();
    ContactResult {
        velocity: v_out,
        spin_rate: spin_out,
        energy_lost,
        is_rolling: speed_out < ROLLING_SPEED_THRESHOLD,
    }
}

// ── WASM bindings ────────────────────────────────────────────────────────────
// ContactParameters and ContactResult contain Vector3 (cross-crate),
// so we use serde-wasm-bindgen for JS ↔ Rust conversion instead of
// struct-level #[wasm_bindgen] derives.

/// WASM-exposed contact impact calculation.
#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = "calculateImpact")]
pub fn wasm_calculate_impact(
    velocity_js: wasm_bindgen::JsValue,
    spin_rate: f64,
    params_js: wasm_bindgen::JsValue,
) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue> {
    let velocity: Vector3 = serde_wasm_bindgen::from_value(velocity_js)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Invalid velocity: {e}")))?;
    let params: ContactParameters = serde_wasm_bindgen::from_value(params_js)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Invalid params: {e}")))?;

    let result = calculate_impact(&velocity, spin_rate, &params);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Serialization error: {e}")))
}

// ── Tests (TDD) ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MASS: f64 = DEFAULT_BALL_MASS_KG;
    const TEST_RADIUS: f64 = DEFAULT_BALL_RADIUS_M;

    fn test_inertia() -> f64 {
        sphere_moment_of_inertia(TEST_MASS, TEST_RADIUS)
    }

    fn params_with_normal(cor: f64, friction: f64, normal: Vector3) -> ContactParameters {
        ContactParameters {
            cor,
            friction,
            normal,
            mass: TEST_MASS,
            radius: TEST_RADIUS,
        }
    }

    fn spin_axis_momentum_scalar(
        velocity: &Vector3,
        spin: f64,
        normal: &Vector3,
        tangent: &Vector3,
    ) -> f64 {
        let spin_dir = spin_axis(normal, tangent);
        let r_contact_to_center = *normal * TEST_RADIUS;
        let orbital = r_contact_to_center.cross(&(*velocity * TEST_MASS));
        orbital.dot(&spin_dir) + test_inertia() * spin
    }

    fn expected_energy_lost_joules(
        v_in: &Vector3,
        v_out: &Vector3,
        spin_in: f64,
        spin_out: f64,
    ) -> f64 {
        let inertia = test_inertia();
        let e_in = translational_rotational_energy(TEST_MASS, v_in, inertia, spin_in);
        let e_out = translational_rotational_energy(TEST_MASS, v_out, inertia, spin_out);
        e_in - e_out
    }

    #[test]
    fn test_default_contact_parameters_use_z_up_normal() {
        let params = ContactParameters::default();

        assert_eq!(params.normal.x, 0.0);
        assert_eq!(params.normal.y, 0.0);
        assert_eq!(params.normal.z, 1.0);
        assert_eq!(params.mass, TEST_MASS);
        assert_eq!(params.radius, TEST_RADIUS);
    }

    #[test]
    fn test_default_contact_reflects_descending_z_velocity() {
        let params = ContactParameters::default();
        let v_in = Vector3::new(1.0, 0.0, -10.0);

        let result = calculate_impact(&v_in, 0.0, &params);

        assert!(
            result.velocity.z > 0.0,
            "default contact normal should reflect descending z velocity, got {}",
            result.velocity.z
        );
    }

    #[test]
    fn test_receding_contact_applies_no_impulse() {
        let params = params_with_normal(0.78, 0.4, Vector3::new(0.0, 1.0, 0.0));
        let v_in = Vector3::new(2.0, 10.0, 0.0);

        let result = calculate_impact(&v_in, 5000.0, &params);

        assert_eq!(result.velocity.x, v_in.x);
        assert_eq!(result.velocity.y, v_in.y);
        assert_eq!(result.velocity.z, v_in.z);
        assert_eq!(result.spin_rate, 5000.0);
        assert_eq!(result.energy_lost, 0.0);
    }

    #[test]
    fn test_elastic_bounce() {
        let params = params_with_normal(1.0, 0.0, Vector3::new(0.0, 1.0, 0.0));

        let v_in = Vector3::new(0.0, -10.0, 0.0);
        let result = calculate_impact(&v_in, 0.0, &params);

        assert!(
            (result.velocity.y - 10.0).abs() < 1e-6,
            "Expected vy=10.0, got {}",
            result.velocity.y
        );
        assert!(
            result.energy_lost.abs() < 1e-9,
            "No energy should be lost for elastic head-on bounce"
        );
    }

    #[test]
    fn test_inelastic_bounce() {
        let params = params_with_normal(0.0, 0.0, Vector3::new(0.0, 1.0, 0.0));

        let v_in = Vector3::new(5.0, -10.0, 0.0);
        let spin_in = 100.0;
        let result = calculate_impact(&v_in, spin_in, &params);

        assert!(
            result.velocity.y.abs() < 1e-6,
            "Normal velocity should be zero"
        );
        assert!(
            (result.velocity.x - 5.0).abs() < 1e-6,
            "Tangential velocity preserved without friction"
        );
        assert!(
            (result.spin_rate - spin_in).abs() < 1e-6,
            "Spin unchanged without friction"
        );
    }

    #[test]
    fn test_friction_reduces_tangential_physics() {
        let params = params_with_normal(0.8, 0.5, Vector3::new(0.0, 1.0, 0.0));

        let v_in = Vector3::new(10.0, -5.0, 0.0);
        let result = calculate_impact(&v_in, 0.0, &params);

        let normal_impulse = TEST_MASS * (1.0 + params.cor) * 5.0;
        let max_friction = params.friction * normal_impulse;
        let effective_mass = tangential_effective_mass(TEST_MASS, TEST_RADIUS, test_inertia());
        let desired_friction = -10.0 * effective_mass;
        let friction_impulse = desired_friction.clamp(-max_friction, max_friction);
        let expected_vx = 10.0 + friction_impulse / TEST_MASS;
        let expected_vy = -params.cor * v_in.y;

        assert!(
            (result.velocity.x - expected_vx).abs() < 1e-9,
            "Expected vx={expected_vx}, got {}",
            result.velocity.x
        );
        assert!(
            (result.velocity.y - expected_vy).abs() < 1e-9,
            "Expected vy={expected_vy}, got {}",
            result.velocity.y
        );
    }

    #[test]
    fn test_friction_couples_spin_with_tangential_slip() {
        let params = params_with_normal(0.8, 0.35, Vector3::new(0.0, 1.0, 0.0));
        let v_in = Vector3::new(0.0, -8.0, 0.0);
        let spin_in = 2500.0;

        let result = calculate_impact(&v_in, spin_in, &params);

        let n = Vector3::new(0.0, 1.0, 0.0);
        let t_hat = primary_tangent(&Vector3::zero(), &n);
        let l_before = spin_axis_momentum_scalar(&v_in, spin_in, &n, &t_hat);
        let l_after = spin_axis_momentum_scalar(&result.velocity, result.spin_rate, &n, &t_hat);

        assert!(
            result.velocity.x.abs() > 1e-6,
            "Pure-spin slip should drive tangential rebound"
        );
        assert!(
            (result.spin_rate - spin_in).abs() > 1e-6,
            "Spin must change when friction opposes spin-generated slip"
        );
        assert!(
            (l_before - l_after).abs() < 1e-7,
            "Coupled friction must conserve spin-axis momentum about contact"
        );
    }

    #[test]
    fn test_typical_green_impact() {
        let params = ContactParameters::default();

        let v_in = Vector3::new(15.0, 0.0, -8.0);
        let result = calculate_impact(&v_in, 3000.0, &params);

        assert!(
            result.velocity.z > 0.0,
            "Should rebound off the z-up surface"
        );
        assert!(result.energy_lost > 0.0);
    }

    #[test]
    fn test_rolling_threshold() {
        let params = ContactParameters::default();

        let v_in = Vector3::new(0.3, -0.1, 0.0);
        let result = calculate_impact(&v_in, 100.0, &params);

        assert!(result.is_rolling, "Ball should be rolling at low speed");
    }

    #[test]
    fn test_energy_non_negative() {
        let params = params_with_normal(0.6, 0.3, Vector3::new(0.0, 1.0, 0.0));

        let v_in = Vector3::new(20.0, -15.0, 5.0);
        let result = calculate_impact(&v_in, 5000.0, &params);

        assert!(
            result.energy_lost >= 0.0,
            "Energy lost must be non-negative"
        );
    }

    #[test]
    fn test_angled_surface() {
        let angle = std::f64::consts::FRAC_PI_6;
        let params = params_with_normal(0.8, 0.2, Vector3::new(angle.sin(), angle.cos(), 0.0));

        let v_in = Vector3::new(0.0, -10.0, 0.0);
        let result = calculate_impact(&v_in, 0.0, &params);

        assert!(
            result.velocity.x.abs() > 0.1,
            "Should deflect on angled surface"
        );
    }

    #[test]
    #[should_panic(expected = "COR must be in [0, 1]")]
    fn test_dbc_cor_out_of_range() {
        let params = params_with_normal(1.5, 0.0, Vector3::new(0.0, 1.0, 0.0));
        let _r = calculate_impact(&Vector3::new(0.0, -10.0, 0.0), 0.0, &params);
    }

    #[test]
    #[should_panic(expected = "Friction must be non-negative")]
    fn test_dbc_negative_friction() {
        let params = params_with_normal(0.5, -0.1, Vector3::new(0.0, 1.0, 0.0));
        let _r = calculate_impact(&Vector3::new(0.0, -10.0, 0.0), 0.0, &params);
    }

    #[test]
    fn test_energy_lost_is_joules_including_rotation() {
        let params = params_with_normal(0.8, 0.3, Vector3::new(0.0, 1.0, 0.0));

        let v_in = Vector3::new(20.0, -15.0, 5.0);
        let spin_in = 5000.0;
        let result = calculate_impact(&v_in, spin_in, &params);

        let expected =
            expected_energy_lost_joules(&v_in, &result.velocity, spin_in, result.spin_rate);

        assert!(
            (result.energy_lost - expected).abs() < 1e-9,
            "energy_lost must be full mechanical energy change in joules"
        );
        assert!(
            result.energy_lost > TEST_MASS * 0.5 * v_in.magnitude().powi(2) * 0.01,
            "Scaled joule loss must exceed the old per-kg mistake"
        );
    }

    #[test]
    fn test_angular_momentum_coupling_about_contact() {
        let params = params_with_normal(0.75, 0.4, Vector3::new(0.0, 1.0, 0.0));
        let v_in = Vector3::new(6.0, -12.0, 0.0);
        let spin_in = 1800.0;
        let inertia = test_inertia();

        let result = calculate_impact(&v_in, spin_in, &params);
        let n = Vector3::new(0.0, 1.0, 0.0);
        let v_tangential = Vector3::new(v_in.x, 0.0, 0.0);
        let t_hat = primary_tangent(&v_tangential, &n);
        let l_before = spin_axis_momentum_scalar(&v_in, spin_in, &n, &t_hat);
        let l_after = spin_axis_momentum_scalar(&result.velocity, result.spin_rate, &n, &t_hat);
        assert!(
            (l_before - l_after).abs() < 1e-7,
            "Coupled impulse must conserve angular momentum about contact"
        );
        let _ = inertia;
    }
}
