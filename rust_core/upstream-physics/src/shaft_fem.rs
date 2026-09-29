//! Flexible shaft finite-element primitives.
//!
//! This module is the first Rust slice for the Python
//! `FiniteElementShaftModel` migration. It keeps the scope to local
//! Euler-Bernoulli beam element math so later slices can assemble global
//! matrices, add Newmark integration, and wire PyO3 without changing these
//! tested primitives.
//!
//! # Design by Contract
//! - Element length must be finite and positive.
//! - Bending stiffness `EI` must be finite and positive.
//! - Linear mass density must be finite and non-negative.

use serde::{Deserialize, Serialize};

/// Dense 4x4 beam element matrix.
pub type BeamMatrix4 = [[f64; 4]; 4];

/// Euler-Bernoulli beam element parameters for shaft bending.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BeamElement {
    /// Start node index in the global shaft mesh.
    pub node_i: usize,
    /// End node index in the global shaft mesh.
    pub node_j: usize,
    /// Element length [m].
    pub length: f64,
    /// Bending stiffness [N*m^2].
    pub ei: f64,
    /// Linear mass density [kg/m].
    pub mass_per_length: f64,
    /// Structural damping ratio or coefficient carried with the element.
    pub damping: f64,
}

impl BeamElement {
    /// Create an element after validating public preconditions.
    pub fn try_new(
        node_i: usize,
        node_j: usize,
        length: f64,
        ei: f64,
        mass_per_length: f64,
        damping: f64,
    ) -> Result<Self, &'static str> {
        let element = Self {
            node_i,
            node_j,
            length,
            ei,
            mass_per_length,
            damping,
        };
        element.validate()?;
        Ok(element)
    }

    /// Validate public element invariants in all build modes.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.node_i == self.node_j {
            return Err("BeamElement nodes must be distinct");
        }
        if !self.length.is_finite() || self.length <= 0.0 {
            return Err("BeamElement.length must be finite and positive");
        }
        if !self.ei.is_finite() || self.ei <= 0.0 {
            return Err("BeamElement.ei must be finite and positive");
        }
        if !self.mass_per_length.is_finite() || self.mass_per_length < 0.0 {
            return Err("BeamElement.mass_per_length must be finite and non-negative");
        }
        if !self.damping.is_finite() {
            return Err("BeamElement.damping must be finite");
        }
        Ok(())
    }
}

/// Compute the 4x4 Euler-Bernoulli element stiffness matrix.
///
/// Matches `FiniteElementShaftModel._element_stiffness_matrix`:
///
/// `K_e = EI/L^3 * [[12, 6L, -12, 6L], ...]`
pub fn element_stiffness_matrix(element: &BeamElement) -> Result<BeamMatrix4, &'static str> {
    element.validate()?;

    let l = element.length;
    let l2 = l * l;
    let l3 = l2 * l;
    let scale = element.ei / l3;

    Ok(scale_matrix(
        [
            [12.0, 6.0 * l, -12.0, 6.0 * l],
            [6.0 * l, 4.0 * l2, -6.0 * l, 2.0 * l2],
            [-12.0, -6.0 * l, 12.0, -6.0 * l],
            [6.0 * l, 2.0 * l2, -6.0 * l, 4.0 * l2],
        ],
        scale,
    ))
}

/// Compute the 4x4 consistent Euler-Bernoulli element mass matrix.
///
/// Matches `FiniteElementShaftModel._element_mass_matrix`:
///
/// `M_e = mu*L/420 * [[156, 22L, 54, -13L], ...]`
pub fn element_mass_matrix(element: &BeamElement) -> Result<BeamMatrix4, &'static str> {
    element.validate()?;

    let l = element.length;
    let l2 = l * l;
    let scale = element.mass_per_length * l / 420.0;

    Ok(scale_matrix(
        [
            [156.0, 22.0 * l, 54.0, -13.0 * l],
            [22.0 * l, 4.0 * l2, 13.0 * l, -3.0 * l2],
            [54.0, 13.0 * l, 156.0, -22.0 * l],
            [-13.0 * l, -3.0 * l2, -22.0 * l, 4.0 * l2],
        ],
        scale,
    ))
}

fn scale_matrix(mut matrix: BeamMatrix4, scale: f64) -> BeamMatrix4 {
    for row in &mut matrix {
        for value in row {
            *value *= scale;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample_element() -> BeamElement {
        BeamElement::try_new(0, 1, 0.25, 3.2, 0.11, 0.02).expect("valid sample element")
    }

    #[test]
    fn stiffness_matrix_matches_python_formula() {
        let matrix = element_stiffness_matrix(&sample_element()).expect("stiffness matrix");

        let expected = [
            [2457.6, 307.2, -2457.6, 307.2],
            [307.2, 51.2, -307.2, 25.6],
            [-2457.6, -307.2, 2457.6, -307.2],
            [307.2, 25.6, -307.2, 51.2],
        ];

        assert_matrix_close(matrix, expected);
    }

    #[test]
    fn mass_matrix_matches_python_formula() {
        let matrix = element_mass_matrix(&sample_element()).expect("mass matrix");

        let expected = [
            [
                0.010214285714285714,
                0.00036011904761904764,
                0.003535714285714286,
                -0.00021279761904761904,
            ],
            [
                0.00036011904761904764,
                0.00001636904761904762,
                0.00021279761904761904,
                -0.000012276785714285716,
            ],
            [
                0.003535714285714286,
                0.00021279761904761904,
                0.010214285714285714,
                -0.00036011904761904764,
            ],
            [
                -0.00021279761904761904,
                -0.000012276785714285716,
                -0.00036011904761904764,
                0.00001636904761904762,
            ],
        ];

        assert_matrix_close(matrix, expected);
    }

    #[test]
    fn matrices_are_symmetric() {
        let element = sample_element();

        assert_symmetric(element_stiffness_matrix(&element).expect("stiffness matrix"));
        assert_symmetric(element_mass_matrix(&element).expect("mass matrix"));
    }

    #[test]
    fn rejects_invalid_element_contracts() {
        assert!(BeamElement::try_new(0, 0, 0.25, 3.2, 0.11, 0.02).is_err());
        assert!(BeamElement::try_new(0, 1, 0.0, 3.2, 0.11, 0.02).is_err());
        assert!(BeamElement::try_new(0, 1, 0.25, -1.0, 0.11, 0.02).is_err());
        assert!(BeamElement::try_new(0, 1, 0.25, 3.2, -0.01, 0.02).is_err());
        assert!(BeamElement::try_new(0, 1, 0.25, 3.2, 0.11, f64::NAN).is_err());
    }

    fn assert_matrix_close(actual: BeamMatrix4, expected: BeamMatrix4) {
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                assert_relative_eq!(actual_value, expected_value, epsilon = 1e-12);
            }
        }
    }

    fn assert_symmetric(matrix: BeamMatrix4) {
        for (row_index, row_values) in matrix.iter().enumerate() {
            for (col_index, value) in row_values.iter().enumerate() {
                assert_relative_eq!(*value, matrix[col_index][row_index], epsilon = 1e-12);
            }
        }
    }
}
