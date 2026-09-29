use approx::assert_relative_eq;
use upstream_muscle::{ActivationDynamics, HillMuscleModel, MuscleParameters, MuscleState};

#[test]
fn activation_update_matches_python_formula() {
    let dynamics = ActivationDynamics::new(0.010, 0.040, 0.001).expect("valid dynamics");

    let cases = [
        (1.0, 0.001, 0.001, 0.20020239282153543),
        (0.2, 0.6, 0.005, 0.53),
        (0.0, 0.2, 0.010, 0.1602),
        (0.9, 0.8, 0.002, 0.8117647058823529),
    ];

    for (u, a, dt, expected) in cases {
        let actual = dynamics.update(u, a, dt).expect("positive dt");
        assert_relative_eq!(actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn activation_rejects_invalid_contract_inputs() {
    assert!(ActivationDynamics::new(0.0, 0.040, 0.001).is_err());
    assert!(ActivationDynamics::new(0.010, -1.0, 0.001).is_err());
    assert!(ActivationDynamics::new(0.010, 0.040, 1.0).is_err());

    let dynamics = ActivationDynamics::default();
    assert!(dynamics.update(1.0, 0.0, 0.0).is_err());
    assert!(dynamics.update_batch(&[1.0], &[0.1, 0.2], 0.001).is_err());
}

#[test]
fn hill_model_compute_force_matches_python_formula() {
    let params = MuscleParameters::new(1000.0, 0.15, 0.20, 10.0, 0.1, 0.05).expect("valid params");
    let model = HillMuscleModel::new(params, None);
    let state = MuscleState {
        activation: 0.8,
        l_ce: 0.16,
        v_ce: -0.5,
        l_mt: 0.35,
    };

    let actual = model.compute_force(&state).expect("valid activation");
    assert_relative_eq!(actual, 229.87747417119178, epsilon = 1e-10);
}

#[test]
fn hill_model_batch_matches_scalar_results() {
    let params = MuscleParameters::new(1000.0, 0.15, 0.20, 10.0, 0.1, 0.05).expect("valid params");
    let model = HillMuscleModel::new(params, None);
    let states = vec![
        MuscleState {
            activation: 0.2,
            l_ce: 0.15,
            v_ce: 0.0,
            l_mt: 0.35,
        },
        MuscleState {
            activation: 0.8,
            l_ce: 0.16,
            v_ce: -0.5,
            l_mt: 0.35,
        },
    ];

    let batch = model.compute_force_batch(&states).expect("valid batch");

    assert_eq!(batch.len(), states.len());
    for (state, actual) in states.iter().zip(batch.iter()) {
        let expected = model.compute_force(state).expect("valid scalar");
        assert_relative_eq!(*actual, expected, epsilon = 1e-12);
    }
}

#[test]
fn hill_model_rejects_invalid_contract_inputs() {
    assert!(MuscleParameters::new(0.0, 0.15, 0.20, 10.0, 0.0, 0.05).is_err());
    assert!(MuscleParameters::new(1000.0, 0.0, 0.20, 10.0, 0.0, 0.05).is_err());
    assert!(MuscleParameters::new(1000.0, 0.15, -1.0, 10.0, 0.0, 0.05).is_err());

    let params = MuscleParameters::new(1000.0, 0.15, 0.20, 10.0, 0.0, 0.05).expect("valid params");
    let model = HillMuscleModel::new(params, None);
    let state = MuscleState {
        activation: 1.1,
        l_ce: 0.15,
        v_ce: 0.0,
        l_mt: 0.35,
    };
    assert!(model.compute_force(&state).is_err());
}
